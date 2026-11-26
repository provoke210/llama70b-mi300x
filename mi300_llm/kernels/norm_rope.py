"""Elementwise / row-reduction kernels: RMSNorm (+ fused residual add), RoPE, SiLU-mul.

All of these are memory-bound; the goal is one read and one write per element.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def add_rmsnorm_row(
    x_ptr, res_ptr, w_ptr, res_out_ptr, row, cols, mask, N, stride_x, stride_r, stride_ro, eps,
    HAS_RES: tl.constexpr,
):
    """Normalized row in fp32, before rounding to the activation dtype.

    Shared by the plain and the quantizing RMSNorm kernels. Sharing the source
    does NOT make them bit-identical: Triton picks the register layout from
    the whole kernel, and storing 1-byte FP8 instead of 2-byte bf16 changes
    sizePerThread from 8 to 16 (16-byte accesses). That changes the order of
    ``tl.sum``, so ``rstd`` differs by 1 ulp on ~19% of rows, and about 1 in
    5e6 outputs then lands on the other side of a bf16 rounding midpoint.
    Greedy 70B decoding diverges after ~14 tokens from that alone. Both
    orders are equally accurate; tests allow such one-step code differences.
    """
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(res_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        # Round the new residual to the activation dtype before normalizing,
        # exactly as the reference does.
        new_res = (x + r).to(res_out_ptr.dtype.element_ty)
        tl.store(res_out_ptr + row * stride_ro + cols, new_res, mask=mask)
        x = new_res.to(tl.float32)
    rstd = tl.rsqrt(tl.div_rn(tl.sum(x * x, axis=0), N.to(tl.float32)) + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    return x * rstd * w


@triton.jit
def _rmsnorm_kernel(
    x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr,
    N, stride_x, stride_r, stride_o, stride_ro, eps,
    HAS_RES: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    y = add_rmsnorm_row(x_ptr, res_ptr, w_ptr, res_out_ptr, row, cols, mask, N, stride_x, stride_r, stride_ro, eps, HAS_RES)
    tl.store(out_ptr + row * stride_o + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_rmsnorm(x, residual, weight, eps):
    assert x.stride(-1) == 1 and x.dim() == 2
    M, N = x.shape
    out = torch.empty_like(x)
    res_out = torch.empty_like(x) if residual is not None else out
    BLOCK = triton.next_power_of_2(N)
    _rmsnorm_kernel[(M,)](
        x, residual if residual is not None else x, weight, out, res_out,
        N, x.stride(0), residual.stride(0) if residual is not None else 0,
        out.stride(0), res_out.stride(0), eps,
        HAS_RES=residual is not None, BLOCK=BLOCK, num_warps=8 if BLOCK >= 4096 else 4,
    )
    return out, res_out


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _launch_rmsnorm(x, None, weight, eps)[0]


def fused_add_rmsnorm(x, residual, weight, eps):
    return _launch_rmsnorm(x, residual, weight, eps)


@triton.jit
def _rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    H, stride_xt, stride_xh, stride_ot, stride_oh, stride_ct,
    HALF: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # One program per token, all heads at once: cos/sin are loaded once per token.
    t = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK_H)[:, None]
    d = tl.arange(0, BLOCK_D)[None, :]
    mask = (h < H) & (d < HALF)
    c = tl.load(cos_ptr + t * stride_ct + d, mask=d < HALF, other=0.0)
    s = tl.load(sin_ptr + t * stride_ct + d, mask=d < HALF, other=0.0)
    base = x_ptr + t * stride_xt + h * stride_xh + d
    x1 = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(base + HALF, mask=mask, other=0.0).to(tl.float32)
    obase = out_ptr + t * stride_ot + h * stride_oh + d
    ty = out_ptr.dtype.element_ty
    tl.store(obase, (x1 * c - x2 * s).to(ty), mask=mask)
    tl.store(obase + HALF, (x2 * c + x1 * s).to(ty), mask=mask)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, H, D] (may be a strided view of the fused QKV output)."""
    T, H, D = x.shape
    assert x.stride(-1) == 1 and cos.is_contiguous() and sin.is_contiguous()
    out = torch.empty((T, H, D), dtype=x.dtype, device=x.device)
    half = D // 2
    _rope_kernel[(T,)](
        x, cos, sin, out, H, x.stride(0), x.stride(1), out.stride(0), out.stride(1), cos.stride(0),
        HALF=half, BLOCK_H=triton.next_power_of_2(H), BLOCK_D=triton.next_power_of_2(half),
    )
    return out


@triton.jit
def _silu_mul_kernel(x_ptr, out_ptr, N, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    g = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * stride_x + N + cols, mask=mask, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    tl.store(out_ptr + row * stride_o + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    M, N2 = gate_up.shape
    N = N2 // 2
    out = torch.empty((M, N), dtype=gate_up.dtype, device=gate_up.device)
    BLOCK = 1024
    _silu_mul_kernel[(M, triton.cdiv(N, BLOCK))](gate_up, out, N, gate_up.stride(0), out.stride(0), BLOCK=BLOCK)
    return out
