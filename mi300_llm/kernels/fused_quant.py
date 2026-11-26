"""Per-token FP8 activation quantization, fused with the op that produces it.

Every FP8 GEMM input is quantized per token. Done as a separate pass, that is
one more read and write of the activation and one more launch per GEMM; and a
single program per row runs the whole K=28672 row of the down-proj input on
one CU at batch 1 (13.9 us of a ~294 us layer). Here:

- ``rmsnorm_quant`` / ``fused_add_rmsnorm_quant``: the normalized row is
  quantized in registers; the bf16 normalized output is never written.
- ``silu_mul_quant`` and ``quantize_per_token``: two stages split across
  column blocks, so a single row uses many CUs: stage 1 writes per-block
  absolute maxima, stage 2 reduces them, then quantizes its block.

The fused value is rounded to the activation dtype before quantizing, as
the unfused path does. SiLU-mul + quant and the two-stage quant are
bit-identical to the unfused kernels. RMSNorm + quant is identical except for
about 1 in 5e6 codes, off by one FP8 step: its different register layout
reorders the row sum (see ``norm_rope.add_rmsnorm_row``).

All divisions use ``tl.div_rn`` (IEEE round-to-nearest) so codes match a
float64 reference exactly (verified on gfx942). A 1-ulp error in the scale
pushes values that sit just below a rounding midpoint (12.499999) over it;
that is what tripped torch's tensor / scalar division on ROCm (see
``fp8.quantize_per_token``). The fp32 -> fp8 cast itself is
round-to-nearest-even and bit-identical to torch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .norm_rope import add_rmsnorm_row

QUANT_EPS = 1e-12


@triton.jit
def _rmsnorm_quant_kernel(
    x_ptr, res_ptr, w_ptr, q_ptr, s_ptr, res_out_ptr,
    N, stride_x, stride_r, stride_q, stride_ro, eps, fp8_max, q_eps,
    HAS_RES: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    y = add_rmsnorm_row(x_ptr, res_ptr, w_ptr, res_out_ptr, row, cols, mask, N, stride_x, stride_r, stride_ro, eps, HAS_RES)
    y = y.to(x_ptr.dtype.element_ty).to(tl.float32)  # same rounding as the unfused output
    scale = tl.div_rn(tl.maximum(tl.max(tl.abs(y), axis=0), q_eps), fp8_max)
    tl.store(s_ptr + row, scale)
    q = tl.clamp(tl.div_rn(y, scale), -fp8_max, fp8_max)
    tl.store(q_ptr + row * stride_q + cols, q.to(q_ptr.dtype.element_ty), mask=mask)


def _rmsnorm_quant(x, residual, weight, eps, fp8_dtype):
    assert x.dim() == 2 and x.stride(-1) == 1
    M, N = x.shape
    q = torch.empty((M, N), dtype=fp8_dtype, device=x.device)
    s = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    res_out = torch.empty_like(x) if residual is not None else x
    BLOCK = triton.next_power_of_2(N)
    _rmsnorm_quant_kernel[(M,)](
        x, residual if residual is not None else x, weight, q, s, res_out,
        N, x.stride(0), residual.stride(0) if residual is not None else 0, q.stride(0), res_out.stride(0),
        eps, float(torch.finfo(fp8_dtype).max), QUANT_EPS,
        HAS_RES=residual is not None, BLOCK=BLOCK, num_warps=8 if BLOCK >= 4096 else 4,
    )
    return q, s, res_out


def rmsnorm_quant(x, weight, eps, fp8_dtype):
    """Returns (q, scale) of rmsnorm(x)."""
    q, s, _ = _rmsnorm_quant(x, None, weight, eps, fp8_dtype)
    return q, s


def fused_add_rmsnorm_quant(x, residual, weight, eps, fp8_dtype):
    """Returns (q, scale, x + residual) of rmsnorm(x + residual)."""
    return _rmsnorm_quant(x, residual, weight, eps, fp8_dtype)


@triton.jit
def _load_act(x_ptr, row, cols, K, stride_x, SILU: tl.constexpr):
    """Row ``row``, columns ``cols`` of the value to quantize, in fp32, rounded
    through the activation dtype exactly like the unfused kernels' output."""
    mask = cols < K
    base = x_ptr + row * stride_x
    if SILU:
        g = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(base + K + cols, mask=mask, other=0.0).to(tl.float32)
        return (g * tl.sigmoid(g) * u).to(x_ptr.dtype.element_ty).to(tl.float32)
    return tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)


@triton.jit
def _block_amax_kernel(x_ptr, part_ptr, K, stride_x, nblk, SILU: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    y = _load_act(x_ptr, row, cols, K, stride_x, SILU)
    tl.store(part_ptr + row * nblk + blk, tl.max(tl.abs(y), axis=0))


@triton.jit
def _block_quant_kernel(
    x_ptr, part_ptr, q_ptr, s_ptr, K, stride_x, stride_q, nblk, fp8_max, q_eps,
    SILU: tl.constexpr, BLOCK: tl.constexpr, BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    p = tl.arange(0, BLOCK_P)
    amax = tl.max(tl.load(part_ptr + row * nblk + p, mask=p < nblk, other=0.0), axis=0)
    scale = tl.div_rn(tl.maximum(amax, q_eps), fp8_max)
    if blk == 0:
        tl.store(s_ptr + row, scale)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    y = _load_act(x_ptr, row, cols, K, stride_x, SILU)
    q = tl.clamp(tl.div_rn(y, scale), -fp8_max, fp8_max)
    tl.store(q_ptr + row * stride_q + cols, q.to(q_ptr.dtype.element_ty), mask=cols < K)


def _two_stage_quant(x, K, fp8_dtype, silu, block=1024):
    assert x.dim() == 2 and x.stride(-1) == 1
    M = x.shape[0]
    nblk = triton.cdiv(K, block)
    part = torch.empty((M, nblk), dtype=torch.float32, device=x.device)
    q = torch.empty((M, K), dtype=fp8_dtype, device=x.device)
    s = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    _block_amax_kernel[(M, nblk)](x, part, K, x.stride(0), nblk, SILU=silu, BLOCK=block)
    _block_quant_kernel[(M, nblk)](
        x, part, q, s, K, x.stride(0), q.stride(0), nblk, float(torch.finfo(fp8_dtype).max), QUANT_EPS,
        SILU=silu, BLOCK=block, BLOCK_P=triton.next_power_of_2(nblk),
    )
    return q, s


def quantize_per_token(x: torch.Tensor, fp8_dtype: torch.dtype):
    """Returns (q [M, K], scale [M, 1] fp32)."""
    return _two_stage_quant(x, x.shape[1], fp8_dtype, silu=False)


def silu_mul_quant(gate_up: torch.Tensor, fp8_dtype: torch.dtype):
    """Returns (q, scale) of silu(gate) * up for gate_up = [gate | up]."""
    return _two_stage_quant(gate_up, gate_up.shape[1] // 2, fp8_dtype, silu=True)
