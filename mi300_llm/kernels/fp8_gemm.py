"""W8A8 FP8 linear: dynamic per-token activation quant + GEMM with a scaled epilogue.

    y[m, n] = (sum_k xq[m, k] * wq[n, k]) * sx[m] * sw[n]

Decode is the skinny case (M = batch, often < 64): the GEMM is bound by
reading W once (~70 GB of FP8 weights per step for 70B). With few output
tiles (N = 8192 at BLOCK_N = 64 is only 128 programs for 304 CUs) most CUs
idle, so the K dimension is split across programs (split-K). Partial sums go
to an fp32 workspace and a second kernel reduces them deterministically.

Per-shape tile configs come from ``gemm_configs.json`` (written by
``scripts/tune_gemm.py``), with a heuristic fallback.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from .fused_quant import quantize_per_token

CONFIG_FILE = Path(__file__).with_name("gemm_configs.json")


@triton.jit
def _fp8_gemm_kernel(
    a_ptr, b_ptr, c_ptr, ws_ptr, sa_ptr, sb_ptr,
    M, N, K, k_per_split,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    # Grouped ordering keeps a band of W tiles hot in L2 across neighbouring programs.
    group = pid // (GROUP_M * num_n)
    first_m = group * GROUP_M
    group_size = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_size)
    pid_n = (pid % (GROUP_M * num_n)) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)
    a_ptrs = a_ptr + rm[:, None].to(tl.int64) * stride_am + (k_start + rk[None, :]) * stride_ak
    # B is stored [N, K]; read it as a [BLOCK_K, BLOCK_N] tile of B^T.
    b_ptrs = b_ptr + rn[None, :].to(tl.int64) * stride_bn + (k_start + rk[:, None]) * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        k_mask = (k0 + rk) < k_end
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT_K == 1:
        sa = tl.load(sa_ptr + rm, mask=rm < M, other=0.0)
        sb = tl.load(sb_ptr + rn, mask=rn < N, other=0.0)
        c = acc * sa[:, None] * sb[None, :]
        c_ptrs = c_ptr + rm[:, None].to(tl.int64) * stride_cm + rn[None, :] * stride_cn
        tl.store(c_ptrs, c.to(c_ptr.dtype.element_ty), mask=mask)
    else:
        # Workspace is [SPLIT_K, M, N] fp32, contiguous.
        w_ptrs = ws_ptr + (pid_k.to(tl.int64) * M + rm[:, None]) * N + rn[None, :]
        tl.store(w_ptrs, acc, mask=mask)


@triton.jit
def _splitk_reduce_kernel(
    ws_ptr, c_ptr, sa_ptr, sb_ptr, M, N, stride_cm, stride_cn,
    SPLIT_K: tl.constexpr, BLOCK: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    rn = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = rn < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):  # fixed order: deterministic
        acc += tl.load(ws_ptr + (s * M + m) * N + rn, mask=mask, other=0.0)
    sa = tl.load(sa_ptr + m)
    sb = tl.load(sb_ptr + rn, mask=mask, other=0.0)
    tl.store(c_ptr + m * stride_cm + rn * stride_cn, (acc * sa * sb).to(c_ptr.dtype.element_ty), mask=mask)


def _heuristic_config(M: int, N: int, K: int) -> dict:
    if M <= 16:
        cfg = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=256, GROUP_M=1, num_warps=4, num_stages=2)
    elif M <= 64:
        cfg = dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=256, GROUP_M=1, num_warps=4, num_stages=2)
    else:
        return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, SPLIT_K=1, num_warps=8, num_stages=2)
    tiles = triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"])
    # Aim for at least ~2 programs per CU (304 CUs on MI300X).
    split = 1
    while tiles * split < 608 and K // (split * 2) >= 4 * cfg["BLOCK_K"]:
        split *= 2
    cfg["SPLIT_K"] = split
    return cfg


def _m_bucket(M: int) -> int:
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        if M <= b:
            return b
    return 512


@lru_cache(maxsize=1)
def _tuned() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def get_config(M: int, N: int, K: int) -> dict:
    tuned = _tuned()
    key = f"{N}x{K}"
    if key in tuned:
        buckets = tuned[key]
        mb = _m_bucket(M)
        # Nearest tuned bucket at or above M, else the largest tuned one.
        cands = sorted(int(b) for b in buckets)
        pick = next((b for b in cands if b >= mb), cands[-1])
        if mb <= 256:
            return dict(buckets[str(pick)])
    return _heuristic_config(M, N, K)


def fp8_gemm(a_q, a_scale, b_q, b_scale, out_dtype, config: dict | None = None):
    M, K = a_q.shape
    N = b_q.shape[0]
    cfg = dict(config or get_config(M, N, K))
    split = cfg.pop("SPLIT_K", 1)
    c = torch.empty((M, N), dtype=out_dtype, device=a_q.device)
    # Each split covers a whole number of BLOCK_K tiles.
    k_per_split = triton.cdiv(triton.cdiv(K, split), cfg["BLOCK_K"]) * cfg["BLOCK_K"]
    split = triton.cdiv(K, k_per_split)
    ws = torch.empty((split, M, N), dtype=torch.float32, device=a_q.device) if split > 1 else c
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]), split)
    _fp8_gemm_kernel[grid](
        a_q, b_q, c, ws, a_scale, b_scale, M, N, K, k_per_split,
        a_q.stride(0), a_q.stride(1), b_q.stride(0), b_q.stride(1), c.stride(0), c.stride(1),
        SPLIT_K=split, **cfg,
    )
    if split > 1:
        BLOCK = 1024
        _splitk_reduce_kernel[(M, triton.cdiv(N, BLOCK))](
            ws, c, a_scale, b_scale, M, N, c.stride(0), c.stride(1), SPLIT_K=split, BLOCK=BLOCK
        )
    return c


def fp8_linear(x, weight_fp8, weight_scale, out_dtype=None, config=None):
    x_q, x_scale = quantize_per_token(x, weight_fp8.dtype)
    return fp8_gemm(x_q, x_scale, weight_fp8, weight_scale, out_dtype or x.dtype, config)
