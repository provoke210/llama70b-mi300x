"""Paged decode attention with split-K (flash-decoding).

With one query token per sequence and few sequences, one program per
(sequence, head) cannot fill 304 CUs. Splitting each sequence's KV range across
``num_splits`` programs adds parallelism; a second kernel merges the partial
softmax results using their log-sum-exp.

GQA: each program handles all ``group = Hq / Hkv`` query heads that share one KV
head (8 for Llama-3.3-70B), so every K/V tile is read once per group, not once
per query head.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_split_kernel(
    q_ptr, k_ptr, v_ptr, bt_ptr, len_ptr, o_ptr, lse_ptr,
    scale, tokens_per_split,
    stride_qb, stride_qh,
    stride_kslot, stride_kh,
    stride_btb,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh,
    GROUP: tl.constexpr, BLOCK_G: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr, PAGE: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1).to(tl.int64)
    split = tl.program_id(2)
    seq_len = tl.load(len_ptr + b)
    start = split * tokens_per_split
    end = tl.minimum(start + tokens_per_split, seq_len)

    g = tl.arange(0, BLOCK_G)
    d = tl.arange(0, HEAD_DIM)
    qh = kvh * GROUP + g
    g_mask = g < GROUP
    q = tl.load(q_ptr + b * stride_qb + qh[:, None] * stride_qh + d[None, :], mask=g_mask[:, None], other=0.0)

    m_i = tl.full((BLOCK_G,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_G,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_G, HEAD_DIM), dtype=tl.float32)

    for n0 in range(start, end, BLOCK_N):
        pos = n0 + tl.arange(0, BLOCK_N)
        n_mask = pos < end
        block = tl.load(bt_ptr + b * stride_btb + pos // PAGE, mask=n_mask, other=0).to(tl.int64)
        slot = block * PAGE + pos % PAGE
        kv_off = slot[:, None] * stride_kslot + kvh * stride_kh + d[None, :]
        k = tl.load(k_ptr + kv_off, mask=n_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_G, BLOCK_N]
        s = tl.where(n_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv_off, mask=n_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Empty splits (sequence shorter than this split's start) write lse = -inf
    # and a zero partial so the merge ignores them.
    has = l_i > 0
    out = tl.where(has[:, None], acc / tl.where(has, l_i, 1.0)[:, None], 0.0)
    lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), float("-inf"))
    tl.store(o_ptr + b * stride_ob + qh[:, None] * stride_oh + split * stride_os + d[None, :], out, mask=g_mask[:, None])
    tl.store(lse_ptr + b * stride_lb + qh * stride_lh + split, lse, mask=g_mask)


@triton.jit
def _decode_merge_kernel(
    o_part_ptr, lse_ptr, out_ptr, num_splits,
    stride_pb, stride_ph, stride_ps, stride_lb, stride_lh, stride_ob, stride_oh,
    HEAD_DIM: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, HEAD_DIM)
    s_mask = s < num_splits
    lse = tl.load(lse_ptr + b * stride_lb + h * stride_lh + s, mask=s_mask, other=float("-inf"))
    m = tl.max(lse, axis=0)
    w = tl.exp(lse - m)
    w = w / tl.sum(w, axis=0)
    part = tl.load(o_part_ptr + b * stride_pb + h * stride_ph + s[:, None] * stride_ps + d[None, :], mask=s_mask[:, None], other=0.0)
    out = tl.sum(part * w[:, None], axis=0)
    tl.store(out_ptr + b * stride_ob + h * stride_oh + d, out.to(out_ptr.dtype.element_ty))


def default_num_splits(batch: int, kv_heads: int, max_len: int) -> int:
    """Measured on MI300X (scripts in README): aim for ~2048 programs, but keep
    at least 128 tokens per split so the merge stays cheap."""
    want = triton.next_power_of_2(triton.cdiv(2048, batch * kv_heads))
    return max(1, min(want, max_len // 128, 128))


def paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens, scale, num_splits=None, block_n=32):
    """Same contract as ``ref_ops.paged_decode_attention``.

    The split count is derived from the block-table width (an upper bound on
    sequence length) so no device-to-host sync is needed, and the launch is
    capturable in a HIP graph.
    """
    B, Hq, D = q.shape
    _, page, Hkv, _ = k_cache.shape
    assert k_cache.is_contiguous() and v_cache.is_contiguous() and q.stride(-1) == 1
    assert D == triton.next_power_of_2(D)
    group = Hq // Hkv
    max_len = block_tables.shape[1] * page
    if num_splits is None:
        num_splits = default_num_splits(B, Hkv, max_len)
    tokens_per_split = triton.cdiv(triton.cdiv(max_len, num_splits), block_n) * block_n

    o_part = torch.empty((B, Hq, num_splits, D), dtype=torch.float32, device=q.device)
    lse = torch.empty((B, Hq, num_splits), dtype=torch.float32, device=q.device)
    k_flat = k_cache.view(-1, Hkv, D)
    _decode_split_kernel[(B, Hkv, num_splits)](
        q, k_flat, v_cache.view(-1, Hkv, D), block_tables, seq_lens, o_part, lse,
        scale, tokens_per_split,
        q.stride(0), q.stride(1),
        k_flat.stride(0), k_flat.stride(1),
        block_tables.stride(0),
        o_part.stride(0), o_part.stride(1), o_part.stride(2),
        lse.stride(0), lse.stride(1),
        GROUP=group, BLOCK_G=max(16, triton.next_power_of_2(group)), HEAD_DIM=D,
        BLOCK_N=block_n, PAGE=page,
    )
    out = torch.empty_like(q)
    _decode_merge_kernel[(B, Hq)](
        o_part, lse, out, num_splits,
        o_part.stride(0), o_part.stride(1), o_part.stride(2), lse.stride(0), lse.stride(1),
        out.stride(0), out.stride(1),
        HEAD_DIM=D, BLOCK_S=triton.next_power_of_2(num_splits),
    )
    return out
