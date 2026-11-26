"""Pure-PyTorch reference implementations.

Every Triton kernel in ``kernels/`` is tested against the function of the same
name here. These favour clarity over speed and run on CPU.
"""

from __future__ import annotations

import math

import torch

from .fp8 import quantize_per_token


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * weight.float()).to(x.dtype)


def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (rmsnorm(x + residual), x + residual)."""
    new_residual = (x.float() + residual.float()).to(x.dtype)
    return rmsnorm(new_residual, weight, eps), new_residual


def rope_inv_freq(head_dim: int, theta: float, scaling: dict | None) -> torch.Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    if scaling is None:
        return inv_freq.float()
    rope_type = scaling.get("rope_type", scaling.get("type"))
    if rope_type != "llama3":
        raise NotImplementedError(f"rope_type={rope_type}")
    factor = scaling["factor"]
    low, high = scaling["low_freq_factor"], scaling["high_freq_factor"]
    old_ctx = scaling["original_max_position_embeddings"]
    low_wavelen, high_wavelen = old_ctx / low, old_ctx / high
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_ctx / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(medium, smoothed, scaled).float()


def rope_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape [T, head_dim // 2], fp32."""
    freqs = positions.float()[:, None] * inv_freq[None, :]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE (Hugging Face Llama layout). x: [T, H, D]."""
    half = x.shape[-1] // 2
    x32 = x.float()
    x1, x2 = x32[..., :half], x32[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


def fp8_gemm(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    config: dict | None = None,
) -> torch.Tensor:
    """y = (x_q @ W^T) * x_scale * w_scale for pre-quantized x [M, K]."""
    acc = x_q.float() @ weight_fp8.float().t()
    return (acc * x_scale * weight_scale.float().view(1, -1)).to(out_dtype)


def fp8_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """y = x @ W^T with W per-channel FP8 [N, K] and x quantized per token."""
    x_q, x_scale = quantize_per_token(x, weight_fp8.dtype)
    return fp8_gemm(x_q, x_scale, weight_fp8, weight_scale, out_dtype or x.dtype)


# Fused producers of FP8 GEMM inputs. The reference is literally the unfused
# composition; the Triton versions must match it bit for bit (same rounding
# of the intermediate to the activation dtype).


def rmsnorm_quant(x, weight, eps, fp8_dtype):
    return quantize_per_token(rmsnorm(x, weight, eps), fp8_dtype)


def fused_add_rmsnorm_quant(x, residual, weight, eps, fp8_dtype):
    y, new_residual = fused_add_rmsnorm(x, residual, weight, eps)
    q, s = quantize_per_token(y, fp8_dtype)
    return q, s, new_residual


def silu_mul_quant(gate_up, fp8_dtype):
    return quantize_per_token(silu_mul(gate_up), fp8_dtype)


def prefill_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Causal attention for one sequence. q: [T, Hq, D], k/v: [T, Hkv, D]."""
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * scale
    t = q.shape[0]
    mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    out = torch.einsum("hts,shd->thd", scores.softmax(-1), v.float())
    return out.to(q.dtype)


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One query token per sequence against a paged KV cache.

    q: [B, Hq, D]; k_cache/v_cache: [num_blocks, block_size, Hkv, D];
    block_tables: [B, max_blocks] int32; seq_lens: [B] int32 (includes the
    current token, whose K/V must already be in the cache).
    """
    block_size = k_cache.shape[1]
    out = torch.empty_like(q)
    for b in range(q.shape[0]):
        n = int(seq_lens[b])
        nblk = (n + block_size - 1) // block_size
        blocks = block_tables[b, :nblk].long()
        k = k_cache[blocks].flatten(0, 1)[:n]
        v = v_cache[blocks].flatten(0, 1)[:n]
        group = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(group, dim=1).float()
        v = v.repeat_interleave(group, dim=1).float()
        s = torch.einsum("hd,shd->hs", q[b].float(), k) * scale
        out[b] = torch.einsum("hs,shd->hd", s.softmax(-1), v).to(q.dtype)
    return out


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate_up.dtype)
