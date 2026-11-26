"""Llama forward pass over a paged KV cache.

Weights are held as plain tensors (no nn.Module) so that the layout the kernels
see is explicit: fused QKV [q_size + 2*kv_size, H] and fused gate/up [2*I, H].
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from . import ops
from .config import LlamaConfig
from .kv_cache import KVCache
from .ref_ops import rope_cos_sin, rope_inv_freq
from .tp import SINGLE, TPGroup

# MI300_FUSE_QUANT=0 quantizes GEMM inputs in a separate pass (for A/B tests).
FUSE_QUANT = os.environ.get("MI300_FUSE_QUANT", "1") != "0"


@dataclass
class Linear:
    weight: torch.Tensor  # [N, K]; FP8 when scale is set
    scale: torch.Tensor | None = None  # [N] fp32 per-channel

    @property
    def is_fp8(self) -> bool:
        return self.scale is not None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            return x @ self.weight.t()
        return ops.fp8_linear(x, self.weight, self.scale, out_dtype=x.dtype)

    def gemm(self, x_q: torch.Tensor, x_scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """FP8 GEMM on an input already quantized per token (by a fused producer)."""
        return ops.fp8_gemm(x_q, x_scale, self.weight, self.scale, out_dtype)


@dataclass
class Layer:
    input_norm: torch.Tensor
    post_norm: torch.Tensor
    qkv: Linear
    o: Linear
    gate_up: Linear
    down: Linear


@dataclass
class AttnMeta:
    """Describes one forward batch.

    Prefill batches carry ``cu_seqlens`` (every sequence starts from an empty
    cache). Decode batches carry one token per sequence plus block tables.
    """

    is_prefill: bool
    slot_mapping: torch.Tensor  # [T] int64: flat cache slot of every token
    cu_seqlens: list[int] | None = None
    block_tables: torch.Tensor | None = None  # [B, max_blocks] int32
    seq_lens: torch.Tensor | None = None  # [B] int32, including the new token


class LlamaModel:
    """``cfg`` is this rank's local view (head counts, FFN width and vocab
    divided by the tensor-parallel size); ``lm_head`` holds the local vocab
    slice. With ``tp.world == 1`` this is the plain single-GPU model."""

    def __init__(self, cfg: LlamaConfig, embed: torch.Tensor, layers: list[Layer], norm: torch.Tensor, lm_head: torch.Tensor, tp: TPGroup = SINGLE):
        self.cfg = cfg
        self.tp = tp
        self.embed = embed
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.inv_freq = rope_inv_freq(cfg.head_dim, cfg.rope_theta, cfg.rope_scaling).to(embed.device)
        self.scale = cfg.head_dim ** -0.5

    @property
    def device(self) -> torch.device:
        return self.embed.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed.dtype

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, meta: AttnMeta, cache: KVCache) -> torch.Tensor:
        """Returns logits [num_seqs, vocab] for the last token of each sequence."""
        cfg = self.cfg
        eps = cfg.rms_norm_eps
        act = self.dtype
        cos, sin = rope_cos_sin(positions, self.inv_freq)
        h = self.embed[input_ids]
        residual = None
        for i, layer in enumerate(self.layers):
            # With FP8 weights the norm emits the quantized GEMM input directly
            # (fused), instead of a bf16 tensor that a separate pass quantizes.
            fp8 = layer.qkv.is_fp8 and FUSE_QUANT
            if residual is None:
                residual = h
                if fp8:
                    xq, xs = ops.rmsnorm_quant(h, layer.input_norm, eps, layer.qkv.weight.dtype)
                else:
                    x = ops.rmsnorm(h, layer.input_norm, eps)
            elif fp8:
                xq, xs, residual = ops.fused_add_rmsnorm_quant(h, residual, layer.input_norm, eps, layer.qkv.weight.dtype)
            else:
                x, residual = ops.fused_add_rmsnorm(h, residual, layer.input_norm, eps)

            qkv = layer.qkv.gemm(xq, xs, act) if fp8 else layer.qkv(x)
            q, k, v = qkv.split([cfg.q_size, cfg.kv_size, cfg.kv_size], dim=-1)
            q = ops.apply_rope(q.view(-1, cfg.num_heads, cfg.head_dim), cos, sin)
            k = ops.apply_rope(k.view(-1, cfg.num_kv_heads, cfg.head_dim), cos, sin)
            v = v.reshape(-1, cfg.num_kv_heads, cfg.head_dim)

            k_cache, v_cache = cache.k[i], cache.v[i]
            k_cache.view(-1, cfg.num_kv_heads, cfg.head_dim)[meta.slot_mapping] = k
            v_cache.view(-1, cfg.num_kv_heads, cfg.head_dim)[meta.slot_mapping] = v

            if meta.is_prefill:
                outs = []
                cu = meta.cu_seqlens
                for s in range(len(cu) - 1):
                    a, b = cu[s], cu[s + 1]
                    outs.append(ops.prefill_attention(q[a:b], k[a:b], v[a:b], self.scale))
                attn = torch.cat(outs, dim=0)
            else:
                attn = ops.paged_decode_attention(q, k_cache, v_cache, meta.block_tables, meta.seq_lens, self.scale)

            h = self.tp.all_reduce(layer.o(attn.reshape(-1, cfg.q_size)))  # row-parallel partial sums
            if fp8:
                fp8_dt = layer.gate_up.weight.dtype
                xq, xs, residual = ops.fused_add_rmsnorm_quant(h, residual, layer.post_norm, eps, fp8_dt)
                aq, a_s = ops.silu_mul_quant(layer.gate_up.gemm(xq, xs, act), fp8_dt)
                h = layer.down.gemm(aq, a_s, act)
            else:
                x, residual = ops.fused_add_rmsnorm(h, residual, layer.post_norm, eps)
                h = layer.down(ops.silu_mul(layer.gate_up(x)))
            h = self.tp.all_reduce(h)

        h, _ = ops.fused_add_rmsnorm(h, residual, self.norm, eps)
        if meta.is_prefill:
            last = torch.tensor(meta.cu_seqlens[1:], device=h.device) - 1
            h = h[last]
        # Vocab-parallel head: local logits, then every rank gets the full row.
        return self.tp.all_gather_last_dim((h @ self.lm_head.t()).float())
