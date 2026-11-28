"""Causal prefill attention via PyTorch SDPA.

The reference (``ref_ops.prefill_attention``) materializes the full [H, T, T]
score matrix in fp32: 17 GB for one 8k-token prompt with 64 heads, so long
prompts don't fit. SDPA dispatches to a fused flash/memory-efficient kernel on
ROCm with memory linear in T. Prefill is not the focus of this project, so it
uses the library kernel rather than a hand-written one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prefill_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Causal attention for one sequence. q: [T, Hq, D], k/v: [T, Hkv, D]."""
    qt, kt, vt = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))  # [1, H, T, D]
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, scale=scale, enable_gqa=True)
    return out.squeeze(0).transpose(0, 1).contiguous()
