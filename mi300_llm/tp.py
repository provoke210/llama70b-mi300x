"""Tensor parallelism (Megatron-style) for the Llama decoder.

Per layer, with world size W:

- qkv (fused):   column-parallel. Rank r keeps query heads [r*Hq/W, (r+1)*Hq/W)
                 and KV heads [r*Hkv/W, (r+1)*Hkv/W). Attention then runs on
                 local heads only, with no communication.
- o:             row-parallel. Input columns split the same way; each rank
                 produces a partial [M, H] sum, followed by an all-reduce.
- gate_up:       column-parallel over the intermediate dimension (gate and up
                 halves are each sliced, so silu(gate) * up stays local).
- down:          row-parallel, followed by an all-reduce.
- lm_head:       vocab-parallel. Each rank computes logits for its slice of the
                 vocabulary; an all-gather assembles the full row. Replicating
                 it instead would make every rank read the full 2.1 GB head
                 each step, +23% weight traffic per rank at W=8.
- embed, norms:  replicated.

So a decode step does 2 all-reduces per layer (160 for 70B) plus one
all-gather of the logits. With per-channel FP8 weights, column slices keep
their own scales and row slices keep the full per-output-row scale; each rank
quantizes its local activation slice with its own per-token scale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from .config import LlamaConfig


@dataclass(frozen=True)
class TPGroup:
    rank: int
    world: int

    @property
    def enabled(self) -> bool:
        return self.world > 1

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(x)
        return x

    def all_gather_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate every rank's ``x`` along the last dimension.

        Uses the single-buffer ``all_gather_into_tensor`` (one collective into
        one allocation), which is friendlier to graph capture than the
        list-of-tensors form.
        """
        if not self.enabled:
            return x
        x = x.contiguous()
        # Ranks stacked along dim 0 (the layout every backend accepts, gloo included).
        out = torch.empty((self.world * x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        # [W * M, n] -> [M, W * n], rank-major along the last dim.
        return out.view(self.world, *x.shape).movedim(0, -2).reshape(*x.shape[:-1], self.world * x.shape[-1])


SINGLE = TPGroup(0, 1)


def init_from_env(backend: str | None = None) -> TPGroup:
    """Initialize torch.distributed from torchrun's environment variables."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return SINGLE
    rank = int(os.environ["RANK"])
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"  # "nccl" is RCCL on ROCm
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group(backend)
    return TPGroup(rank, world)


def local_config(cfg: LlamaConfig, tp: TPGroup) -> LlamaConfig:
    """The per-rank view of the model: local head counts and FFN width."""
    w = tp.world
    for name, n in (("num_heads", cfg.num_heads), ("num_kv_heads", cfg.num_kv_heads),
                    ("intermediate_size", cfg.intermediate_size), ("vocab_size", cfg.vocab_size)):
        if n % w:
            raise ValueError(f"{name}={n} is not divisible by tensor-parallel size {w}")
    return replace(
        cfg,
        num_heads=cfg.num_heads // w,
        num_kv_heads=cfg.num_kv_heads // w,
        intermediate_size=cfg.intermediate_size // w,
        vocab_size=cfg.vocab_size // w,
    )


def _part(n: int, tp: TPGroup) -> slice:
    size = n // tp.world
    return slice(tp.rank * size, (tp.rank + 1) * size)


def shard_rows(t: torch.Tensor, kind: str, cfg: LlamaConfig, tp: TPGroup) -> torch.Tensor:
    """Slice the output (row) dimension of a full [N, ...] weight or [N] scale.

    ``cfg`` is the full (not local) config. Fused tensors are sliced per part
    so that each rank keeps matching pieces of q / k / v and of gate / up.
    """
    if not tp.enabled:
        return t
    if kind == "qkv":
        q, k, v = t.split([cfg.q_size, cfg.kv_size, cfg.kv_size], dim=0)
        return torch.cat([q[_part(cfg.q_size, tp)], k[_part(cfg.kv_size, tp)], v[_part(cfg.kv_size, tp)]], 0)
    if kind == "gate_up":
        gate, up = t.split([cfg.intermediate_size, cfg.intermediate_size], dim=0)
        part = _part(cfg.intermediate_size, tp)
        return torch.cat([gate[part], up[part]], 0)
    if kind == "vocab":
        return t[_part(t.shape[0], tp)]
    raise ValueError(kind)


def shard_cols(t: torch.Tensor, cfg: LlamaConfig, tp: TPGroup) -> torch.Tensor:
    """Slice the input (column) dimension of a row-parallel [N, K] weight."""
    if not tp.enabled:
        return t
    return t[:, _part(t.shape[1], tp)]
