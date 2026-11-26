"""Paged KV cache: fixed-size blocks plus a per-sequence block table."""

from __future__ import annotations

import torch

from .config import LlamaConfig


class KVCache:
    def __init__(
        self,
        cfg: LlamaConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        shape = (cfg.num_layers, num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self._free = list(range(num_blocks - 1, -1, -1))

    @staticmethod
    def bytes_per_token(cfg: LlamaConfig, dtype: torch.dtype) -> int:
        elem = torch.empty((), dtype=dtype).element_size()
        return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * elem

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise RuntimeError(f"KV cache out of blocks: need {n}, have {len(self._free)}")
        return [self._free.pop() for _ in range(n)]

    def free(self, blocks: list[int]) -> None:
        self._free.extend(reversed(blocks))


class Sequence:
    """Token ids and KV blocks for one request."""

    def __init__(self, seq_id: int, prompt: list[int]):
        self.seq_id = seq_id
        self.tokens = list(prompt)
        self.num_prompt = len(prompt)
        self.blocks: list[int] = []
        self.num_cached = 0  # tokens whose K/V are in the cache

    def __len__(self) -> int:
        return len(self.tokens)

    def ensure_blocks(self, cache: KVCache, num_tokens: int) -> None:
        need = (num_tokens + cache.block_size - 1) // cache.block_size - len(self.blocks)
        if need > 0:
            self.blocks.extend(cache.allocate(need))

    def slots(self, start: int, end: int, block_size: int) -> list[int]:
        return [self.blocks[i // block_size] * block_size + i % block_size for i in range(start, end)]
