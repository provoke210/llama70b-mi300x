"""Model configuration, read from a Hugging Face ``config.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LlamaConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    # llama3-style RoPE scaling; None means plain RoPE.
    rope_scaling: dict | None = None
    tie_word_embeddings: bool = False
    # True when the checkpoint stores FP8 linear weights (compressed-tensors).
    fp8_weights: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def q_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @classmethod
    def from_hf_dict(cls, d: dict) -> "LlamaConfig":
        # transformers v5 moved rope settings into ``rope_parameters``.
        rope = d.get("rope_parameters") or d.get("rope_scaling")
        rope_theta = d.get("rope_theta") or (rope or {}).get("rope_theta", 500000.0)
        if rope is not None and rope.get("rope_type", rope.get("type")) in (None, "default"):
            rope = None
        quant = d.get("quantization_config") or {}
        fp8 = quant.get("quant_method") in ("compressed-tensors", "fp8")
        return cls(
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_layers=d["num_hidden_layers"],
            num_heads=d["num_attention_heads"],
            num_kv_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
            head_dim=d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"],
            vocab_size=d["vocab_size"],
            rms_norm_eps=d.get("rms_norm_eps", 1e-5),
            rope_theta=float(rope_theta),
            rope_scaling=dict(rope) if rope else None,
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            fp8_weights=fp8,
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "LlamaConfig":
        return cls.from_hf_dict(json.loads((Path(path) / "config.json").read_text()))
