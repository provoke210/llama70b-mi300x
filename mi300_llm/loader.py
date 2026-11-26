"""Build a LlamaModel from Hugging Face weights.

Two sources share one code path, a ``get(name) -> tensor`` callable:
- a safetensors checkpoint directory (BF16, or FP8 compressed-tensors with
  per-channel ``weight_scale``), loaded one tensor at a time to the device;
- an in-memory state dict (used by tests with tiny random models).

Fusing QKV and gate/up is exact for per-channel FP8: each output row keeps its
own scale, so concatenating rows never forces a shared, requantized scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch

from .config import LlamaConfig
from .fp8 import E4M3FN, E4M3FNUZ, e4m3fn_to_e4m3fnuz, quantize_per_channel
from .model import Layer, Linear, LlamaModel
from .tp import SINGLE, TPGroup, local_config, shard_cols, shard_rows

Getter = Callable[[str], torch.Tensor]


def _safetensors_getter(path: Path) -> tuple[Getter, set[str]]:
    from safetensors import safe_open

    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text())["weight_map"]
    else:
        with safe_open(path / "model.safetensors", "pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
    handles: dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        fname = weight_map[name]
        if fname not in handles:
            handles[fname] = safe_open(path / fname, "pt")
        return handles[fname].get_tensor(name)

    return get, set(weight_map)


def _linear(
    get: Getter,
    names: set[str],
    prefixes: list[str],
    dtype: torch.dtype,
    device: torch.device,
    fp8_dtype: torch.dtype | None,
    quantize: bool,
    shard=None,
) -> Linear:
    """Load and concatenate the [N_i, K] weights under ``prefixes`` into one Linear.

    ``shard(w, s)`` slices the full weight and its per-row scale for tensor
    parallelism. It runs after quantization, so a row's scale always covers
    the whole (unsharded) row, exactly as in the checkpoint.
    """
    stored_fp8 = f"{prefixes[0]}.weight_scale" in names
    if stored_fp8:
        ws = [get(f"{p}.weight") for p in prefixes]
        ss = [get(f"{p}.weight_scale").float().reshape(-1) for p in prefixes]
        w, s = torch.cat(ws, 0), torch.cat(ss, 0)
        assert w.dtype == E4M3FN, w.dtype
        if fp8_dtype == E4M3FNUZ:
            w, s = e4m3fn_to_e4m3fnuz(w, s)
    else:
        w, s = torch.cat([get(f"{p}.weight") for p in prefixes], 0), None
        if quantize:
            w, s = quantize_per_channel(w.to(device), fp8_dtype)
            s = s.reshape(-1)
    if shard is not None:
        w, s = shard(w, s)
    if s is None:
        return Linear(w.contiguous().to(device=device, dtype=dtype))
    return Linear(w.contiguous().to(device), s.contiguous().to(device))


def build_model(
    cfg: LlamaConfig,
    get: Getter,
    names: set[str],
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    fp8_dtype: torch.dtype | None = None,
    quantize: bool = False,
    tp: TPGroup = SINGLE,
) -> LlamaModel:
    """``fp8_dtype``: compute format for FP8 weights (fnuz on MI300X).
    ``quantize``: quantize BF16 weights to per-channel FP8 on load.
    ``tp``: load only this rank's shard (see ``tp.py`` for the layout)."""
    device = torch.device(device)
    if (cfg.fp8_weights or quantize) and fp8_dtype is None:
        raise ValueError("FP8 weights need an fp8_dtype")
    local_cfg = local_config(cfg, tp)  # also checks divisibility

    def t(name: str) -> torch.Tensor:
        return get(name).to(device=device, dtype=dtype)

    def rows(kind):  # column-parallel: slice output rows and their scales
        return lambda w, s: (shard_rows(w, kind, cfg, tp), None if s is None else shard_rows(s, kind, cfg, tp))

    def cols(w, s):  # row-parallel: slice input columns; per-row scales stay whole
        return shard_cols(w, cfg, tp), s

    def lin(prefixes, shard):
        return _linear(get, names, prefixes, dtype, device, fp8_dtype, quantize, shard if tp.enabled else None)

    layers = []
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}"
        layers.append(
            Layer(
                input_norm=t(f"{p}.input_layernorm.weight"),
                post_norm=t(f"{p}.post_attention_layernorm.weight"),
                qkv=lin([f"{p}.self_attn.{x}_proj" for x in "qkv"], rows("qkv")),
                o=lin([f"{p}.self_attn.o_proj"], cols),
                gate_up=lin([f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"], rows("gate_up")),
                down=lin([f"{p}.mlp.down_proj"], cols),
            )
        )
    embed = t("model.embed_tokens.weight")
    head_name = "model.embed_tokens.weight" if cfg.tie_word_embeddings or "lm_head.weight" not in names else "lm_head.weight"
    lm_head = shard_rows(get(head_name), "vocab", cfg, tp).contiguous().to(device=device, dtype=dtype)
    return LlamaModel(local_cfg, embed, layers, t("model.norm.weight"), lm_head, tp=tp)


def load_pretrained(path: str | Path, device="cuda", dtype=torch.bfloat16, fp8_dtype=None, tp: TPGroup = SINGLE) -> LlamaModel:
    path = Path(path)
    cfg = LlamaConfig.from_pretrained(path)
    get, names = _safetensors_getter(path)
    return build_model(cfg, get, names, dtype=dtype, device=device, fp8_dtype=fp8_dtype, tp=tp)


def from_state_dict(cfg: LlamaConfig, sd: dict[str, torch.Tensor], **kw) -> LlamaModel:
    return build_model(cfg, sd.__getitem__, set(sd), **kw)
