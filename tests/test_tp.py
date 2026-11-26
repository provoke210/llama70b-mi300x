"""Tensor parallelism on CPU: W processes (gloo) must reproduce the single-process model.

Runs anywhere, no GPU: each rank loads only its shard, runs the reference ops,
and communicates through torch.distributed. The GPU path differs only in the
backend (RCCL) and the kernels, which test_kernels.py covers.
"""

from __future__ import annotations

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mi300_llm import KVCache, LlamaConfig, from_state_dict, ops
from mi300_llm.engine import generate
from mi300_llm.fp8 import E4M3FNUZ
from mi300_llm.tp import SINGLE, TPGroup, local_config

# Like conftest.TINY but with 4 KV heads, so TP up to 4 has whole heads per rank.
TINY_TP = dict(
    hidden_size=128, intermediate_size=320, num_hidden_layers=3, num_attention_heads=8,
    num_key_value_heads=4, head_dim=16, vocab_size=304, rms_norm_eps=1e-5, rope_theta=500000.0,
    max_position_embeddings=512, tie_word_embeddings=False,
    rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                  "high_freq_factor": 4.0, "original_max_position_embeddings": 64},
)
PROMPTS = [[1, 17, 42, 99, 5, 250, 3], [7, 8, 9], list(range(20, 50))]
NEW = 8


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from transformers import LlamaConfig as HFConfig
    from transformers import LlamaForCausalLM

    torch.manual_seed(0)
    hf = LlamaForCausalLM(HFConfig(**TINY_TP, attn_implementation="eager")).float().eval()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" not in name:
                p.mul_(3.0)
    path = tmp_path_factory.mktemp("tp") / "ckpt.pt"
    torch.save({"sd": hf.state_dict(), "cfg": hf.config.to_dict()}, path)
    return path


def _run(ckpt_path, tp: TPGroup, fp8: bool):
    ck = torch.load(ckpt_path)
    cfg = LlamaConfig.from_hf_dict(ck["cfg"])
    kw = dict(fp8_dtype=E4M3FNUZ, quantize=True) if fp8 else {}
    ops.set_backend("ref")
    model = from_state_dict(cfg, ck["sd"], dtype=torch.float32, tp=tp, **kw)
    cache = KVCache(model.cfg, num_blocks=64, block_size=4, dtype=torch.float32, device="cpu")
    tokens, logits = generate(model, cache, PROMPTS, NEW, return_logits=True)
    return model, tokens, torch.stack(logits)


def _worker(rank, world, port, ckpt_path, out_path, fp8):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        tp = TPGroup(rank, world)
        model, tokens, logits = _run(ckpt_path, tp, fp8)
        n_params = sum(t.numel() for l in model.layers for t in (l.qkv.weight, l.o.weight, l.gate_up.weight, l.down.weight))
        torch.save({"tokens": tokens, "logits": logits, "local_heads": model.cfg.num_kv_heads, "params": n_params},
                   f"{out_path}.{rank}")
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_tp(ckpt, world, fp8, tmp_path):
    out = tmp_path / f"out_w{world}_fp8{int(fp8)}"
    mp.spawn(_worker, args=(world, _free_port(), str(ckpt), str(out), fp8), nprocs=world, join=True)
    return [torch.load(f"{out}.{r}") for r in range(world)]


@pytest.mark.parametrize("world", [2, 4])
def test_tp_matches_single_process(checkpoint, world, tmp_path):
    """fp32: the sharded model generates the same tokens; logits differ only by
    the summation order of the all-reduce."""
    _, ref_tokens, ref_logits = _run(checkpoint, SINGLE, fp8=False)
    results = _run_tp(checkpoint, world, False, tmp_path)
    for r, res in enumerate(results):
        assert res["tokens"] == ref_tokens, f"rank {r}"
        torch.testing.assert_close(res["logits"], ref_logits, rtol=1e-4, atol=1e-4)
        assert res["local_heads"] == TINY_TP["num_key_value_heads"] // world
    # Every rank holds 1/world of the layer weights.
    single_params = sum(t.numel() for l in _run(checkpoint, SINGLE, False)[0].layers
                        for t in (l.qkv.weight, l.o.weight, l.gate_up.weight, l.down.weight))
    assert all(res["params"] * world == single_params for res in results)


def test_tp_fp8_tracks_single_process(checkpoint, tmp_path):
    """FP8: row-parallel layers quantize their local activation slice with a
    local per-token scale, so numerics change slightly but stay close."""
    _, ref_tokens, ref_logits = _run(checkpoint, SINGLE, fp8=True)
    res = _run_tp(checkpoint, 2, True, tmp_path)[0]
    assert [t[0] for t in res["tokens"]] == [t[0] for t in ref_tokens]
    cos = torch.nn.functional.cosine_similarity(res["logits"][0], ref_logits[0], dim=-1)
    assert (cos > 0.99).all(), cos


def test_local_config_rejects_uneven_split():
    cfg = LlamaConfig.from_hf_dict({**TINY_TP, "num_key_value_heads": 4})
    assert local_config(cfg, TPGroup(0, 4)).num_kv_heads == 1
    with pytest.raises(ValueError, match="num_kv_heads"):
        local_config(cfg, TPGroup(0, 8))
