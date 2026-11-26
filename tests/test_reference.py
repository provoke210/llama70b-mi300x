"""The reference path must match Hugging Face exactly (fp32) before any kernel work."""

import torch

from mi300_llm import KVCache, from_state_dict
from mi300_llm import ops
from mi300_llm.engine import generate
from mi300_llm.fp8 import E4M3FN, E4M3FNUZ, e4m3fn_to_e4m3fnuz, quantize_per_channel
from mi300_llm.ref_ops import fp8_linear, rope_inv_freq

PROMPTS = [[1, 17, 42, 99, 5, 250, 3], [7, 8, 9], [200, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]


def _ours(hf, cfg, **kw):
    ops.set_backend("ref")
    model = from_state_dict(cfg, hf.state_dict(), dtype=torch.float32, **kw)
    cache = KVCache(cfg, num_blocks=64, block_size=4, dtype=torch.float32, device="cpu")
    return model, cache


def test_config_parsed(tiny_cfg):
    assert tiny_cfg.num_kv_heads == 1 and tiny_cfg.head_dim == 16
    assert tiny_cfg.rope_scaling["rope_type"] == "llama3"


def test_rope_inv_freq_matches_hf(hf_tiny, tiny_cfg):
    hf_inv = hf_tiny.model.rotary_emb.inv_freq.float()
    ours = rope_inv_freq(tiny_cfg.head_dim, tiny_cfg.rope_theta, tiny_cfg.rope_scaling)
    torch.testing.assert_close(ours, hf_inv, rtol=1e-6, atol=0)


def test_greedy_logits_match_hf(hf_tiny, tiny_cfg):
    """Prefill + paged decode (block_size=4, so sequences span several blocks)."""
    model, cache = _ours(hf_tiny, tiny_cfg)
    new, step_logits = generate(model, cache, PROMPTS, max_new_tokens=8, return_logits=True)
    for b, prompt in enumerate(PROMPTS):
        full = prompt + new[b]
        with torch.no_grad():
            ref = hf_tiny(torch.tensor([full])).logits[0].float()
        for step in range(8):
            torch.testing.assert_close(
                step_logits[step][b], ref[len(prompt) - 1 + step], rtol=1e-4, atol=1e-4
            )
    assert cache.num_free_blocks == cache.num_blocks


def test_fnuz_conversion_is_lossless():
    torch.manual_seed(0)
    w = torch.randn(64, 32) * 3
    q, s = quantize_per_channel(w, E4M3FN)
    q = q.clone()
    q.view(torch.int8)[0, 0] = -128  # -0.0 in e4m3fn, NaN in fnuz
    qz, sz = e4m3fn_to_e4m3fnuz(q, s)
    deq_fn = q.float() * s
    deq_fnuz = qz.float() * sz
    assert not torch.isnan(deq_fnuz).any()
    torch.testing.assert_close(deq_fnuz, deq_fn, rtol=0, atol=0)


def test_fp8_linear_close_to_bf16():
    torch.manual_seed(0)
    x, w = torch.randn(5, 256), torch.randn(96, 256) / 16
    exact = x @ w.t()
    for dt in (E4M3FN, E4M3FNUZ):
        q, s = quantize_per_channel(w, dt)
        y = fp8_linear(x, q, s.reshape(-1))
        rel = (y - exact).norm() / exact.norm()
        assert rel < 0.05, (dt, rel)


def _save_compressed_tensors_fp8(hf, path):
    """Write a tiny checkpoint in the RedHatAI FP8-dynamic layout: e4m3fn
    per-channel weights + ``weight_scale`` [N, 1]; lm_head left in high precision."""
    import json

    from safetensors.torch import save_file

    sd = {}
    for name, t in hf.state_dict().items():
        if name.endswith("_proj.weight"):
            q, s = quantize_per_channel(t.float(), E4M3FN)
            sd[name], sd[name.replace(".weight", ".weight_scale")] = q, s.to(torch.bfloat16)
        else:
            sd[name] = t.to(torch.bfloat16).contiguous()
    save_file(sd, str(path / "model.safetensors"))
    cfg = hf.config.to_dict()
    cfg["quantization_config"] = {"quant_method": "compressed-tensors", "format": "float-quantized"}
    (path / "config.json").write_text(json.dumps(cfg))


def test_load_compressed_tensors_checkpoint(hf_tiny, tiny_cfg, tmp_path):
    from mi300_llm import load_pretrained

    _save_compressed_tensors_fp8(hf_tiny, tmp_path)
    ops.set_backend("ref")
    cache = KVCache(tiny_cfg, num_blocks=64, block_size=4, dtype=torch.bfloat16, device="cpu")
    fn = load_pretrained(tmp_path, device="cpu", fp8_dtype=E4M3FN)
    fnuz = load_pretrained(tmp_path, device="cpu", fp8_dtype=E4M3FNUZ)
    assert fn.cfg.fp8_weights and fn.layers[0].qkv.weight.shape == (tiny_cfg.q_size + 2 * tiny_cfg.kv_size, 128)
    assert fnuz.layers[0].qkv.weight.dtype == E4M3FNUZ
    # Converting the weights to fnuz is lossless.
    torch.testing.assert_close(
        fn.layers[0].qkv.weight.float() * fn.layers[0].qkv.scale[:, None],
        fnuz.layers[0].qkv.weight.float() * fnuz.layers[0].qkv.scale[:, None],
        rtol=0, atol=0,
    )
    tok_a, a = generate(fn, cache, PROMPTS, 3, return_logits=True)
    tok_b, b = generate(fnuz, cache, PROMPTS, 3, return_logits=True)
    # Only the activation quantization grid differs (max 448 vs 240), which can
    # flip a near-tie; compare each sequence only while its tokens agree.
    for seq in range(len(PROMPTS)):
        for step in range(3):
            cos = torch.nn.functional.cosine_similarity(a[step][seq], b[step][seq], dim=0)
            assert cos > 0.98, (seq, step, cos)
            if tok_a[seq][step] != tok_b[seq][step]:
                break


def test_fp8_model_tracks_fp32(hf_tiny, tiny_cfg):
    """Quantizing every linear to FP8 (fnuz, as on MI300X) keeps the prefill
    logits close to the fp32 model.

    A 3-layer random model with inflated weights is a weak accuracy signal
    (cosine lands around 0.99); real accuracy is checked with GSM8K on 70B."""
    ref_model, cache = _ours(hf_tiny, tiny_cfg)
    q_model, _ = _ours(hf_tiny, tiny_cfg, fp8_dtype=E4M3FNUZ, quantize=True)
    _, ref_logits = generate(ref_model, cache, PROMPTS, 1, return_logits=True)
    _, q_logits = generate(q_model, cache, PROMPTS, 1, return_logits=True)
    a, b = ref_logits[0], q_logits[0]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    assert (cos > 0.98).all(), cos
