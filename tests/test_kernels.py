"""Each Triton kernel against its reference. Runs on a GPU, or on CPU under
TRITON_INTERPRET=1 (Linux only: Triton has no macOS wheels)."""

import os

import pytest
import torch

triton = pytest.importorskip("triton")

from mi300_llm import kernels, ref_ops  # noqa: E402
from mi300_llm.fp8 import compute_fp8_dtype, quantize_per_channel  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
if not torch.cuda.is_available() and not INTERPRET:
    pytest.skip("needs a GPU or TRITON_INTERPRET=1", allow_module_level=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16


def _close(a, b, tol=2e-2):
    torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("M,N", [(1, 8192), (7, 8192), (33, 1000)])
def test_rmsnorm(M, N):
    x = torch.randn(M, N, device=DEV, dtype=DT)
    r = torch.randn(M, N, device=DEV, dtype=DT)
    w = torch.randn(N, device=DEV, dtype=DT)
    _close(kernels.rmsnorm(x, w, 1e-5), ref_ops.rmsnorm(x, w, 1e-5))
    out, res = kernels.fused_add_rmsnorm(x, r, w, 1e-5)
    ref_out, ref_res = ref_ops.fused_add_rmsnorm(x, r, w, 1e-5)
    _close(res, ref_res, 0)
    _close(out, ref_out)


@pytest.mark.parametrize("T,H,D", [(1, 64, 128), (5, 8, 128), (3, 32, 64)])
def test_rope_on_strided_view(T, H, D):
    inv = ref_ops.rope_inv_freq(D, 500000.0, None).to(DEV)
    cos, sin = ref_ops.rope_cos_sin(torch.arange(T, device=DEV) * 37, inv)
    # Mimic q sliced out of the fused QKV output.
    qkv = torch.randn(T, H * D + 2 * 8 * D, device=DEV, dtype=DT)
    x = qkv[:, : H * D].view(T, H, D)
    _close(kernels.apply_rope(x, cos.contiguous(), sin.contiguous()), ref_ops.apply_rope(x, cos, sin))


def test_silu_mul():
    x = torch.randn(9, 2 * 3000, device=DEV, dtype=DT)
    _close(kernels.silu_mul(x), ref_ops.silu_mul(x))


@pytest.mark.parametrize("M,N,K", [(1, 1280, 8192), (16, 10240, 8192), (40, 8192, 28672), (200, 512, 1024)])
def test_fp8_linear(M, N, K):
    dt = compute_fp8_dtype()
    x = torch.randn(M, K, device=DEV, dtype=DT)
    wq, ws = quantize_per_channel(torch.randn(N, K, device=DEV) / K**0.5, dt)
    ws = ws.reshape(-1)
    ours = kernels.fp8_linear(x, wq, ws)
    ref = ref_ops.fp8_linear(x, wq, ws)
    # Same quantized inputs; only accumulation order and bf16 rounding differ.
    _close(ours, ref, 3e-2)


@pytest.mark.parametrize("split", [1, 2, 3, 8])
@pytest.mark.parametrize("M,N,K", [(1, 8192, 28672), (5, 1000, 8192 + 64), (32, 640, 4096)])
def test_fp8_gemm_split_k(M, N, K, split):
    """Split-K (including uneven splits and a ragged K tail) is exact in fp32."""
    dt = compute_fp8_dtype()
    xq, xs = kernels.quantize_per_token(torch.randn(M, K, device=DEV, dtype=DT), dt)
    wq, ws = quantize_per_channel(torch.randn(N, K, device=DEV) / K**0.5, dt)
    ws = ws.reshape(-1)
    exact = (xq.float() @ wq.float().t()) * xs * ws.view(1, -1)
    cfg = dict(BLOCK_M=16 if M <= 16 else 32, BLOCK_N=64, BLOCK_K=256, GROUP_M=1, SPLIT_K=split, num_warps=4, num_stages=2)
    out = kernels.fp8_gemm(xq, xs, wq, ws, torch.float32, config=cfg)
    torch.testing.assert_close(out, exact, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("lens", [[1], [7, 300, 64], [4097, 15, 2000, 1]])
@pytest.mark.parametrize("hq,hkv,d", [(64, 8, 128), (32, 8, 64)])
def test_paged_decode_attention(lens, hq, hkv, d):
    torch.manual_seed(0)
    page, B = 16, len(lens)
    max_blocks = (max(lens) + page - 1) // page
    num_blocks = B * max_blocks + 3
    k_cache = torch.randn(num_blocks, page, hkv, d, device=DEV, dtype=DT)
    v_cache = torch.randn_like(k_cache)
    # Shuffled, non-contiguous block assignment, as a real allocator produces.
    perm = torch.randperm(num_blocks)[: B * max_blocks].view(B, max_blocks)
    tables = perm.to(torch.int32).to(DEV)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=DEV)
    q = torch.randn(B, hq, d, device=DEV, dtype=DT)
    scale = d**-0.5
    ref = ref_ops.paged_decode_attention(q, k_cache, v_cache, tables, seq_lens, scale)
    for splits in (1, 3, None):
        ours = kernels.paged_decode_attention(q, k_cache, v_cache, tables, seq_lens, scale, num_splits=splits)
        _close(ours, ref)


def test_tiny_model_triton_matches_ref(hf_tiny, tiny_cfg):
    """Whole model, FP8 linears: the Triton backend generates the same greedy
    tokens as the reference backend."""
    from mi300_llm import KVCache, from_state_dict, ops
    from mi300_llm.engine import generate

    prompts = [[1, 17, 42, 99, 5, 250, 3], [7, 8, 9], list(range(20, 60))]
    model = from_state_dict(tiny_cfg, hf_tiny.state_dict(), dtype=DT, device=DEV, fp8_dtype=compute_fp8_dtype(), quantize=True)
    cache = KVCache(tiny_cfg, num_blocks=64, block_size=16, dtype=DT, device=DEV)
    results = {}
    for backend in ("ref", "triton"):
        ops.set_backend(backend)
        results[backend] = generate(model, cache, prompts, max_new_tokens=6, return_logits=True)
    ops.set_backend("ref")
    (ref_tok, ref_logits), (tri_tok, tri_logits) = results["ref"], results["triton"]
    # bf16 accumulation order can flip a near-tie; once a sequence's tokens
    # differ its later logits describe different contexts, so stop there.
    for seq in range(len(prompts)):
        for step in range(6):
            cos = torch.nn.functional.cosine_similarity(ref_logits[step][seq], tri_logits[step][seq], dim=0)
            assert cos > 0.995, (seq, step, cos)
            if ref_tok[seq][step] != tri_tok[seq][step]:
                break
    assert [t[0] for t in ref_tok] == [t[0] for t in tri_tok]


def _same_codes(a, b, max_frac_one_step=0.0):
    """Equal scales; codes equal, except at most ``max_frac_one_step`` of them
    may sit one FP8 grid step apart."""
    (qa, sa), (qb, sb) = a, b
    assert torch.equal(sa, sb), (sa - sb).abs().max()
    diff = qa.view(torch.uint8) != qb.view(torch.uint8)
    if not diff.any():
        return
    assert max_frac_one_step > 0.0, diff.sum()
    assert diff.float().mean() <= max_frac_one_step, diff.sum()
    # One grid step: adjacent sign-magnitude codes of the same sign.
    ca, cb = qa.view(torch.uint8)[diff].int(), qb.view(torch.uint8)[diff].int()
    assert ((ca & 0x80) == (cb & 0x80)).all() and ((ca & 0x7F) - (cb & 0x7F)).abs().max() <= 1


@pytest.mark.parametrize("M", [1, 7, 32, 2048])
@pytest.mark.parametrize("residual", [False, True])
def test_rmsnorm_quant_matches_unfused(M, residual):
    """Fused norm+quant equals quantize(norm(...)) of the unfused kernels, up to
    rare one-step codes from a reordered row sum (see add_rmsnorm_row)."""
    dt = compute_fp8_dtype()
    x, r = (torch.randn(M, 8192, device=DEV, dtype=DT) * 3 for _ in range(2))
    w = torch.randn(8192, device=DEV, dtype=DT)
    if residual:
        q, s, res = kernels.fused_add_rmsnorm_quant(x, r, w, 1e-5, dt)
        y, ref_res = kernels.fused_add_rmsnorm(x, r, w, 1e-5)
        assert torch.equal(res, ref_res)
    else:
        q, s = kernels.rmsnorm_quant(x, w, 1e-5, dt)
        y = kernels.rmsnorm(x, w, 1e-5)
    _same_codes((q, s), kernels.quantize_per_token(y, dt), max_frac_one_step=1e-5)


@pytest.mark.parametrize("M,I", [(1, 28672), (9, 3000), (32, 28672)])
def test_silu_mul_quant_matches_unfused(M, I):
    dt = compute_fp8_dtype()
    gate_up = torch.randn(M, 2 * I, device=DEV, dtype=DT) * 2
    _same_codes(kernels.silu_mul_quant(gate_up, dt), kernels.quantize_per_token(kernels.silu_mul(gate_up), dt))


@pytest.mark.parametrize("M,K", [(1, 28672), (3, 1000), (40, 8192)])
def test_quant_against_reference(M, K):
    """Two-stage quantization equals the (float64-exact) torch reference."""
    from mi300_llm.fp8 import quantize_per_token as ref_quant

    dt = compute_fp8_dtype()
    x = torch.randn(M, K, device=DEV, dtype=DT) * 5
    _same_codes(kernels.quantize_per_token(x, dt), ref_quant(x, dt))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="graphs need a GPU")
def test_graph_decode_matches_eager(hf_tiny, tiny_cfg):
    """Replaying a captured decode step gives exactly the eager result."""
    from mi300_llm import KVCache, from_state_dict, ops
    from mi300_llm.engine import generate

    prompts = [[1, 17, 42, 99, 5, 250, 3], [7, 8, 9], list(range(20, 60))]
    model = from_state_dict(tiny_cfg, hf_tiny.state_dict(), dtype=DT, device=DEV, fp8_dtype=compute_fp8_dtype(), quantize=True)
    cache = KVCache(tiny_cfg, num_blocks=64, block_size=16, dtype=DT, device=DEV)
    ops.set_backend("triton")
    try:
        graphs = {}
        g_tok, g_logits = generate(model, cache, prompts, 10, return_logits=True, use_graph=True, graphs=graphs)
        g2_tok = generate(model, cache, prompts, 10, use_graph=True, graphs=graphs)  # replay a cached graph
        e_tok, e_logits = generate(model, cache, prompts, 10, return_logits=True)
    finally:
        ops.set_backend("ref")
    assert g_tok == e_tok == g2_tok
    # Eager pads the block table to the live width, the graph to the final
    # width, so split counts may differ; logits agree to bf16 noise.
    for a, b in zip(g_logits, e_logits):
        torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2)
    assert cache.num_free_blocks == cache.num_blocks


def test_quant_codes_are_correctly_rounded():
    """Triton per-token quantization matches a float64 computation bit for bit."""
    dt = compute_fp8_dtype()
    torch.manual_seed(0)
    x = torch.randn(40, 28672, device=DEV, dtype=DT)
    q, s = kernels.quantize_per_token(x, dt)
    fmax = torch.finfo(dt).max
    exact_s = (x.double().abs().amax(-1, keepdim=True) / fmax).float()
    exact_q = (x.double() / exact_s.double()).clamp(-fmax, fmax).to(dt)
    assert torch.equal(s, exact_s)
    assert torch.equal(q.view(torch.uint8), exact_q.view(torch.uint8))
