"""Time every kernel of one Llama-3.3-70B decoder layer, in isolation.

The sum x 80 layers should roughly match the measured decode step; the
breakdown shows where the remaining time goes.

    python scripts/bench_layer.py --batches 1 8 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mi300_llm import kernels, ref_ops  # noqa: E402
from mi300_llm.fp8 import compute_fp8_dtype, quantize_per_channel  # noqa: E402

H, I, HQ, HKV, D, LAYERS = 8192, 28672, 64, 8, 128, 80


def us(fn) -> float:
    # Replay from a graph: plain do_bench reports ~6.4 us for every tiny kernel,
    # which is the CPU launch floor, not GPU time. Decode runs from a graph too.
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench_cudagraph(fn, rep=100) * 1e3


def layer_ops(m: int, ctx: int, fused: bool):
    dt, bf = compute_fp8_dtype(), torch.bfloat16
    dev = "cuda"
    w = {n: quantize_per_channel(torch.randn(N, K, device=dev) / K**0.5, dt)
         for n, (N, K) in {"qkv": (H + 2 * HKV * D, H), "o": (H, H), "gate_up": (2 * I, H), "down": (H, I)}.items()}
    w = {n: (q, s.reshape(-1)) for n, (q, s) in w.items()}
    x, res, nw = (torch.randn(m, H, device=dev, dtype=bf) for _ in range(3))
    nw = nw[0]
    qkv = torch.randn(m, H + 2 * HKV * D, device=dev, dtype=bf)
    inv = ref_ops.rope_inv_freq(D, 500000.0, None).to(dev)
    cos, sin = ref_ops.rope_cos_sin(torch.full((m,), ctx, device=dev), inv)
    cos, sin = cos.contiguous(), sin.contiguous()
    q = qkv[:, : HQ * D].view(m, HQ, D)
    page = 16
    nblk = (ctx + page - 1) // page
    kc = torch.randn(m * nblk, page, HKV, D, device=dev, dtype=bf)
    vc = torch.randn_like(kc)
    tables = torch.arange(m * nblk, device=dev, dtype=torch.int32).view(m, nblk)
    lens = torch.full((m,), ctx, device=dev, dtype=torch.int32)
    gate_up = torch.randn(m, 2 * I, device=dev, dtype=bf)
    act = torch.randn(m, I, device=dev, dtype=bf)
    slots = torch.arange(m, device=dev)
    k_new = torch.randn(m, HKV, D, device=dev, dtype=bf)

    def gemm(name, inp):
        xq, xs = kernels.quantize_per_token(inp, dt)
        wq, ws = w[name]
        return {f"quant({name})": lambda: kernels.quantize_per_token(inp, dt),
                f"gemm({name})": lambda: kernels.fp8_gemm(xq, xs, wq, ws, bf)}

    def only_gemm(name, inp):
        return {f"gemm({name})": gemm(name, inp)[f"gemm({name})"]}

    ops = {}
    if fused:
        # What model.py runs with FP8 weights: norms and SiLU emit FP8 directly.
        ops["add_rmsnorm_quant(in)"] = lambda: kernels.fused_add_rmsnorm_quant(x, res, nw, 1e-5, dt)
        ops.update(only_gemm("qkv", x))
    else:
        ops["add_rmsnorm(in)"] = lambda: kernels.fused_add_rmsnorm(x, res, nw, 1e-5)
        ops.update(gemm("qkv", x))
    ops["rope(q)"] = lambda: kernels.apply_rope(q, cos, sin)
    ops["rope(k)"] = lambda: kernels.apply_rope(qkv[:, HQ * D : HQ * D + HKV * D].view(m, HKV, D), cos, sin)
    ops["kv_write"] = lambda: (kc.view(-1, HKV, D).index_put_((slots,), k_new), vc.view(-1, HKV, D).index_put_((slots,), k_new))
    ops["decode_attn"] = lambda: kernels.paged_decode_attention(q.contiguous(), kc, vc, tables, lens, D**-0.5)
    ops.update(gemm("o", x))
    if fused:
        ops["add_rmsnorm_quant(post)"] = lambda: kernels.fused_add_rmsnorm_quant(x, res, nw, 1e-5, dt)
        ops.update(only_gemm("gate_up", x))
        ops["silu_mul_quant"] = lambda: kernels.silu_mul_quant(gate_up, dt)
        ops.update(only_gemm("down", act))
    else:
        ops["add_rmsnorm(post)"] = lambda: kernels.fused_add_rmsnorm(x, res, nw, 1e-5)
        ops.update(gemm("gate_up", x))
        ops["silu_mul"] = lambda: kernels.silu_mul(gate_up)
        ops.update(gemm("down", act))
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--ctx", type=int, default=128, help="KV length; ~the e2e benchmark's context")
    args = ap.parse_args()
    for m in args.batches:
        for fused in (False, True):
            rows = {name: us(fn) for name, fn in layer_ops(m, args.ctx, fused).items()}
            total = sum(rows.values())
            tag = "fused quant" if fused else "separate quant"
            print(f"\n== batch={m}, ctx={args.ctx}, {tag}: {total:.0f} us/layer -> {total * LAYERS / 1e3:.1f} ms/step (x{LAYERS})")
            for name, t in sorted(rows.items(), key=lambda r: -r[1]):
                print(f"  {name:<24}{t:8.1f} us  {100 * t / total:5.1f}%")
            gemm = sum(t for n, t in rows.items() if n.startswith("gemm("))
            quant = sum(t for n, t in rows.items() if "quant" in n)
            attn = rows["decode_attn"]
            print(f"  by group: gemm {gemm / total:.0%}, quant(+fused producers) {quant / total:.0%}, "
                  f"attention {attn / total:.0%}, other {(total - gemm - quant - attn) / total:.0%}")


if __name__ == "__main__":
    main()
