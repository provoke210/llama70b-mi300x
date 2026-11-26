"""Sweep FP8 GEMM tile configs on the Llama-3.3-70B decode shapes.

Every candidate is checked against an fp32 reference before it is timed, and
the winners are written to ``mi300_llm/kernels/gemm_configs.json``.

    python scripts/tune_gemm.py --batches 1 8 32
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib  # noqa: E402

from mi300_llm.fp8 import compute_fp8_dtype, quantize_per_channel  # noqa: E402

# The package re-exports a function named fp8_gemm, so import the module by path.
G = importlib.import_module("mi300_llm.kernels.fp8_gemm")

H, I, KV = 8192, 28672, 8 * 128
SHAPES = {"qkv": (H + 2 * KV, H), "o": (H, H), "gate_up": (2 * I, H), "down": (H, I)}


def candidates(M: int):
    bms = [16] if M <= 16 else [32] if M <= 32 else [64]
    for bm, bn, bk, split, nw in itertools.product(bms, [32, 64, 128], [128, 256, 512], [1, 2, 4, 8], [4, 8]):
        if bn * bk > 64 * 512:  # keep the B tile within a sane register/LDS budget
            continue
        yield dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=1, SPLIT_K=split, num_warps=nw, num_stages=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    args = ap.parse_args()
    dt = compute_fp8_dtype()
    table = json.loads(G.CONFIG_FILE.read_text()) if G.CONFIG_FILE.exists() else {}
    for name, (n, k) in SHAPES.items():
        wq, ws = quantize_per_channel(torch.randn(n, k, device="cuda") / k**0.5, dt)
        ws = ws.reshape(-1)
        for m in args.batches:
            xq, xs = G.quantize_per_token(torch.randn(m, k, device="cuda", dtype=torch.bfloat16), dt)
            exact = (xq.float() @ wq.float().t()) * xs * ws.view(1, -1)
            base = G._heuristic_config(m, n, k)
            t_base = triton.testing.do_bench(lambda: G.fp8_gemm(xq, xs, wq, ws, torch.bfloat16, base), warmup=5, rep=30)
            best, t_best, tried, t0 = base, t_base, 0, time.time()
            for cfg in candidates(m):
                try:
                    out = G.fp8_gemm(xq, xs, wq, ws, torch.float32, cfg)
                    if not torch.allclose(out, exact, rtol=1e-3, atol=1e-3):
                        print(f"  WRONG RESULT {cfg}")
                        continue
                    t = triton.testing.do_bench(lambda: G.fp8_gemm(xq, xs, wq, ws, torch.bfloat16, cfg), warmup=5, rep=30)
                except Exception as e:  # noqa: BLE001  (e.g. out of shared memory)
                    print(f"  skip {cfg}: {type(e).__name__}")
                    continue
                tried += 1
                if t < t_best:
                    best, t_best = cfg, t
            tb = (n * k) / (t_best * 1e-3) / 1e12
            print(f"{name:8s} M={m:<3} heuristic {t_base * 1e3:7.1f} us -> best {t_best * 1e3:7.1f} us ({tb:.2f} TB/s) "
                  f"[{tried} cfgs, {time.time() - t0:.0f}s] {best}", flush=True)
            table.setdefault(f"{n}x{k}", {})[str(m)] = best
    G.CONFIG_FILE.write_text(json.dumps(table, indent=1, sort_keys=True))
    print(f"wrote {G.CONFIG_FILE}")


if __name__ == "__main__":
    main()
