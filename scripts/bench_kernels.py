"""Per-kernel timings on Llama-3.3-70B decode shapes, with achieved bandwidth.

Decode kernels are memory-bound, so the figure of merit is bytes moved / time
against the MI300X's peak HBM bandwidth (5.3 TB/s spec).

    python scripts/bench_kernels.py --csv results/kernels.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mi300_llm import kernels  # noqa: E402
from mi300_llm.fp8 import compute_fp8_dtype, quantize_per_channel  # noqa: E402

PEAK_BW = 5.3e12
H, I, HQ, HKV, D = 8192, 28672, 64, 8, 128
GEMMS = {"qkv": (H + 2 * HKV * D, H), "o": (H, H), "gate_up": (2 * I, H), "down": (H, I)}


def bench(fn) -> float:
    return triton.testing.do_bench(fn, warmup=25, rep=100) * 1e-3  # seconds


def gemm_rows(batches):
    dt = compute_fp8_dtype()
    for name, (n, k) in GEMMS.items():
        wq, ws = quantize_per_channel(torch.randn(n, k, device="cuda") / k**0.5, dt)
        ws = ws.reshape(-1)
        for m in batches:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            xq, xs = kernels.quantize_per_token(x, dt)
            nbytes = n * k + m * k + 2 * m * n
            yield dict(op=f"fp8_gemm.{name}", M=m, N=n, K=k, bytes=nbytes, t=bench(lambda: kernels.fp8_gemm(xq, xs, wq, ws, torch.bfloat16)))
            try:  # hipBLASLt via torch, as a vendor-library baseline
                f = lambda: torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws.view(1, -1), out_dtype=torch.bfloat16)  # noqa: E731
                f()
                yield dict(op=f"scaled_mm.{name}", M=m, N=n, K=k, bytes=nbytes, t=bench(f))
            except Exception as e:  # noqa: BLE001
                print(f"torch._scaled_mm unavailable for {name}: {type(e).__name__}: {e}"[:200])


def attn_rows(batches, ctxs, page=16):
    for b in batches:
        for ctx in ctxs:
            nblk = (ctx + page - 1) // page
            k = torch.randn(b * nblk, page, HKV, D, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            tables = torch.arange(b * nblk, device="cuda", dtype=torch.int32).view(b, nblk)
            lens = torch.full((b,), ctx, device="cuda", dtype=torch.int32)
            q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
            nbytes = 2 * b * ctx * HKV * D * 2
            t = bench(lambda: kernels.paged_decode_attention(q, k, v, tables, lens, D**-0.5))
            yield dict(op="decode_attn", M=b, N=ctx, K=D, bytes=nbytes, t=t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 4096, 16384])
    ap.add_argument("--csv")
    args = ap.parse_args()

    rows = list(gemm_rows(args.batches)) + list(attn_rows(args.batches, args.ctx))
    print(f"{'op':<20}{'M':>6}{'N':>8}{'K':>7}{'us':>10}{'TB/s':>8}{'%peak':>7}")
    for r in rows:
        r["us"] = r["t"] * 1e6
        r["tbps"] = r["bytes"] / r["t"] / 1e12
        r["pct_peak"] = 100 * r["bytes"] / r["t"] / PEAK_BW
        print(f"{r['op']:<20}{r['M']:>6}{r['N']:>8}{r['K']:>7}{r['us']:>10.1f}{r['tbps']:>8.2f}{r['pct_peak']:>6.0f}%")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
