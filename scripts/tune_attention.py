"""Sweep decode-attention launch parameters over (batch, context) points.

GPU time from graph replay (as decode runs), every candidate checked against
the reference first. Prints the best launch per point, the v1 kernel and the
current defaults for comparison, and the single launch config that is best
overall, which becomes ``DEFAULT_LAUNCH``.

    python scripts/tune_attention.py | tee results/attention_tune.txt
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mi300_llm import ref_ops  # noqa: E402
from mi300_llm.kernels import decode_attention as v2  # noqa: E402
from mi300_llm.kernels import decode_attention_v1 as v1  # noqa: E402

HQ, HKV, D, PAGE = 64, 8, 128, 16
POINTS = [(1, 1024), (1, 8192), (1, 32768), (8, 2048), (8, 8192), (32, 128), (32, 2048), (32, 8192)]
LAUNCHES = list(itertools.product([16, 32, 64, 128], [2, 4, 8], [1, 2, 3]))  # block_n, warps, stages


def gpu_us(fn) -> float:
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench_cudagraph(fn, rep=50) * 1e3


def setup(b, ctx):
    nblk = triton.cdiv(ctx, PAGE)
    k = torch.randn(b * nblk, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    tables = torch.randperm(b * nblk, device="cuda").to(torch.int32).view(b, nblk)
    lens = torch.full((b,), ctx, device="cuda", dtype=torch.int32)
    q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
    return q, k, v, tables, lens


def main():
    scale = D**-0.5
    table = {}  # (point, launch) -> best us over split choices
    for b, ctx in POINTS:
        q, k, v, tables, lens = setup(b, ctx)
        ref = ref_ops.paged_decode_attention(q, k, v, tables, lens, scale) if b * ctx <= 65536 else None
        nbytes = 2 * b * ctx * HKV * D * 2
        h = v2.default_num_splits(b, HKV, tables.shape[1] * PAGE)
        splits = sorted({max(1, h // 2), h, min(128, h * 2)})
        t_v1 = gpu_us(lambda: v1.paged_decode_attention(q, k, v, tables, lens, scale))
        t_def = gpu_us(lambda: v2.paged_decode_attention(q, k, v, tables, lens, scale))
        best = (math.inf, None)
        for (bn, w, st), s in itertools.product(LAUNCHES, splits):
            if s * bn > ctx and s > 1:
                continue
            f = lambda: v2.paged_decode_attention(q, k, v, tables, lens, scale, num_splits=s,  # noqa: E731
                                                  block_n=bn, num_warps=w, num_stages=st)
            try:
                out = f()
                if ref is not None and not torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-2):
                    print(f"  WRONG RESULT b={b} ctx={ctx} split={s} launch={(bn, w, st)}")
                    continue
                t = gpu_us(f)
            except Exception as e:  # noqa: BLE001  (e.g. out of LDS)
                print(f"  skip {(bn, w, st)} split={s}: {type(e).__name__}")
                continue
            key = ((b, ctx), (bn, w, st))
            table[key] = min(table.get(key, math.inf), t)
            if t < best[0]:
                best = (t, (s, bn, w, st))
        t, (s, bn, w, st) = best
        print(f"B={b:<3} ctx={ctx:<6} v1 {t_v1:7.1f} us | default {t_def:7.1f} us | best {t:7.1f} us "
              f"({nbytes / t / 1e6:.2f} TB/s) split={s} block_n={bn} warps={w} stages={st}", flush=True)

    # One launch config for every point: minimize the geometric mean of time / per-point best.
    best_at = {p: min(t for (pp, _), t in table.items() if pp == p) for p, _ in {k for k in table}}
    scores = {}
    for launch in LAUNCHES:
        ratios = [table[(p, launch)] / best_at[p] for p in best_at if (p, launch) in table]
        if len(ratios) == len(best_at):
            scores[launch] = math.exp(sum(map(math.log, ratios)) / len(ratios))
    for launch, sc in sorted(scores.items(), key=lambda x: x[1])[:5]:
        print(f"overall: block_n={launch[0]} warps={launch[1]} stages={launch[2]}  geomean time/best = {sc:.3f}")


if __name__ == "__main__":
    main()
