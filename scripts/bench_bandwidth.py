"""Achievable HBM read bandwidth on this GPU, the realistic roofline ceiling.

The spec number (5.3 TB/s on MI300X) is not reachable in practice; decode
kernels should be judged against what a pure streaming read achieves.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _stream_read(x_ptr, out_ptr, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    base = pid.to(tl.int64) * BLOCK * ITERS
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + offs, mask=offs < n, other=0).to(tl.int32)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


def main():
    for gib in (0.25, 1, 4):
        n = int(gib * 2**30)
        x = torch.randint(0, 100, (n,), dtype=torch.uint8, device="cuda")
        y = torch.empty_like(x)
        best = {}
        t = triton.testing.do_bench(lambda: y.copy_(x), warmup=10, rep=50) * 1e-3
        best["torch copy (r+w)"] = 2 * n / t
        for block, iters in ((4096, 16), (8192, 8), (16384, 4), (8192, 32)):
            grid = triton.cdiv(n, block * iters)
            out = torch.empty(grid, dtype=torch.int32, device="cuda")
            for nw in (4, 8):
                t = triton.testing.do_bench(
                    lambda: _stream_read[(grid,)](x, out, n, BLOCK=block, ITERS=iters, num_warps=nw), warmup=10, rep=50
                ) * 1e-3
                key = "triton stream read"
                best[key] = max(best.get(key, 0), n / t)
        print(f"{gib:>5} GiB: " + ", ".join(f"{k} {v / 1e12:.2f} TB/s" for k, v in best.items()))


if __name__ == "__main__":
    main()
