"""All-reduce latency at the sizes a TP decode step uses.

Each decode step all-reduces a [batch, 8192] bf16 tensor twice per layer, 160
times for Llama-3.3-70B. At batch 1 that is 16 KB: far too small to be
bandwidth-bound, so latency per call is what matters. Measured both eagerly
and replayed from a graph (as the engine runs it).

    torchrun --nproc-per-node 8 scripts/bench_allreduce.py
    torchrun --nproc-per-node 2 scripts/bench_allreduce.py   # CPU/gloo smoke test
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mi300_llm.tp import init_from_env  # noqa: E402

H, CALLS_PER_STEP = 8192, 160


def time_us(fn, iters: int, cuda: bool) -> float:
    for _ in range(5):
        fn()
    if cuda:
        torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    tp = init_from_env()
    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    dtype = torch.bfloat16 if cuda else torch.float32  # gloo lacks bf16 all-reduce
    say = print if tp.rank == 0 else (lambda *a, **k: None)
    say(f"world={tp.world} backend={dist.get_backend()} device={dev}")
    say(f"{'batch':>6}{'bytes':>10}{'eager us':>10}{'graph us':>10}{'x160 ms':>9}")
    for m in (1, 8, 32, 128):
        x = torch.ones(m, H, device=dev, dtype=dtype)
        eager = time_us(lambda: dist.all_reduce(x), 200, cuda)
        graph = float("nan")
        if cuda:
            # Capture 20 back-to-back all-reduces, as consecutive layers would.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                dist.all_reduce(x)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(20):
                    dist.all_reduce(x)
            graph = time_us(g.replay, 50, cuda) / 20
        best = graph if cuda else eager
        say(f"{m:>6}{m * H * x.element_size():>10}{eager:>10.1f}{graph:>10.1f}{best * CALLS_PER_STEP / 1e3:>9.2f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
