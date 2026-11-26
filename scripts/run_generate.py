"""Load a checkpoint, generate greedily, report decode speed.

    python scripts/run_generate.py ~/models/Llama-3.2-1B-Instruct-FP8-dynamic --compare-ref
    python scripts/run_generate.py ~/models/Llama-3.3-70B-Instruct-FP8-dynamic --batch 1 8 32

Tensor parallel: launch one process per GPU with torchrun; every rank runs the
same loop in lockstep and only rank 0 prints.

    torchrun --nproc-per-node 4 scripts/run_generate.py <model> --graph --batch 1 8 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mi300_llm import KVCache, load_pretrained, ops  # noqa: E402
from mi300_llm.engine import generate  # noqa: E402
from mi300_llm.fp8 import compute_fp8_dtype  # noqa: E402
from mi300_llm.tp import init_from_env  # noqa: E402

PROMPT = "Explain in three sentences why LLM decoding is memory-bandwidth bound."
SHORT = 16  # tokens in the short timing run; keep in sync with bench_vllm.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--backend", default="triton", choices=["ref", "triton"])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--kv-gib", type=float, default=40.0)
    ap.add_argument("--compare-ref", action="store_true", help="also run the reference backend and diff tokens")
    ap.add_argument("--graph", action="store_true", help="replay decode steps from a captured HIP graph")
    ap.add_argument("--save-tokens", help="write batch-0 prompt and output token ids as JSON")
    ap.add_argument("--compare-tokens", help="JSON from bench_vllm.py: report the first diverging token")
    args = ap.parse_args()
    tp = init_from_env()
    import builtins

    print = builtins.print if tp.rank == 0 else (lambda *a, **k: None)  # noqa: A001

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    # Render to text, then encode: the return type of apply_chat_template with
    # tokenize=True changed across transformers versions.
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text, add_special_tokens=False)

    t0 = time.time()
    model = load_pretrained(args.model, device="cuda", fp8_dtype=compute_fp8_dtype(), tp=tp)
    torch.cuda.synchronize()
    print(f"TP={tp.world}: loaded in {time.time() - t0:.1f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB per GPU")

    block = 16
    per_block = KVCache.bytes_per_token(model.cfg, torch.bfloat16) * block
    cache = KVCache(model.cfg, int(args.kv_gib * 2**30 // per_block), block, torch.bfloat16, "cuda")
    print(f"KV cache: {cache.num_blocks * block} tokens")

    ops.set_backend(args.backend)
    generate(model, cache, [ids], 4)  # warm up / JIT compile
    graphs: dict = {}

    for bs in args.batch:
        prompts = [ids] * bs
        if args.graph:  # capture outside the timed region
            generate(model, cache, prompts, args.new_tokens, use_graph=True, graphs=graphs)
        # Time decode only: the difference between a short and a long run,
        # which contain the same prefill. (Subtracting a 1-token run instead
        # is fragile: one slow prefill skews every step by ~3 ms.) Best of 3
        # for each, as bench_vllm.py does. Graphs are captured per width, so
        # warm both lengths first.
        def run(n):
            torch.cuda.synchronize()
            t0 = time.time()
            out = generate(model, cache, prompts, n, use_graph=args.graph, graphs=graphs)
            torch.cuda.synchronize()
            return time.time() - t0, out
        short, long = SHORT, SHORT + args.new_tokens
        run(short), run(long)
        t_short = min(run(short)[0] for _ in range(3))
        t_long, out = min((run(long) for _ in range(3)), key=lambda r: r[0])
        step = (t_long - t_short) / args.new_tokens
        print(f"TP={tp.world} batch={bs}: {step * 1e3:.2f} ms/step decode, {bs / step:.0f} tok/s")
    print("---\n" + tok.decode(out[0]))
    if args.save_tokens and tp.rank == 0:
        import json

        Path(args.save_tokens).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_tokens).write_text(json.dumps({"prompt": ids, "tokens": out[0]}))
    if args.compare_tokens:
        import json

        other = json.loads(Path(args.compare_tokens).read_text())
        assert other["prompt"] == ids, "different prompt tokens"
        same = next((i for i, (a, b) in enumerate(zip(other["tokens"], out[0])) if a != b), min(len(out[0]), len(other["tokens"])))
        print(f"vs {args.compare_tokens}: first {same}/{len(out[0])} greedy tokens identical")

    if args.compare_ref:
        ops.set_backend("ref")
        ref = generate(model, cache, [ids], args.new_tokens)[0]
        same = next((i for i, (a, b) in enumerate(zip(ref, out[0])) if a != b), len(ref))
        print(f"triton vs ref: first {same}/{len(ref)} tokens identical")

    if tp.enabled:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
