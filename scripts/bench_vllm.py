"""Measure vLLM on exactly the workload ``run_generate.py`` measures.

Same prompt tokens, same batch sizes, greedy, EOS ignored, and decode time
isolated the same way: (time for SHORT+N tokens - time for SHORT tokens) / N,
best of 3 each.
Prefix caching is off, since the batch repeats one prompt and our engine has
no prefix cache. Run inside a vLLM ROCm container:

    python scripts/bench_vllm.py /models/Llama-3.3-70B-Instruct-FP8-dynamic --batch 1 8 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROMPT = "Explain in three sentences why LLM decoding is memory-bandwidth bound."
SHORT = 16  # keep in sync with run_generate.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--save-tokens", default="results/vllm_tokens.json")
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text, add_special_tokens=False)

    t0 = time.time()
    llm = LLM(
        model=args.model,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        max_num_seqs=max(args.batch),
        tensor_parallel_size=args.tp,
    )
    print(f"vLLM ready in {time.time() - t0:.0f}s")

    def run(bs: int, n: int):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        t = time.time()
        outs = llm.generate([{"prompt_token_ids": ids}] * bs, sp, use_tqdm=False)
        return time.time() - t, outs

    run(1, 4)  # warm-up
    results = {}
    short, long = SHORT, SHORT + args.new_tokens
    for bs in args.batch:
        run(bs, short), run(bs, long)  # warm-up at this batch size
        t_short = min(run(bs, short)[0] for _ in range(3))
        t_long, outs = min((run(bs, long) for _ in range(3)), key=lambda r: r[0])
        step = (t_long - t_short) / args.new_tokens
        print(f"TP={args.tp} batch={bs}: {step * 1e3:.2f} ms/step decode, {bs / step:.0f} tok/s")
        results[bs] = list(outs[0].outputs[0].token_ids)
    print("---\n" + tok.decode(results[args.batch[0]]))
    Path(args.save_tokens).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_tokens).write_text(json.dumps({"prompt": ids, "tokens": results[args.batch[0]]}))


if __name__ == "__main__":
    main()
