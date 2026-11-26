"""GSM8K accuracy of this engine vs vLLM, with identical prompts and greedy decoding.

Zero-shot chain of thought on the full test split (1319 questions). Both
engines get the same prompt token ids, decode greedily for up to
--max-new-tokens and stop at the model's EOS tokens; the answer is the number
after the last "####" (falling back to the last number in the response).

    python scripts/eval_gsm8k.py prepare                       # once, anywhere: writes data/gsm8k_test.jsonl
    python scripts/eval_gsm8k.py ours <model> --out results/gsm8k_ours.jsonl
    python scripts/eval_gsm8k.py vllm <model> --out results/gsm8k_vllm.jsonl   # in a vLLM container
    python scripts/eval_gsm8k.py report results/gsm8k_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path("data/gsm8k_test.jsonl")
INSTRUCTION = (
    "Solve the following math problem step by step. At the end, write the final "
    'answer as a number on its own line, in the form "#### <number>".'
)
NUM = r"-?[\d,]*\.?\d+"


def prepare(_args):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with DATA.open("w") as f:
        for i, row in df.iterrows():
            gold = row["answer"].split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({"id": int(i), "question": row["question"], "gold": gold}) + "\n")
    print(f"wrote {len(df)} questions to {DATA}")


def load_questions(limit: int | None):
    qs = [json.loads(line) for line in DATA.read_text().splitlines()]
    return qs[:limit] if limit else qs


def prompt_ids(tok, question: str) -> list[int]:
    msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\n{question}"}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def extract(text: str) -> str | None:
    marked = re.findall(r"####\s*\$?\s*(" + NUM + ")", text)
    nums = marked or re.findall(NUM, text)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return math.isclose(float(pred), float(gold), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return False


def stop_token_ids(model_path: str, tok) -> set[int]:
    ids = {tok.eos_token_id}
    gc = Path(model_path) / "generation_config.json"
    if gc.exists():
        eos = json.loads(gc.read_text()).get("eos_token_id")
        ids |= set(eos if isinstance(eos, list) else [eos])
    return {i for i in ids if i is not None}


def write_results(out: str, qs, outputs, tok, meta: dict):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for q, toks in zip(qs, outputs):
            text = tok.decode(toks, skip_special_tokens=True)
            pred = extract(text)
            ok = is_correct(pred, q["gold"])
            n_ok += ok
            f.write(json.dumps({"id": q["id"], "gold": q["gold"], "pred": pred, "correct": ok,
                                "marked": "####" in text, "n_tokens": len(toks), "response": text}) + "\n")
    print(f"{meta['engine']}: {n_ok}/{len(qs)} = {n_ok / len(qs):.2%}  ({meta['seconds']:.0f}s) -> {out}")


def run_ours(args):
    import torch
    from transformers import AutoTokenizer

    from mi300_llm import KVCache, load_pretrained, ops
    from mi300_llm.engine import generate
    from mi300_llm.fp8 import compute_fp8_dtype

    tok = AutoTokenizer.from_pretrained(args.model)
    qs = load_questions(args.limit)
    prompts = [prompt_ids(tok, q["question"]) for q in qs]
    stops = stop_token_ids(args.model, tok)
    model = load_pretrained(args.model, device="cuda", fp8_dtype=compute_fp8_dtype())
    block = 16
    per_block = KVCache.bytes_per_token(model.cfg, torch.bfloat16) * block
    cache = KVCache(model.cfg, int(args.kv_gib * 2**30 // per_block), block, torch.bfloat16, "cuda")
    ops.set_backend("triton")
    # One graph width for every batch, so each batch size is captured once.
    width = (max(map(len, prompts)) + args.max_new_tokens + block - 1) // block
    graphs: dict = {}
    outputs, t0 = [], time.time()
    for b in range(0, len(prompts), args.batch):
        chunk = prompts[b : b + args.batch]
        outputs += generate(model, cache, chunk, args.max_new_tokens, use_graph=True, graphs=graphs,
                            stop_ids=stops, graph_width=width)
        done = b + len(chunk)
        print(f"  {done}/{len(prompts)} questions, {time.time() - t0:.0f}s", flush=True)
    meta = dict(engine="ours", model=args.model, max_new_tokens=args.max_new_tokens, batch=args.batch,
                seconds=time.time() - t0, fuse_quant=__import__("mi300_llm.model").model.FUSE_QUANT)
    write_results(args.out, qs, outputs, tok, meta)


def run_vllm(args):
    import os

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    qs = load_questions(args.limit)
    prompts = [{"prompt_token_ids": prompt_ids(tok, q["question"])} for q in qs]
    stops = stop_token_ids(args.model, tok)
    llm = LLM(model=args.model, max_model_len=4096, gpu_memory_utilization=0.9, enable_prefix_caching=False,
              max_num_seqs=args.batch)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop_token_ids=sorted(stops))
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    meta = dict(engine="vllm" + ("+aiter" if os.environ.get("VLLM_ROCM_USE_AITER") == "1" else ""),
                model=args.model, max_new_tokens=args.max_new_tokens, batch=args.batch, seconds=time.time() - t0)
    write_results(args.out, qs, [list(o.outputs[0].token_ids) for o in outs], tok, meta)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def report(args):
    runs = {}
    for path in args.files:
        lines = Path(path).read_text().splitlines()
        meta, rows = json.loads(lines[0])["meta"], [json.loads(x) for x in lines[1:]]
        runs[meta["engine"]] = {r["id"]: r for r in rows}
        k, n = sum(r["correct"] for r in rows), len(rows)
        lo, hi = wilson(k, n)
        unmarked = sum(not r["marked"] for r in rows)
        trunc = sum(r["n_tokens"] >= meta["max_new_tokens"] for r in rows)
        print(f"{meta['engine']:<12} {k:>5}/{n} = {k / n:6.2%}  95% CI [{lo:.2%}, {hi:.2%}]  "
              f"no '####': {unmarked}  hit max tokens: {trunc}  ({meta['seconds']:.0f}s)")
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ids = runs[a].keys() & runs[b].keys()
            both = sum(runs[a][j]["correct"] and runs[b][j]["correct"] for j in ids)
            only_a = sum(runs[a][j]["correct"] and not runs[b][j]["correct"] for j in ids)
            only_b = sum(runs[b][j]["correct"] and not runs[a][j]["correct"] for j in ids)
            same_pred = sum(runs[a][j]["pred"] == runs[b][j]["pred"] for j in ids)
            same_text = sum(runs[a][j]["response"] == runs[b][j]["response"] for j in ids)
            print(f"{a} vs {b}: both correct {both}, only {a} {only_a}, only {b} {only_b}; "
                  f"same answer {same_pred}/{len(ids)}, identical response {same_text}/{len(ids)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    for name in ("ours", "vllm"):
        p = sub.add_parser(name)
        p.add_argument("model")
        p.add_argument("--out", required=True)
        p.add_argument("--limit", type=int)
        p.add_argument("--batch", type=int, default=32)
        p.add_argument("--max-new-tokens", type=int, default=512)
        p.add_argument("--kv-gib", type=float, default=20.0)
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    args = ap.parse_args()
    {"prepare": prepare, "ours": run_ours, "vllm": run_vllm, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
