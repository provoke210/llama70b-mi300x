"""Benchmark prompts shared by run_generate.py (this engine) and bench_vllm.py,
so both engines decode after exactly the same prompt token ids."""

from __future__ import annotations

PROMPT = "Explain in three sentences why LLM decoding is memory-bandwidth bound."

FILLER = (
    "The memory hierarchy of a GPU spans registers, shared memory, caches and "
    "high-bandwidth memory. Each level trades capacity for latency and bandwidth. "
)


def prompt_ids(tok, prompt_len: int | None = None) -> list[int]:
    """Chat-formatted PROMPT; with ``prompt_len``, filler text tokens are inserted
    after the begin-of-text token until the prompt has exactly that many tokens.
    The content is meaningless; it sets the KV length that decode reads."""
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text, add_special_tokens=False)
    if not prompt_len:
        return ids
    need = prompt_len - len(ids)
    if need < 0:
        raise ValueError(f"prompt_len={prompt_len} is shorter than the base prompt ({len(ids)} tokens)")
    unit = tok.encode(FILLER, add_special_tokens=False)
    filler = (unit * (need // len(unit) + 1))[:need]
    return ids[:1] + filler + ids[1:]
