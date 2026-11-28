"""Minimal greedy generation: batched prefill, then batched decode steps.

Decode steps can run through a captured HIP/CUDA graph: one decode step is
~1600 small kernel launches for 70B, and launching them from Python costs more
than the GPU work at small batch. The graph is captured once per
(batch, block-table width) with static input buffers and replayed every step.

Continuous batching (admitting new requests mid-decode) comes later; this loop
is what the correctness tests and benchmarks drive.
"""

from __future__ import annotations

import torch

from .kv_cache import KVCache, Sequence
from .model import AttnMeta, LlamaModel


def _prefill(model: LlamaModel, cache: KVCache, seqs: list[Sequence]) -> torch.Tensor:
    dev = model.device
    ids, pos, slots, cu = [], [], [], [0]
    for s in seqs:
        s.ensure_blocks(cache, len(s))
        ids += s.tokens
        pos += range(len(s))
        slots += s.slots(0, len(s), cache.block_size)
        cu.append(cu[-1] + len(s))
        s.num_cached = len(s)
    meta = AttnMeta(True, torch.tensor(slots, device=dev), cu_seqlens=cu)
    return model.forward(torch.tensor(ids, device=dev), torch.tensor(pos, device=dev), meta, cache)


def _decode_inputs(cache: KVCache, seqs: list[Sequence], width: int | None = None):
    """Host-side tensors for one decode step; the last token of each sequence is new."""
    width = width or max(len(s.blocks) for s in seqs)
    tables = torch.zeros(len(seqs), width, dtype=torch.int32)
    for i, s in enumerate(seqs):
        tables[i, : len(s.blocks)] = torch.tensor(s.blocks, dtype=torch.int32)
    return dict(
        ids=torch.tensor([s.tokens[-1] for s in seqs]),
        pos=torch.tensor([len(s) - 1 for s in seqs]),
        slots=torch.tensor([s.slots(len(s) - 1, len(s), cache.block_size)[0] for s in seqs]),
        tables=tables,
        lens=torch.tensor([len(s) for s in seqs], dtype=torch.int32),
    )


def decode_meta(cache: KVCache, seqs: list[Sequence], device) -> AttnMeta:
    x = _decode_inputs(cache, seqs)
    return AttnMeta(False, x["slots"].to(device), block_tables=x["tables"].to(device), seq_lens=x["lens"].to(device))


class GraphDecoder:
    """One captured decode step for a fixed batch size and block-table width."""

    def __init__(self, model: LlamaModel, cache: KVCache, batch: int, width: int):
        dev = model.device
        self.model, self.cache, self.width = model, cache, width
        self.ids = torch.zeros(batch, dtype=torch.long, device=dev)
        self.pos = torch.zeros(batch, dtype=torch.long, device=dev)
        self.slots = torch.zeros(batch, dtype=torch.long, device=dev)
        self.tables = torch.zeros(batch, width, dtype=torch.int32, device=dev)
        # Dummy lens of 1 for warm-up/capture: attention reads slot 0 only.
        self.lens = torch.ones(batch, dtype=torch.int32, device=dev)
        self.meta = AttnMeta(False, self.slots, block_tables=self.tables, seq_lens=self.lens)
        # Warm up on a side stream (Triton JIT, workspace allocation), then capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                model.forward(self.ids, self.pos, self.meta, cache)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = model.forward(self.ids, self.pos, self.meta, cache)

    def __call__(self, cache: KVCache, seqs: list[Sequence]) -> torch.Tensor:
        return self.replay(_decode_inputs(cache, seqs, self.width))

    def replay(self, x: dict) -> torch.Tensor:
        """Run the captured step on host tensors shaped like ``_decode_inputs``
        output (rows = the captured batch, tables = the captured width)."""
        self.ids.copy_(x["ids"], non_blocking=True)
        self.pos.copy_(x["pos"], non_blocking=True)
        self.slots.copy_(x["slots"], non_blocking=True)
        self.tables.copy_(x["tables"], non_blocking=True)
        self.lens.copy_(x["lens"], non_blocking=True)
        self.graph.replay()
        return self.logits


def _decode(model: LlamaModel, cache: KVCache, seqs: list[Sequence], graph: GraphDecoder | None) -> torch.Tensor:
    for s in seqs:
        s.ensure_blocks(cache, len(s))
    if graph is not None:
        out = graph(cache, seqs)
    else:
        x = _decode_inputs(cache, seqs)
        dev = model.device
        meta = AttnMeta(False, x["slots"].to(dev), block_tables=x["tables"].to(dev), seq_lens=x["lens"].to(dev))
        out = model.forward(x["ids"].to(dev), x["pos"].to(dev), meta, cache)
    for s in seqs:
        s.num_cached = len(s)
    return out


def generate(
    model: LlamaModel,
    cache: KVCache,
    prompts: list[list[int]],
    max_new_tokens: int,
    return_logits: bool = False,
    use_graph: bool = False,
    graphs: dict | None = None,
    stop_ids: set[int] | None = None,
    graph_width: int | None = None,
):
    """Greedy decoding. Returns new token ids per prompt (and per-step logits).

    ``graphs`` caches GraphDecoders across calls, keyed by (batch, width).
    ``stop_ids``: stop once every sequence has produced one of these tokens;
    each output is cut after its first stop token. Finished sequences keep
    decoding (discarded) until the last one stops: the batch is static.
    ``graph_width``: fixed block-table width for the graph, so batches with
    different prompt lengths reuse one captured graph instead of capturing
    a new one each time.
    """
    seqs = [Sequence(i, p) for i, p in enumerate(prompts)]
    graph = None
    if use_graph:
        width = (max(len(p) for p in prompts) + max_new_tokens + cache.block_size - 1) // cache.block_size
        if graph_width is not None:
            if graph_width < width:
                raise ValueError(f"graph_width={graph_width} < needed {width}")
            width = graph_width
        for s in seqs:  # reserve every block up front: the graph's table width is fixed
            s.ensure_blocks(cache, len(s) + max_new_tokens)
        graphs = graphs if graphs is not None else {}
        key = (len(seqs), width)
        # Capture must happen before prefill: warm-up and capture write dummy
        # K/V into slot 0, which prefill then overwrites if a sequence owns it.
        if key not in graphs:
            graphs[key] = GraphDecoder(model, cache, len(seqs), width)
        graph = graphs[key]
    step_logits = []
    done = [False] * len(seqs)
    logits = _prefill(model, cache, seqs)
    for step in range(max_new_tokens):
        step_logits.append(logits.clone() if graph is not None else logits)
        nxt = logits.argmax(-1).tolist()
        for i, (s, t) in enumerate(zip(seqs, nxt)):
            s.tokens.append(t)
            if stop_ids and t in stop_ids:
                done[i] = True
        if stop_ids and all(done):
            break
        if step + 1 < max_new_tokens:
            logits = _decode(model, cache, seqs, graph)
    for s in seqs:
        cache.free(s.blocks)
    new = [s.tokens[s.num_prompt :] for s in seqs]
    if stop_ids:
        new = [t[: next((j + 1 for j, x in enumerate(t) if x in stop_ids), len(t))] for t in new]
    return (new, step_logits) if return_logits else new
