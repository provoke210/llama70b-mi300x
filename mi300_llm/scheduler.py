"""Continuous batching: requests join and leave the running batch every step.

The static ``engine.generate`` runs a fixed batch until its longest sequence
ends, so finished slots sit idle; on GSM8K that made it 2x slower than vLLM at
the same per-step speed. Here each ``step()``:

1. Reserves KV for the token every running sequence writes this step. If the
   cache is full it preempts the most recently admitted sequence (frees its
   blocks, requeues it at the front; it is recomputed from its tokens later).
2. Admits waiting requests into free slots, within a prefill-token budget and
   the free KV blocks. Admission waits until ``admit_min`` slots are free
   (unless nothing is running): an admitting step costs a full weight read
   plus an eager launch, so admitting one request per step would roughly
   double the cost of those steps.
3. Runs one forward. With admissions it is a mixed batch: the new requests'
   whole prompts (prefill) and one decode token per running sequence, in one
   pass over the weights. Without admissions it is a pure decode step, which
   replays a HIP graph captured for the smallest batch bucket that fits; the
   rows beyond the live batch are padding that writes to a reserved scratch
   KV block and is discarded.
4. Samples greedily, appends, and retires sequences that hit a stop token,
   ``max_new_tokens`` or ``max_model_len``, freeing their blocks at once.

Greedy outputs are the same as running each request alone (tests check this),
up to floating-point effects of the batch composition.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from .engine import GraphDecoder
from .kv_cache import KVCache, Sequence
from .model import AttnMeta, LlamaModel


@dataclass(eq=False)
class Request:
    req_id: int
    prompt: list[int]
    max_new_tokens: int
    stop_ids: frozenset = frozenset()
    output: list[int] = field(default_factory=list)
    seq: Sequence | None = None
    arrival: float = 0.0
    first_token_time: float | None = None
    finish_time: float | None = None
    preemptions: int = 0
    rejected: bool = False

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.output


@dataclass
class Stats:
    steps: int = 0
    mixed_steps: int = 0  # steps that admitted requests (eager)
    graph_steps: int = 0  # pure decode steps replayed from a graph
    eager_decode_steps: int = 0
    decode_rows: int = 0  # live decode rows summed over steps
    padded_rows: int = 0  # padding rows added to reach a graph bucket
    prefill_tokens: int = 0
    preemptions: int = 0

    @property
    def mean_decode_batch(self) -> float:
        n = self.graph_steps + self.eager_decode_steps + self.mixed_steps
        return self.decode_rows / max(1, n)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class ContinuousBatchingEngine:
    def __init__(
        self,
        model: LlamaModel,
        cache: KVCache,
        max_num_seqs: int = 32,
        max_model_len: int = 4096,
        max_prefill_tokens: int = 8192,
        admit_min: int | None = None,
        use_graph: bool = False,
        pad_decode: bool | None = None,
    ):
        """``pad_decode``: pad pure-decode steps to a batch bucket even without
        graphs (defaults to ``use_graph``); lets CPU tests cover the padding."""
        self.model, self.cache = model, cache
        self.max_num_seqs, self.max_model_len = max_num_seqs, max_model_len
        self.max_prefill_tokens = max_prefill_tokens
        self.admit_min = admit_min if admit_min is not None else max(1, max_num_seqs // 8)
        self.use_graph = use_graph
        self.pad_decode = use_graph if pad_decode is None else pad_decode
        if self.pad_decode and cache.num_usable_blocks == cache.num_blocks:
            raise ValueError("padded decode writes to block 0: build the KVCache with reserve_scratch_block=True")
        b, self.buckets = 1, []
        while b < max_num_seqs:
            self.buckets.append(b)
            b *= 2
        self.buckets.append(max_num_seqs)
        self.width = _cdiv(max_model_len, cache.block_size)
        self.graphs: dict[int, GraphDecoder] = {}
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self._rejected: list[Request] = []
        self.stats = Stats()

    # ----------------------------------------------------------------- queue
    def add_request(self, req: Request) -> None:
        req.arrival = time.perf_counter()
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _blocks(self, n_tokens: int) -> int:
        return _cdiv(n_tokens, self.cache.block_size)

    def _preempt(self, req: Request) -> None:
        self.cache.free(req.seq.blocks)
        req.seq = None
        req.preemptions += 1
        self.stats.preemptions += 1
        self.waiting.appendleft(req)

    def _finish(self, req: Request) -> None:
        if req.seq is not None:
            self.cache.free(req.seq.blocks)
            req.seq = None
        req.finish_time = time.perf_counter()

    # ------------------------------------------------------------- scheduling
    def _reserve_decode_blocks(self) -> bool:
        """Make room for the KV every running sequence writes this step,
        oldest first; preempt from the newest end. Returns True if anything
        was preempted."""
        preempted = False
        for req in list(self.running):
            if req.seq is None:  # preempted earlier in this loop
                continue
            need = self._blocks(len(req.seq)) - len(req.seq.blocks)
            while need > self.cache.num_free_blocks and self.running[-1] is not req:
                self._preempt(self.running.pop())
                preempted = True
            if need > self.cache.num_free_blocks:  # req is the newest: preempt it
                self._preempt(self.running.pop())
                preempted = True
                continue
            req.seq.ensure_blocks(self.cache, len(req.seq))
        return preempted

    def _admit(self) -> list[Request]:
        free_slots = self.max_num_seqs - len(self.running)
        if not self.waiting or free_slots == 0 or (self.running and free_slots < self.admit_min):
            return []
        admitted, budget = [], self.max_prefill_tokens
        while self.waiting and len(admitted) < free_slots:
            req = self.waiting[0]
            n = len(req.tokens)
            if n >= self.max_model_len or self._blocks(n) > self.cache.num_usable_blocks:
                self.waiting.popleft()  # can never run
                req.rejected = True
                self._finish(req)
                self._rejected.append(req)
                continue
            if admitted and n > budget:
                break
            if self._blocks(n) > self.cache.num_free_blocks:
                break
            self.waiting.popleft()
            req.seq = Sequence(req.req_id, req.tokens)
            req.seq.ensure_blocks(self.cache, n)
            admitted.append(req)
            budget -= n
        return admitted

    # -------------------------------------------------------------- execution
    def _decode_host_inputs(self, dec: list[Request], rows: int, width: int) -> dict:
        bs = self.cache.block_size
        pad = rows - len(dec)
        tables = torch.zeros(rows, width, dtype=torch.int32)
        for i, r in enumerate(dec):
            tables[i, : len(r.seq.blocks)] = torch.tensor(r.seq.blocks, dtype=torch.int32)
        return dict(
            ids=torch.tensor([r.seq.tokens[-1] for r in dec] + [0] * pad),
            pos=torch.tensor([len(r.seq) - 1 for r in dec] + [0] * pad),
            slots=torch.tensor([r.seq.slots(len(r.seq) - 1, len(r.seq), bs)[0] for r in dec] + [0] * pad),
            tables=tables,
            lens=torch.tensor([len(r.seq) for r in dec] + [1] * pad, dtype=torch.int32),
        )

    def _bucket(self, n: int) -> int:
        return next(b for b in self.buckets if b >= n)

    def _forward(self, pre: list[Request], dec: list[Request]) -> torch.Tensor:
        """Logits, one row per request: ``pre`` first, then ``dec``."""
        dev = self.model.device
        if not pre and self.pad_decode:
            rows = self._bucket(len(dec))
            self.stats.padded_rows += rows - len(dec)
            x = self._decode_host_inputs(dec, rows, self.width)
            if self.use_graph:
                self.stats.graph_steps += 1
                if rows not in self.graphs:
                    self.graphs[rows] = GraphDecoder(self.model, self.cache, rows, self.width)
                return self.graphs[rows].replay(x)[: len(dec)]
            self.stats.eager_decode_steps += 1
            meta = AttnMeta(False, x["slots"].to(dev), block_tables=x["tables"].to(dev), seq_lens=x["lens"].to(dev))
            return self.model.forward(x["ids"].to(dev), x["pos"].to(dev), meta, self.cache)[: len(dec)]

        bs = self.cache.block_size
        ids, pos, slots, cu = [], [], [], [0]
        for r in pre:
            n = len(r.seq)
            ids += r.seq.tokens
            pos += range(n)
            slots += r.seq.slots(0, n, bs)
            cu.append(cu[-1] + n)
        meta_kw = {}
        if dec:
            x = self._decode_host_inputs(dec, len(dec), max(len(r.seq.blocks) for r in dec))
            ids += x["ids"].tolist()
            pos += x["pos"].tolist()
            slots += x["slots"].tolist()
            meta_kw = dict(block_tables=x["tables"].to(dev), seq_lens=x["lens"].to(dev))
        if pre:
            self.stats.mixed_steps += 1
            self.stats.prefill_tokens += cu[-1]
        else:
            self.stats.eager_decode_steps += 1
        meta = AttnMeta(not dec, torch.tensor(slots, device=dev), cu_seqlens=cu if pre else None, **meta_kw)
        return self.model.forward(torch.tensor(ids, device=dev), torch.tensor(pos, device=dev), meta, self.cache)

    def step(self) -> list[Request]:
        """Run one scheduler step; returns the requests that finished in it."""
        preempted = self._reserve_decode_blocks()
        dec = list(self.running)
        # After a preemption, don't admit in the same step (the cache is
        # tight) unless nothing is left running.
        pre = self._admit() if (not preempted or not dec) else []
        finished, self._rejected = self._rejected, []
        if not pre and not dec:
            if self.waiting and not finished:
                raise RuntimeError("no request fits in the KV cache")
            return finished
        self.stats.steps += 1
        self.stats.decode_rows += len(dec)
        logits = self._forward(pre, dec)
        next_tokens = logits.argmax(-1).tolist()
        now = time.perf_counter()
        for req, tok in zip(pre + dec, next_tokens):
            req.output.append(tok)
            req.seq.tokens.append(tok)
            req.seq.num_cached = len(req.seq) - 1
            if req.first_token_time is None:
                req.first_token_time = now
            if tok in req.stop_ids or len(req.output) >= req.max_new_tokens or len(req.tokens) >= self.max_model_len:
                self._finish(req)
                finished.append(req)
        self.running = [r for r in dec if r.finish_time is None] + [r for r in pre if r.finish_time is None]
        return finished

    def capture_graphs(self) -> None:
        """Capture every bucket's decode graph up front (as vLLM does at
        startup), so no capture lands inside a timed run. Requires an idle
        engine: capture writes dummy K/V to the scratch block only."""
        assert self.use_graph and not self.running
        for b in self.buckets:
            if b not in self.graphs:
                self.graphs[b] = GraphDecoder(self.model, self.cache, b, self.width)

    def run(self, requests: list[Request]) -> list[Request]:
        """Serve ``requests`` (all arriving now) to completion; returns them in order."""
        for r in requests:
            self.add_request(r)
        while self.has_work():
            self.step()
        return requests
