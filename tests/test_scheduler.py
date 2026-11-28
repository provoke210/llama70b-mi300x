"""Continuous batching: every request's greedy output must equal running it alone.

CPU, reference ops, fp32. Covers queueing beyond max_num_seqs, mixed
prefill+decode steps, stop tokens, bucket padding with a scratch KV block,
preemption under a tight KV cache, and rejection of over-long prompts.
"""

import random

import pytest
import torch

from mi300_llm import KVCache, from_state_dict, ops
from mi300_llm.engine import generate
from mi300_llm.scheduler import ContinuousBatchingEngine, Request

random.seed(0)
PROMPTS = [[random.randrange(1, 300) for _ in range(n)] for n in (3, 17, 9, 40, 5, 26, 12, 33, 7, 21)]
MAX_NEW = [6, 12, 3, 9, 1, 11, 8, 4, 10, 7]


@pytest.fixture(scope="module")
def model(hf_tiny, tiny_cfg):
    ops.set_backend("ref")
    return from_state_dict(tiny_cfg, hf_tiny.state_dict(), dtype=torch.float32)


@pytest.fixture(scope="module")
def solo(model, tiny_cfg):
    """Each request run alone with the static engine, plus a stop token per
    request taken from the middle of its output (so stops really trigger)."""
    cache = KVCache(tiny_cfg, num_blocks=64, block_size=4, dtype=torch.float32, device="cpu")
    full = [generate(model, cache, [p], n)[0] for p, n in zip(PROMPTS, MAX_NEW)]
    stops = [frozenset({out[len(out) // 2]}) if i % 3 == 0 else frozenset() for i, out in enumerate(full)]
    want = [generate(model, cache, [p], n, stop_ids=set(s) or None)[0] for p, n, s in zip(PROMPTS, MAX_NEW, stops)]
    return stops, want


def _requests(stops):
    return [Request(i, p, n, s) for i, (p, n, s) in enumerate(zip(PROMPTS, MAX_NEW, stops))]


def _serve(model, tiny_cfg, stops, num_blocks=64, reserve=False, **kw):
    cache = KVCache(tiny_cfg, num_blocks=num_blocks, block_size=4, dtype=torch.float32, device="cpu",
                    reserve_scratch_block=reserve)
    eng = ContinuousBatchingEngine(model, cache, max_model_len=128, **kw)
    reqs = eng.run(_requests(stops))
    assert cache.num_free_blocks == cache.num_usable_blocks, "blocks leaked"
    assert not eng.running and not eng.waiting
    return reqs, eng.stats


def test_matches_running_each_request_alone(model, tiny_cfg, solo):
    stops, want = solo
    reqs, st = _serve(model, tiny_cfg, stops, max_num_seqs=3, admit_min=1)
    assert [r.output for r in reqs] == want
    assert st.mixed_steps > 1 and st.mean_decode_batch > 1.5  # requests really overlapped
    assert all(r.first_token_time <= r.finish_time for r in reqs)


def test_admit_min_batches_admissions(model, tiny_cfg, solo):
    stops, want = solo
    reqs, st_lazy = _serve(model, tiny_cfg, stops, max_num_seqs=4, admit_min=3)
    assert [r.output for r in reqs] == want
    _, st_eager = _serve(model, tiny_cfg, stops, max_num_seqs=4, admit_min=1)
    assert st_lazy.mixed_steps < st_eager.mixed_steps


def test_bucket_padding_uses_scratch_block(model, tiny_cfg, solo):
    """Pure decode steps padded to 1/2/4/8 rows (as the graph path does); the
    padding rows write to reserved block 0 and must not disturb live rows."""
    stops, want = solo
    reqs, st = _serve(model, tiny_cfg, stops, reserve=True, max_num_seqs=8, admit_min=1, pad_decode=True)
    assert [r.output for r in reqs] == want
    assert st.padded_rows > 0 and st.eager_decode_steps > 0


def test_preemption_recomputes_correctly(model, tiny_cfg):
    """Four 8-token prompts fit at admission (2 blocks each of 12), but each
    grows to 28 tokens (7 blocks): the cache runs out mid-decode, the newest
    sequences are preempted and later recomputed from their tokens."""
    prompts = [[random.randrange(1, 300) for _ in range(8)] for _ in range(4)]
    ref_cache = KVCache(tiny_cfg, num_blocks=64, block_size=4, dtype=torch.float32, device="cpu")
    want = [generate(model, ref_cache, [p], 20)[0] for p in prompts]
    cache = KVCache(tiny_cfg, num_blocks=12, block_size=4, dtype=torch.float32, device="cpu")
    eng = ContinuousBatchingEngine(model, cache, max_num_seqs=4, max_model_len=64, admit_min=1)
    reqs = eng.run([Request(i, p, 20) for i, p in enumerate(prompts)])
    assert eng.stats.preemptions > 0
    assert [r.output for r in reqs] == want
    assert sum(r.preemptions for r in reqs) == eng.stats.preemptions
    assert cache.num_free_blocks == cache.num_usable_blocks


def test_rejects_prompts_longer_than_max_model_len(model, tiny_cfg):
    cache = KVCache(tiny_cfg, num_blocks=64, block_size=4, dtype=torch.float32, device="cpu")
    eng = ContinuousBatchingEngine(model, cache, max_num_seqs=4, max_model_len=20, admit_min=1)
    reqs = eng.run([Request(0, PROMPTS[3], 5), Request(1, PROMPTS[0], 5)])  # 40 tokens > 20
    assert reqs[0].rejected and reqs[0].output == []
    assert not reqs[1].rejected and len(reqs[1].output) == 5


def test_max_model_len_caps_generation(model, tiny_cfg):
    cache = KVCache(tiny_cfg, num_blocks=64, block_size=4, dtype=torch.float32, device="cpu")
    eng = ContinuousBatchingEngine(model, cache, max_num_seqs=2, max_model_len=12, admit_min=1)
    (req,) = eng.run([Request(0, PROMPTS[2], 50)])  # 9-token prompt
    assert len(req.tokens) == 12
