# llama70b-mi300x

Llama-3.3-70B FP8 decode on a single AMD Instinct MI300X, with a from-scratch
engine and hand-written Triton kernels, measured against the hardware roofline
and against vLLM-ROCm on the same workload.

## Results at a glance

Decode ms/step on 1x MI300X (lower is better), same prompt, greedy, all three
run back to back. Details and method: [Results](#results-1x-mi300x-vf-rocm-102-pytorch-215-triton-39).

| batch | this repo | vLLM 0.32 default | vLLM 0.32 + AITER |
|---|---|---|---|
| 1 | **26.2 ms** (38 tok/s) | 28.8 ms | 27.3 ms |
| 8 | **27.4 ms** (292 tok/s) | 29.5 ms | 28.0 ms |
| 32 | 30.5 ms (1048 tok/s) | 31.5 ms | **30.1 ms** |

- Batch-1 decode went from 25 to 38 tok/s over the project; it now streams the
  weights at ~77% of the *measured* HBM bandwidth (3.6 TB/s, not the 5.3 TB/s
  spec).
- The tuned split-K FP8 GEMMs take 18% less time per layer than hipBLASLt on
  the same shapes.
- Same accuracy as vLLM: GSM8K (all 1319 test questions, greedy, identical
  prompts) scores 95.45% vs 95.75% for vLLM and 95.68% for vLLM + AITER; the
  differences are not significant (paired McNemar p = 0.58 / 0.69). See
  [Accuracy](#accuracy-gsm8k).
- Along the way: three FP8 numerical pitfalls on ROCm / Triton, written up in
  [Findings worth knowing](#findings-worth-knowing).

## Why this is interesting

- **Decode is memory-bound.** Every step reads all ~73 GB of FP8 weights. The
  MI300X spec says 5.3 TB/s (a ~13 ms floor), but a streaming read on this
  instance reaches 3.6 TB/s, so the realistic batch-1 floor is ~20 ms per
  step. How close the engine gets to that is the headline metric.
- **192 GB on one GPU.** FP8 weights take ~70 GB, leaving ~110 GB of KV cache
  (~0.31 MB/token in BF16, ~350k tokens). Tensor parallelism is therefore not
  needed to fit the model; it is here to cut per-step latency.
- **FP8 is a different format on MI300X.** Checkpoints use OCP `e4m3fn` (max 448);
  gfx942 computes in `e4m3fnuz` (max 240, exponent bias 8, no negative zero).
  `mi300_llm/fp8.py` converts by reinterpreting bits and doubling the scale,
  which is lossless (tested).

## Layout

| Path | What |
|---|---|
| `mi300_llm/ref_ops.py` | Pure-PyTorch reference for every op; each Triton kernel is tested against it |
| `mi300_llm/kernels/fp8_gemm.py` | W8A8 FP8 GEMM with deterministic split-K; per-shape configs in `gemm_configs.json` |
| `mi300_llm/kernels/decode_attention.py` | Split-K (flash-decoding) paged attention, GQA-aware |
| `mi300_llm/kernels/fused_quant.py` | Per-token FP8 quantization, column-parallel, fused with RMSNorm and SiLU-mul |
| `mi300_llm/kernels/norm_rope.py` | RMSNorm (+ residual add), RoPE, SiLU-mul |
| `mi300_llm/ops.py` | Switches each op between `ref` and `triton` |
| `mi300_llm/model.py` | Llama forward over a paged KV cache (fused QKV, fused gate/up) |
| `mi300_llm/loader.py` | Loads BF16 or compressed-tensors FP8 safetensors, optionally sharded for TP |
| `mi300_llm/tp.py` | Tensor-parallel layout and communication |
| `mi300_llm/engine.py` | Batched greedy generation, HIP-graph decode |
| `tests/` | Reference vs Hugging Face, kernels vs reference, TP on CPU (gloo) |
| `scripts/` | Droplet setup, generation + timing, benchmarks, GEMM tuning, vLLM comparison |
| `scripts/investigations/` | One-off scripts behind the numerical findings |
| `results/` | Raw logs and CSVs; [results/README.md](results/README.md) indexes them |
| `notebooks/roofline.ipynb` | Plots: kernel bandwidth, attention scaling, end-to-end vs vLLM |

## Status

| Phase | State |
|---|---|
| 1. Reference path matches Hugging Face (fp32, GQA, llama3 RoPE, multi-block paged KV) | Done |
| 1. FP8 loader (compressed-tensors, fnuz conversion, QKV fusion) | Done |
| 2. Triton kernels on MI300X, 70B generates coherent text | Done |
| 3. Split-K GEMM + per-shape tuning, attention split heuristic, HIP graph decode | Done |
| 4. Compare with vLLM-ROCm on the same workload | Done |
| 5. Column-parallel quantization; RMSNorm + quant and SiLU-mul + quant fusion | Done |
| 6. Tensor parallelism: sharded loading, all-reduce, vocab-parallel head; verified on CPU (gloo, TP=2/4) | Done; **not yet run on GPUs** |
| 7. GSM8K accuracy vs vLLM on identical prompts | Done: on par |
| Next: TP scaling on 8x MI300X vs vLLM (`scripts/compare_tp.sh`); faster decode attention; continuous batching | |

Tests: 58 pass on the MI300X (kernels, model, early stopping, and the CPU TP
tests); 12 run on a CPU-only machine.

## Limitations

- **Accuracy is checked on one benchmark.** GSM8K matches vLLM; other tasks
  (long context, code) are not evaluated.
- **Decode-focused engine.** No continuous batching (requests can't join
  mid-decode), so a batch runs until its longest answer ends: GSM8K took
  618 s here vs ~300 s in vLLM at the same per-step speed. Prefill uses the
  reference attention, not a tuned kernel.
- **One virtualized GPU.** All numbers are from a single MI300X VF (SR-IOV) on
  AMD Developer Cloud; bare-metal bandwidth may differ.
- **Tensor parallelism is verified for correctness only**, on CPU. Its GPU
  path (RCCL, all-reduce inside a HIP graph) has not run yet.

## Tensor parallelism

Megatron-style, one process per GPU (`torchrun`), see `mi300_llm/tp.py`:

| Layer | Split | Communication |
|---|---|---|
| qkv | column-parallel: local query heads and KV heads (Llama-3.3-70B has 8 KV heads, so TP <= 8) | none |
| attention | local heads only | none |
| o | row-parallel | all-reduce |
| gate_up | column-parallel over the FFN width | none |
| down | row-parallel | all-reduce |
| lm_head | vocab-parallel | all-gather of logits |
| embed, norms | replicated | none |

Per decode step: 160 all-reduces of a [batch, 8192] bf16 tensor (16 KB at
batch 1, so latency-bound) and one logits all-gather. Weights are sharded at
load time; with per-channel FP8, each row keeps the scale computed over the
whole row, and row-parallel layers quantize their local activation slice with
a local per-token scale.

`tests/test_tp.py` runs TP=2 and 4 as gloo processes on CPU: fp32 tokens match
the single-process model exactly (logits to 1e-4), FP8 tracks it, and every
rank holds 1/W of the layer weights. Disabling the all-reduce makes the tests
fail, so they do exercise it.

```bash
torchrun --nproc-per-node 8 scripts/run_generate.py <model> --graph --batch 1 8 32
torchrun --nproc-per-node 8 scripts/bench_allreduce.py   # RCCL latency, eager and in a graph
```

On macOS, add `--master-addr 127.0.0.1` to torchrun (hostname rendezvous hangs).

## Results (1x MI300X VF, ROCm 10.2, PyTorch 2.15, Triton 3.9)

Measured streaming-read bandwidth is **3.6 TB/s** (`scripts/bench_bandwidth.py`),
not the 5.3 TB/s spec, so the realistic batch-1 floor for 72.7 GB of weights is
~20 ms/step.

### Against vLLM

Llama-3.3-70B-Instruct-FP8-dynamic, same prompt tokens, greedy, EOS ignored.
Decode ms/step = (time for 80 tokens - time for 16 tokens) / 64, best of 3,
warm, all three run back to back. vLLM is
`rocm/vllm:rocm10.2.0_ubuntu24.04_py3.14-pytorch_2.14.0_vllm-0.32.0`
(vLLM 0.32.1.dev), prefix caching off. Raw log: `results/compare_vllm_fused.log`.

| batch | this repo (Triton + HIP graph, fused quant) | vLLM default (hipBLASLt GEMM) | vLLM `VLLM_ROCM_USE_AITER=1` |
|---|---|---|---|
| 1 | **26.2 ms** (38 tok/s) | 28.8 ms (35 tok/s) | 27.3 ms (37 tok/s) |
| 8 | **27.4 ms** (292 tok/s) | 29.5 ms (271 tok/s) | 28.0 ms (286 tok/s) |
| 32 | 30.5 ms (1048 tok/s) | 31.5 ms (1015 tok/s) | **30.1 ms** (1062 tok/s) |

Faster than default vLLM at every batch size, 4% / 2% faster than vLLM with
AITER at batch 1 / 8, and 1% slower at batch 32 (where decode attention
starts to matter). AITER's 70B GEMM shapes are untuned in that image, per its
own log. At batch 1 the engine streams 72.7 GB in 26.2 ms = 2.8 TB/s, ~77% of
measured bandwidth.

Before the quantization work (`results/compare_vllm.log`) this repo was at
27.8 / 28.9 / 32.0 ms, 2-8% behind vLLM-AITER.

Greedy outputs agree for the first 15 tokens with both vLLM runs; the two vLLM
backends agree with each other for 42. Different FP8 accumulation orders flip
near-ties, then the sequences diverge; all outputs read coherently. Token
agreement is not an accuracy metric; a task benchmark is the next check.

### Accuracy (GSM8K)

All 1319 GSM8K test questions, zero-shot chain of thought, greedy, up to 512
new tokens, stopping at the model's EOS tokens. All three engines receive the
same prompt token ids; the answer is the number after the last `####`.
`scripts/eval_gsm8k.py`, run by `scripts/run_gsm8k.sh`; per-question outputs in
`results/gsm8k_*.jsonl`.

| engine | correct | accuracy | 95% CI (Wilson) |
|---|---|---|---|
| this repo | 1259 / 1319 | 95.45% | 94.19 - 96.45% |
| vLLM 0.32 default | 1263 / 1319 | 95.75% | 94.53 - 96.72% |
| vLLM 0.32 + AITER | 1262 / 1319 | 95.68% | 94.44 - 96.65% |

Paired by question: this repo and vLLM disagree on 30 questions (13 only this
repo gets right, 17 only vLLM; exact McNemar p = 0.58). vLLM's own two
backends disagree on 23 (12 vs 11, p = 1.00). So FP8 kernel differences flip a
handful of answers in both directions, no more than switching vLLM's backend
does. Mean response length is the same within 1 token (~205).

### Where a layer's time goes (`scripts/bench_layer.py`, batch 1)

GPU time per kernel, replayed from a graph (plain `do_bench` reports ~6.4 us
for every tiny kernel: that is the CPU launch floor). GEMMs whose weights fit
the 256 MB Infinity Cache (qkv, o, down) read faster in this isolated loop
than in real decode, where 70 GB of weights stream through.

| | before | column-parallel quant | + fusion |
|---|---|---|---|
| us / layer | 294 | 280 | **274** |
| down-proj input quant | 13.9 us (one program per row: one CU at batch 1) | 3.5 us | fused into SiLU-mul |

### Optimization steps

| batch | first working version | + split-K, tuned tiles, HIP graph | + parallel, fused quantization |
|---|---|---|---|
| 1 | ~38 ms/step | 27.8 ms/step | **26.2 ms/step** |
| 32 | ~62 ms/step | 32.0 ms/step | **30.5 ms/step** |

(The first column was measured with an older, less careful method; treat it as
approximate.)

Per-layer FP8 GEMM time at batch 1 (us): 464 -> **328** after split-K and
tuning, vs 402 for hipBLASLt (`torch._scaled_mm`) on the same shapes.

Without the graph, split-K makes eager decode *slower* (52 ms/step at batch 1):
the extra reduce launches add CPU launch overhead, which already dominates at
~1600 launches per step. The graph removes it.

### A measurement mistake worth remembering

An earlier version timed decode as (N-token run - 1-token run). One of the
1-token runs hit a first-time compile (261 ms instead of ~76 ms), which made
every decode step look ~2.9 ms faster (25.0 instead of 27.8 ms at batch 1).
Differencing two multi-token runs, best of 3, after warm-up fixes it.

## Findings worth knowing

- **torch tensor / Python-scalar division is not correctly rounded on ROCm GPUs.**
  It made the reference FP8 quantizer disagree with a float64 computation on
  0.08% of codes (values at rounding midpoints). Tensor / tensor is exact. The
  Triton quantizer uses `tl.div_rn` and matches float64 bit for bit.
- Triton's plain fp32 -> fp8 (fnuz) cast is round-to-nearest-even and
  bit-identical to torch; requesting `fp_downcast_rounding="rtz"` for fnuz
  crashes the Triton 3.9 AMD backend (assertion in `ConvertFpCastOpToLLVM`).
- Under-occupancy, not bandwidth, limited the narrow GEMMs: N=8192 at
  BLOCK_N=64 is 128 programs for 304 CUs. Split-K fixed it.
- **Fusing a kernel can change its numerics even with identical source.**
  Triton picks the register layout from the whole kernel: storing 1-byte FP8
  instead of 2-byte bf16 changes sizePerThread from 8 to 16, which reorders
  the RMSNorm row sum. `rstd` then differs by 1 ulp on ~19% of rows, about 1
  in 5e6 outputs land on the other side of a bf16 rounding midpoint, and
  greedy 70B decoding diverges from the unfused build after 14 tokens. The
  fp32-to-bf16 casts themselves were checked to be exact round-to-nearest-even.
  `MI300_FUSE_QUANT=0` switches the fusion off for A/B runs.

## On the droplet

On the AMD Developer Cloud PyTorch image, PyTorch and Triton live in a Docker
container named `rocm`, not on the host; `/shared-docker` is shared between the
two. Put the project and models there and run everything inside the container.

### From VS Code

1. After creating a droplet, set `HostName` of `Host mi300x` in `~/.ssh/config`
   to its new IP.
2. *Remote-SSH: Connect to Host* → `mi300x`.
3. *Dev Containers: Attach to Running Container* → `rocm` (needs the Dev
   Containers extension), then open `/shared-docker/llama70b-mi300x`.
4. Open `notebooks/roofline.ipynb` and pick the container's Python kernel.

### From a shell (inside the container: `ssh mi300x`, then `docker exec -it rocm bash`)

```bash
bash scripts/setup_mi300x.sh            # deps, 1B FP8 model, all tests
python -m pytest tests -q               # kernel tests run on the GPU
python scripts/run_generate.py ~/models/Llama-3.2-1B-Instruct-FP8-dynamic --compare-ref
bash scripts/setup_mi300x.sh --with-70b # adds the ~73 GB 70B checkpoint
python scripts/run_generate.py ~/models/Llama-3.3-70B-Instruct-FP8-dynamic --batch 1 8 32 --graph
python scripts/bench_bandwidth.py       # achievable HBM bandwidth
python scripts/tune_gemm.py             # writes mi300_llm/kernels/gemm_configs.json
python scripts/bench_kernels.py --csv results/kernels.csv
```

Billing runs while a droplet exists, even powered off: destroy it after each session.

## Locally (CPU, e.g. macOS)

```bash
uv venv --python 3.12 && uv pip install torch numpy pytest safetensors transformers
.venv/bin/python -m pytest tests -q   # reference, loader and TP tests; GPU kernel tests skip
```
