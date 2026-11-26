# Results

All measured on one AMD Instinct MI300X VF (AMD Developer Cloud, ROCm 10.2,
PyTorch 2.15, Triton 3.9), 2026-11-25, with Llama-3.3-70B-Instruct-FP8-dynamic.

| File | What | Made by |
|---|---|---|
| `compare_vllm_fused.log` | **Final** end-to-end decode: this repo vs vLLM default vs vLLM+AITER, back to back | `scripts/compare_vllm.sh` |
| `compare_vllm.log` | Same comparison before the quantization work (its first "ours" block used a 128-token window by mistake; see the note at the end) | `scripts/compare_vllm.sh` |
| `layer_breakdown.txt` | GPU time of every kernel in one decoder layer, separate vs fused quantization | `scripts/bench_layer.py` |
| `bandwidth.txt` | Achievable HBM bandwidth (3.6 TB/s, vs 5.3 spec) | `scripts/bench_bandwidth.py` |
| `kernels_baseline.csv` | Per-kernel bandwidth of the first working version, Triton vs hipBLASLt | `scripts/bench_kernels.py` |
| `kernels.csv` | Same after split-K and tuning (before the quantization fusion) | `scripts/bench_kernels.py` |
| `tune_gemm.txt` | GEMM tile sweep; winners in `mi300_llm/kernels/gemm_configs.json` | `scripts/tune_gemm.py` |
| `attention_sweep.txt` | Decode-attention split sweep behind `default_num_splits` | `scripts/investigations/sweep_attn.py` |
| `ours_unfused_tokens.json`, `ours_fused_tokens.json` | Greedy batch-0 output with `MI300_FUSE_QUANT=0` / `1`; they diverge at token 14 | `scripts/run_generate.py --save-tokens` |
| `ours_tokens.json`, `vllm_tokens.json`, `vllm_aiter_tokens.json` | Greedy outputs from the pre-fusion comparison run | `run_generate.py`, `bench_vllm.py` |

The token files of the final comparison run were overwritten while syncing;
its agreement numbers (15 / 15 / 42 tokens) are in `compare_vllm_fused.log`.

`scripts/investigations/` holds the one-off scripts behind the numerical
findings in the top-level README (FP8 rounding, torch division on ROCm, bf16
casts, layout-dependent row sums). They expect the repo at
`/shared-docker/llama70b-mi300x` inside the droplet's `rocm` container.
