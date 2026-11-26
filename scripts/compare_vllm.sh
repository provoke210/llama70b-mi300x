#!/bin/bash
cd /shared-docker/llama70b-mi300x
M=/shared-docker/models/Llama-3.3-70B-Instruct-FP8-dynamic
IMG=rocm/vllm:rocm10.2.0_ubuntu24.04_py3.14-pytorch_2.14.0_vllm-0.32.0
V="docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 16g --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v /shared-docker:/shared-docker -w $PWD -e HF_HUB_OFFLINE=1"
echo "== ours (Triton + HIP graph)"
docker exec -w $PWD rocm python3 scripts/run_generate.py $M --batch 1 8 32 --new-tokens 64 --kv-gib 40 --graph --save-tokens results/ours_tokens.json 2>&1 | grep -E "batch="
echo "== vLLM default"
$V $IMG python3 scripts/bench_vllm.py $M --save-tokens results/vllm_tokens.json 2>&1 | grep -E "^batch="
echo "== vLLM AITER"
$V -e VLLM_ROCM_USE_AITER=1 $IMG python3 scripts/bench_vllm.py $M --save-tokens results/vllm_aiter_tokens.json 2>&1 | grep -E "^batch="
python3 - <<'P'
import json
def load(n): return json.load(open(f"results/{n}.json"))["tokens"]
def same(a, b): return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
o, v, a = load("ours_tokens"), load("vllm_tokens"), load("vllm_aiter_tokens")
print(f"== greedy agreement (of {len(o)}): ours~vLLM {same(o, v)}, ours~vLLM-AITER {same(o, a)}, vLLM~vLLM-AITER {same(v, a)}")
P
echo DONE
