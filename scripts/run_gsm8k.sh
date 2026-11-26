#!/bin/bash
# GSM8K on one MI300X droplet: download the model, run the tests, then score
# this engine, vLLM default and vLLM + AITER on identical prompts.
# Run on the droplet host:  nohup bash scripts/run_gsm8k.sh > results/gsm8k.log 2>&1 &
set -u
cd /shared-docker/llama70b-mi300x
M=/shared-docker/models/Llama-3.3-70B-Instruct-FP8-dynamic
IMG=rocm/vllm:rocm10.2.0_ubuntu24.04_py3.14-pytorch_2.14.0_vllm-0.32.0
OURS="docker exec -w $PWD rocm"
V="docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 16g --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v /shared-docker:/shared-docker -w $PWD -e HF_HUB_OFFLINE=1"

echo "== $(date -u +%H:%M) setup"
docker pull -q $IMG > /dev/null 2>&1 &
PULL=$!
$OURS python3 -m pip install -q --root-user-action=ignore safetensors transformers huggingface_hub pytest
$OURS hf download RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic --local-dir $M --quiet

echo "== $(date -u +%H:%M) tests"
$OURS python3 -m pytest tests -q -p no:cacheprovider -W ignore::UserWarning 2>&1 | tail -3

echo "== $(date -u +%H:%M) ours"
$OURS python3 scripts/eval_gsm8k.py ours $M --out results/gsm8k_ours.jsonl 2>&1 | grep -v -i warn | tail -4

wait $PULL
echo "== $(date -u +%H:%M) vLLM default"
$V $IMG python3 scripts/eval_gsm8k.py vllm $M --out results/gsm8k_vllm.jsonl 2>&1 | grep -E "^vllm"
echo "== $(date -u +%H:%M) vLLM + AITER"
$V -e VLLM_ROCM_USE_AITER=1 $IMG python3 scripts/eval_gsm8k.py vllm $M --out results/gsm8k_vllm_aiter.jsonl 2>&1 | grep -E "^vllm"

echo "== $(date -u +%H:%M) report"
$OURS python3 scripts/eval_gsm8k.py report results/gsm8k_ours.jsonl results/gsm8k_vllm.jsonl results/gsm8k_vllm_aiter.jsonl
echo DONE
