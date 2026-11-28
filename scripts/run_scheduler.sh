#!/bin/bash
# Continuous-batching session on one MI300X droplet. Run on the droplet host:
#   nohup bash scripts/run_scheduler.sh > results/scheduler.log 2>&1 < /dev/null &
# GSM8K (1319 questions, max 32 concurrent sequences) through the scheduler with
# two admission policies, and vLLM + AITER in the same session for reference.
set -u
cd /shared-docker/llama70b-mi300x
M=/shared-docker/models/Llama-3.3-70B-Instruct-FP8-dynamic
IMG=rocm/vllm:rocm10.2.0_ubuntu24.04_py3.14-pytorch_2.14.0_vllm-0.32.0
OURS="docker exec -w $PWD rocm"
V="docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 16g --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v /shared-docker:/shared-docker -w $PWD -e HF_HUB_OFFLINE=1"
stamp() { echo "== $(date -u +%H:%M:%S) $*"; }

stamp setup
docker pull -q $IMG > /dev/null 2>&1 &
PULL=$!
$OURS python3 -m pip install -q --root-user-action=ignore safetensors transformers huggingface_hub pytest
$OURS hf download RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic --local-dir $M --quiet
stamp tests
$OURS python3 -m pytest tests -q -p no:cacheprovider -W ignore::UserWarning 2>&1 | tail -3

stamp "continuous, admit_min=4 (default)"
$OURS python3 scripts/eval_gsm8k.py ours $M --continuous --out results/gsm8k_ours_continuous.jsonl 2>&1 | grep -E "scheduler|^ours|Error|Traceback"
stamp "continuous, admit_min=1"
$OURS python3 scripts/eval_gsm8k.py ours $M --continuous --admit-min 1 --out results/gsm8k_ours_continuous_admit1.jsonl 2>&1 | grep -E "scheduler|^ours|Error|Traceback"

wait $PULL
stamp "vLLM + AITER"
$V -e VLLM_ROCM_USE_AITER=1 $IMG python3 scripts/eval_gsm8k.py vllm $M --out results/gsm8k_vllm_aiter_rerun.jsonl 2>&1 | grep -E "^vllm|Error"

stamp report
$OURS python3 scripts/eval_gsm8k.py report results/gsm8k_ours_continuous.jsonl results/gsm8k_ours_continuous_admit1.jsonl \
    results/gsm8k_ours_attn_v2.jsonl results/gsm8k_vllm_aiter_rerun.jsonl results/gsm8k_vllm_aiter.jsonl
stamp DONE
