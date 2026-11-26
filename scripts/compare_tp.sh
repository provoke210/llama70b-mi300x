#!/bin/bash
# One 8x MI300X session, end to end: setup, tests, RCCL all-reduce latency,
# then TP=1/2/4/8 decode for this engine and for vLLM + AITER.
# Run on the droplet host:  nohup bash scripts/compare_tp.sh > results/compare_tp.log 2>&1 &
# Every step logs and moves on when it fails, so one failure doesn't waste the session.
set -u
cd /shared-docker/llama70b-mi300x
M=/shared-docker/models/Llama-3.3-70B-Instruct-FP8-dynamic
IMG=rocm/vllm:rocm10.2.0_ubuntu24.04_py3.14-pytorch_2.14.0_vllm-0.32.0
OURS="docker exec -w $PWD rocm"
V="docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 64g --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v /shared-docker:/shared-docker -w $PWD -e HF_HUB_OFFLINE=1"
stamp() { echo "== $(date -u +%H:%M:%S) $*"; }

stamp setup
docker pull -q $IMG > /dev/null 2>&1 &
PULL=$!
$OURS python3 -m pip install -q --root-user-action=ignore safetensors transformers huggingface_hub pytest
$OURS hf download RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic --local-dir $M --quiet
$OURS rocm-smi --showproductname 2>&1 | grep -c "Card Series" | sed 's/^/GPUs visible in container: /'
$OURS sh -c 'command -v torchrun || echo "torchrun MISSING"'

stamp tests
$OURS python3 -m pytest tests -q -p no:cacheprovider -W ignore::UserWarning 2>&1 | tail -2

stamp all-reduce latency
for W in 2 4 8; do
  $OURS torchrun --nproc-per-node $W scripts/bench_allreduce.py 2>&1 | grep -v -i warn
done

stamp "ours (Triton + HIP graph, fused quant)"
for W in 1 2 4 8; do
  out=$($OURS torchrun --nproc-per-node $W scripts/run_generate.py $M --batch 1 8 32 --new-tokens 64 \
        --kv-gib 40 --graph --save-tokens results/ours_tp${W}_tokens.json 2>&1)
  if echo "$out" | grep -q "^TP=$W batch=32"; then
    echo "$out" | grep -E "^TP="
  else
    echo "TP=$W with graph FAILED; last lines:"; echo "$out" | grep -v -i warn | tail -6
    echo "TP=$W retry without graph (eager):"
    $OURS torchrun --nproc-per-node $W scripts/run_generate.py $M --batch 1 8 32 --new-tokens 64 \
        --kv-gib 40 2>&1 | grep -E "^TP=|Error" | head -6
  fi
done

wait $PULL
stamp "vLLM + AITER"
for W in 1 2 4 8; do
  $V -e VLLM_ROCM_USE_AITER=1 $IMG python3 scripts/bench_vllm.py $M --tp $W \
      --save-tokens results/vllm_aiter_tp${W}_tokens.json 2>&1 | grep -E "^TP=|Error" | head -6
done
stamp DONE
