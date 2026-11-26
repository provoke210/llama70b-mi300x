#!/bin/bash
# One 8x MI300X session: TP scaling of this engine vs vLLM, plus all-reduce
# latency. Run on the droplet host (not inside a container); ~30-40 minutes.
#
#   nohup bash scripts/compare_tp.sh > results/compare_tp.log 2>&1 &
set -u
cd /shared-docker/llama70b-mi300x
M=/shared-docker/models/Llama-3.3-70B-Instruct-FP8-dynamic
IMG=rocm/vllm:rocm10.2.0_ubuntu24.04_py3.14-pytorch_2.14.0_vllm-0.32.0
V="docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 16g --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v /shared-docker:/shared-docker -w $PWD -e HF_HUB_OFFLINE=1"
OURS="docker exec -w $PWD rocm"

echo "== all-reduce latency"
for W in 2 4 8; do
  $OURS torchrun --nproc-per-node $W scripts/bench_allreduce.py 2>&1 | grep -v -i warn
done

echo "== ours (Triton + HIP graph, fused quant)"
for W in 1 2 4 8; do
  $OURS torchrun --nproc-per-node $W scripts/run_generate.py $M --batch 1 8 32 --new-tokens 64 \
      --kv-gib 40 --graph --save-tokens results/ours_tp${W}_tokens.json 2>&1 | grep -E "^TP="
done

echo "== vLLM + AITER"
for W in 1 2 4 8; do
  $V -e VLLM_ROCM_USE_AITER=1 $IMG python3 scripts/bench_vllm.py $M --tp $W \
      --save-tokens results/vllm_aiter_tp${W}_tokens.json 2>&1 | grep -E "^TP="
done
echo DONE
