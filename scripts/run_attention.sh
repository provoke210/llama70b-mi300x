#!/bin/bash
# Decode-attention session on one MI300X droplet. Run on the droplet host:
#   nohup bash scripts/run_attention.sh prep    > results/attention_prep.log 2>&1 &   # setup, tests, sweep
#   nohup bash scripts/run_attention.sh compare > results/attention_compare.log 2>&1 &
# "compare" runs this engine with the new kernel, with the v1 kernel
# (MI300_ATTN=v1; everything else identical), and vLLM + AITER, at short and
# long contexts.
set -u
cd /shared-docker/llama70b-mi300x
M=/shared-docker/models/Llama-3.3-70B-Instruct-FP8-dynamic
IMG=rocm/vllm:rocm10.2.0_ubuntu24.04_py3.14-pytorch_2.14.0_vllm-0.32.0
OURS="docker exec -w $PWD rocm"
V="docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 16g --security-opt seccomp=unconfined --security-opt apparmor=unconfined -v /shared-docker:/shared-docker -w $PWD -e HF_HUB_OFFLINE=1"
stamp() { echo "== $(date -u +%H:%M:%S) $*"; }

# ctx:batches. "base" is the 50-token chat prompt used in earlier comparisons.
CASES="base:1,8,32 2048:1,8,32 8192:1,8 32768:1"

ours() {  # $1 = label, $2 = extra env, $3 = ctx, $4 = batches
  local plen=""; [ "$3" != base ] && plen="--prompt-len $3"
  $OURS env $2 python3 scripts/run_generate.py $M --graph --batch ${4//,/ } --new-tokens 64 --kv-gib 60 $plen 2>&1 \
    | grep -E "^TP=|Error|error" | sed "s/^/$1 /"
}

case "${1:-}" in
prep)
  stamp setup
  docker pull -q $IMG > /dev/null 2>&1 &
  $OURS python3 -m pip install -q --root-user-action=ignore safetensors transformers huggingface_hub pytest
  $OURS hf download RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic --local-dir $M --quiet
  stamp tests
  $OURS python3 -m pytest tests -q -p no:cacheprovider -W ignore::UserWarning 2>&1 | tail -3
  stamp sweep
  $OURS python3 scripts/tune_attention.py 2>&1 | grep -v -i warn | tee results/attention_tune.txt
  wait
  stamp DONE
  ;;
compare)
  stamp layer breakdown
  $OURS python3 scripts/bench_layer.py --batches 1 32 2>&1 | grep -E "^== .*fused|decode_attn" | tee results/layer_breakdown_attn_v2.txt
  for c in $CASES; do
    ctx=${c%%:*}; bs=${c#*:}
    stamp "ctx=$ctx batches=$bs"
    ours "new" "" $ctx $bs
    ours "v1 " "MI300_ATTN=v1" $ctx $bs
    plen=""; [ "$ctx" != base ] && plen="--prompt-len $ctx"
    $V -e VLLM_ROCM_USE_AITER=1 $IMG python3 scripts/bench_vllm.py $M --batch ${bs//,/ } $plen \
        --save-tokens results/vllm_aiter_ctx${ctx}_tokens.json 2>&1 | grep -E "^TP=|Error" | sed "s/^/vllm+aiter /"
  done
  stamp DONE
  ;;
*) echo "usage: $0 prep|compare"; exit 1 ;;
esac
