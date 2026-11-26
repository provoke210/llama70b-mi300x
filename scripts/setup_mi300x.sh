#!/usr/bin/env bash
# First-boot setup on an AMD Developer Cloud MI300X droplet (PyTorch/ROCm image).
# Usage: bash scripts/setup_mi300x.sh [--with-70b]
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
mkdir -p "$MODEL_DIR"

echo "== GPU"
rocm-smi --showproductname || true
python3 -c "import torch; p=torch.cuda.get_device_properties(0); print(torch.__version__, torch.version.hip, p.name, p.gcnArchName, p.multi_processor_count, 'CUs', round(p.total_memory/2**30), 'GiB')"
python3 -c "import triton; print('triton', triton.__version__)"

echo "== Python deps"
python3 -m pip install -q safetensors transformers pytest huggingface_hub
# For notebooks in VS Code (Remote-SSH, then Dev Containers: Attach to Running Container).
python3 -m pip install -q ipykernel matplotlib pandas

echo "== Disk"
df -h "$MODEL_DIR" | tail -1

export HF_HUB_ENABLE_HF_TRANSFER=1
dl() {
  local repo=$1
  local t0=$(date +%s)
  hf download "$repo" --local-dir "$MODEL_DIR/$(basename "$repo")" --exclude "*.pth" "original/*"
  echo "downloaded $repo in $(( $(date +%s) - t0 ))s"
}

# 1B with the same FP8 format: for fast iteration and correctness checks.
dl RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic
if [[ "${1:-}" == "--with-70b" ]]; then
  dl RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic  # ~73 GB
fi

echo "== Unit tests"
python3 -m pytest tests -q
