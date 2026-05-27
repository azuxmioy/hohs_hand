#!/bin/bash
# Launch FLUX.1-Fill + ControlNet training on the GPU server.
# Checks GPU usage and runs on a single free GPU.
set -e

source /etc/profile
module load cuda/12.2 gcc/13

DATA_DIR="/data/hohs2"
VENV_DIR="$DATA_DIR/envs/hohs_hand"
REPO_DIR="$HOME/hohs_hand"
CONFIG="${1:-configs/train_flux.yaml}"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "==> Current GPU usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

# Pick first free GPU (< 500 MiB used, 0 % util)
FREE_GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
  --format=csv,noheader,nounits | awk -F', ' '$2 == 0 && $3 < 500 {print $1; exit}')

if [ -z "$FREE_GPU" ]; then
    echo "ERROR: No free GPU found. Aborting." >&2
    exit 1
fi
echo "==> Using GPU $FREE_GPU"

# Step 1: pre-compute text embeddings if not already done
EMBED_CACHE="$DATA_DIR/outputs/flux_controlnet/text_embeddings.pt"
if [ ! -f "$EMBED_CACHE" ]; then
    echo "==> Pre-computing text embeddings (run once) …"
    mkdir -p "$(dirname "$EMBED_CACHE")"
    CUDA_VISIBLE_DEVICES="$FREE_GPU" python scripts/precompute_flux_embeddings.py \
        --out "$EMBED_CACHE"
fi

# Step 2: train
echo "==> Starting training …"
CUDA_VISIBLE_DEVICES="$FREE_GPU" accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    training/train_flux_controlnet.py --config "$CONFIG"
