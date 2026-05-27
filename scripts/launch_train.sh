#!/bin/bash
# Launch training on the GPU server.
# Defaults to 1 GPU. Pass --multi-gpu to use all free GPUs (check first!).
set -e

DATA_DIR="/data/hohs2"
VENV_DIR="$DATA_DIR/envs/hohs_hand"
REPO_DIR="$HOME/hohs_hand"
CONFIG="${1:-configs/train_ddpm.yaml}"
MULTI_GPU=false

for arg in "$@"; do
  [ "$arg" = "--multi-gpu" ] && MULTI_GPU=true
done

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "==> Current GPU usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

# Identify free GPUs (utilization == 0 % and memory used < 500 MiB)
FREE_GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
  --format=csv,noheader,nounits | awk -F', ' '$2 == 0 && $3 < 500 {print $1}')

if [ -z "$FREE_GPUS" ]; then
    echo "ERROR: No free GPUs found. Aborting." >&2
    exit 1
fi

FREE_COUNT=$(echo "$FREE_GPUS" | wc -w)
FIRST_FREE=$(echo "$FREE_GPUS" | awk '{print $1}')

if $MULTI_GPU && [ "$FREE_COUNT" -gt 1 ]; then
    GPU_LIST=$(echo "$FREE_GPUS" | tr '\n' ',' | sed 's/,$//')
    echo "==> Launching on $FREE_COUNT free GPUs: $GPU_LIST"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" accelerate launch \
        --multi_gpu --num_processes="$FREE_COUNT" \
        training/train.py --config "$CONFIG"
else
    echo "==> Launching on 1 GPU (index $FIRST_FREE)"
    CUDA_VISIBLE_DEVICES="$FIRST_FREE" accelerate launch \
        training/train.py --config "$CONFIG"
fi
