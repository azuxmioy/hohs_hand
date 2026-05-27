#!/bin/bash
# Launch distributed training on the GPU server
set -e

VENV_DIR="$HOME/envs/hohs_hand"
REPO_DIR="$HOME/hohs_hand"
CONFIG="${1:-configs/train_ddpm.yaml}"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "==> Found $NUM_GPUS GPU(s)"

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch --multi_gpu --num_processes="$NUM_GPUS" training/train.py --config "$CONFIG"
else
    accelerate launch training/train.py --config "$CONFIG"
fi
