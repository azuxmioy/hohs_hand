#!/bin/bash
# Launch FLUX.1-Fill + ControlNet training on the GPU server.
# Usage: launch_flux_train.sh [config_path] [--num-gpus N]
#   Default config: configs/train_flux.yaml
#   Default GPUs:   4
# Checks GPU usage before reserving any GPUs (server policy).
set -e

source /etc/profile
module load cuda/12.2 gcc/13

export DATA_DIR="${DATA_DIR:-/data/${USER}}"
VENV_DIR="${VENV_DIR:-$DATA_DIR/envs/hohs_hand}"
REPO_DIR="${REPO_DIR:-$HOME/hohs_hand}"
CONFIG="configs/train_flux.yaml"
NUM_GPUS=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)
            NUM_GPUS="$2"; shift 2 ;;
        --num-gpus=*)
            NUM_GPUS="${1#*=}"; shift ;;
        *)
            CONFIG="$1"; shift ;;
    esac
done

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "==> Current GPU usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

# Collect free GPUs (< 100 MiB used, 0 % util)
FREE_GPU_LIST=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits \
    | awk -F', ' '$2 == 0 && $3 < 100 {print $1}')

NUM_FREE=$(echo "$FREE_GPU_LIST" | grep -c '[0-9]' || true)

if [ "$NUM_FREE" -lt "$NUM_GPUS" ]; then
    echo "ERROR: Need $NUM_GPUS free GPU(s), found $NUM_FREE. Aborting." >&2
    echo "Free GPUs: $FREE_GPU_LIST" >&2
    exit 1
fi

# Take the first NUM_GPUS free GPUs
SELECTED=$(echo "$FREE_GPU_LIST" | head -n "$NUM_GPUS" | tr '\n' ',')
CUDA_VISIBLE="${SELECTED%,}"   # strip trailing comma
echo "==> Using GPU(s): $CUDA_VISIBLE ($NUM_GPUS process(es))"

# Step 1: pre-compute text embeddings if not already done
EMBED_CACHE="$DATA_DIR/outputs/flux_controlnet/text_embeddings.pt"
if [ ! -f "$EMBED_CACHE" ]; then
    echo "==> Pre-computing text embeddings (run once) …"
    mkdir -p "$(dirname "$EMBED_CACHE")"
    CUDA_VISIBLE_DEVICES="$(echo "$CUDA_VISIBLE" | cut -d, -f1)" \
        python scripts/precompute_flux_embeddings.py --out "$EMBED_CACHE"
fi

# Step 2: train
echo "==> Starting training with $NUM_GPUS GPU(s) …"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 30-minute NCCL watchdog so rank 0 can run eval/checkpoint without other
# ranks timing out on the next gradient sync.
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

if [ "$NUM_GPUS" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" accelerate launch \
        --num_processes=1 \
        --mixed_precision=bf16 \
        training/train_flux_controlnet.py --config "$CONFIG"
else
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" accelerate launch \
        --multi_gpu \
        --num_processes="$NUM_GPUS" \
        --mixed_precision=bf16 \
        training/train_flux_controlnet.py --config "$CONFIG"
fi
