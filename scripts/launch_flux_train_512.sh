#!/bin/bash
# Launch FLUX.1-Fill + ControlNet training at 512×512 using cached latents.
# Pre-computes text + VAE latent caches on first run, then trains.
set -e

source /etc/profile
module load cuda/12.2 gcc/13

export DATA_DIR="${DATA_DIR:-/data/${USER}}"
VENV_DIR="${VENV_DIR:-$DATA_DIR/envs/hohs_hand}"
REPO_DIR="${REPO_DIR:-$HOME/hohs_hand}"
CONFIG="configs/train_flux_512.yaml"
NUM_GPUS=4
RESOLUTION=512

# Parse args (config path or --num-gpus)
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

# Pick first N free GPUs (< 100 MiB used, 0 % util)
FREE_GPU_LIST=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits \
    | awk -F', ' '$2 == 0 && $3 < 100 {print $1}')
NUM_FREE=$(echo "$FREE_GPU_LIST" | grep -c '[0-9]' || true)
if [ "$NUM_FREE" -lt "$NUM_GPUS" ]; then
    echo "ERROR: Need $NUM_GPUS free GPU(s), found $NUM_FREE. Aborting." >&2
    exit 1
fi
SELECTED=$(echo "$FREE_GPU_LIST" | head -n "$NUM_GPUS" | tr '\n' ',')
CUDA_VISIBLE="${SELECTED%,}"
echo "==> Using GPU(s): $CUDA_VISIBLE ($NUM_GPUS process(es))"

# Step 1: text-embedding cache
EMBED_CACHE="$DATA_DIR/outputs/flux_controlnet/text_embeddings.pt"
if [ ! -f "$EMBED_CACHE" ]; then
    echo "==> Pre-computing text embeddings …"
    mkdir -p "$(dirname "$EMBED_CACHE")"
    CUDA_VISIBLE_DEVICES="$(echo "$CUDA_VISIBLE" | cut -d, -f1)" \
        python scripts/precompute_flux_embeddings.py --out "$EMBED_CACHE"
fi

# Step 2: latent cache (runs once, ~5 min for 3500 samples at 512)
LATENT_CACHE="$DATA_DIR/datasets/data_latents_${RESOLUTION}.h5"
if [ ! -f "$LATENT_CACHE" ]; then
    echo "==> Pre-computing VAE latents at ${RESOLUTION}×${RESOLUTION} …"
    CUDA_VISIBLE_DEVICES="$(echo "$CUDA_VISIBLE" | cut -d, -f1)" \
        python scripts/precompute_latents.py \
            --src "$DATA_DIR/datasets/data.h5" \
            --dst "$LATENT_CACHE" \
            --resolution "$RESOLUTION"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

echo "==> Starting training with $NUM_GPUS GPU(s) …"
if [ "$NUM_GPUS" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" accelerate launch \
        --num_processes=1 \
        --mixed_precision=bf16 \
        training/train_flux_controlnet_latent.py --config "$CONFIG"
else
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE" accelerate launch \
        --multi_gpu \
        --num_processes="$NUM_GPUS" \
        --mixed_precision=bf16 \
        training/train_flux_controlnet_latent.py --config "$CONFIG"
fi
