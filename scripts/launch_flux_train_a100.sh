#!/bin/bash
# Launch FLUX.1-Fill + ControlNet training on a single A100 (e.g. ait-server-05).
# - No NF4 quantization (bf16 transformer)
# - No gradient checkpointing (faster fwd/bwd)
# - Single-GPU only
# Pre-computes text + VAE latent caches on first run, then trains.
set -e

source /etc/profile
module load cuda/12.2 gcc/13

export DATA_DIR="${DATA_DIR:-/data/${USER}}"
VENV_DIR="${VENV_DIR:-$DATA_DIR/envs/hohs_hand}"
REPO_DIR="${REPO_DIR:-$HOME/hohs_hand}"
CONFIG="configs/train_flux_a100.yaml"
RESOLUTION=512
PROMPT="${PROMPT:-a hand wearing a black glove}"

# Allow overriding the config path
if [[ -n "$1" && "$1" != --* ]]; then
    CONFIG="$1"; shift
fi

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "==> Current GPU usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

# Pick first free GPU (< 100 MiB used, 0 % util) — server etiquette.
FREE_GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits \
    | awk -F', ' '$2 == 0 && $3 < 100 {print $1; exit}')
if [ -z "$FREE_GPU" ]; then
    echo "ERROR: No free GPU found. Aborting." >&2
    exit 1
fi
echo "==> Using GPU $FREE_GPU"

# Step 1: text-embedding cache (uses default prompt unless PROMPT env override).
EMBED_CACHE="$DATA_DIR/outputs/flux_controlnet/text_embeddings.pt"
if [ ! -f "$EMBED_CACHE" ]; then
    echo "==> Pre-computing text embeddings with prompt: $PROMPT"
    mkdir -p "$(dirname "$EMBED_CACHE")"
    CUDA_VISIBLE_DEVICES="$FREE_GPU" \
        python scripts/precompute_flux_embeddings.py \
            --prompt "$PROMPT" --out "$EMBED_CACHE"
fi

# Step 2: latent cache.
LATENT_CACHE="$DATA_DIR/datasets/data_latents_${RESOLUTION}.h5"
if [ ! -f "$LATENT_CACHE" ]; then
    echo "==> Pre-computing VAE latents at ${RESOLUTION}×${RESOLUTION} …"
    CUDA_VISIBLE_DEVICES="$FREE_GPU" \
        python scripts/precompute_latents.py \
            --src "$DATA_DIR/datasets/data.h5" \
            --dst "$LATENT_CACHE" \
            --resolution "$RESOLUTION"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==> Starting training on GPU $FREE_GPU …"
CUDA_VISIBLE_DEVICES="$FREE_GPU" \
    python training/train_flux_controlnet_a100.py --config "$CONFIG"
