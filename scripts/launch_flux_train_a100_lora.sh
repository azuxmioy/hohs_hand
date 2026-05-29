#!/bin/bash
# Launch FLUX.1-Fill: ControlNet + transformer LoRA training on a single A100.
# Same shape as launch_flux_train_a100.sh but uses the LoRA training script
# and a different output namespace.
set -e

source /etc/profile
module load cuda/12.2 gcc/13

export DATA_DIR="${DATA_DIR:-/data/${USER}}"
VENV_DIR="${VENV_DIR:-$DATA_DIR/envs/hohs_hand}"
REPO_DIR="${REPO_DIR:-$HOME/hohs_hand}"
CONFIG="configs/train_flux_a100_lora.yaml"
PROMPT="${PROMPT:-a hand wearing a black glove}"

if [[ -n "$1" && "$1" != --* ]]; then
    CONFIG="$1"; shift
fi

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "==> Current GPU usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

FREE_GPU=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits \
    | awk -F', ' '$2 == 0 && $3 < 100 {print $1; exit}')
if [ -z "$FREE_GPU" ]; then
    echo "ERROR: No free GPU found. Aborting." >&2
    exit 1
fi
echo "==> Using GPU $FREE_GPU"

EMBED_CACHE="$DATA_DIR/outputs/flux_controlnet/text_embeddings.pt"
if [ ! -f "$EMBED_CACHE" ]; then
    echo "==> Pre-computing text embeddings with prompt: $PROMPT"
    mkdir -p "$(dirname "$EMBED_CACHE")"
    CUDA_VISIBLE_DEVICES="$FREE_GPU" \
        python scripts/precompute_flux_embeddings.py \
            --prompt "$PROMPT" --out "$EMBED_CACHE"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==> Starting LoRA + ControlNet training on GPU $FREE_GPU …"
CUDA_VISIBLE_DEVICES="$FREE_GPU" \
    python training/train_flux_controlnet_lora.py --config "$CONFIG"
