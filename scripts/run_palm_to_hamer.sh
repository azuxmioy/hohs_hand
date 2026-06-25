#!/usr/bin/env bash
# End-to-end pipeline: PALM subjects -> inpainted images -> HaMeR training data.
#
# Steps:
#   1. Generate conditioning HDF5 from PALM subjects  (palm_make_conditions.py)
#   2. Batch-inpaint all samples with FLUX ControlNet  (batch_inpaint.py)
#   3. Convert to HaMeR-compatible NPZ                (h5_to_hamer_npz.py)
#
# Usage:
#   bash scripts/run_palm_to_hamer.sh
#
# Expects the following env vars (or defaults in the paths below):
#   DATA_DIR      — root data directory   (default: /data/$USER)
#   CUDA_VISIBLE_DEVICES — GPU to use     (default: 0)

set -euo pipefail
cd "$(dirname "$0")/.."

# ─── Paths ────────────────────────────────────────────────────────────────
DATA="${DATA_DIR:-/data/${USER}}"

# PALM dataset: one or more subject directories
PALM_SUBJECTS=(
    "${DATA}/palm/_peek/0000"
)
GESTURES=("000001")

# MANO assets
MANO_DIR="${DATA}/datasets/arctic_dl/mano_v1_2/models"
UV_RIGHT="${DATA}/arctic/MANO_UV_right.obj"

# Inpainting checkpoint
INPAINT_CONFIG="configs/train_flux_a100_lora_calib.yaml"
INPAINT_CKPT="${DATA}/checkpoints/flux_controlnet_lora/20260611_235529/step_005600"
EMBEDDINGS="${DATA}/outputs/flux_controlnet/text_embeddings.pt"

# Output layout
OUT_ROOT="${DATA}/palm/pipeline_output"
H5_DIR="${OUT_ROOT}/conditioning"
IMG_DIR="${OUT_ROOT}/inpainted"
NPZ_OUT="${OUT_ROOT}/train_hamer.npz"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ─── Step 1: Generate conditioning HDF5 ──────────────────────────────────
echo "=== Step 1: Generate conditioning HDF5 ==="
H5_FILES=()
for subj_dir in "${PALM_SUBJECTS[@]}"; do
    subj_name="$(basename "$subj_dir")"
    for gesture in "${GESTURES[@]}"; do
        h5_out="${H5_DIR}/${subj_name}_g${gesture}.h5"
        if [ -f "$h5_out" ]; then
            echo "  [skip] $h5_out exists"
        else
            mkdir -p "${H5_DIR}"
            python inference/palm_make_conditions.py \
                --subj-dir "$subj_dir" \
                --mano-dir "$MANO_DIR" \
                --uv-right "$UV_RIGHT" \
                --gesture "$gesture" \
                --scale 3 --mask-dilate 8 \
                --out "$h5_out"
        fi
        H5_FILES+=("$h5_out")
    done
done

# ─── Step 2: Batch inpaint ───────────────────────────────────────────────
echo ""
echo "=== Step 2: Batch inpaint ==="
IMG_DIRS=()
for h5 in "${H5_FILES[@]}"; do
    stem="$(basename "$h5" .h5)"
    img_out="${IMG_DIR}/${stem}"
    if [ -f "${img_out}/manifest.json" ]; then
        echo "  [skip] $img_out already inpainted"
    else
        python scripts/batch_inpaint.py \
            --config "$INPAINT_CONFIG" \
            --checkpoint "$INPAINT_CKPT" \
            --h5 "$h5" \
            --embeddings "$EMBEDDINGS" \
            --out-dir "$img_out" \
            --num-steps 30 --guidance 30.0
    fi
    IMG_DIRS+=("$img_out")
done

# ─── Step 3: Convert to HaMeR NPZ ───────────────────────────────────────
echo ""
echo "=== Step 3: Convert to HaMeR NPZ ==="
python scripts/h5_to_hamer_npz.py \
    --h5 "${H5_FILES[@]}" \
    --images "${IMG_DIRS[@]}" \
    --output "$NPZ_OUT" \
    --image-root "$IMG_DIR"

echo ""
echo "=== Pipeline complete ==="
echo "  Conditioning HDF5s : ${H5_DIR}/"
echo "  Inpainted images   : ${IMG_DIR}/"
echo "  HaMeR training NPZ : ${NPZ_OUT}"
echo ""
echo "To train HaMeR on this data, copy the NPZ to the regressor repo:"
echo "  cp ${NPZ_OUT} <regressor>/data/artic/processed/train_hamer.npz"
echo "  # Update configs/train_artic.yaml with image_root: ${IMG_DIR}"
echo "  bash scripts/train_artic.sh"
