#!/usr/bin/env bash
# End-to-end pipeline: PALM subjects -> inpainted images -> HaMeR training data.
#
# Steps:
#   0. (optional) Download PALM subjects from HuggingFace
#   1. Generate conditioning HDF5 from PALM subjects  (palm_make_conditions.py)
#   2. Batch-inpaint all samples with FLUX ControlNet  (batch_inpaint.py)
#   3. Convert to HaMeR-compatible NPZ                (h5_to_hamer_npz.py)
#
# Usage:
#   # Quick test with one subject already on disk
#   bash scripts/run_palm_to_hamer.sh
#
#   # Download 10 subjects first, then run the full pipeline
#   PALM_DOWNLOAD="0000-0009" bash scripts/run_palm_to_hamer.sh
#
# Env vars:
#   DATA_DIR             — root data directory      (default: /data/$USER)
#   CUDA_VISIBLE_DEVICES — GPU to use               (default: 0)
#   PALM_DOWNLOAD        — subjects to download     (default: empty = skip)
#   PALM_SUBJ_DIR        — directory with subjects   (default: $DATA/palm/subjects)
#   PALM_SUBJECTS        — specific subjects         (default: auto-detect all in PALM_SUBJ_DIR)
#   NUM_GESTURES         — gestures per subject      (default: 42)

set -euo pipefail
cd "$(dirname "$0")/.."

# ─── Paths ────────────────────────────────────────────────────────────────
DATA="${DATA_DIR:-/data/${USER}}"

# PALM subjects
PALM_SUBJ_DIR="${PALM_SUBJ_DIR:-${DATA}/palm/subjects}"
NUM_GESTURES="${NUM_GESTURES:-56}"

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
export HF_HOME="${HF_HOME:-/data/${USER}/cache/huggingface}"

# ─── Step 0: Download PALM subjects (optional) ──────────────────────────
if [ -n "${PALM_DOWNLOAD:-}" ]; then
    echo "=== Step 0: Download PALM subjects ==="
    python scripts/download_palm.py \
        --subjects "$PALM_DOWNLOAD" \
        --out "$PALM_SUBJ_DIR"
    echo ""
fi

# ─── Discover subjects ───────────────────────────────────────────────────
if [ -n "${PALM_SUBJECTS:-}" ]; then
    IFS=',' read -ra SUBJECTS <<< "$PALM_SUBJECTS"
else
    SUBJECTS=()
    for d in "$PALM_SUBJ_DIR"/*/; do
        [ -f "${d}poses.npy" ] && SUBJECTS+=("$(basename "$d")")
    done
fi

if [ ${#SUBJECTS[@]} -eq 0 ]; then
    # Fall back to _peek if no subjects downloaded yet
    if [ -d "${DATA}/palm/_peek/0000" ]; then
        echo "No subjects in ${PALM_SUBJ_DIR}; falling back to _peek/0000"
        PALM_SUBJ_DIR="${DATA}/palm/_peek"
        SUBJECTS=("0000")
        NUM_GESTURES=1
    else
        echo "ERROR: No PALM subjects found. Run with PALM_DOWNLOAD=0000-0009 to download."
        exit 1
    fi
fi

echo "Subjects: ${SUBJECTS[*]}  (${NUM_GESTURES} gestures each)"
echo "Output:   ${OUT_ROOT}"
echo ""

# Build gesture list (1-indexed, zero-padded to 6 digits)
GESTURES=()
for ((g=1; g<=NUM_GESTURES; g++)); do
    GESTURES+=("$(printf '%06d' "$g")")
done

# ─── Step 1: Generate conditioning HDF5 ──────────────────────────────────
echo "=== Step 1: Generate conditioning HDF5 ==="
H5_FILES=()
for subj in "${SUBJECTS[@]}"; do
    subj_dir="${PALM_SUBJ_DIR}/${subj}"
    for gesture in "${GESTURES[@]}"; do
        # Skip gestures without images
        if [ ! -f "${subj_dir}/images/MCU_01/${gesture}.jpg" ]; then
            continue
        fi
        h5_out="${H5_DIR}/${subj}_g${gesture}.h5"
        if [ -f "$h5_out" ]; then
            echo "  [skip] ${subj}/g${gesture}"
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
echo "  ${#H5_FILES[@]} conditioning HDF5 files"

# ─── Step 2: Batch inpaint ───────────────────────────────────────────────
echo ""
echo "=== Step 2: Batch inpaint ==="
IMG_DIRS=()
for h5 in "${H5_FILES[@]}"; do
    stem="$(basename "$h5" .h5)"
    img_out="${IMG_DIR}/${stem}"
    if [ -f "${img_out}/manifest.json" ]; then
        echo "  [skip] ${stem}"
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
echo "  Subjects processed : ${#SUBJECTS[@]}"
echo "  Conditioning HDF5s : ${#H5_FILES[@]} files in ${H5_DIR}/"
echo "  Inpainted images   : ${IMG_DIR}/"
echo "  HaMeR training NPZ : ${NPZ_OUT}"
echo ""
echo "To train HaMeR on this data, copy the NPZ to the regressor repo:"
echo "  cp ${NPZ_OUT} <regressor>/data/artic/processed/train_hamer.npz"
echo "  # Update configs/train_artic.yaml with image_root: ${IMG_DIR}"
echo "  bash scripts/train_artic.sh"
