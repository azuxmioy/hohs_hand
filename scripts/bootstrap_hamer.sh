#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HAMER_ROOT="${HAMER_ROOT:-$ROOT/third_party/hamer}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
CUDA_INDEX="${CUDA_INDEX:-cu118}"
CREATE_VENV="${CREATE_VENV:-1}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
FETCH_DEMO_DATA="${FETCH_DEMO_DATA:-1}"
MODULES="${MODULES:-gcc/11 cuda/11.8 cudnn/8.4_cuda11.x}"
INSTALL_DETECTRON2="${INSTALL_DETECTRON2:-0}"
INSTALL_VITPOSE="${INSTALL_VITPOSE:-0}"
if [[ -d /data/hohs2 ]]; then
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/hohs2/pip-cache}"
fi
if [[ -d /data/hohs2 ]]; then
  DEFAULT_VENV_DIR="/data/hohs2/venvs/hohs_mano_regressor"
else
  DEFAULT_VENV_DIR="$ROOT/.venv"
fi
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"

if ! type module >/dev/null 2>&1; then
  for init_script in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
    if [[ -f "$init_script" ]]; then
      # shellcheck disable=SC1090
      source "$init_script"
      break
    fi
  done
fi

if type module >/dev/null 2>&1 && [[ -n "$MODULES" ]]; then
  module load $MODULES
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  FALLBACK_PYTHON="/data/hohs2/python/cpython-3.10.13/bin/python3.10"
  if [[ -x "$FALLBACK_PYTHON" ]]; then
    PYTHON_BIN="$FALLBACK_PYTHON"
  else
    echo "Missing $PYTHON_BIN. On AIT, run: bash scripts/install_python310.sh"
    exit 1
  fi
fi

if [[ "$CREATE_VENV" == "1" && -z "${VIRTUAL_ENV:-}" ]]; then
  mkdir -p "$(dirname "$VENV_DIR")"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel

if [[ "$INSTALL_TORCH" == "1" ]]; then
  python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_INDEX"
fi

if [[ ! -d "$HAMER_ROOT/.git" ]]; then
  git clone --recursive https://github.com/geopavlakos/hamer.git "$HAMER_ROOT"
else
  git -C "$HAMER_ROOT" submodule update --init --recursive
fi

python -m pip install "setuptools==69.5.1" "numpy==1.23.5"
python -m pip install --no-build-isolation -r "$ROOT/requirements/hamer_runtime.txt"
if [[ "$INSTALL_DETECTRON2" == "1" ]]; then
  MAX_JOBS="${MAX_JOBS:-8}" python -m pip install --no-build-isolation \
    "detectron2 @ git+https://github.com/facebookresearch/detectron2"
fi
python -m pip install --no-deps -e "$HAMER_ROOT"
if [[ "$INSTALL_VITPOSE" == "1" ]]; then
  python -m pip install "mmcv==1.3.9"
  python -m pip install -v -e "$HAMER_ROOT/third-party/ViTPose"
fi
python -m pip install -e "$ROOT"

if [[ "$FETCH_DEMO_DATA" == "1" ]]; then
  (cd "$HAMER_ROOT" && bash fetch_demo_data.sh)
fi

MANO_PATH="$HAMER_ROOT/_DATA/data/mano/MANO_RIGHT.pkl"
if [[ ! -f "$MANO_PATH" ]]; then
  echo "Missing MANO model: $MANO_PATH"
  echo "Download MANO_RIGHT.pkl from the MANO website and place it there before training."
fi

python "$ROOT/scripts/smoke_test.py" --config "$ROOT/configs/train_artic.yaml" --soft
