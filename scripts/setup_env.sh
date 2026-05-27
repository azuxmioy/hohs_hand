#!/bin/bash
# Run this on the GPU server to create the virtual environment
set -e

VENV_DIR="$HOME/envs/hohs_hand"
REPO_DIR="$HOME/hohs_hand"

echo "==> Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing PyTorch (CUDA 12.1)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing project requirements"
pip install -r "$REPO_DIR/requirements.txt"

echo "==> Done. Activate with: source $VENV_DIR/bin/activate"
