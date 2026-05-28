#!/bin/bash
# Run this on the GPU server to create the virtual environment.
# Large files (venv, data, checkpoints) live under /data, not $HOME.
set -e

source /etc/profile
module load cuda/12.2 gcc/13

DATA_DIR="${DATA_DIR:-/data/${USER}}"
VENV_DIR="${VENV_DIR:-$DATA_DIR/envs/hohs_hand}"
REPO_DIR="${REPO_DIR:-$HOME/hohs_hand}"

echo "==> CUDA: $(nvcc --version | grep release)"
echo "==> Python: $(python3 --version)"

echo "==> Creating virtual environment at $VENV_DIR"
~/.local/bin/virtualenv -p python3 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing PyTorch (cu121 wheels, compatible with CUDA 12.2+)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing project requirements"
pip install -r "$REPO_DIR/requirements.txt"

echo "==> Done. Activate with:"
echo "    source /etc/profile && module load cuda/12.2 gcc/13"
echo "    source $VENV_DIR/bin/activate"
