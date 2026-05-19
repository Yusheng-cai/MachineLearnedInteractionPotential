#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
RAW_ARCHIVE="data/raw/training-set.zip"
PROCESSED_FILE="data/processed/cheng-water.extxyz"
SPLIT_DIR="data/processed/splits"
RUN_DIR="${RUN_DIR:-runs/mace-gpu}"
CONFIG_FILE="${CONFIG_FILE:-configs/mace-gpu.yaml}"
TRAINING_URL="https://archive.materialscloud.org/api/records/eg3pn-1fw83/files/training-set.zip/content"
EXPECTED_MD5="8cf0da8a72ddcb778529d2869990a53c"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PYTORCH_SPEC="${PYTORCH_SPEC:-torch==2.5.1}"
MAX_TORCH_CUDA="${MAX_TORCH_CUDA:-12.2}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "Installing PyTorch from: $PYTORCH_INDEX_URL"
python -m pip install "$PYTORCH_SPEC" --index-url "$PYTORCH_INDEX_URL"
python - <<PY
import torch

max_cuda = tuple(int(part) for part in "$MAX_TORCH_CUDA".split("."))
cuda_version = torch.version.cuda
print(f"torch={torch.__version__}, cuda={cuda_version}, cuda_available={torch.cuda.is_available()}")

if cuda_version is None:
    raise SystemExit("Installed Torch is CPU-only; expected a CUDA-enabled build.")

installed_cuda = tuple(int(part) for part in cuda_version.split(".")[:2])
if installed_cuda > max_cuda:
    raise SystemExit(
        f"Torch CUDA runtime {cuda_version} is newer than workstation driver capability $MAX_TORCH_CUDA. "
        "Set PYTORCH_SPEC/PYTORCH_INDEX_URL to a compatible wheel."
    )

if not torch.cuda.is_available():
    raise SystemExit("Torch installed, but CUDA is not available. Check NVIDIA driver and GPU visibility.")
PY

python -m pip install -e ".[dev,mace]"

mkdir -p data/raw data/processed "$RUN_DIR"

if [ ! -f "$RAW_ARCHIVE" ]; then
  curl -L -o "$RAW_ARCHIVE" "$TRAINING_URL"
fi

python - <<'PY'
from pathlib import Path
import hashlib

archive = Path("data/raw/training-set.zip")
expected = "8cf0da8a72ddcb778529d2869990a53c"
actual = hashlib.md5(archive.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"MD5 mismatch for {archive}: expected {expected}, got {actual}")
print(f"MD5 verified: {actual}")
PY

water-mlip probe-archive "$RAW_ARCHIVE"
water-mlip convert "$RAW_ARCHIVE" "$PROCESSED_FILE"
water-mlip split "$PROCESSED_FILE" "$SPLIT_DIR" --train-fraction 0.8 --valid-fraction 0.1 --seed 20260518

echo "Starting MACE training with config: $CONFIG_FILE"
mace_run_train \
  --train_file "$SPLIT_DIR/train.extxyz" \
  --valid_file "$SPLIT_DIR/valid.extxyz" \
  --config "$CONFIG_FILE" \
  --results_dir "$RUN_DIR/results" \
  --checkpoints_dir "$RUN_DIR/checkpoints" \
  --name "$(basename "$RUN_DIR")"
