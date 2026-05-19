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

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "Installing PyTorch from: $PYTORCH_INDEX_URL"
python -m pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}')"

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
