# Water MLIP Benchmark

This repository benchmarks whether static energy and force errors predict liquid-water MD observables for machine-learned interatomic potentials.

The first study uses the public Cheng et al. revPBE0-D3 water dataset and trains multiple MACE checkpoints from the same data. Each checkpoint is evaluated with static energy/force metrics and liquid-water observables such as RDFs, diffusion, and NVE energy drift.

## First Workflow

```bash
python -m pip install -e ".[dev]"
water-mlip config-summary configs/experiment.yaml
water-mlip probe-archive data/raw/training-set.zip
water-mlip convert data/raw/training-set.zip data/processed/cheng-water.extxyz
water-mlip split data/processed/cheng-water.extxyz data/processed/splits
```

## GPU Training

On a Linux machine with an NVIDIA GPU, CUDA-capable PyTorch, and `python3` available:

```bash
bash scripts/train_mace_gpu.sh
```

The script creates `.venv`, installs `.[dev,mace]`, downloads and verifies the Cheng archive when needed, converts it to extended XYZ, creates deterministic train/validation/test splits, and starts MACE training with `configs/mace-gpu.yaml`.

By default, the script installs PyTorch from the CUDA 12.1 wheel index:

```bash
https://download.pytorch.org/whl/cu121
```

If your workstation needs a different PyTorch CUDA wheel, override it:

```bash
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/train_mace_gpu.sh
```

Useful overrides:

```bash
RUN_DIR=runs/mace-gpu-large CONFIG_FILE=configs/mace-gpu.yaml bash scripts/train_mace_gpu.sh
```

## Documentation

- Design spec: `docs/superpowers/specs/2026-05-18-water-mlip-metric-observable-design.md`
- First smoke workflow: `docs/workflows/first-smoke-run.md`

## Verified Local Checks

Before launching MACE training, run:

```bash
pytest -q
water-mlip config-summary configs/experiment.yaml
water-mlip probe-archive data/raw/training-set.zip
water-mlip convert data/raw/training-set.zip data/processed/cheng-water.extxyz
water-mlip split data/processed/cheng-water.extxyz data/processed/splits
```

The `data/` directory is intentionally ignored by git. In the current local checkout, the Cheng archive conversion produces 1,593 frames.
