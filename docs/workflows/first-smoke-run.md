# First Smoke Run

This workflow validates the repository before running expensive training or MD.

## 1. Install

```bash
python -m pip install -e ".[dev]"
```

## 2. Inspect The Configuration

```bash
water-mlip config-summary configs/experiment.yaml
```

## 3. Place Dataset Archive

Download `training-set.zip` from the Cheng et al. Materials Cloud record:

https://archive.materialscloud.org/record/2018.0020/v1

Place it at:

```text
data/raw/training-set.zip
```

## 4. Probe Archive

```bash
water-mlip probe-archive data/raw/training-set.zip
```

## 5. Convert To Extended XYZ

```bash
water-mlip convert data/raw/training-set.zip data/processed/cheng-water.extxyz
```

## 6. Generate MACE Training Command

After splitting the converted file into train and validation files, generate the first MACE command:

```bash
water-mlip mace-train-command data/processed/train.extxyz data/processed/valid.extxyz configs/mace-small.yaml runs/mace-small
```

The generated command is printed instead of executed so it can be inspected before launching a long run.
