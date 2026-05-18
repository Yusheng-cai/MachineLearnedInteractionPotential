# Water MLIP Benchmark

This repository benchmarks whether static energy and force errors predict liquid-water MD observables for machine-learned interatomic potentials.

The first study uses the public Cheng et al. revPBE0-D3 water dataset and trains multiple MACE checkpoints from the same data. Each checkpoint is evaluated with static energy/force metrics and liquid-water observables such as RDFs, diffusion, and NVE energy drift.

## First Workflow

```bash
python -m pip install -e ".[dev]"
water-mlip config-summary configs/experiment.yaml
water-mlip probe-archive data/raw/training-set.zip
water-mlip convert data/raw/training-set.zip data/processed/cheng-water.extxyz
```
