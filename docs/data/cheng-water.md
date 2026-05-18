# Cheng et al. revPBE0-D3 Water Dataset

This project uses the public Cheng et al. water dataset as its first benchmark data source.

Primary source:

- Materials Cloud record: https://archive.materialscloud.org/record/2018.0020/v1
- Record identifier: `materialscloud:2018.0020/v1`
- Dataset title: `Ab initio thermodynamics of liquid and solid water: supplemental materials`
- License: Creative Commons Attribution 4.0 International
- Related repository: https://github.com/BingqingCheng/neural-network-potential-for-water-revPBE0-D3
- Related publication: https://doi.org/10.1073/pnas.1815117116

## Manual Download

Download `training-set.zip` from the Materials Cloud record and place it at:

```text
data/raw/training-set.zip
```

Then run:

```bash
water-mlip probe-archive data/raw/training-set.zip
water-mlip convert data/raw/training-set.zip data/processed/cheng-water.extxyz
```

The repository does not commit the downloaded archive or converted dataset.
