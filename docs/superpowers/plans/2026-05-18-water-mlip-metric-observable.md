# Water MLIP Metric-To-Observable Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python research scaffold for testing whether static energy/force errors predict liquid-water MD observables across MACE checkpoints trained on the Cheng et al. revPBE0-D3 water dataset.

**Architecture:** The project is a small Python package with focused modules for configuration, dataset probing/conversion, static metric evaluation, MD observable analysis, MACE command generation, and plotting. Expensive external steps such as dataset download, MACE training, and production MD are exposed as explicit CLI commands so the core logic can be unit-tested with synthetic data first.

**Tech Stack:** Python 3.11+, NumPy, ASE, Matplotlib, pytest, PyYAML, and MACE Tools for training and checkpoint deployment.

---

## File Structure

Create this repository structure:

- `pyproject.toml`: package metadata, runtime dependencies, pytest configuration, CLI entry point.
- `.gitignore`: ignores generated datasets, models, trajectories, caches, and figures.
- `README.md`: short project overview and first workflow commands.
- `docs/data/cheng-water.md`: dataset provenance and manual download instructions.
- `configs/experiment.yaml`: default benchmark configuration.
- `configs/mace-small.yaml`: small MACE training configuration used for the first smoke run.
- `src/water_mlip_benchmark/__init__.py`: package version.
- `src/water_mlip_benchmark/config.py`: typed configuration loading and path resolution.
- `src/water_mlip_benchmark/data_sources.py`: dataset manifest and archive probing.
- `src/water_mlip_benchmark/convert.py`: conversion from Cheng-style archives or ASE-readable structures to extended XYZ.
- `src/water_mlip_benchmark/metrics.py`: energy/force error metrics.
- `src/water_mlip_benchmark/observables.py`: RDF, MSD diffusion, and NVE drift calculations.
- `src/water_mlip_benchmark/mace.py`: MACE command generation and checkpoint discovery.
- `src/water_mlip_benchmark/plots.py`: metric-to-observable plotting helpers.
- `src/water_mlip_benchmark/cli.py`: command-line interface.
- `tests/fixtures/`: tiny synthetic structures and metric inputs.
- `tests/test_config.py`: config tests.
- `tests/test_data_sources.py`: dataset manifest and archive-probing tests.
- `tests/test_convert.py`: parser/converter tests with synthetic structures.
- `tests/test_metrics.py`: static metric tests.
- `tests/test_observables.py`: RDF/MSD/drift tests.
- `tests/test_mace.py`: command-generation tests.
- `tests/test_cli.py`: CLI smoke tests.

Generated files will live outside source control:

- `data/raw/`: manually downloaded archives or extracted raw files.
- `data/processed/`: converted `.xyz` train/validation/test splits.
- `runs/`: training, evaluation, and MD outputs.
- `figures/`: generated plots.

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/water_mlip_benchmark/__init__.py`
- Create: `tests/fixtures/.gitkeep`

- [ ] **Step 1: Create project files**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "water-mlip-benchmark"
version = "0.1.0"
description = "Metric-to-observable benchmark for water MLIPs."
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "ase>=3.23.0",
  "matplotlib>=3.8.0",
  "numpy>=1.26.0",
  "pyyaml>=6.0.1",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
]
mace = [
  "mace-torch>=0.3.6",
]

[project.scripts]
water-mlip = "water_mlip_benchmark.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Create `.gitignore`:

```gitignore
__pycache__/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.DS_Store

data/raw/
data/processed/
runs/
figures/
*.egg-info/
```

Create `README.md`:

```markdown
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
```

Create `src/water_mlip_benchmark/__init__.py`:

```python
"""Water MLIP metric-to-observable benchmark."""

__version__ = "0.1.0"
```

Create an empty tracked fixture marker:

```text
tests/fixtures/.gitkeep
```

- [ ] **Step 2: Install in editable mode**

Run:

```bash
python -m pip install -e ".[dev]"
```

Expected: installation succeeds and `water-mlip` is available.

- [ ] **Step 3: Run package import smoke check**

Run:

```bash
python -c "import water_mlip_benchmark; print(water_mlip_benchmark.__version__)"
```

Expected: prints `0.1.0`.

- [ ] **Step 4: Commit scaffold**

```bash
git add pyproject.toml .gitignore README.md src/water_mlip_benchmark/__init__.py tests/fixtures/.gitkeep
git commit -m "chore: scaffold water mlip benchmark"
```

## Task 2: Configuration Loading

**Files:**
- Create: `configs/experiment.yaml`
- Create: `configs/mace-small.yaml`
- Create: `src/water_mlip_benchmark/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for configuration loading**

Create `tests/test_config.py`:

```python
from pathlib import Path

from water_mlip_benchmark.config import BenchmarkConfig, load_config


def test_load_config_resolves_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(
        """
project_root: .
raw_data_dir: data/raw
processed_data_dir: data/processed
runs_dir: runs
figures_dir: figures
random_seed: 7
dataset:
  archive_name: training-set.zip
  converted_extxyz: cheng-water.extxyz
  train_fraction: 0.8
  validation_fraction: 0.1
  test_fraction: 0.1
md:
  temperature_K: 300.0
  timestep_fs: 0.5
  equilibration_steps: 10
  production_steps: 20
  sample_interval: 2
  rdf_r_max_A: 6.0
  rdf_bin_width_A: 0.1
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert isinstance(config, BenchmarkConfig)
    assert config.random_seed == 7
    assert config.raw_data_dir == tmp_path / "data/raw"
    assert config.dataset.converted_extxyz == "cheng-water.extxyz"
    assert config.md.temperature_K == 300.0


def test_load_config_rejects_bad_split_sum(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(
        """
project_root: .
raw_data_dir: data/raw
processed_data_dir: data/processed
runs_dir: runs
figures_dir: figures
random_seed: 7
dataset:
  archive_name: training-set.zip
  converted_extxyz: cheng-water.extxyz
  train_fraction: 0.7
  validation_fraction: 0.2
  test_fraction: 0.2
md:
  temperature_K: 300.0
  timestep_fs: 0.5
  equilibration_steps: 10
  production_steps: 20
  sample_interval: 2
  rdf_r_max_A: 6.0
  rdf_bin_width_A: 0.1
""",
        encoding="utf-8",
    )

    try:
        load_config(config_file)
    except ValueError as exc:
        assert "split fractions must sum to 1.0" in str(exc)
    else:
        raise AssertionError("Expected split fraction validation failure")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_config.py -q
```

Expected: FAIL because `water_mlip_benchmark.config` does not exist.

- [ ] **Step 3: Implement configuration module**

Create `src/water_mlip_benchmark/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    archive_name: str
    converted_extxyz: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class MDConfig:
    temperature_K: float
    timestep_fs: float
    equilibration_steps: int
    production_steps: int
    sample_interval: int
    rdf_r_max_A: float
    rdf_bin_width_A: float


@dataclass(frozen=True)
class BenchmarkConfig:
    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    runs_dir: Path
    figures_dir: Path
    random_seed: int
    dataset: DatasetConfig
    md: MDConfig


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a mapping")
    return data


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = _require_mapping(raw, "config")

    base_dir = config_path.parent
    project_root = _resolve(base_dir, str(data["project_root"]))
    dataset_data = _require_mapping(data["dataset"], "dataset")
    md_data = _require_mapping(data["md"], "md")

    dataset = DatasetConfig(
        archive_name=str(dataset_data["archive_name"]),
        converted_extxyz=str(dataset_data["converted_extxyz"]),
        train_fraction=float(dataset_data["train_fraction"]),
        validation_fraction=float(dataset_data["validation_fraction"]),
        test_fraction=float(dataset_data["test_fraction"]),
    )
    split_sum = dataset.train_fraction + dataset.validation_fraction + dataset.test_fraction
    if abs(split_sum - 1.0) > 1.0e-8:
        raise ValueError("dataset split fractions must sum to 1.0")

    md = MDConfig(
        temperature_K=float(md_data["temperature_K"]),
        timestep_fs=float(md_data["timestep_fs"]),
        equilibration_steps=int(md_data["equilibration_steps"]),
        production_steps=int(md_data["production_steps"]),
        sample_interval=int(md_data["sample_interval"]),
        rdf_r_max_A=float(md_data["rdf_r_max_A"]),
        rdf_bin_width_A=float(md_data["rdf_bin_width_A"]),
    )

    return BenchmarkConfig(
        project_root=project_root,
        raw_data_dir=_resolve(project_root, str(data["raw_data_dir"])),
        processed_data_dir=_resolve(project_root, str(data["processed_data_dir"])),
        runs_dir=_resolve(project_root, str(data["runs_dir"])),
        figures_dir=_resolve(project_root, str(data["figures_dir"])),
        random_seed=int(data["random_seed"]),
        dataset=dataset,
        md=md,
    )
```

Create `configs/experiment.yaml`:

```yaml
project_root: ..
raw_data_dir: data/raw
processed_data_dir: data/processed
runs_dir: runs
figures_dir: figures
random_seed: 20260518
dataset:
  archive_name: training-set.zip
  converted_extxyz: cheng-water.extxyz
  train_fraction: 0.8
  validation_fraction: 0.1
  test_fraction: 0.1
md:
  temperature_K: 300.0
  timestep_fs: 0.5
  equilibration_steps: 1000
  production_steps: 10000
  sample_interval: 10
  rdf_r_max_A: 6.0
  rdf_bin_width_A: 0.02
```

Create `configs/mace-small.yaml`:

```yaml
name: mace-small-water
model: MACE
r_max: 5.0
num_channels: 32
max_L: 1
correlation: 2
batch_size: 4
max_num_epochs: 200
valid_batch_size: 4
default_dtype: float64
device: cpu
energy_weight: 1.0
forces_weight: 100.0
ema: true
ema_decay: 0.99
```

- [ ] **Step 4: Run configuration tests**

Run:

```bash
pytest tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit configuration**

```bash
git add configs/experiment.yaml configs/mace-small.yaml src/water_mlip_benchmark/config.py tests/test_config.py
git commit -m "feat: add benchmark configuration loading"
```

## Task 3: Dataset Provenance, Archive Probe, And CLI Skeleton

**Files:**
- Create: `docs/data/cheng-water.md`
- Create: `src/water_mlip_benchmark/data_sources.py`
- Create: `src/water_mlip_benchmark/cli.py`
- Create: `tests/test_data_sources.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for dataset source and CLI**

Create `tests/test_data_sources.py`:

```python
from pathlib import Path
from zipfile import ZipFile

from water_mlip_benchmark.data_sources import CHENG_WATER, probe_zip_archive


def test_cheng_dataset_manifest_has_sources() -> None:
    assert CHENG_WATER.name == "Cheng revPBE0-D3 water"
    assert "materialscloud:2018.0020/v1" in CHENG_WATER.materials_cloud_record
    assert CHENG_WATER.license == "CC-BY-4.0"


def test_probe_zip_archive_lists_files(tmp_path: Path) -> None:
    archive = tmp_path / "training-set.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("training-set/input.data", "begin\nend\n")
        zip_file.writestr("README.txt", "synthetic archive")

    summary = probe_zip_archive(archive)

    assert summary.archive_path == archive
    assert summary.file_count == 2
    assert summary.total_uncompressed_bytes > 0
    assert "training-set/input.data" in summary.names
```

Create `tests/test_cli.py`:

```python
from pathlib import Path
from zipfile import ZipFile

from water_mlip_benchmark.cli import main


def test_cli_config_summary(capsys) -> None:
    exit_code = main(["config-summary", "configs/experiment.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "raw_data_dir" in captured.out


def test_cli_probe_archive(tmp_path: Path, capsys) -> None:
    archive = tmp_path / "training-set.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("input.data", "begin\nend\n")

    exit_code = main(["probe-archive", str(archive)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "file_count: 1" in captured.out
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_data_sources.py tests/test_cli.py -q
```

Expected: FAIL because dataset source and CLI modules do not exist.

- [ ] **Step 3: Implement dataset source and CLI**

Create `docs/data/cheng-water.md`:

```markdown
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
```

Create `src/water_mlip_benchmark/data_sources.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile


@dataclass(frozen=True)
class DatasetSource:
    name: str
    materials_cloud_record: str
    landing_page: str
    github_url: str
    publication_doi: str
    license: str


@dataclass(frozen=True)
class ArchiveSummary:
    archive_path: Path
    file_count: int
    total_uncompressed_bytes: int
    names: tuple[str, ...]


CHENG_WATER = DatasetSource(
    name="Cheng revPBE0-D3 water",
    materials_cloud_record="materialscloud:2018.0020/v1",
    landing_page="https://archive.materialscloud.org/record/2018.0020/v1",
    github_url="https://github.com/BingqingCheng/neural-network-potential-for-water-revPBE0-D3",
    publication_doi="https://doi.org/10.1073/pnas.1815117116",
    license="CC-BY-4.0",
)


def probe_zip_archive(path: str | Path) -> ArchiveSummary:
    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(f"archive does not exist: {archive_path}")

    with ZipFile(archive_path) as zip_file:
        infos = zip_file.infolist()
        names = tuple(info.filename for info in infos)
        total_size = sum(info.file_size for info in infos)

    return ArchiveSummary(
        archive_path=archive_path,
        file_count=len(names),
        total_uncompressed_bytes=total_size,
        names=names,
    )
```

Create `src/water_mlip_benchmark/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from water_mlip_benchmark.config import load_config
from water_mlip_benchmark.data_sources import probe_zip_archive


def _config_summary(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(f"project_root: {config.project_root}")
    print(f"raw_data_dir: {config.raw_data_dir}")
    print(f"processed_data_dir: {config.processed_data_dir}")
    print(f"runs_dir: {config.runs_dir}")
    print(f"figures_dir: {config.figures_dir}")
    print(f"random_seed: {config.random_seed}")
    return 0


def _probe_archive(args: argparse.Namespace) -> int:
    summary = probe_zip_archive(args.archive)
    print(f"archive_path: {summary.archive_path}")
    print(f"file_count: {summary.file_count}")
    print(f"total_uncompressed_bytes: {summary.total_uncompressed_bytes}")
    print("files:")
    for name in summary.names:
        print(f"  - {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="water-mlip")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config-summary")
    config_parser.add_argument("config", type=Path)
    config_parser.set_defaults(func=_config_summary)

    probe_parser = subparsers.add_parser("probe-archive")
    probe_parser.add_argument("archive", type=Path)
    probe_parser.set_defaults(func=_probe_archive)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
```

- [ ] **Step 4: Run dataset and CLI tests**

Run:

```bash
pytest tests/test_data_sources.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit dataset provenance and CLI skeleton**

```bash
git add docs/data/cheng-water.md src/water_mlip_benchmark/data_sources.py src/water_mlip_benchmark/cli.py tests/test_data_sources.py tests/test_cli.py
git commit -m "feat: add dataset provenance and cli skeleton"
```

## Task 4: Dataset Conversion To Extended XYZ

**Files:**
- Create: `src/water_mlip_benchmark/convert.py`
- Create: `tests/test_convert.py`
- Modify: `src/water_mlip_benchmark/cli.py`

- [ ] **Step 1: Write failing converter tests**

Create `tests/test_convert.py`:

```python
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from ase.io import read

from water_mlip_benchmark.convert import convert_archive_to_extxyz, read_runner_input_data


RUNNER_TEXT = """begin
comment synthetic water
lattice 10.0 0.0 0.0
lattice 0.0 10.0 0.0
lattice 0.0 0.0 10.0
atom 0.0 0.0 0.0 O 0.0 0.0 0.1 0.2 0.3
atom 0.8 0.0 0.0 H 0.0 0.0 -0.1 -0.2 -0.3
atom 0.0 0.8 0.0 H 0.0 0.0 0.0 0.1 0.0
energy -76.0
charge 0.0
end
"""


def test_read_runner_input_data() -> None:
    atoms_list = read_runner_input_data(RUNNER_TEXT)

    assert len(atoms_list) == 1
    atoms = atoms_list[0]
    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
    assert atoms.info["REF_energy"] == -76.0
    np.testing.assert_allclose(atoms.arrays["REF_forces"][0], [0.1, 0.2, 0.3])
    assert atoms.cell.lengths()[0] == 10.0


def test_convert_archive_to_extxyz(tmp_path: Path) -> None:
    archive = tmp_path / "training-set.zip"
    output = tmp_path / "converted.extxyz"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("training-set/input.data", RUNNER_TEXT)

    count = convert_archive_to_extxyz(archive, output)

    assert count == 1
    frames = read(output, ":")
    assert len(frames) == 1
    assert frames[0].info["REF_energy"] == -76.0
```

- [ ] **Step 2: Run converter tests to verify failure**

Run:

```bash
pytest tests/test_convert.py -q
```

Expected: FAIL because `convert.py` does not exist.

- [ ] **Step 3: Implement converter**

Create `src/water_mlip_benchmark/convert.py`:

```python
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from ase import Atoms
from ase.io import read, write


def read_runner_input_data(text: str) -> list[Atoms]:
    frames: list[Atoms] = []
    lattice: list[list[float]] = []
    symbols: list[str] = []
    positions: list[list[float]] = []
    forces: list[list[float]] = []
    energy: float | None = None
    in_block = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        keyword = fields[0].lower()

        if keyword == "begin":
            lattice = []
            symbols = []
            positions = []
            forces = []
            energy = None
            in_block = True
            continue

        if keyword == "end":
            if not in_block:
                raise ValueError("encountered end outside a begin block")
            atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))
            if lattice:
                atoms.set_cell(np.asarray(lattice, dtype=float))
                atoms.set_pbc(True)
            if energy is not None:
                atoms.info["REF_energy"] = energy
            if forces:
                atoms.arrays["REF_forces"] = np.asarray(forces, dtype=float)
            frames.append(atoms)
            in_block = False
            continue

        if not in_block:
            continue

        if keyword == "lattice":
            if len(fields) != 4:
                raise ValueError(f"bad lattice line: {line}")
            lattice.append([float(fields[1]), float(fields[2]), float(fields[3])])
        elif keyword == "atom":
            if len(fields) < 10:
                raise ValueError(f"bad atom line: {line}")
            positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
            symbols.append(fields[4])
            forces.append([float(fields[-3]), float(fields[-2]), float(fields[-1])])
        elif keyword == "energy":
            if len(fields) != 2:
                raise ValueError(f"bad energy line: {line}")
            energy = float(fields[1])

    if in_block:
        raise ValueError("unterminated begin block")
    return frames


def _read_structure_file(path: Path) -> list[Atoms]:
    if path.name == "input.data":
        return read_runner_input_data(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".xyz", ".extxyz"}:
        frames = read(path, ":")
        if isinstance(frames, Atoms):
            return [frames]
        return list(frames)
    return []


def convert_archive_to_extxyz(archive_path: str | Path, output_path: str | Path) -> int:
    archive_path = Path(archive_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Atoms] = []
    with TemporaryDirectory() as tmpdir:
        extract_dir = Path(tmpdir)
        with ZipFile(archive_path) as zip_file:
            zip_file.extractall(extract_dir)
        for path in sorted(extract_dir.rglob("*")):
            if path.is_file():
                frames.extend(_read_structure_file(path))

    if not frames:
        raise ValueError(f"no supported structure files found in {archive_path}")

    write(output_path, frames, format="extxyz")
    return len(frames)
```

- [ ] **Step 4: Add CLI convert command**

Modify `src/water_mlip_benchmark/cli.py` to import and expose conversion:

```python
from water_mlip_benchmark.convert import convert_archive_to_extxyz
```

Add this function:

```python
def _convert(args: argparse.Namespace) -> int:
    count = convert_archive_to_extxyz(args.archive, args.output)
    print(f"converted_frames: {count}")
    print(f"output: {args.output}")
    return 0
```

Add this parser block inside `build_parser()`:

```python
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("archive", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.set_defaults(func=_convert)
```

- [ ] **Step 5: Run converter and CLI tests**

Run:

```bash
pytest tests/test_convert.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit conversion pipeline**

```bash
git add src/water_mlip_benchmark/convert.py src/water_mlip_benchmark/cli.py tests/test_convert.py
git commit -m "feat: convert cheng water data to extxyz"
```

## Task 5: Static Energy And Force Metrics

**Files:**
- Create: `src/water_mlip_benchmark/metrics.py`
- Create: `tests/test_metrics.py`
- Modify: `src/water_mlip_benchmark/cli.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_metrics.py`:

```python
import numpy as np

from water_mlip_benchmark.metrics import MetricSummary, summarize_errors


def test_summarize_errors() -> None:
    reference_energy = np.array([1.0, 2.0, 3.0])
    predicted_energy = np.array([1.1, 1.8, 3.3])
    reference_forces = np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]])
    predicted_forces = np.array([[[0.1, 0.0, 0.0]], [[0.8, 0.0, 0.0]], [[0.0, 1.5, 0.0]]])

    summary = summarize_errors(reference_energy, predicted_energy, reference_forces, predicted_forces)

    assert isinstance(summary, MetricSummary)
    assert round(summary.energy_mae, 6) == 0.2
    assert round(summary.force_component_mae, 6) == round((0.1 + 0.2 + 0.5) / 9.0, 6)
    assert summary.force_vector_p95 > 0.0


def test_summarize_errors_rejects_shape_mismatch() -> None:
    try:
        summarize_errors(np.array([1.0]), np.array([1.0, 2.0]), np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    except ValueError as exc:
        assert "energy arrays must have matching shapes" in str(exc)
    else:
        raise AssertionError("Expected shape mismatch failure")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_metrics.py -q
```

Expected: FAIL because `metrics.py` does not exist.

- [ ] **Step 3: Implement metrics**

Create `src/water_mlip_benchmark/metrics.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MetricSummary:
    energy_mae: float
    energy_rmse: float
    force_component_mae: float
    force_component_rmse: float
    force_vector_mae: float
    force_vector_rmse: float
    force_vector_p95: float
    force_vector_max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _validate_same_shape(name: str, reference: np.ndarray, predicted: np.ndarray) -> None:
    if reference.shape != predicted.shape:
        raise ValueError(f"{name} arrays must have matching shapes: {reference.shape} != {predicted.shape}")


def summarize_errors(
    reference_energy: np.ndarray,
    predicted_energy: np.ndarray,
    reference_forces: np.ndarray,
    predicted_forces: np.ndarray,
) -> MetricSummary:
    reference_energy = np.asarray(reference_energy, dtype=float)
    predicted_energy = np.asarray(predicted_energy, dtype=float)
    reference_forces = np.asarray(reference_forces, dtype=float)
    predicted_forces = np.asarray(predicted_forces, dtype=float)

    _validate_same_shape("energy", reference_energy, predicted_energy)
    _validate_same_shape("force", reference_forces, predicted_forces)

    energy_error = predicted_energy - reference_energy
    force_error = predicted_forces - reference_forces
    force_vector_error = np.linalg.norm(force_error, axis=-1)

    return MetricSummary(
        energy_mae=float(np.mean(np.abs(energy_error))),
        energy_rmse=float(np.sqrt(np.mean(energy_error**2))),
        force_component_mae=float(np.mean(np.abs(force_error))),
        force_component_rmse=float(np.sqrt(np.mean(force_error**2))),
        force_vector_mae=float(np.mean(force_vector_error)),
        force_vector_rmse=float(np.sqrt(np.mean(force_vector_error**2))),
        force_vector_p95=float(np.percentile(force_vector_error, 95.0)),
        force_vector_max=float(np.max(force_vector_error)),
    )
```

- [ ] **Step 4: Run metric tests**

Run:

```bash
pytest tests/test_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit static metrics**

```bash
git add src/water_mlip_benchmark/metrics.py tests/test_metrics.py
git commit -m "feat: add static mlip error metrics"
```

## Task 6: MD Observable Analysis

**Files:**
- Create: `src/water_mlip_benchmark/observables.py`
- Create: `tests/test_observables.py`

- [ ] **Step 1: Write failing observable tests**

Create `tests/test_observables.py`:

```python
import numpy as np
from ase import Atoms

from water_mlip_benchmark.observables import compute_energy_drift, compute_msd, compute_rdf


def test_compute_energy_drift() -> None:
    times_fs = np.array([0.0, 1.0, 2.0, 3.0])
    energies_eV = np.array([10.0, 10.1, 10.2, 10.3])

    drift = compute_energy_drift(times_fs, energies_eV)

    assert round(drift.slope_eV_per_ps, 6) == 100.0
    assert drift.total_drift_eV == 0.3000000000000007


def test_compute_msd_linear_motion() -> None:
    positions = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ]
    )

    times, msd = compute_msd(positions, timestep_fs=2.0)

    np.testing.assert_allclose(times, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(msd, [0.0, 1.0, 4.0])


def test_compute_rdf_has_expected_bins() -> None:
    atoms = Atoms(
        symbols=["O", "O"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    rdf = compute_rdf([atoms], pair=("O", "O"), r_max_A=2.0, bin_width_A=0.5)

    assert rdf.r_A.shape == rdf.g_r.shape
    assert rdf.counts.sum() == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_observables.py -q
```

Expected: FAIL because `observables.py` does not exist.

- [ ] **Step 3: Implement observable functions**

Create `src/water_mlip_benchmark/observables.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class EnergyDrift:
    slope_eV_per_ps: float
    total_drift_eV: float
    intercept_eV: float


@dataclass(frozen=True)
class RDFResult:
    r_A: np.ndarray
    g_r: np.ndarray
    counts: np.ndarray


def compute_energy_drift(times_fs: np.ndarray, energies_eV: np.ndarray) -> EnergyDrift:
    times_fs = np.asarray(times_fs, dtype=float)
    energies_eV = np.asarray(energies_eV, dtype=float)
    if times_fs.shape != energies_eV.shape:
        raise ValueError("times and energies must have matching shapes")
    if times_fs.size < 2:
        raise ValueError("at least two time points are required")

    times_ps = times_fs / 1000.0
    slope, intercept = np.polyfit(times_ps, energies_eV, deg=1)
    return EnergyDrift(
        slope_eV_per_ps=float(slope),
        total_drift_eV=float(energies_eV[-1] - energies_eV[0]),
        intercept_eV=float(intercept),
    )


def compute_msd(positions_A: np.ndarray, timestep_fs: float) -> tuple[np.ndarray, np.ndarray]:
    positions_A = np.asarray(positions_A, dtype=float)
    if positions_A.ndim != 3 or positions_A.shape[-1] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3)")

    displacements = positions_A - positions_A[0:1, :, :]
    squared = np.sum(displacements**2, axis=-1)
    msd = np.mean(squared, axis=1)
    times_fs = np.arange(positions_A.shape[0], dtype=float) * float(timestep_fs)
    return times_fs, msd


def compute_rdf(frames: list[Atoms], pair: tuple[str, str], r_max_A: float, bin_width_A: float) -> RDFResult:
    if not frames:
        raise ValueError("at least one frame is required")
    edges = np.arange(0.0, r_max_A + bin_width_A, bin_width_A)
    counts = np.zeros(len(edges) - 1, dtype=float)
    pair_a, pair_b = pair

    for atoms in frames:
        symbols = atoms.get_chemical_symbols()
        indices_a = [idx for idx, symbol in enumerate(symbols) if symbol == pair_a]
        indices_b = [idx for idx, symbol in enumerate(symbols) if symbol == pair_b]
        for local_i, atom_i in enumerate(indices_a):
            for atom_j in indices_b:
                if pair_a == pair_b and atom_j <= atom_i:
                    continue
                distance = atoms.get_distance(atom_i, atom_j, mic=True)
                if 0.0 < distance < r_max_A:
                    bin_index = int(distance / bin_width_A)
                    counts[bin_index] += 1.0

    r = 0.5 * (edges[:-1] + edges[1:])
    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = float(np.mean([atoms.get_volume() for atoms in frames]))
    number_density = len(frames[0]) / volume
    normalization = len(frames) * max(len(frames[0]), 1) * number_density * shell_volumes
    g_r = np.divide(counts, normalization, out=np.zeros_like(counts), where=normalization > 0.0)
    return RDFResult(r_A=r, g_r=g_r, counts=counts)
```

- [ ] **Step 4: Run observable tests**

Run:

```bash
pytest tests/test_observables.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit observables**

```bash
git add src/water_mlip_benchmark/observables.py tests/test_observables.py
git commit -m "feat: add md observable analysis"
```

## Task 7: MACE Command Generation

**Files:**
- Create: `src/water_mlip_benchmark/mace.py`
- Create: `tests/test_mace.py`
- Modify: `src/water_mlip_benchmark/cli.py`

- [ ] **Step 1: Write failing MACE command tests**

Create `tests/test_mace.py`:

```python
from pathlib import Path

from water_mlip_benchmark.mace import build_mace_train_command, discover_checkpoints


def test_build_mace_train_command() -> None:
    command = build_mace_train_command(
        train_file=Path("data/processed/train.extxyz"),
        valid_file=Path("data/processed/valid.extxyz"),
        config_file=Path("configs/mace-small.yaml"),
        run_dir=Path("runs/mace-small"),
    )

    assert command[0] == "mace_run_train"
    assert "--train_file" in command
    assert "data/processed/train.extxyz" in command
    assert "--valid_file" in command
    assert "--config" in command


def test_discover_checkpoints(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "model_epoch-001.model").write_text("one", encoding="utf-8")
    (checkpoints / "model_epoch-010.model").write_text("ten", encoding="utf-8")

    found = discover_checkpoints(run_dir)

    assert [path.name for path in found] == ["model_epoch-001.model", "model_epoch-010.model"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_mace.py -q
```

Expected: FAIL because `mace.py` does not exist.

- [ ] **Step 3: Implement MACE helpers**

Create `src/water_mlip_benchmark/mace.py`:

```python
from __future__ import annotations

from pathlib import Path


def build_mace_train_command(
    train_file: Path,
    valid_file: Path,
    config_file: Path,
    run_dir: Path,
) -> list[str]:
    return [
        "mace_run_train",
        "--train_file",
        str(train_file),
        "--valid_file",
        str(valid_file),
        "--config",
        str(config_file),
        "--results_dir",
        str(run_dir / "results"),
        "--checkpoints_dir",
        str(run_dir / "checkpoints"),
        "--name",
        run_dir.name,
    ]


def discover_checkpoints(run_dir: Path) -> list[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("*.model"))
```

- [ ] **Step 4: Add CLI command generation**

Modify `src/water_mlip_benchmark/cli.py` to import:

```python
from water_mlip_benchmark.mace import build_mace_train_command
```

Add function:

```python
def _mace_train_command(args: argparse.Namespace) -> int:
    command = build_mace_train_command(args.train_file, args.valid_file, args.config, args.run_dir)
    print(" ".join(command))
    return 0
```

Add parser block:

```python
    mace_parser = subparsers.add_parser("mace-train-command")
    mace_parser.add_argument("train_file", type=Path)
    mace_parser.add_argument("valid_file", type=Path)
    mace_parser.add_argument("config", type=Path)
    mace_parser.add_argument("run_dir", type=Path)
    mace_parser.set_defaults(func=_mace_train_command)
```

- [ ] **Step 5: Run MACE tests**

Run:

```bash
pytest tests/test_mace.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit MACE helpers**

```bash
git add src/water_mlip_benchmark/mace.py src/water_mlip_benchmark/cli.py tests/test_mace.py
git commit -m "feat: add mace training command helpers"
```

## Task 8: Plotting Metric-To-Observable Relationships

**Files:**
- Create: `src/water_mlip_benchmark/plots.py`
- Create: `tests/test_plots.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write failing plotting test**

Create `tests/test_plots.py`:

```python
from pathlib import Path

from water_mlip_benchmark.plots import plot_metric_relationship


def test_plot_metric_relationship(tmp_path: Path) -> None:
    output = tmp_path / "force_mae_vs_rdf_error.png"
    plot_metric_relationship(
        x=[0.3, 0.2, 0.1],
        y=[1.0, 0.6, 0.4],
        labels=["early", "mid", "best"],
        x_label="Force MAE",
        y_label="RDF error",
        title="Static error vs RDF error",
        output_path=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
pytest tests/test_plots.py -q
```

Expected: FAIL because `plots.py` does not exist.

- [ ] **Step 3: Implement plotting helper**

Create `src/water_mlip_benchmark/plots.py`:

```python
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt


def plot_metric_relationship(
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[str],
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    if not (len(x) == len(y) == len(labels)):
        raise ValueError("x, y, and labels must have matching lengths")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.scatter(x, y, color="#2458a7")
    for x_value, y_value, label in zip(x, y, labels, strict=True):
        ax.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
```

- [ ] **Step 4: Force noninteractive Matplotlib backend for tests**

Create `tests/conftest.py`:

```python
import matplotlib

matplotlib.use("Agg")
```

- [ ] **Step 5: Run plotting tests**

Run:

```bash
pytest tests/test_plots.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit plotting helper**

```bash
git add src/water_mlip_benchmark/plots.py tests/test_plots.py tests/conftest.py
git commit -m "feat: add metric relationship plotting"
```

## Task 9: End-To-End Smoke Workflow

**Files:**
- Create: `tests/test_smoke_workflow.py`
- Create: `docs/workflows/first-smoke-run.md`
- Modify: `README.md`

- [ ] **Step 1: Write smoke workflow test**

Create `tests/test_smoke_workflow.py`:

```python
from pathlib import Path
from zipfile import ZipFile

from ase.io import read

from water_mlip_benchmark.convert import convert_archive_to_extxyz
from water_mlip_benchmark.metrics import summarize_errors
from water_mlip_benchmark.observables import compute_rdf


RUNNER_TEXT = """begin
lattice 10.0 0.0 0.0
lattice 0.0 10.0 0.0
lattice 0.0 0.0 10.0
atom 0.0 0.0 0.0 O 0.0 0.0 0.0 0.0 0.0
atom 0.8 0.0 0.0 H 0.0 0.0 0.0 0.0 0.0
atom 0.0 0.8 0.0 H 0.0 0.0 0.0 0.0 0.0
energy -76.0
charge 0.0
end
begin
lattice 10.0 0.0 0.0
lattice 0.0 10.0 0.0
lattice 0.0 0.0 10.0
atom 0.1 0.0 0.0 O 0.0 0.0 0.1 0.0 0.0
atom 0.9 0.0 0.0 H 0.0 0.0 -0.1 0.0 0.0
atom 0.1 0.8 0.0 H 0.0 0.0 0.0 0.0 0.0
energy -75.9
charge 0.0
end
"""


def test_synthetic_end_to_end_workflow(tmp_path: Path) -> None:
    archive = tmp_path / "training-set.zip"
    extxyz = tmp_path / "converted.extxyz"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("input.data", RUNNER_TEXT)

    count = convert_archive_to_extxyz(archive, extxyz)
    frames = read(extxyz, ":")
    energies = [atoms.info["REF_energy"] for atoms in frames]
    forces = [atoms.arrays["REF_forces"] for atoms in frames]

    summary = summarize_errors(energies, energies, forces, forces)
    rdf = compute_rdf(frames, pair=("O", "H"), r_max_A=2.0, bin_width_A=0.1)

    assert count == 2
    assert summary.energy_mae == 0.0
    assert rdf.counts.sum() > 0.0
```

- [ ] **Step 2: Run smoke test**

Run:

```bash
pytest tests/test_smoke_workflow.py -q
```

Expected: PASS.

- [ ] **Step 3: Write first workflow documentation**

Create `docs/workflows/first-smoke-run.md`:

```markdown
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
```

Modify `README.md` by adding:

```markdown
## Documentation

- Design spec: `docs/superpowers/specs/2026-05-18-water-mlip-metric-observable-design.md`
- First smoke workflow: `docs/workflows/first-smoke-run.md`
```

- [ ] **Step 4: Run full test suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 5: Commit smoke workflow**

```bash
git add tests/test_smoke_workflow.py docs/workflows/first-smoke-run.md README.md
git commit -m "test: add synthetic benchmark smoke workflow"
```

## Task 10: Manual Dataset Probe Check

**Files:**
- Modify: `docs/data/cheng-water.md`

- [ ] **Step 1: Download official archive manually**

Open the Materials Cloud record:

```text
https://archive.materialscloud.org/record/2018.0020/v1
```

Download `training-set.zip` and place it at:

```text
data/raw/training-set.zip
```

- [ ] **Step 2: Probe official archive**

Run:

```bash
water-mlip probe-archive data/raw/training-set.zip
```

Expected: the command prints a nonzero `file_count` and shows at least one supported structure file such as `input.data`, `.xyz`, or `.extxyz`.

- [ ] **Step 3: Convert official archive**

Run:

```bash
water-mlip convert data/raw/training-set.zip data/processed/cheng-water.extxyz
```

Expected: the command prints `converted_frames:` with a positive integer.

- [ ] **Step 4: If conversion needs a parser adjustment, add a focused fixture test**

If the official archive contains a supported-looking text format that is not parsed, copy five non-sensitive lines from one representative structure block into `tests/test_convert.py` as a synthetic fixture and update `read_runner_input_data()` or `_read_structure_file()` to parse that exact format.

Run:

```bash
pytest tests/test_convert.py tests/test_smoke_workflow.py -q
```

Expected: PASS.

- [ ] **Step 5: Record the observed archive structure**

Append this section to `docs/data/cheng-water.md` after copying the real `file_count` value and the supported structure-file path from the `probe-archive` output:

```markdown
## Observed Archive Structure

The local `training-set.zip` archive was probed with:

```bash
water-mlip probe-archive data/raw/training-set.zip
```

Observed file count: copy the integer printed after `file_count:`.

Supported structure files used by the converter:

- Copy the supported structure-file path printed under `files:`.
```

Before committing, make the two copied values concrete in `docs/data/cheng-water.md` so the committed document contains the observed integer and path, not instructions.

- [ ] **Step 6: Commit dataset probe documentation and parser fixes**

```bash
git add docs/data/cheng-water.md src/water_mlip_benchmark/convert.py tests/test_convert.py
git commit -m "docs: record cheng dataset archive structure"
```

## Task 11: Verification Before Training

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run full unit suite**

Run:

```bash
pytest -q
```

Expected: PASS.

- [ ] **Step 2: Run CLI checks**

Run:

```bash
water-mlip config-summary configs/experiment.yaml
water-mlip mace-train-command data/processed/train.extxyz data/processed/valid.extxyz configs/mace-small.yaml runs/mace-small
```

Expected: the first command prints resolved project paths. The second command prints a `mace_run_train` command.

- [ ] **Step 3: Update README with verified commands**

Add this section to `README.md`:

```markdown
## Verified Local Checks

Before launching MACE training, run:

```bash
pytest -q
water-mlip config-summary configs/experiment.yaml
water-mlip probe-archive data/raw/training-set.zip
water-mlip convert data/raw/training-set.zip data/processed/cheng-water.extxyz
```

The `data/` directory is intentionally ignored by git.
```

- [ ] **Step 4: Commit verification docs**

```bash
git add README.md
git commit -m "docs: add pretraining verification checks"
```

## Self-Review

Spec coverage:

- Existing public Cheng dataset: covered by Tasks 3, 4, and 10.
- One MLIP architecture with MACE as default: covered by Tasks 2 and 7.
- Multiple checkpoints: checkpoint discovery is covered by Task 7; actual long training execution is the next project phase after the scaffold.
- Static energy/force metrics: covered by Task 5.
- MD observables: covered by Task 6.
- Reproducible scripts and written analysis foundation: covered by Tasks 1, 8, 9, and 11.
- Avoid generating DFT data: documented in Task 3 and enforced by workflow using downloaded public data.

Known boundary:

- This plan builds the first end-to-end scaffold and local verification workflow. The next implementation plan should cover production MACE training, train/validation/test splitting, checkpoint evaluation against real model predictions, ASE-MACE MD execution, and final figure generation.
