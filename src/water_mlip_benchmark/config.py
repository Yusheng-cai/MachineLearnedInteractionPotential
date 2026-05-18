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
