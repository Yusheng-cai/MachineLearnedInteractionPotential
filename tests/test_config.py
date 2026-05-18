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
