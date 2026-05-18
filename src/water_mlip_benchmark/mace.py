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
