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
