from pathlib import Path
from zipfile import ZipFile

from ase import Atoms
from ase.io import write

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


def test_cli_split(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.extxyz"
    frames = [
        Atoms(symbols=["O", "H", "H"], positions=[[0.0, 0.0, index], [0.8, 0.0, index], [0.0, 0.8, index]])
        for index in range(5)
    ]
    write(source, frames, format="extxyz")

    exit_code = main(["split", str(source), str(tmp_path / "splits"), "--train-fraction", "0.6", "--valid-fraction", "0.2"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "train: 3" in captured.out
    assert "valid: 1" in captured.out
    assert "test: 1" in captured.out
