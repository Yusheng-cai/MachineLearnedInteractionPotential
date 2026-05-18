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
