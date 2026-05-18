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
