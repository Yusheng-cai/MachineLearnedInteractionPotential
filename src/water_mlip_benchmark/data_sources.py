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
