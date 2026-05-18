from pathlib import Path

from ase import Atoms
from ase.io import read, write

from water_mlip_benchmark.split import split_extxyz


def _write_frames(path: Path, count: int) -> None:
    frames = [
        Atoms(
            symbols=["O", "H", "H"],
            positions=[[0.0, 0.0, float(index)], [0.8, 0.0, float(index)], [0.0, 0.8, float(index)]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        for index in range(count)
    ]
    write(path, frames, format="extxyz")


def test_split_extxyz_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.extxyz"
    _write_frames(source, 10)

    first = split_extxyz(source, tmp_path / "first", train_fraction=0.6, validation_fraction=0.2, seed=17)
    second = split_extxyz(source, tmp_path / "second", train_fraction=0.6, validation_fraction=0.2, seed=17)

    assert first.counts == {"train": 6, "valid": 2, "test": 2}
    assert second.counts == first.counts
    assert [atoms.positions[0, 2] for atoms in read(first.train_file, ":")] == [
        atoms.positions[0, 2] for atoms in read(second.train_file, ":")
    ]


def test_split_extxyz_rejects_invalid_fractions(tmp_path: Path) -> None:
    source = tmp_path / "source.extxyz"
    _write_frames(source, 3)

    try:
        split_extxyz(source, tmp_path / "splits", train_fraction=0.8, validation_fraction=0.3, seed=1)
    except ValueError as exc:
        assert "train and validation fractions must leave a positive test fraction" in str(exc)
    else:
        raise AssertionError("Expected invalid split fractions to fail")
