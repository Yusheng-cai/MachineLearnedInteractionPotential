from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


@dataclass(frozen=True)
class SplitResult:
    train_file: Path
    valid_file: Path
    test_file: Path
    counts: dict[str, int]


def _read_frames(path: Path) -> list[Atoms]:
    frames = read(path, ":")
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def split_extxyz(
    source_file: str | Path,
    output_dir: str | Path,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> SplitResult:
    if train_fraction <= 0.0 or validation_fraction <= 0.0:
        raise ValueError("train and validation fractions must be positive")
    test_fraction = 1.0 - train_fraction - validation_fraction
    if test_fraction <= 0.0:
        raise ValueError("train and validation fractions must leave a positive test fraction")

    source_file = Path(source_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = _read_frames(source_file)
    if len(frames) < 3:
        raise ValueError("at least three frames are required for train/valid/test splits")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(frames))
    train_count = int(np.floor(len(frames) * train_fraction))
    valid_count = int(np.floor(len(frames) * validation_fraction))
    test_count = len(frames) - train_count - valid_count
    if min(train_count, valid_count, test_count) <= 0:
        raise ValueError("split fractions produce an empty train, validation, or test split")

    train_indices = indices[:train_count]
    valid_indices = indices[train_count : train_count + valid_count]
    test_indices = indices[train_count + valid_count :]

    train_file = output_dir / "train.extxyz"
    valid_file = output_dir / "valid.extxyz"
    test_file = output_dir / "test.extxyz"

    write(train_file, [frames[index] for index in train_indices], format="extxyz")
    write(valid_file, [frames[index] for index in valid_indices], format="extxyz")
    write(test_file, [frames[index] for index in test_indices], format="extxyz")

    return SplitResult(
        train_file=train_file,
        valid_file=valid_file,
        test_file=test_file,
        counts={"train": train_count, "valid": valid_count, "test": test_count},
    )
