from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from ase import Atoms
from ase.io import read, write


def read_runner_input_data(text: str) -> list[Atoms]:
    frames: list[Atoms] = []
    lattice: list[list[float]] = []
    symbols: list[str] = []
    positions: list[list[float]] = []
    forces: list[list[float]] = []
    energy: float | None = None
    in_block = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        keyword = fields[0].lower()

        if keyword == "begin":
            lattice = []
            symbols = []
            positions = []
            forces = []
            energy = None
            in_block = True
            continue

        if keyword == "end":
            if not in_block:
                raise ValueError("encountered end outside a begin block")
            atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))
            if lattice:
                atoms.set_cell(np.asarray(lattice, dtype=float))
                atoms.set_pbc(True)
            if energy is not None:
                atoms.info["REF_energy"] = energy
            if forces:
                atoms.arrays["REF_forces"] = np.asarray(forces, dtype=float)
            frames.append(atoms)
            in_block = False
            continue

        if not in_block:
            continue

        if keyword == "lattice":
            if len(fields) != 4:
                raise ValueError(f"bad lattice line: {line}")
            lattice.append([float(fields[1]), float(fields[2]), float(fields[3])])
        elif keyword == "atom":
            if len(fields) < 10:
                raise ValueError(f"bad atom line: {line}")
            positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
            symbols.append(fields[4])
            forces.append([float(fields[-3]), float(fields[-2]), float(fields[-1])])
        elif keyword == "energy":
            if len(fields) != 2:
                raise ValueError(f"bad energy line: {line}")
            energy = float(fields[1])

    if in_block:
        raise ValueError("unterminated begin block")
    return frames


def _read_structure_file(path: Path) -> list[Atoms]:
    if path.name == "input.data":
        return read_runner_input_data(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".xyz", ".extxyz"}:
        frames = read(path, ":")
        if isinstance(frames, Atoms):
            return [frames]
        return list(frames)
    return []


def convert_archive_to_extxyz(archive_path: str | Path, output_path: str | Path) -> int:
    archive_path = Path(archive_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Atoms] = []
    with TemporaryDirectory() as tmpdir:
        extract_dir = Path(tmpdir)
        with ZipFile(archive_path) as zip_file:
            zip_file.extractall(extract_dir)
        files = [path for path in sorted(extract_dir.rglob("*")) if path.is_file()]
        runner_files = [path for path in files if path.name == "input.data"]
        structure_files = runner_files if runner_files else files
        for path in structure_files:
            frames.extend(_read_structure_file(path))

    if not frames:
        raise ValueError(f"no supported structure files found in {archive_path}")

    write(output_path, frames, format="extxyz")
    return len(frames)
