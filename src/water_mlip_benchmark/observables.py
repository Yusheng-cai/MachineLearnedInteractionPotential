from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class EnergyDrift:
    slope_eV_per_ps: float
    total_drift_eV: float
    intercept_eV: float


@dataclass(frozen=True)
class RDFResult:
    r_A: np.ndarray
    g_r: np.ndarray
    counts: np.ndarray


def compute_energy_drift(times_fs: np.ndarray, energies_eV: np.ndarray) -> EnergyDrift:
    times_fs = np.asarray(times_fs, dtype=float)
    energies_eV = np.asarray(energies_eV, dtype=float)
    if times_fs.shape != energies_eV.shape:
        raise ValueError("times and energies must have matching shapes")
    if times_fs.size < 2:
        raise ValueError("at least two time points are required")

    times_ps = times_fs / 1000.0
    slope, intercept = np.polyfit(times_ps, energies_eV, deg=1)
    return EnergyDrift(
        slope_eV_per_ps=float(slope),
        total_drift_eV=float(energies_eV[-1] - energies_eV[0]),
        intercept_eV=float(intercept),
    )


def compute_msd(positions_A: np.ndarray, timestep_fs: float) -> tuple[np.ndarray, np.ndarray]:
    positions_A = np.asarray(positions_A, dtype=float)
    if positions_A.ndim != 3 or positions_A.shape[-1] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3)")

    displacements = positions_A - positions_A[0:1, :, :]
    squared = np.sum(displacements**2, axis=-1)
    msd = np.mean(squared, axis=1)
    times_fs = np.arange(positions_A.shape[0], dtype=float) * float(timestep_fs)
    return times_fs, msd


def compute_rdf(frames: list[Atoms], pair: tuple[str, str], r_max_A: float, bin_width_A: float) -> RDFResult:
    if not frames:
        raise ValueError("at least one frame is required")
    edges = np.arange(0.0, r_max_A + bin_width_A, bin_width_A)
    counts = np.zeros(len(edges) - 1, dtype=float)
    pair_a, pair_b = pair

    for atoms in frames:
        symbols = atoms.get_chemical_symbols()
        indices_a = [idx for idx, symbol in enumerate(symbols) if symbol == pair_a]
        indices_b = [idx for idx, symbol in enumerate(symbols) if symbol == pair_b]
        for atom_i in indices_a:
            for atom_j in indices_b:
                if pair_a == pair_b and atom_j <= atom_i:
                    continue
                distance = atoms.get_distance(atom_i, atom_j, mic=True)
                if 0.0 < distance < r_max_A:
                    bin_index = int(distance / bin_width_A)
                    counts[bin_index] += 1.0

    r = 0.5 * (edges[:-1] + edges[1:])
    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = float(np.mean([atoms.get_volume() for atoms in frames]))
    number_density = len(frames[0]) / volume
    normalization = len(frames) * max(len(frames[0]), 1) * number_density * shell_volumes
    g_r = np.divide(counts, normalization, out=np.zeros_like(counts), where=normalization > 0.0)
    return RDFResult(r_A=r, g_r=g_r, counts=counts)
