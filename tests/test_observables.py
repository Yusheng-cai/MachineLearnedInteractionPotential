import numpy as np
from ase import Atoms

from water_mlip_benchmark.observables import compute_energy_drift, compute_msd, compute_rdf


def test_compute_energy_drift() -> None:
    times_fs = np.array([0.0, 1.0, 2.0, 3.0])
    energies_eV = np.array([10.0, 10.1, 10.2, 10.3])

    drift = compute_energy_drift(times_fs, energies_eV)

    assert round(drift.slope_eV_per_ps, 6) == 100.0
    assert drift.total_drift_eV == 0.3000000000000007


def test_compute_msd_linear_motion() -> None:
    positions = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ]
    )

    times, msd = compute_msd(positions, timestep_fs=2.0)

    np.testing.assert_allclose(times, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(msd, [0.0, 1.0, 4.0])


def test_compute_rdf_has_expected_bins() -> None:
    atoms = Atoms(
        symbols=["O", "O"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    rdf = compute_rdf([atoms], pair=("O", "O"), r_max_A=2.0, bin_width_A=0.5)

    assert rdf.r_A.shape == rdf.g_r.shape
    assert rdf.counts.sum() == 1
