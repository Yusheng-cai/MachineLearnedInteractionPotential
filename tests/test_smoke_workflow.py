from pathlib import Path
from zipfile import ZipFile

from ase.io import read

from water_mlip_benchmark.convert import convert_archive_to_extxyz
from water_mlip_benchmark.metrics import summarize_errors
from water_mlip_benchmark.observables import compute_rdf


RUNNER_TEXT = """begin
lattice 10.0 0.0 0.0
lattice 0.0 10.0 0.0
lattice 0.0 0.0 10.0
atom 0.0 0.0 0.0 O 0.0 0.0 0.0 0.0 0.0
atom 0.8 0.0 0.0 H 0.0 0.0 0.0 0.0 0.0
atom 0.0 0.8 0.0 H 0.0 0.0 0.0 0.0 0.0
energy -76.0
charge 0.0
end
begin
lattice 10.0 0.0 0.0
lattice 0.0 10.0 0.0
lattice 0.0 0.0 10.0
atom 0.1 0.0 0.0 O 0.0 0.0 0.1 0.0 0.0
atom 0.9 0.0 0.0 H 0.0 0.0 -0.1 0.0 0.0
atom 0.1 0.8 0.0 H 0.0 0.0 0.0 0.0 0.0
energy -75.9
charge 0.0
end
"""


def test_synthetic_end_to_end_workflow(tmp_path: Path) -> None:
    archive = tmp_path / "training-set.zip"
    extxyz = tmp_path / "converted.extxyz"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("input.data", RUNNER_TEXT)

    count = convert_archive_to_extxyz(archive, extxyz)
    frames = read(extxyz, ":")
    energies = [atoms.info["REF_energy"] for atoms in frames]
    forces = [atoms.arrays["REF_forces"] for atoms in frames]

    summary = summarize_errors(energies, energies, forces, forces)
    rdf = compute_rdf(frames, pair=("O", "H"), r_max_A=2.0, bin_width_A=0.1)

    assert count == 2
    assert summary.energy_mae == 0.0
    assert rdf.counts.sum() > 0.0
