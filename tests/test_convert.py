from pathlib import Path
from zipfile import ZipFile

import numpy as np
from ase.io import read

from water_mlip_benchmark.convert import convert_archive_to_extxyz, read_runner_input_data


RUNNER_TEXT = """begin
comment synthetic water
lattice 10.0 0.0 0.0
lattice 0.0 10.0 0.0
lattice 0.0 0.0 10.0
atom 0.0 0.0 0.0 O 0.0 0.0 0.1 0.2 0.3
atom 0.8 0.0 0.0 H 0.0 0.0 -0.1 -0.2 -0.3
atom 0.0 0.8 0.0 H 0.0 0.0 0.0 0.1 0.0
energy -76.0
charge 0.0
end
"""


def test_read_runner_input_data() -> None:
    atoms_list = read_runner_input_data(RUNNER_TEXT)

    assert len(atoms_list) == 1
    atoms = atoms_list[0]
    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
    assert atoms.info["REF_energy"] == -76.0
    np.testing.assert_allclose(atoms.arrays["REF_forces"][0], [0.1, 0.2, 0.3])
    assert atoms.cell.lengths()[0] == 10.0


def test_convert_archive_to_extxyz(tmp_path: Path) -> None:
    archive = tmp_path / "training-set.zip"
    output = tmp_path / "converted.extxyz"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("training-set/input.data", RUNNER_TEXT)

    count = convert_archive_to_extxyz(archive, output)

    assert count == 1
    frames = read(output, ":")
    assert len(frames) == 1
    assert frames[0].info["REF_energy"] == -76.0
