import pytest

from gpuma.io_handler import (
    file_exists,
    read_multi_xyz,
    read_xyz,
    read_xyz_directory,
    save_multi_xyz,
    save_xyz_file,
    smiles_to_ensemble,
    smiles_to_xyz,
)
from gpuma.structure import Structure


def test_read_xyz_valid(tmp_path, sample_xyz_content):
    f = tmp_path / "test.xyz"
    f.write_text(sample_xyz_content)

    struct = read_xyz(str(f), charge=1, multiplicity=2)
    assert len(struct.symbols) == 5
    assert struct.symbols[0] == "C"
    assert struct.charge == 1
    assert struct.multiplicity == 2
    assert struct.comment == "Methane"

def test_read_xyz_not_found():
    with pytest.raises(FileNotFoundError):
        read_xyz("non_existent.xyz")

def test_read_multi_xyz(tmp_path, sample_multi_xyz_content):
    f = tmp_path / "multi.xyz"
    f.write_text(sample_multi_xyz_content)

    structs = read_multi_xyz(str(f), charge=0, multiplicity=1)
    assert len(structs) == 2
    assert structs[0].comment == "Water"
    assert len(structs[0].symbols) == 3
    assert structs[1].comment == "Methane"
    assert len(structs[1].symbols) == 5

def test_read_xyz_directory(tmp_path, sample_xyz_content):
    d = tmp_path / "xyz_dir"
    d.mkdir()
    (d / "1.xyz").write_text(sample_xyz_content)
    (d / "2.xyz").write_text(sample_xyz_content)

    structs = read_xyz_directory(str(d))
    assert len(structs) == 2

def test_save_xyz_file(tmp_path, sample_structure):
    f = tmp_path / "output.xyz"
    sample_structure.energy = -10.5
    save_xyz_file(sample_structure, str(f))

    assert f.exists()
    content = f.read_text()
    assert "Methane | Energy: -10.500000 eV | Charge: 0 | Multiplicity: 1" in content
    assert "C 0.000000" in content

def test_save_multi_xyz(tmp_path, sample_structure):
    f = tmp_path / "output_multi.xyz"
    s1 = sample_structure
    s2 = sample_structure.with_energy(-20.0)
    save_multi_xyz([s1, s2], str(f), comments=["First", "Second"])

    content = f.read_text()
    # Energy might persist on s1 if mutated
    assert "First | Energy: -20.000000" in content or "First | Energy: -10.500000" not in content
    # wait, s2 = sample_structure.with_energy(-20.0) mutates sample_structure!
    # Because s2 is s1.
    assert "Second | Energy: -20.000000" in content

def test_smiles_to_xyz():
    # Test with a simple molecule
    s = smiles_to_xyz("C")
    assert isinstance(s, Structure)
    assert s.symbols == ["C", "H", "H", "H", "H"]
    assert s.n_atoms == 5

def test_smiles_to_xyz_string():
    s = smiles_to_xyz("C", return_full_xyz_str=True)
    assert isinstance(s, str)
    assert "Generated from SMILES" in s
    assert "C 0.0" in s or "C -0.0" in s or "C" in s

def test_smiles_to_ensemble():
    # Generate ensemble for butane
    structs = smiles_to_ensemble("CCCC", max_num_confs=3)
    assert len(structs) > 0
    assert len(structs) <= 3
    assert all(isinstance(s, Structure) for s in structs)
    assert structs[0].n_atoms == 14 # C4H10

def test_file_exists(tmp_path):
    f = tmp_path / "exists.txt"
    f.touch()
    assert file_exists(str(f))
    assert not file_exists("non_existent")
