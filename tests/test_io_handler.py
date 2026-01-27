import pytest
from gpuma.io_handler import read_multi_xyz, read_xyz, read_xyz_directory, save_multi_xyz, save_xyz_file
from gpuma.structure import Structure


def test_read_and_save_single_xyz_roundtrip(tmp_path):
    xyz = """3
Water | Energy: -1.000 eV | Charge: 0 | Multiplicity: 1
O 0.000000 0.000000 0.000000
H 0.000000 0.000000 1.000000
H 0.000000 1.000000 0.000000
"""
    src = tmp_path / "w.xyz"
    src.write_text(xyz)

    s = read_xyz(str(src), charge=-1, multiplicity=2)
    assert isinstance(s, Structure)
    assert s.n_atoms == 3
    # Energies should not be parsed from comments when reading
    assert s.energy is None
    assert s.charge == -1 and s.multiplicity == 2

    # If we set an energy and save, it should be written
    s.energy = -1.0
    out = tmp_path / "out.xyz"
    save_xyz_file(s, str(out))
    assert out.exists()
    data = out.read_text()
    assert "Energy: -1.000000" in data
    assert "Charge: -1" in data and "Multiplicity: 2" in data


def test_read_multi_xyz(tmp_path):
    content = """2
Mol A | Energy: -2.0 eV | Charge: 0 | Multiplicity: 1
H 0 0 0
H 0 0 1

2
Mol B | Charge: 0 | Multiplicity: 1
He 0 0 0
He 0 0 1
"""
    src = tmp_path / "m.xyz"
    src.write_text(content)
    items = read_multi_xyz(str(src), charge=0, multiplicity=1)
    assert len(items) == 2
    # Energies should not be parsed when reading multi-XYZ either
    assert items[0].energy is None
    assert items[1].energy is None
    assert all(s.charge == 0 and s.multiplicity == 1 for s in items)


def test_save_multi_xyz(tmp_path):
    structs = [
        Structure(["H"], [(0.0, 0.0, 0.0)], charge=0, multiplicity=1, energy=-1.0),
        Structure(["He"], [(1.0, 0.0, 0.0)], charge=0, multiplicity=1, energy=None),
    ]
    out = tmp_path / "multi.xyz"
    save_multi_xyz(structs, str(out), comments=["A", "B"])
    data = out.read_text()
    assert "A | Energy: -1.000000" in data
    assert "Charge: 0" in data and "Multiplicity: 1" in data
    assert "B | Charge: 0 | Multiplicity: 1" in data


def test_read_xyz_directory(tmp_path):
    # 1. Test empty directory raises ValueError
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No XYZ files found"):
        read_xyz_directory(str(empty_dir))

    # 2. Test directory with valid files
    xyz_dir = tmp_path / "xyzs"
    xyz_dir.mkdir()
    (xyz_dir / "1.xyz").write_text("1\nH\nH 0 0 0")
    (xyz_dir / "2.xyz").write_text("1\nHe\nHe 1 0 0")

    structs = read_xyz_directory(str(xyz_dir), charge=1, multiplicity=2)
    assert len(structs) == 2
    # Verify properties passed down
    assert all(s.charge == 1 and s.multiplicity == 2 for s in structs)
    symbols = sorted([s.symbols[0] for s in structs])
    assert symbols == ["H", "He"]

    # 3. Test directory with only invalid files raises ValueError
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "bad.xyz").write_text("Not an XYZ file")

    with pytest.raises(ValueError, match="No valid structures could be read"):
        read_xyz_directory(str(bad_dir))

    # 4. Test non-existent directory raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        read_xyz_directory(str(tmp_path / "non_existent"))
