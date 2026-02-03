from gpuma.structure import Structure


def test_structure_initialization(sample_structure):
    s = sample_structure
    assert s.symbols == ["C", "H", "H", "H", "H"]
    assert len(s.coordinates) == 5
    assert s.charge == 0
    assert s.multiplicity == 1
    assert s.comment == "Methane"
    assert s.energy is None
    assert s.metadata == {}

def test_structure_n_atoms(sample_structure):
    assert sample_structure.n_atoms == 5

def test_structure_with_energy(sample_structure):
    s = sample_structure
    assert s.energy is None
    s.with_energy(-123.456)
    assert s.energy == -123.456

    # Check chaining
    s2 = s.with_energy(-100.0)
    assert s2 is s
    assert s.energy == -100.0

def test_structure_metadata():
    s = Structure(
        symbols=["H"],
        coordinates=[(0.0, 0.0, 0.0)],
        charge=0,
        multiplicity=2,
        metadata={"source": "test"}
    )
    assert s.metadata["source"] == "test"
