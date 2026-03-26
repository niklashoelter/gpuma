"""Tests for the Structure dataclass."""

from gpuma.structure import Structure


def test_structure_initialization(methane):
    """Structure stores symbols, coordinates, charge, multiplicity, and comment."""
    s = methane
    assert s.symbols == ["C", "H", "H", "H", "H"]
    assert len(s.coordinates) == 5
    assert s.charge == 0
    assert s.multiplicity == 1
    assert s.comment == "Methane"
    assert s.energy is None
    assert s.metadata == {}


def test_structure_n_atoms(methane):
    """n_atoms property returns the number of symbols."""
    assert methane.n_atoms == 5


def test_structure_with_energy(methane):
    """with_energy() sets energy and supports chaining."""
    assert methane.energy is None
    methane.with_energy(-123.456)
    assert methane.energy == -123.456

    s2 = methane.with_energy(-100.0)
    assert s2 is methane
    assert methane.energy == -100.0


def test_structure_metadata():
    """Structure accepts and stores arbitrary metadata."""
    s = Structure(
        symbols=["H"],
        coordinates=[(0.0, 0.0, 0.0)],
        charge=0,
        multiplicity=2,
        metadata={"source": "test"},
    )
    assert s.metadata["source"] == "test"
