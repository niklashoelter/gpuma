import numpy as np
import pytest

from gpuma.mol_utils import (
    _to_coord_list,
    _to_symbol_list,
    smiles_to_conformer_ensemble,
    smiles_to_structure,
)
from gpuma.structure import Structure


def test_to_symbol_list_with_strings_and_atomic_numbers():
    elems = ["H", "C", 8]
    out = _to_symbol_list(elems)
    assert out == ["H", "C", "O"]


def test_to_symbol_list_with_numpy_array():
    arr = np.array([1, 6])
    out = _to_symbol_list(arr)
    assert out == ["H", "C"]


def test_to_coord_list_with_numpy_array():
    coords = np.array([[0, 1, 2], [3.5, 4.5, 5.5]])
    out = _to_coord_list(coords)
    assert isinstance(out, list) and all(isinstance(t, tuple) and len(t) == 3 for t in out)
    assert out[0] == (0.0, 1.0, 2.0)


def test_smiles_to_conformer_ensemble_invalid_inputs():
    with pytest.raises(ValueError):
        smiles_to_conformer_ensemble("", max_num_confs=1)
    with pytest.raises(ValueError):
        smiles_to_conformer_ensemble("H", max_num_confs=0)


def test_smiles_generation_if_available():
    pytest.importorskip("morfeus.conformer")
    res = smiles_to_conformer_ensemble("O", max_num_confs=1)
    assert isinstance(res, list) and res and isinstance(res[0], Structure)


def test_smiles_to_structure_if_available():
    pytest.importorskip("morfeus.conformer")
    s = smiles_to_structure("O")
    assert isinstance(s, Structure)


def test_smiles_charge_is_inferred_and_multiplicity_default_if_available():
    pytest.importorskip("morfeus.conformer")
    s = smiles_to_structure("[NH4+]")
    # formal charge +1 from SMILES, multiplicity default 1
    assert s.charge == 1
    assert s.multiplicity == 1
