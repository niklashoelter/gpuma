import os
import sys
from unittest.mock import MagicMock


# Helper to mock a module structure if it's missing
def mock_module(name):
    if name in sys.modules:
        return sys.modules[name]

    try:
        # Check if we can import it
        __import__(name)
        return sys.modules[name]
    except ImportError:
        pass

    m = MagicMock()
    sys.modules[name] = m
    return m


# Mock missing dependencies to allow running tests in environments without heavy libs
mock_module("torch")
mock_module("torch_sim")
mock_module("torch_sim.models")
mock_module("torch_sim.models.fairchem")

ase = mock_module("ase")
ase_data = mock_module("ase.data")
ase_opt = mock_module("ase.optimize")

if isinstance(ase, MagicMock):
    ase.data = ase_data
    ase.optimize = ase_opt

fairchem = mock_module("fairchem")
fairchem_core = mock_module("fairchem.core")

if isinstance(fairchem, MagicMock):
    fairchem.core = fairchem_core

morfeus = mock_module("morfeus")
morfeus_conformer = mock_module("morfeus.conformer")

if isinstance(morfeus, MagicMock):
    morfeus.conformer = morfeus_conformer

rdkit = mock_module("rdkit")
rdkit_chem = mock_module("rdkit.Chem")

if isinstance(rdkit, MagicMock):
    rdkit.Chem = rdkit_chem

mock_module("tables")
mock_module("hf_xet")

# Setup reasonable defaults for mocks if they are indeed mocks
if isinstance(ase_data, MagicMock):
    # Minimal chemical symbols list
    # 0 is 'X', 1 is 'H', 6 is 'C', 7 is 'N', 8 is 'O'
    # We populate a list of strings for 118 elements
    symbols = ["X"] * 119
    symbols[1] = "H"
    symbols[6] = "C"
    symbols[7] = "N"
    symbols[8] = "O"
    ase_data.chemical_symbols = symbols

if isinstance(morfeus_conformer, MagicMock):
    # Setup ConformerEnsemble to behave like a list
    class MockConformer:
        def __init__(self, elements, coordinates):
            self.elements = elements
            self.coordinates = coordinates

    class MockEnsemble(list):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.multiplicity = 1

        @classmethod
        def from_rdkit(cls, mol):
            e = cls()
            # Check if mol has specific smiles attached (from our rdkit mock)
            smiles = getattr(mol, "_smiles", "")

            if smiles == "O":
                # O -> H-O-H approx
                e.append(
                    MockConformer(
                        elements=[8, 1, 1],
                        coordinates=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.2, 0.9, 0.0]],
                    )
                )
            elif smiles == "[NH4+]":
                # NH4+ -> N + 4H
                e.append(
                    MockConformer(
                        elements=[7, 1, 1, 1, 1],
                        coordinates=[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
                    )
                )
            else:
                # Default dummy
                e.append(MockConformer(elements=[6], coordinates=[[0.0, 0.0, 0.0]]))

            return e

        def prune_rmsd(self, *args, **kwargs):
            pass

        def sort(self, *args, **kwargs):
            pass

    morfeus_conformer.ConformerEnsemble = MockEnsemble

if isinstance(rdkit_chem, MagicMock):

    def mock_mol_from_smiles(smiles):
        if not smiles:
            return None
        m = MagicMock()
        m._smiles = smiles
        return m

    rdkit_chem.MolFromSmiles.side_effect = mock_mol_from_smiles

    def mock_get_formal_charge(mol):
        if getattr(mol, "_smiles", "") == "[NH4+]":
            return 1
        return 0

    rdkit_chem.GetFormalCharge.side_effect = mock_get_formal_charge

    rdkit_chem.AddHs.side_effect = lambda m: m


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
