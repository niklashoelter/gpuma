import sys
from unittest.mock import MagicMock, patch, Mock
import numpy as np
import pytest

# --- Mocking Heavy Dependencies ---

def _mock_module(name):
    """Mocks a module and registers it in sys.modules."""
    parts = name.split('.')
    for i in range(1, len(parts) + 1):
        subname = ".".join(parts[:i])
        if subname not in sys.modules:
            sys.modules[subname] = MagicMock()
            if i > 1:
                parent_name = ".".join(parts[:i-1])
                child_name = parts[i-1]
                setattr(sys.modules[parent_name], child_name, sys.modules[subname])
    return sys.modules[name]

# Mock Torch
try:
    import torch
except ImportError:
    if "torch" not in sys.modules:
        torch_mock = MagicMock()

        class MockDevice:
            def __init__(self, arg):
                if isinstance(arg, str):
                    if ":" in arg:
                        self.type, idx = arg.split(":")
                        self.index = int(idx)
                    else:
                        self.type = arg
                        self.index = None
                else:
                    self.type = "cpu"
                    self.index = None

            def __eq__(self, other):
                if isinstance(other, MockDevice):
                    return self.type == other.type and self.index == other.index
                return False

            def __repr__(self):
                return f"device(type='{self.type}', index={self.index})"

        torch_mock.device = MockDevice

        torch_mock.cuda = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        torch_mock.float64 = "float64"

        def mock_tensor_factory(*args, **kwargs):
            m = MagicMock()
            m.to.return_value = m
            m.item.return_value = 0.0
            return m
        torch_mock.zeros = MagicMock(side_effect=mock_tensor_factory)
        torch_mock.tensor = MagicMock(side_effect=mock_tensor_factory)
        sys.modules["torch"] = torch_mock

# Mock ASE
try:
    import ase
except ImportError:
    ase = _mock_module("ase")

    def mock_atoms_factory(*args, **kwargs):
        atoms = MagicMock()
        atoms.calc = None

        positions_arg = kwargs.get('positions')
        symbols_arg = kwargs.get('symbols')

        n_atoms = 5
        if positions_arg is not None:
            n_atoms = len(positions_arg)
        elif symbols_arg is not None:
            n_atoms = len(symbols_arg)

        def get_potential_energy(*args, **kwargs):
            if atoms.calc:
                return atoms.calc.get_potential_energy(atoms)
            return 0.0

        def get_forces(*args, **kwargs):
            if atoms.calc:
                return atoms.calc.get_forces(atoms)
            return np.zeros((len(atoms), 3))

        atoms.get_potential_energy.side_effect = get_potential_energy
        atoms.get_forces.side_effect = get_forces

        def get_positions():
            return np.zeros((n_atoms, 3))

        atoms.get_positions.side_effect = get_positions
        atoms.get_chemical_symbols.return_value = ["X"] * n_atoms
        atoms.__len__.return_value = n_atoms

        return atoms

    ase.Atoms = MagicMock(side_effect=mock_atoms_factory)
    ase_optimize = _mock_module("ase.optimize")
    ase_optimize.BFGS = MagicMock()

    ase_data = _mock_module("ase.data")
    dummy_symbols = ["X"] + ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"] * 20
    ase_data.chemical_symbols = dummy_symbols
    _mock_module("ase.io")

# Mock Fairchem
try:
    import fairchem.core
except ImportError:
    fairchem_core = _mock_module("fairchem.core")
    fairchem_core.FAIRChemCalculator = MagicMock()
    fairchem_core.pretrained_mlip = MagicMock()

# Mock RDKit
try:
    from rdkit import Chem
except ImportError:
    rdkit_chem = _mock_module("rdkit.Chem")

    def mock_mol_from_smiles(smiles):
        m = MagicMock()
        m._smiles = smiles
        return m

    rdkit_chem.MolFromSmiles = MagicMock(side_effect=mock_mol_from_smiles)
    rdkit_chem.AddHs = MagicMock(side_effect=lambda m: m)
    rdkit_chem.GetFormalCharge = MagicMock(return_value=0)

# Mock Morfeus
try:
    from morfeus import conformer
except ImportError:
    morfeus_conformer = _mock_module("morfeus.conformer")

    class MockConformerEnsemble:
        def __init__(self, *args, **kwargs):
            self._conformers = []
            self.multiplicity = 1

        def __iter__(self):
            return iter(self._conformers)

        def prune_rmsd(self):
            pass

        def sort(self):
            pass

        @classmethod
        def from_rdkit(cls, mol):
            ens = cls()
            class MockConformer:
                def __init__(self, smiles):
                    if smiles == "CCCC":
                        # Butane: 4 C, 10 H
                        self.elements = ["C"] * 4 + ["H"] * 10
                        self.coordinates = [[0.0, 0.0, 0.0]] * 14
                    else:
                        self.elements = ["C", "H", "H", "H", "H"]
                        self.coordinates = [[0.0,0.0,0.0], [1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0], [0.0,0.0,-1.0]]

            smiles = getattr(mol, "_smiles", "")
            conf = MockConformer(smiles)
            ens._conformers = [conf]
            return ens

    morfeus_conformer.ConformerEnsemble = MockConformerEnsemble

# Mock TorchSim
try:
    import torch_sim
except ImportError:
    _mock_module("torch_sim.models.fairchem")
    _mock_module("torch_sim.autobatching")
    _mock_module("torch_sim.io")

    torch_sim = sys.modules["torch_sim"]
    torch_sim.models.fairchem.FairChemModel = MagicMock()
    torch_sim.autobatching.InFlightAutoBatcher = MagicMock()
    torch_sim.io.atoms_to_state = MagicMock()

# Ensure we don't spec mocks in a way that causes issues
# The error "Cannot spec a Mock object" usually comes from autospec=True
# We are not using it, but maybe pytest or something else is.
# We will ensure our mocks are simple.

# --- End Mocking ---

from gpuma.structure import Structure
from gpuma.config import Config

@pytest.fixture
def mock_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake_token")

@pytest.fixture
def sample_structure():
    return Structure(
        symbols=["C", "H", "H", "H", "H"],
        coordinates=[
            (0.0, 0.0, 0.0),
            (0.63, 0.63, 0.63),
            (-0.63, -0.63, 0.63),
            (-0.63, 0.63, -0.63),
            (0.63, -0.63, -0.63),
        ],
        charge=0,
        multiplicity=1,
        comment="Methane",
    )

@pytest.fixture
def sample_xyz_content():
    return """5
Methane
C 0.000000 0.000000 0.000000
H 0.630000 0.630000 0.630000
H -0.630000 -0.630000 0.630000
H -0.630000 0.630000 -0.630000
H 0.630000 -0.630000 -0.630000
"""

@pytest.fixture
def sample_multi_xyz_content():
    return """3
Water
O 0.000000 0.000000 0.000000
H 0.757000 0.586000 0.000000
H -0.757000 0.586000 0.000000
5
Methane
C 0.000000 0.000000 0.000000
H 0.630000 0.630000 0.630000
H -0.630000 -0.630000 0.630000
H -0.630000 0.630000 -0.630000
H 0.630000 -0.630000 -0.630000
"""

@pytest.fixture(autouse=True)
def mock_load_models(request):
    """Automatically mock model loading functions to prevent network access."""
    if "real_model" in request.keywords:
        return

    with patch("gpuma.optimizer.load_model_fairchem") as mock_load_fc, \
         patch("gpuma.optimizer._get_cached_calculator") as mock_get_cached_fc, \
         patch("gpuma.optimizer.load_model_torchsim") as mock_load_ts, \
         patch("gpuma.optimizer._get_cached_torchsim_model") as mock_get_cached_ts:

        mock_calc = MagicMock()
        mock_calc.results = {}

        def calculate(atoms=None, properties=None, system_changes=None):
            if properties is None:
                properties = ['energy']
            mock_calc.results['energy'] = -50.0
            if atoms:
                mock_calc.results['forces'] = np.zeros((len(atoms), 3))
            else:
                mock_calc.results['forces'] = np.zeros((1, 3))

        mock_calc.calculate = calculate

        def get_potential_energy(atoms=None, force_consistent=False):
            if atoms:
                calculate(atoms)
            return mock_calc.results.get('energy', -50.0)

        def get_forces(atoms=None):
            if atoms:
                calculate(atoms)
            return mock_calc.results.get('forces', np.zeros((len(atoms), 3)))

        mock_calc.get_potential_energy.side_effect = get_potential_energy
        mock_calc.get_forces.side_effect = get_forces

        mock_load_fc.return_value = mock_calc
        mock_get_cached_fc.return_value = mock_calc

        # Setup TorchSim model mock
        mock_ts_model = MagicMock()
        mock_ts_model.model_name = "mock-uma"
        mock_load_ts.return_value = mock_ts_model
        mock_get_cached_ts.return_value = mock_ts_model

        yield
