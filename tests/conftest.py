from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from gpuma.structure import Structure


@pytest.fixture
def mock_hf_token(monkeypatch):
    """Ensure HF_TOKEN is set for tests that check for it, or unset it."""
    # By default, we might want to unset it to ensure our mocking works even without it
    # But some code paths check for it.
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

@pytest.fixture
def mock_fairchem_calculator():
    """Returns a mock Fairchem calculator."""
    mock_calc = MagicMock()
    # Mocking what ASE calculator expects
    mock_calc.get_potential_energy.return_value = -100.0
    mock_calc.get_forces.return_value = np.zeros((5, 3)) # 5 atoms, 3 coords

    # We also need to mock the internal implementation of FAIRChemCalculator if needed
    # but since we mock the class or the object returned by load_model_fairchem,
    # the ASE interface methods are what matters for optimize_single_structure (BFGS)

    # ASE optimizer calls get_potential_energy and get_forces on the Atoms object,
    # which delegates to calc.
    def calculate(atoms, properties, system_changes):
        # update results
        mock_calc.results = {
            'energy': -100.0,
            'forces': np.zeros((len(atoms), 3))
        }

    mock_calc.calculate = calculate
    return mock_calc

@pytest.fixture
def mock_torchsim_model():
    """Returns a mock TorchSim model."""
    mock_model = MagicMock()
    mock_model.model_name = "mock-uma"

    # TorchSim model is called with a batched state
    # and returns energy, forces, etc.
    # However, torch_sim.optimize calls model(system) -> output

    def forward(system):
        n_systems = system.n_systems
        n_atoms = system.n_atoms
        device = system.positions.device

        # Mock output
        # energy: (n_systems,)
        # forces: (n_atoms, 3)
        return MagicMock(
            energy=torch.zeros(n_systems, device=device),
            forces=torch.zeros((n_atoms, 3), device=device)
        )

    mock_model.side_effect = forward
    return mock_model

@pytest.fixture(autouse=True)
def mock_load_models(request):
    """Automatically mock model loading functions to prevent network access."""
    # Check if the test is marked to use real models (optional, for future)
    if "real_model" in request.keywords:
        return

    # Mock fairchem loading
    # We mock _get_cached_calculator and load_model_fairchem

    with patch("gpuma.optimizer.load_model_fairchem") as mock_load_fc, \
         patch("gpuma.optimizer._get_cached_calculator") as mock_get_cached_fc, \
         patch("gpuma.optimizer.load_model_torchsim") as mock_load_ts, \
         patch("gpuma.optimizer._get_cached_torchsim_model") as mock_get_cached_ts:

        # Setup mocks
        mock_calc = MagicMock()
        # Setup ASE calculator mock behavior
        mock_calc.get_potential_energy.return_value = -50.0
        mock_calc.get_forces.return_value = np.zeros((5, 3))
        # Ensure it works when assigned to atoms.calc
        def side_effect_calc(atoms=None, **kwargs):
             pass
        mock_calc.calculate = MagicMock(side_effect=side_effect_calc)
        mock_calc.results = {'energy': -50.0, 'forces': np.zeros((1, 3))} # Default

        # Better mock for ASE calculator
        class MockCalculator:
            def __init__(self):
                self.results = {}
                self.pars = {}
                self.atoms = None
            def calculate(self, atoms=None, properties=None, system_changes=None):
                if properties is None:
                    properties = ['energy']
                self.results['energy'] = -50.0
                self.results['forces'] = np.zeros((len(atoms), 3))

            def get_potential_energy(self, atoms=None, force_consistent=False):
                if atoms:
                    self.calculate(atoms)
                return self.results['energy']

            def get_forces(self, atoms=None):
                if atoms:
                    self.calculate(atoms)
                return self.results['forces']
            def reset(self):
                pass

        mock_instance = MockCalculator()
        mock_load_fc.return_value = mock_instance
        mock_get_cached_fc.return_value = mock_instance

        # Setup TorchSim model mock
        mock_ts_model = MagicMock()
        mock_ts_model.model_name = "mock-uma"

        def ts_forward(system):
            n_systems = system.n_systems
            n_atoms = system.n_atoms
            # Create tensors on the same device as system
            # Ensure we return objects that behave like tensors
            energy = torch.zeros(n_systems).to(system.positions.device)
            forces = torch.zeros((n_atoms, 3)).to(system.positions.device)

            output = MagicMock()
            output.energy = energy
            output.forces = forces
            return output

        mock_ts_model.side_effect = ts_forward
        mock_load_ts.return_value = mock_ts_model
        mock_get_cached_ts.return_value = mock_ts_model

        yield
