import logging
from unittest.mock import MagicMock, patch

import pytest

from gpuma.config import Config
from gpuma.optimizer import optimize_single_structure, optimize_structure_batch
from gpuma.structure import Structure


def test_optimize_single_structure(sample_structure):
    config = Config()
    # The autouse fixture mock_load_models will ensure we get a mock calculator

    optimized = optimize_single_structure(sample_structure, config)

    # Check if energy was updated (mock returns -50.0)
    assert optimized.energy == -50.0
    # Check coordinates are updated (mock returns all zeros)
    # The result is typically a list of lists or list of tuples depending on ASE
    # optimize_single_structure converts to list
    coords = optimized.coordinates[0]
    assert list(coords) == [0.0, 0.0, 0.0]

def test_optimize_sequential(sample_structure):
    config = Config({"optimization": {"batch_optimization_mode": "sequential"}})

    structures = [sample_structure, sample_structure]
    results = optimize_structure_batch(structures, config)

    assert len(results) == 2
    assert results[0].energy == -50.0
    assert results[1].energy == -50.0

def test_optimize_batch_gpu_fallback(sample_structure):
    # If we request batch mode but have no GPU, it might fall back or raise error
    # depending on implementation.
    # The code says: if mode == "batch" and not force_cpu: return _optimize_batch
    # if mode == "sequential" or force_cpu: ...

    # Let's mock device to be cpu
    config = Config({
        "optimization": {"batch_optimization_mode": "batch"},
        "technical": {"device": "cpu"},
    })

    # Should fallback to sequential
    with patch("gpuma.optimizer._optimize_sequential") as mock_seq:
        optimize_structure_batch([sample_structure], config)
        mock_seq.assert_called()

def test_optimize_batch_call(sample_structure):
    # Test that batch mode calls _optimize_batch when device is cuda
    config = Config({
        "optimization": {"batch_optimization_mode": "batch"},
        "technical": {"device": "cuda"},
    })

    with patch("gpuma.optimizer._parse_device_string", return_value="cuda"), \
         patch("gpuma.optimizer._optimize_batch") as mock_batch:

        optimize_structure_batch([sample_structure], config)
        mock_batch.assert_called()

def test_optimize_batch_implementation(sample_structure):
    # Test the actual implementation of _optimize_batch with mocked torchsim
    config = Config({
        "optimization": {"batch_optimization_mode": "batch"},
        "technical": {"device": "cuda"},
    })

    with patch("gpuma.optimizer._parse_device_string", return_value="cuda"), \
         patch("torch.cuda.is_available", return_value=True):

        # We need to ensure _optimize_batch runs
        # It uses torch_sim.io.atoms_to_state and torch_sim.optimize
        # We need to mock those too because we don't want to run real
        # torchsim optimization logic potentially

        with patch("torch_sim.io.atoms_to_state") as mock_ats, \
             patch("torch_sim.optimize") as mock_optimize, \
             patch("torch_sim.autobatching.InFlightAutoBatcher") as _:

             mock_state = MagicMock()
             mock_state.n_atoms = 5
             mock_ats.return_value = mock_state

             mock_final_state = MagicMock()
             # Mock final state energy/charge/spin/positions
             mock_final_state.energy = [MagicMock(item=lambda: -60.0)]
             mock_final_state.charge = [MagicMock(item=lambda: 0)]
             mock_final_state.spin = [MagicMock(item=lambda: 1)]

             mock_atoms = MagicMock()
             mock_atoms.get_chemical_symbols.return_value = ["C", "H", "H", "H", "H"]

             # get_positions returns a numpy array usually, which has tolist()
             # If we return a list, it fails .tolist() call in optimizer.py
             mock_pos = MagicMock()
             mock_pos.tolist.return_value = [[0.0, 0.0, 0.0]] * 5
             mock_atoms.get_positions.return_value = mock_pos

             mock_final_state.to_atoms.return_value = [mock_atoms]

             mock_optimize.return_value = mock_final_state

             results = optimize_structure_batch([sample_structure], config)

             assert len(results) == 1
             assert results[0].energy == -60.0

def test_optimize_single_structure_convergence_warnings(sample_structure, caplog):
    # Case 1: Both force and energy
    config = Config({"optimization": {
        "force_convergence_criterion": 0.01,
        "energy_convergence_criterion": 0.001
    }})
    optimize_single_structure(sample_structure, config)
    assert "Both force and energy convergence criteria given" in caplog.text

    caplog.clear()

    # Case 2: Only energy
    config = Config({"optimization": {
        "force_convergence_criterion": None,
        "energy_convergence_criterion": 0.001
    }})
    optimize_single_structure(sample_structure, config)
    assert "Energy convergence criterion requested but only force" in caplog.text


# --- ORB model_type routing tests ---

def test_optimize_single_structure_orb(sample_structure):
    """ORB model_type should work the same as fairchem for single structure."""
    config = Config({"model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"}})
    optimized = optimize_single_structure(sample_structure, config)
    assert optimized.energy == -50.0


def test_optimize_sequential_orb(sample_structure):
    config = Config({
        "optimization": {"batch_optimization_mode": "sequential"},
        "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
    })
    results = optimize_structure_batch([sample_structure], config)
    assert len(results) == 1
    assert results[0].energy == -50.0


def test_optimize_batch_orb_calls_batch(sample_structure):
    """Batch mode with ORB should route to _optimize_batch."""
    config = Config({
        "optimization": {"batch_optimization_mode": "batch"},
        "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
        "technical": {"device": "cuda"},
    })

    with patch("gpuma.optimizer._parse_device_string", return_value="cuda"), \
         patch("gpuma.optimizer._optimize_batch") as mock_batch:
        optimize_structure_batch([sample_structure], config)
        mock_batch.assert_called()


def test_optimization_summary_logged(sample_structure, caplog):
    """Verify the optimization summary is logged after a batch run."""
    config = Config({"optimization": {"batch_optimization_mode": "sequential"}})

    with caplog.at_level(logging.INFO, logger="gpuma.logging_utils"):
        optimize_structure_batch([sample_structure], config)

    assert "GPUMA Optimization Summary" in caplog.text
    assert "Structures input:    1" in caplog.text
    assert "Structures output:   1" in caplog.text
    assert "Success rate:" in caplog.text
    assert "Total time:" in caplog.text
    assert "Energy min:" in caplog.text


def test_optimize_batch_empty_list():
    """Empty input returns empty output without error."""
    config = Config()
    assert optimize_structure_batch([], config) == []


def test_optimize_batch_mismatched_coords():
    """Mismatched symbols/coordinates raises ValueError."""
    bad = Structure(
        symbols=["C", "H"],
        coordinates=[(0, 0, 0)],  # 1 coord for 2 symbols
        charge=0,
        multiplicity=1,
    )
    with pytest.raises(ValueError, match="symbols/coords length mismatch"):
        optimize_structure_batch([bad])


def test_optimize_batch_empty_structure():
    """Empty structure (0 atoms) raises ValueError."""
    empty = Structure(symbols=[], coordinates=[], charge=0, multiplicity=1)
    with pytest.raises(ValueError, match="empty structure"):
        optimize_structure_batch([empty])
