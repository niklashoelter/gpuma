
import logging
from unittest.mock import MagicMock, patch, ANY
import pytest
import numpy as np

from gpuma.config import Config, DEFAULT_CONFIG, validate_config
from gpuma.optimizer import optimize_single_structure, _optimize_batch_structures
from gpuma.structure import Structure

def test_default_config_convergence_criteria():
    """Test that default configuration has correct convergence criteria values."""
    assert "force_convergence_criterion" in DEFAULT_CONFIG["optimization"]
    assert "energy_convergence_criterion" in DEFAULT_CONFIG["optimization"]
    assert DEFAULT_CONFIG["optimization"]["force_convergence_criterion"] == 0.05
    assert DEFAULT_CONFIG["optimization"]["energy_convergence_criterion"] is None

def test_validate_config_convergence_criteria():
    """Test validation of convergence criteria."""
    cfg = Config()
    # These should be valid
    cfg.optimization.force_convergence_criterion = 0.01
    cfg.optimization.energy_convergence_criterion = None
    validate_config(cfg)

    cfg.optimization.force_convergence_criterion = None
    cfg.optimization.energy_convergence_criterion = 1e-6
    validate_config(cfg)

    # Invalid values
    cfg.optimization.force_convergence_criterion = -1.0
    with pytest.raises(ValueError):
        validate_config(cfg)

    cfg.optimization.force_convergence_criterion = 0.05
    cfg.optimization.energy_convergence_criterion = -1.0
    with pytest.raises(ValueError):
        validate_config(cfg)

def test_optimize_single_structure_uses_force_criterion():
    """Test that optimize_single_structure uses the configured force criterion."""
    s = Structure(["H"], [(0.0, 0.0, 0.0)], charge=0, multiplicity=1)
    cfg = Config()
    cfg.optimization.force_convergence_criterion = 0.02
    cfg.optimization.energy_convergence_criterion = None

    # Mock Calculator and Atoms
    mock_atoms_instance = MagicMock()
    mock_atoms_instance.get_positions.return_value = np.array([[0.0, 0.0, 0.0]])
    mock_atoms_instance.get_potential_energy.return_value = -1.0

    with (
        patch("gpuma.optimizer.Atoms", return_value=mock_atoms_instance),
        patch("gpuma.optimizer.BFGS") as MockBFGS,
        patch("gpuma.optimizer._get_cached_calculator")
    ):
        mock_bfgs_instance = MockBFGS.return_value

        optimize_single_structure(s, config=cfg)

        # Verify BFGS.run was called with fmax=0.02
        mock_bfgs_instance.run.assert_called_with(fmax=0.02)

def test_optimize_single_structure_warning_for_energy_criterion(caplog):
    """Test that single structure optimization warns if energy criterion is requested."""
    s = Structure(["H"], [(0.0, 0.0, 0.0)], charge=0, multiplicity=1)
    cfg = Config()
    cfg.optimization.force_convergence_criterion = None
    cfg.optimization.energy_convergence_criterion = 1e-5

    mock_atoms_instance = MagicMock()
    mock_atoms_instance.get_positions.return_value = np.array([[0.0, 0.0, 0.0]])
    mock_atoms_instance.get_potential_energy.return_value = -1.0

    with (
        patch("gpuma.optimizer.Atoms", return_value=mock_atoms_instance),
        patch("gpuma.optimizer.BFGS") as MockBFGS,
        patch("gpuma.optimizer._get_cached_calculator")
    ):
        mock_bfgs_instance = MockBFGS.return_value

        with caplog.at_level(logging.WARNING):
            optimize_single_structure(s, config=cfg)

        # Should warn about falling back to force
        assert "only force convergence criterion can be used" in caplog.text.lower()
        # Should use default force 0.05
        mock_bfgs_instance.run.assert_called_with(fmax=0.05)

def test_optimize_single_structure_warning_both_criteria(caplog):
    """Test warning when both criteria are set for single structure."""
    s = Structure(["H"], [(0.0, 0.0, 0.0)], charge=0, multiplicity=1)
    cfg = Config()
    cfg.optimization.force_convergence_criterion = 0.03
    cfg.optimization.energy_convergence_criterion = 1e-4

    mock_atoms_instance = MagicMock()
    mock_atoms_instance.get_positions.return_value = np.array([[0.0, 0.0, 0.0]])
    mock_atoms_instance.get_potential_energy.return_value = -1.0

    with (
        patch("gpuma.optimizer.Atoms", return_value=mock_atoms_instance),
        patch("gpuma.optimizer.BFGS") as MockBFGS,
        patch("gpuma.optimizer._get_cached_calculator")
    ):
        mock_bfgs_instance = MockBFGS.return_value

        with caplog.at_level(logging.WARNING):
            optimize_single_structure(s, config=cfg)

        # Should warn that force is used
        assert "force criterion should be used" in caplog.text.lower()
        mock_bfgs_instance.run.assert_called_with(fmax=0.03)

def test_batch_structure_optimization_criteria(caplog):
    """Test batch optimization criteria selection logic."""
    s = Structure(["H"], [(0.0, 0.0, 0.0)], charge=0, multiplicity=1)
    cfg = Config()
    cfg.optimization.batch_optimization_mode = "batch"
    cfg.optimization.device = "cuda"

    # Common mocks
    mock_ts = MagicMock()
    mock_ts_autobatching = MagicMock()
    mock_torch = MagicMock()

    # Setup mocks for torch_sim convergence functions
    mock_energy_conv_fn = MagicMock(name="energy_conv_fn")
    mock_force_conv_fn = MagicMock(name="force_conv_fn")
    mock_ts.generate_energy_convergence_fn.return_value = mock_energy_conv_fn
    mock_ts.generate_force_convergence_fn.return_value = mock_force_conv_fn

    # Mock return values for optimization
    mock_final_state = MagicMock()
    mock_final_state.to_atoms.return_value = []
    mock_ts.optimize.return_value = mock_final_state

    # Mock batched state n_atoms
    mock_batched_state = MagicMock()
    mock_batched_state.n_atoms = 10
    mock_ts.io.atoms_to_state.return_value = mock_batched_state

    modules_to_patch = {
        "torch_sim": mock_ts,
        "torch_sim.autobatching": mock_ts_autobatching,
        "torch": mock_torch
    }

    with (
        patch("gpuma.optimizer._parse_device_string", return_value="cuda"),
        patch("gpuma.optimizer._device_for_torch"),
        patch("gpuma.optimizer._get_cached_torchsim_model"),
        patch.dict("sys.modules", modules_to_patch)
    ):
        # Case 1: Force only (should use force)
        cfg.optimization.force_convergence_criterion = 0.02
        cfg.optimization.energy_convergence_criterion = None
        _optimize_batch_structures([s], config=cfg)
        mock_ts.generate_force_convergence_fn.assert_called_with(force_tol=0.02)
        mock_ts.optimize.assert_called_with(
            system=ANY, model=ANY, optimizer=ANY,
            convergence_fn=mock_force_conv_fn, # Check correct fn used
            autobatcher=ANY, steps_between_swaps=ANY
        )

        # Case 2: Energy only (should use energy)
        cfg.optimization.force_convergence_criterion = None
        cfg.optimization.energy_convergence_criterion = 1e-5
        _optimize_batch_structures([s], config=cfg)
        mock_ts.generate_energy_convergence_fn.assert_called_with(energy_tol=1e-5)
        mock_ts.optimize.assert_called_with(
            system=ANY, model=ANY, optimizer=ANY,
            convergence_fn=mock_energy_conv_fn, # Check correct fn used
            autobatcher=ANY, steps_between_swaps=ANY
        )

        # Case 3: Both (should use force and warn)
        cfg.optimization.force_convergence_criterion = 0.03
        cfg.optimization.energy_convergence_criterion = 1e-4
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _optimize_batch_structures([s], config=cfg)

        assert "using force convergence criterion" in caplog.text.lower()
        mock_ts.generate_force_convergence_fn.assert_called_with(force_tol=0.03)
        # Should be called with force fn
        mock_ts.optimize.call_args[1]["convergence_fn"] == mock_force_conv_fn
