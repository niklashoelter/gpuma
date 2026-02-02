import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gpuma.config import Config
from gpuma.optimizer import optimize_single_structure, optimize_structure_batch
from gpuma.structure import Structure


def test_optimize_structure_batch_empty_returns_empty():
    assert optimize_structure_batch([]) == []


def test_optimize_structure_batch_raises_on_mismatch():
    s = Structure(["H", "H"], [(0.0, 0.0, 0.0)], charge=0, multiplicity=1)
    with pytest.raises(ValueError):
        optimize_structure_batch([s])


def test_optimize_structure_batch_raises_on_empty_structure():
    s = Structure([], [], charge=0, multiplicity=1)
    with pytest.raises(ValueError):
        optimize_structure_batch([s])


def test_optimize_single_structure_runs_bfgs():
    s = Structure(["H"], [(0.0, 0.0, 0.0)], charge=0, multiplicity=1)

    # Mock Calculator
    calc = MagicMock()

    # Mock Atoms instance
    mock_atoms_instance = MagicMock()
    mock_atoms_instance.get_positions.return_value = np.array([[1.0, 0.0, 0.0]])
    mock_atoms_instance.get_potential_energy.return_value = -13.6

    with (
        patch("gpuma.optimizer.Atoms", return_value=mock_atoms_instance) as MockAtoms,
        patch("gpuma.optimizer.BFGS") as MockBFGS,
    ):
        mock_bfgs_instance = MockBFGS.return_value

        res = optimize_single_structure(s, calculator=calc)

        MockAtoms.assert_called()
        MockBFGS.assert_called_with(mock_atoms_instance, logfile=None)
        mock_bfgs_instance.run.assert_called_once()

        assert res.coordinates == [[1.0, 0.0, 0.0]]
        assert res.energy == -13.6


def test_optimize_structure_batch_sequential():
    s1 = Structure(["H"], [(0, 0, 0)], 0, 1)
    s2 = Structure(["He"], [(1, 0, 0)], 0, 1)

    cfg = Config()
    cfg.optimization.batch_optimization_mode = "sequential"
    cfg.optimization.device = "cpu"

    # We need to mock _get_cached_calculator to prevent network calls
    with (
        patch("gpuma.optimizer._get_cached_calculator") as mock_get_calc,
        patch("gpuma.optimizer.optimize_single_structure") as mock_opt,
    ):
        mock_get_calc.return_value = MagicMock()
        mock_opt.side_effect = lambda s, c, calc: s.with_energy(-5.0)

        results = optimize_structure_batch([s1, s2], config=cfg)

        assert len(results) == 2
        assert results[0].energy == -5.0
        assert results[1].energy == -5.0
        assert mock_opt.call_count == 2
        mock_get_calc.assert_called_once()


def test_optimize_structure_batch_batch_mode():
    s1 = Structure(["H"], [(0, 0, 0)], 0, 1)

    cfg = Config()
    cfg.optimization.batch_optimization_mode = "batch"
    cfg.optimization.device = "cuda"

    mock_ts = MagicMock()
    mock_ts_autobatching = MagicMock()

    # We patch both torch_sim and torch_sim.autobatching to support the import
    modules_to_patch = {
        "torch_sim": mock_ts,
        "torch_sim.autobatching": mock_ts_autobatching,
    }

    # Patch torch to allow torch.device(mock) if necessary, or just rely on correct mocking
    # Since torch.device(MagicMock) fails if torch is real, we should mock torch in this test too
    # or ensure we pass something torch.device accepts.
    # The safest way for "device=torch.device(device)" where device is a mock, is if torch.device
    # is also a mock that accepts anything.
    mock_torch = MagicMock()
    modules_to_patch["torch"] = mock_torch

    with (
        patch("gpuma.optimizer._parse_device_string", return_value="cuda"),
        patch("gpuma.optimizer._device_for_torch") as mock_dev_fn,
        patch("gpuma.optimizer._get_cached_torchsim_model") as mock_model_fn,
        patch.dict(sys.modules, modules_to_patch),
    ):
        # When _device_for_torch is called, return a string "cuda" (or a mock)
        # But wait, optimizer.py calls:
        # device = _device_for_torch(...) -> returns torch.device object usually
        # then calls torch.device(device)
        # If we mock _device_for_torch to return a string "cuda",
        # and we mock torch.device to accept it, it works.

        # If torch is mocked:
        # torch.device("cuda") -> MagicMock
        # _device_for_torch(...) -> returns that MagicMock
        # torch.device(that_mock) -> MagicMock (if mock accepts args)

        mock_dev_obj = MagicMock()
        mock_dev_fn.return_value = mock_dev_obj
        mock_torch.device.return_value = mock_dev_obj

        mock_model_fn.return_value = MagicMock()

        # Mock batched state
        mock_batched_state = MagicMock()
        mock_batched_state.n_atoms = 1
        mock_ts.io.atoms_to_state.return_value = mock_batched_state

        # Mock final state
        mock_final_state = MagicMock()
        mock_ts.optimize.return_value = mock_final_state

        # Mock final atoms
        mock_final_atom = MagicMock()
        mock_final_atom.get_chemical_symbols.return_value = ["H"]
        mock_final_atom.get_positions.return_value = np.array([[0.5, 0.0, 0.0]])

        mock_final_state.to_atoms.return_value = [mock_final_atom]

        # energy/charge/spin tensors
        mock_final_state.energy = [MagicMock(item=lambda: -10.0)]
        mock_final_state.charge = [MagicMock(item=lambda: 0)]
        mock_final_state.spin = [MagicMock(item=lambda: 1)]

        results = optimize_structure_batch([s1], config=cfg)

        assert len(results) == 1
        assert results[0].energy == -10.0
        assert results[0].coordinates == [[0.5, 0.0, 0.0]]

        mock_ts.optimize.assert_called_once()
