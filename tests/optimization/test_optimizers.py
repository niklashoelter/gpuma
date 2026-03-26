"""Tests for optimizer selection — all 4 optimizers in both ASE and torch-sim."""

import logging

import pytest

from gpuma.config import Config, VALID_BATCH_OPTIMIZERS
from gpuma.optimizer import (
    _resolve_ase_optimizer,
    optimize_single_structure,
    optimize_structure_batch,
)

from conftest import DEVICE, requires_gpu


# ---------------------------------------------------------------------------
# ASE optimizer selection (single-structure mode)
# ---------------------------------------------------------------------------


class TestAseOptimizerSelection:
    """Config-to-ASE-class mapping and fallback behavior."""

    @pytest.mark.parametrize("optimizer_name,expected_cls_name", [
        ("fire", "FIRE"),
        ("bfgs", "BFGS"),
        ("lbfgs", "LBFGS"),
    ])
    def test_resolves_correct_class(self, optimizer_name, expected_cls_name):
        """Each optimizer name maps to the correct ASE class."""
        config = Config({"optimization": {"batch_optimizer": optimizer_name}})
        cls, name = _resolve_ase_optimizer(config)
        assert cls.__name__ == expected_cls_name
        assert name == optimizer_name

    def test_gradient_descent_falls_back_to_fire(self, caplog):
        """gradient_descent has no ASE equivalent; falls back to FIRE."""
        config = Config({"optimization": {"batch_optimizer": "gradient_descent"}})
        with caplog.at_level(logging.WARNING):
            cls, name = _resolve_ase_optimizer(config)
        assert cls.__name__ == "FIRE"
        assert name == "fire"
        assert "no gradient_descent optimizer" in caplog.text

    def test_default_is_fire(self):
        """Default optimizer (from DEFAULT_CONFIG) is FIRE."""
        config = Config()
        cls, name = _resolve_ase_optimizer(config)
        assert cls.__name__ == "FIRE"
        assert name == "fire"

    def test_optimizer_logged(self, methane, caplog):
        """Single-structure optimization logs the selected optimizer."""
        config = Config({
            "optimization": {
                "batch_optimizer": "bfgs",
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE},
        })
        with caplog.at_level(logging.INFO):
            optimize_single_structure(methane, config)
        assert "optimizer=bfgs" in caplog.text


class TestAseOptimizersOnRealStructures:
    """Verify each ASE optimizer produces valid energies on real structures."""

    @pytest.mark.parametrize("optimizer_name", ["fire", "bfgs", "lbfgs"])
    def test_each_ase_optimizer_produces_energy(self, ethanol, optimizer_name):
        """Each ASE optimizer produces a non-zero energy on ethanol."""
        config = Config({
            "optimization": {
                "batch_optimizer": optimizer_name,
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE},
        })
        result = optimize_single_structure(ethanol, config)
        assert result.energy is not None
        assert result.energy != 0.0


# ---------------------------------------------------------------------------
# Torch-sim optimizer selection (batch mode, all 4)
# ---------------------------------------------------------------------------


class TestTorchSimOptimizers:
    """All 4 torch-sim optimizers run successfully on real structures."""

    @requires_gpu
    @pytest.mark.parametrize("optimizer_name", sorted(VALID_BATCH_OPTIMIZERS))
    def test_optimizer_runs(self, ethanol, optimizer_name):
        """Each torch-sim optimizer produces valid energies in batch mode."""
        config = Config({
            "optimization": {
                "batch_optimization_mode": "batch",
                "batch_optimizer": optimizer_name,
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE, "max_atoms_to_try": 1000},
        })
        results = optimize_structure_batch([ethanol] * 3, config)
        assert len(results) == 3
        for r in results:
            assert r.energy is not None
            assert r.energy != 0.0
