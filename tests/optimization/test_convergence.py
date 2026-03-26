"""Tests for convergence criteria, warnings, validation, and summary logging."""

import logging

import pytest

from gpuma.config import Config
from gpuma.optimizer import optimize_single_structure, optimize_structure_batch
from gpuma.structure import Structure

from conftest import DEVICE


class TestConvergenceWarnings:
    """Warnings when convergence criteria are ambiguous or unsupported."""

    def test_both_criteria_warns(self, methane, caplog):
        """Both force and energy criteria set logs a warning."""
        config = Config({
            "optimization": {
                "force_convergence_criterion": 0.5,
                "energy_convergence_criterion": 0.001,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE},
        })
        with caplog.at_level(logging.WARNING):
            optimize_single_structure(methane, config)
        assert "Both force and energy convergence criteria given" in caplog.text

    def test_energy_only_criterion_warns(self, methane, caplog):
        """Energy-only convergence warns and falls back to default force."""
        config = Config({
            "optimization": {
                "force_convergence_criterion": None,
                "energy_convergence_criterion": 0.001,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE},
        })
        with caplog.at_level(logging.WARNING):
            result = optimize_single_structure(methane, config)
        assert "Energy convergence criterion requested" in caplog.text
        assert result.energy is not None


class TestConvergenceBehavior:
    """Verify that convergence criteria affect optimization results."""

    def test_tighter_criterion_lowers_energy(self, ethanol):
        """Tighter force convergence produces lower or equal energy."""
        loose = Config({
            "optimization": {"force_convergence_criterion": 0.5},
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE},
        })
        tight = Config({
            "optimization": {"force_convergence_criterion": 0.01},
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE},
        })

        loose_input = Structure(
            symbols=list(ethanol.symbols),
            coordinates=[list(c) for c in ethanol.coordinates],
            charge=ethanol.charge,
            multiplicity=ethanol.multiplicity,
        )
        tight_input = Structure(
            symbols=list(ethanol.symbols),
            coordinates=[list(c) for c in ethanol.coordinates],
            charge=ethanol.charge,
            multiplicity=ethanol.multiplicity,
        )

        loose_result = optimize_single_structure(loose_input, loose)
        tight_result = optimize_single_structure(tight_input, tight)

        assert tight_result.energy <= loose_result.energy + 0.01


class TestOptimizationSummary:
    """Optimization summary is logged after batch runs."""

    def test_summary_logged(self, methane, orb_sequential_config, caplog):
        """Summary includes structure counts and optimizer info."""
        with caplog.at_level(logging.INFO, logger="gpuma.logging_utils"):
            optimize_structure_batch([methane], orb_sequential_config)
        assert "GPUMA Optimization Summary" in caplog.text
        assert "Structures input:    1" in caplog.text
        assert "Optimizer:" in caplog.text

    def test_summary_includes_optimizer_name(self, methane, caplog):
        """Summary shows the configured optimizer name."""
        config = Config({
            "optimization": {
                "batch_optimization_mode": "sequential",
                "batch_optimizer": "lbfgs",
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE},
        })
        with caplog.at_level(logging.INFO, logger="gpuma.logging_utils"):
            optimize_structure_batch([methane], config)
        assert "Optimizer:           lbfgs" in caplog.text


class TestValidation:
    """Input validation for optimize_structure_batch."""

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert optimize_structure_batch([]) == []

    def test_mismatched_coords(self):
        """Mismatched symbols/coordinates raises ValueError."""
        bad = Structure(
            symbols=["C", "H"],
            coordinates=[(0, 0, 0)],
            charge=0,
            multiplicity=1,
        )
        with pytest.raises(ValueError, match="symbols/coords length mismatch"):
            optimize_structure_batch([bad])

    def test_empty_structure(self):
        """Zero-atom structure raises ValueError."""
        empty = Structure(symbols=[], coordinates=[], charge=0, multiplicity=1)
        with pytest.raises(ValueError, match="empty structure"):
            optimize_structure_batch([empty])
