"""Tests for single-structure optimization — real ORB models."""

import logging

import pytest

from gpuma.config import Config
from gpuma.optimizer import optimize_single_structure

from conftest import DEVICE, requires_hf_token


class TestOptimizeSingleStructure:
    """Single-structure optimization using ASE with real ORB calculator."""

    def test_optimize_methane(self, methane, orb_sequential_config):
        """Methane optimizes to a valid energy."""
        result = optimize_single_structure(methane, orb_sequential_config)
        assert result.energy is not None
        assert isinstance(result.energy, float)
        assert result.energy != 0.0
        assert result.n_atoms == 5

    def test_optimize_ethanol(self, ethanol, orb_sequential_config):
        """Ethanol optimizes to a valid energy."""
        result = optimize_single_structure(ethanol, orb_sequential_config)
        assert result.energy is not None
        assert result.n_atoms == 9

    def test_energy_decreases(self, ethanol, orb_sequential_config, orb_calculator):
        """Optimization produces energy lower than or equal to the initial state."""
        from ase import Atoms

        atoms = Atoms(symbols=ethanol.symbols, positions=ethanol.coordinates)
        atoms.calc = orb_calculator
        atoms.info = {"charge": 0, "spin": 1}
        initial_energy = atoms.get_potential_energy()

        result = optimize_single_structure(ethanol, orb_sequential_config)
        assert result.energy <= initial_energy

    @requires_hf_token
    def test_optimize_with_fairchem(self, methane):
        """Fairchem backend works for single-structure optimization."""
        config = Config({
            "optimization": {"force_convergence_criterion": 0.5},
            "model": {"model_type": "fairchem", "model_name": "uma-s-1p1"},
            "technical": {"device": DEVICE},
        })
        result = optimize_single_structure(methane, config)
        assert result.energy is not None
