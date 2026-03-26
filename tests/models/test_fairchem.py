"""Tests for Fairchem model loading — requires HF_TOKEN."""

import pytest

from gpuma.config import Config
from gpuma.models import (
    AVAILABLE_FAIRCHEM_MODELS,
    load_calculator,
    load_torchsim_model,
)

from conftest import DEVICE, requires_gpu, requires_hf_token

from .conftest import METHANE


class TestFairchemCalculator:
    """Fairchem ASE calculator and torch-sim model loading."""

    @requires_hf_token
    @requires_gpu
    @pytest.mark.parametrize("model_name", AVAILABLE_FAIRCHEM_MODELS)
    def test_load_and_forward_pass(self, model_name):
        """Each Fairchem model loads and produces non-zero energy on methane."""
        config = Config({
            "model": {"model_type": "fairchem", "model_name": model_name},
            "technical": {"device": DEVICE},
        })
        calc = load_calculator(config)
        assert calc is not None

        atoms = METHANE.copy()
        atoms.calc = calc
        atoms.info = {"charge": 0, "spin": 1}
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert energy != 0.0

    @requires_hf_token
    @requires_gpu
    @pytest.mark.parametrize("model_name", AVAILABLE_FAIRCHEM_MODELS)
    def test_load_torchsim(self, model_name):
        """Each Fairchem torch-sim model loads successfully."""
        config = Config({
            "model": {"model_type": "fairchem", "model_name": model_name},
            "technical": {"device": DEVICE},
        })
        model = load_torchsim_model(config)
        assert model is not None


class TestFairchemModelRegistry:
    """Fairchem model name registry."""

    def test_fairchem_model_names_exist(self):
        """AVAILABLE_FAIRCHEM_MODELS contains all expected UMA models."""
        assert len(AVAILABLE_FAIRCHEM_MODELS) >= 3
        assert "uma-s-1p2" in AVAILABLE_FAIRCHEM_MODELS
