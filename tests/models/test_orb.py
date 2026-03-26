"""Tests for ORB model loading — real models, no HF token needed."""

import pytest
import torch

from gpuma.config import Config
from gpuma.models import (
    AVAILABLE_ORB_MODELS,
    load_calculator,
    load_torchsim_model,
)

from conftest import DEVICE, requires_gpu

from .conftest import METHANE

TARGET_GPU = 3


class TestOrbCalculator:
    """ORB ASE calculator loading and inference."""

    @requires_gpu
    def test_load(self, orb_calculator):
        """Session-scoped ORB calculator loads successfully."""
        assert orb_calculator is not None

    @requires_gpu
    def test_forward_pass(self, orb_calculator):
        """ORB calculator produces non-zero energy on methane."""
        atoms = METHANE.copy()
        atoms.calc = orb_calculator
        atoms.info = {"charge": 0, "spin": 1}
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert energy != 0.0

    def test_invalid_model_name(self):
        """Unknown model name raises ValueError."""
        config = Config({
            "model": {"model_type": "orb", "model_name": "nonexistent_model"},
            "technical": {"device": DEVICE},
        })
        with pytest.raises(ValueError, match="Unknown ORB model name"):
            load_calculator(config)

    def test_missing_model_name(self):
        """Empty model name raises ValueError."""
        config = Config({
            "model": {"model_type": "orb", "model_name": ""},
            "technical": {"device": DEVICE},
        })
        with pytest.raises(ValueError, match="Model name must be specified"):
            load_calculator(config)


class TestOrbTorchsim:
    """ORB torch-sim model loading for batch optimization."""

    @requires_gpu
    def test_load(self, orb_torchsim_model):
        """Session-scoped ORB torch-sim model loads successfully."""
        assert orb_torchsim_model is not None

    @requires_gpu
    @pytest.mark.parametrize("model_name", [
        "orb_v3_direct_omol", "orb_v3_conservative_omol",
    ])
    def test_variants(self, model_name):
        """Both direct and conservative ORB variants load."""
        config = Config({
            "model": {"model_type": "orb", "model_name": model_name},
            "technical": {"device": DEVICE},
        })
        model = load_torchsim_model(config)
        assert model is not None


class TestOrbD3Correction:
    """D3 dispersion correction for ORB models."""

    @requires_gpu
    def test_d3_changes_energy(self):
        """Enabling D3 correction produces a different energy than without."""
        config_no_d3 = Config({
            "model": {
                "model_type": "orb",
                "model_name": "orb_v3_direct_omol",
                "d3_correction": False,
            },
            "technical": {"device": DEVICE},
        })
        config_d3 = Config({
            "model": {
                "model_type": "orb",
                "model_name": "orb_v3_direct_omol",
                "d3_correction": True,
                "d3_functional": "PBE",
                "d3_damping": "BJ",
            },
            "technical": {"device": DEVICE},
        })

        calc_no_d3 = load_calculator(config_no_d3)
        calc_d3 = load_calculator(config_d3)

        atoms1 = METHANE.copy()
        atoms1.calc = calc_no_d3
        atoms1.info = {"charge": 0, "spin": 1}
        e_no_d3 = atoms1.get_potential_energy()

        atoms2 = METHANE.copy()
        atoms2.calc = calc_d3
        atoms2.info = {"charge": 0, "spin": 1}
        e_d3 = atoms2.get_potential_energy()

        assert e_no_d3 != e_d3, "D3 correction should change energy"


class TestGpuDevicePlacement:
    """GPU device selection and fallback behavior."""

    @requires_gpu
    def test_orb_calculator_on_cuda3(self):
        """ORB calculator can be placed on a specific GPU (cuda:3)."""
        if torch.cuda.device_count() <= TARGET_GPU:
            pytest.skip(f"GPU {TARGET_GPU} not available")
        config = Config({
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": f"cuda:{TARGET_GPU}"},
        })
        calc = load_calculator(config)
        atoms = METHANE.copy()
        atoms.calc = calc
        atoms.info = {"charge": 0, "spin": 1}
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        torch.cuda.set_device(0)

    @requires_gpu
    def test_invalid_gpu_fallback(self, caplog):
        """Invalid GPU index falls back to cuda:0 with a warning."""
        bad_index = torch.cuda.device_count() + 5
        config = Config({
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": f"cuda:{bad_index}"},
        })
        calc = load_calculator(config)
        assert calc is not None
        assert "Falling back to cuda:0" in caplog.text


class TestModelRegistries:
    """ORB model name registry."""

    def test_orb_model_names_exist(self):
        """AVAILABLE_ORB_MODELS contains all expected model variants."""
        assert len(AVAILABLE_ORB_MODELS) >= 10
        assert "orb_v3_direct_omol" in AVAILABLE_ORB_MODELS
