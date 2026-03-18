"""Slow integration tests that load and instantiate every available model.

These tests are **not** run by default.  Execute them explicitly with::

    pytest -m slow tests/test_model_loading_slow.py

Each test downloads (if not cached) and instantiates the model on the
available device, then runs a minimal forward pass on a small molecule
(methane) to verify correctness.

GPU memory is explicitly freed between parametrized tests to avoid OOM
errors when loading many models sequentially.
"""

import gc
import os

import pytest
import torch
from ase import Atoms

from gpuma.config import Config
from gpuma.models import (
    AVAILABLE_FAIRCHEM_MODELS,
    AVAILABLE_ORB_MODELS,
    load_calculator,
    load_torchsim_model,
)

# All tests in this module require real model loading (no mocks) and are slow.
pytestmark = [pytest.mark.slow, pytest.mark.real_model]

# Use GPU if available, otherwise CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A small molecule for quick forward-pass validation.
METHANE = Atoms(
    symbols=["C", "H", "H", "H", "H"],
    positions=[
        [0.000, 0.000, 0.000],
        [0.629, 0.629, 0.629],
        [-0.629, -0.629, 0.629],
        [-0.629, 0.629, -0.629],
        [0.629, -0.629, -0.629],
    ],
)


def _has_hf_token() -> bool:
    """Check whether a HuggingFace token is available."""
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))


@pytest.fixture(autouse=True)
def _cleanup_gpu_memory():
    """Free GPU memory after each test to avoid OOM when loading many models."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Fairchem — ASE calculator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", AVAILABLE_FAIRCHEM_MODELS)
def test_load_fairchem_calculator(model_name: str):
    """Load a Fairchem calculator and run a single-point energy evaluation."""
    if not _has_hf_token():
        pytest.skip("HF_TOKEN not set — required for Fairchem model download")

    config = Config({
        "model": {
            "model_type": "fairchem",
            "model_name": model_name,
        },
        "technical": {"device": DEVICE},
    })

    calc = load_calculator(config)
    assert calc is not None

    atoms = METHANE.copy()
    atoms.calc = calc
    atoms.info = {"charge": 0, "spin": 1}

    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    assert energy != 0.0, f"Energy is exactly 0.0 for model {model_name}"


# ---------------------------------------------------------------------------
# Fairchem — torch-sim model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", AVAILABLE_FAIRCHEM_MODELS)
def test_load_fairchem_torchsim(model_name: str):
    """Load a Fairchem torch-sim model and verify it is not None."""
    if not _has_hf_token():
        pytest.skip("HF_TOKEN not set — required for Fairchem model download")

    config = Config({
        "model": {
            "model_type": "fairchem",
            "model_name": model_name,
        },
        "technical": {"device": DEVICE},
    })

    model = load_torchsim_model(config)
    assert model is not None


# ---------------------------------------------------------------------------
# ORB-v3 — ASE calculator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", AVAILABLE_ORB_MODELS)
def test_load_orb_calculator(model_name: str):
    """Load an ORB calculator and run a single-point energy evaluation."""
    config = Config({
        "model": {
            "model_type": "orb",
            "model_name": model_name,
        },
        "technical": {"device": DEVICE},
    })

    calc = load_calculator(config)
    assert calc is not None

    atoms = METHANE.copy()
    atoms.calc = calc
    atoms.info = {"charge": 0, "spin": 1}

    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    assert energy != 0.0, f"Energy is exactly 0.0 for model {model_name}"


# ---------------------------------------------------------------------------
# ORB-v3 — torch-sim model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", AVAILABLE_ORB_MODELS)
def test_load_orb_torchsim(model_name: str):
    """Load an ORB torch-sim model and verify it is not None."""
    config = Config({
        "model": {
            "model_type": "orb",
            "model_name": model_name,
        },
        "technical": {"device": DEVICE},
    })

    model = load_torchsim_model(config)
    assert model is not None


# ---------------------------------------------------------------------------
# ORB-v3 with D3 dispersion correction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["orb_v3_direct_omol", "orb_v3_conservative_omol"])
def test_load_orb_calculator_with_d3(model_name: str):
    """Load an ORB calculator with D3 correction and verify energy output."""
    config = Config({
        "model": {
            "model_type": "orb",
            "model_name": model_name,
            "d3_correction": True,
            "d3_functional": "PBE",
            "d3_damping": "BJ",
        },
        "technical": {"device": DEVICE},
    })

    calc = load_calculator(config)
    assert calc is not None

    atoms = METHANE.copy()
    atoms.calc = calc
    atoms.info = {"charge": 0, "spin": 1}

    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    assert energy != 0.0, f"Energy is exactly 0.0 for model {model_name} with D3"


# ---------------------------------------------------------------------------
# GPU device selection tests
#
# These tests verify that models are actually placed on the correct GPU.
# They target cuda:3 which is typically free on multi-GPU machines.
# ---------------------------------------------------------------------------

TARGET_GPU = 3  # The GPU index we test specific device selection with


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _skip_if_gpu_missing(idx: int):
    _skip_if_no_cuda()
    if torch.cuda.device_count() <= idx:
        pytest.skip(f"GPU {idx} not available (only {torch.cuda.device_count()} GPUs)")


def _get_model_device_index(model) -> int:
    """Extract the GPU index from an ORB or Fairchem torch-sim model."""
    # torch-sim models expose a .device property (torch.device)
    dev = model.device if hasattr(model, "device") else None
    if dev is not None and hasattr(dev, "index") and dev.index is not None:
        return dev.index
    # Fallback: check first parameter
    for param in model.parameters() if hasattr(model, "parameters") else []:
        if param.device.type == "cuda":
            return param.device.index or 0
        break
    return torch.cuda.current_device()


# --- ORB-v3: cuda:3 device placement ---


def test_orb_calculator_on_cuda3():
    """ORB calculator on cuda:3 runs inference on the correct GPU."""
    _skip_if_gpu_missing(TARGET_GPU)
    config = Config({
        "model": {
            "model_type": "orb",
            "model_name": "orb_v3_direct_omol",
        },
        "technical": {"device": f"cuda:{TARGET_GPU}"},
    })
    calc = load_calculator(config)
    assert calc is not None

    atoms = METHANE.copy()
    atoms.calc = calc
    atoms.info = {"charge": 0, "spin": 1}
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    assert energy != 0.0


def test_orb_torchsim_on_cuda3():
    """ORB torch-sim model is placed on cuda:3."""
    _skip_if_gpu_missing(TARGET_GPU)
    config = Config({
        "model": {
            "model_type": "orb",
            "model_name": "orb_v3_direct_omol",
        },
        "technical": {"device": f"cuda:{TARGET_GPU}"},
    })
    model = load_torchsim_model(config)
    assert model is not None
    assert _get_model_device_index(model) == TARGET_GPU


# --- Fairchem: cuda:3 device placement ---


def test_fairchem_calculator_on_cuda3():
    """Fairchem calculator on cuda:3 runs inference on the correct GPU."""
    _skip_if_gpu_missing(TARGET_GPU)
    if not _has_hf_token():
        pytest.skip("HF_TOKEN not set")
    config = Config({
        "model": {
            "model_type": "fairchem",
            "model_name": "uma-s-1p1",
        },
        "technical": {"device": f"cuda:{TARGET_GPU}"},
    })
    calc = load_calculator(config)
    assert calc is not None

    atoms = METHANE.copy()
    atoms.calc = calc
    atoms.info = {"charge": 0, "spin": 1}
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    assert energy != 0.0

    # Verify Fairchem actually selected the right GPU via set_device
    assert torch.cuda.current_device() == TARGET_GPU


def test_fairchem_torchsim_on_cuda3():
    """Fairchem torch-sim model is placed on cuda:3 via set_device."""
    _skip_if_gpu_missing(TARGET_GPU)
    if not _has_hf_token():
        pytest.skip("HF_TOKEN not set")
    config = Config({
        "model": {
            "model_type": "fairchem",
            "model_name": "uma-s-1p1",
        },
        "technical": {"device": f"cuda:{TARGET_GPU}"},
    })
    model = load_torchsim_model(config)
    assert model is not None
    # Fairchem uses set_device, so current_device should be TARGET_GPU
    assert torch.cuda.current_device() == TARGET_GPU


# --- Fallback: invalid GPU index ---


def test_invalid_gpu_index_fallback_orb(caplog):
    """Requesting a non-existent GPU falls back to cuda:0 with a warning."""
    _skip_if_no_cuda()
    num_gpus = torch.cuda.device_count()
    bad_index = num_gpus + 5  # guaranteed to not exist
    config = Config({
        "model": {
            "model_type": "orb",
            "model_name": "orb_v3_direct_omol",
        },
        "technical": {"device": f"cuda:{bad_index}"},
    })
    calc = load_calculator(config)
    assert calc is not None
    assert "Falling back to cuda:0" in caplog.text

    atoms = METHANE.copy()
    atoms.calc = calc
    atoms.info = {"charge": 0, "spin": 1}
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)


def test_invalid_gpu_index_fallback_fairchem(caplog):
    """Requesting a non-existent GPU falls back to cuda:0 for Fairchem."""
    _skip_if_no_cuda()
    if not _has_hf_token():
        pytest.skip("HF_TOKEN not set")
    num_gpus = torch.cuda.device_count()
    bad_index = num_gpus + 5
    config = Config({
        "model": {
            "model_type": "fairchem",
            "model_name": "uma-s-1p1",
        },
        "technical": {"device": f"cuda:{bad_index}"},
    })
    calc = load_calculator(config)
    assert calc is not None
    assert "Falling back to cuda:0" in caplog.text
