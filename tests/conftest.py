"""Shared test fixtures for GPUMA.

No mocking — all fixtures use real packages and models.
ORB-v3 is the default test backend (no HuggingFace token required).
Fairchem tests are skipped when HF_TOKEN is not set.
"""

import gc
import os
from pathlib import Path

import pytest
import torch

from gpuma.config import Config
from gpuma.structure import Structure

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "examples" / "example_input_xyzs"
SINGLE_XYZ = TEST_DATA_DIR / "single_xyz_file.xyz"
MULTI_XYZ = TEST_DATA_DIR / "multi_xyz_file.xyz"
SMALL_BATCH_XYZ = TEST_DATA_DIR / "many_structures_small.xyz"
MULTI_XYZ_DIR = TEST_DATA_DIR / "multi_xyz_dir"
BUTENE_SINGLET_XYZ = TEST_DATA_DIR / "butene_singlet.xyz"
BUTENE_TRIPLET_XYZ = TEST_DATA_DIR / "butene_triplet.xyz"
BUTENE_TRIPLET_MULTI_XYZ = TEST_DATA_DIR / "butene_triplet_multi.xyz"


def has_hf_token() -> bool:
    """Check whether a HuggingFace token is available."""
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

requires_hf_token = pytest.mark.skipif(
    not has_hf_token(), reason="HF_TOKEN not set — required for Fairchem models"
)


# ---------------------------------------------------------------------------
# Structure fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def methane():
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
def ethanol():
    return Structure(
        symbols=["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        coordinates=[
            (-0.047, 0.536, 0.000),
            (-1.268, -0.376, 0.000),
            (1.139, -0.227, 0.000),
            (-0.045, 1.163, 0.891),
            (-0.045, 1.163, -0.891),
            (-1.307, -0.998, 0.891),
            (-1.307, -0.998, -0.891),
            (-2.149, 0.259, 0.000),
            (1.960, 0.286, 0.000),
        ],
        charge=0,
        multiplicity=1,
        comment="Ethanol",
    )


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orb_config():
    """ORB config for batch tests — no HF token needed."""
    return Config({
        "optimization": {
            "batch_optimization_mode": "batch",
            "force_convergence_criterion": 0.5,
        },
        "model": {
            "model_type": "orb",
            "model_name": "orb_v3_direct_omol",
        },
        "technical": {
            "device": DEVICE,
            "max_atoms_to_try": 1000,
        },
    })


@pytest.fixture
def orb_sequential_config():
    """ORB config for sequential (CPU-safe) optimization."""
    return Config({
        "optimization": {
            "batch_optimization_mode": "sequential",
            "force_convergence_criterion": 0.5,
        },
        "model": {
            "model_type": "orb",
            "model_name": "orb_v3_direct_omol",
        },
        "technical": {
            "device": DEVICE,
        },
    })


# ---------------------------------------------------------------------------
# Session-scoped model fixtures (loaded once, reused across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def orb_calculator():
    """Real ORB ASE calculator, loaded once per session."""
    from gpuma.models import load_calculator

    config = Config({
        "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
        "technical": {"device": DEVICE},
    })
    calc = load_calculator(config)
    yield calc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def orb_torchsim_model():
    """Real ORB torch-sim model, loaded once per session."""
    from gpuma.models import load_torchsim_model

    config = Config({
        "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
        "technical": {"device": DEVICE},
    })
    model = load_torchsim_model(config)
    yield model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
