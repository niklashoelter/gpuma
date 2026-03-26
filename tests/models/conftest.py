"""Model test fixtures — GPU memory cleanup between tests."""

import gc

import pytest
import torch
from ase import Atoms

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


@pytest.fixture(autouse=True)
def _cleanup_gpu_memory():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
