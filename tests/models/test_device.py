"""Tests for device parsing and setup helpers."""

import pytest
import torch

from gpuma.models import (
    _device_for_torch,
    _parse_device_string,
    _setup_fairchem_device,
)

from conftest import requires_gpu


class TestParseDeviceString:
    """Tests for _parse_device_string — normalizes device strings."""

    def test_cpu(self):
        """'cpu' and 'CPU' both normalize to 'cpu'."""
        assert _parse_device_string("cpu") == "cpu"
        assert _parse_device_string("CPU") == "cpu"

    @requires_gpu
    def test_cuda(self):
        """'cuda' is accepted when GPU is available."""
        assert _parse_device_string("cuda") == "cuda"

    @requires_gpu
    def test_cuda_with_valid_index(self):
        """'cuda:0' is accepted as-is."""
        assert _parse_device_string("cuda:0") == "cuda:0"

    @requires_gpu
    def test_cuda_invalid_index_fallback(self):
        """Out-of-range GPU index falls back to 'cuda:0'."""
        bad_index = torch.cuda.device_count() + 5
        assert _parse_device_string(f"cuda:{bad_index}") == "cuda:0"

    def test_cuda_no_gpu_fallback(self):
        """CUDA requested without GPU falls back to 'cpu'."""
        if torch.cuda.is_available():
            pytest.skip("Test only runs without GPU")
        assert _parse_device_string("cuda") == "cpu"

    def test_unknown_device(self):
        """Unknown device strings fall back to 'cpu'."""
        assert _parse_device_string("tpu") == "cpu"
        assert _parse_device_string("") == "cpu"

    @requires_gpu
    def test_cuda_non_integer_index(self):
        """Non-integer CUDA index falls back to plain 'cuda'."""
        assert _parse_device_string("cuda:abc") == "cuda"


class TestDeviceForTorch:
    """Tests for _device_for_torch — creates torch.device objects."""

    def test_cpu(self):
        """'cpu' returns torch.device('cpu')."""
        assert _device_for_torch("cpu") == torch.device("cpu")

    @requires_gpu
    def test_cuda(self):
        """'cuda:0' returns a CUDA torch.device with index 0."""
        d = _device_for_torch("cuda:0")
        assert d.type == "cuda"
        assert d.index == 0


class TestSetupFairchemDevice:
    """Tests for _setup_fairchem_device — prepares device for Fairchem backend."""

    def test_cpu(self):
        """CPU input returns 'cpu'."""
        assert _setup_fairchem_device("cpu") == "cpu"

    @requires_gpu
    def test_plain_cuda(self):
        """Plain 'cuda' returns 'cuda' without calling set_device."""
        assert _setup_fairchem_device("cuda") == "cuda"

    @requires_gpu
    def test_cuda_with_index(self):
        """'cuda:0' calls set_device and returns plain 'cuda'."""
        assert _setup_fairchem_device("cuda:0") == "cuda"
