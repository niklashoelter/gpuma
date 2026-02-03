from unittest.mock import MagicMock, patch

import pytest
import torch

from gpuma.config import Config
from gpuma.models import (
    _backend_device_for_fairchem,
    _device_for_torch,
    _parse_device_string,
    load_model_fairchem,
    load_model_torchsim,
)


def test_parse_device_string():
    assert _parse_device_string("cpu") == "cpu"
    assert _parse_device_string("CPU") == "cpu"

    # Mock cuda availability
    with patch("torch.cuda.is_available", return_value=True):
        assert _parse_device_string("cuda") == "cuda"
        assert _parse_device_string("cuda:0") == "cuda:0"

    with patch("torch.cuda.is_available", return_value=False):
        assert _parse_device_string("cuda") == "cpu"

def test_backend_device_for_fairchem():
    with patch("torch.cuda.is_available", return_value=True):
        assert _backend_device_for_fairchem("cuda:0") == "cuda"
        assert _backend_device_for_fairchem("cpu") == "cpu"

def test_device_for_torch():
    with patch("torch.cuda.is_available", return_value=True):
        dev = _device_for_torch("cuda:0")
        assert isinstance(dev, torch.device)
        assert dev.type == "cuda"
        assert dev.index == 0

def test_load_model_fairchem_logic(mock_hf_token):
    config = Config({"optimization": {"model_name": "test_model"}})

    # We need to patch fairchem.core inside the function or pre-import it
    # Since it is a local import, patching the module where it lives (fairchem.core) works
    # if we can import it first.
    try:
        import fairchem.core as _  # noqa: F401
    except ImportError:
        pytest.skip("fairchem.core not installed")

    with patch("fairchem.core.pretrained_mlip.get_predict_unit") as mock_get_unit, \
         patch("fairchem.core.FAIRChemCalculator") as mock_calc_cls:

        mock_get_unit.return_value = MagicMock()
        mock_calc_cls.return_value = MagicMock()

        calc = load_model_fairchem(config)

        mock_get_unit.assert_called()
        mock_calc_cls.assert_called()
        assert calc is mock_calc_cls.return_value

def test_load_model_torchsim_logic(mock_hf_token):
    config = Config({"optimization": {"model_name": "test_model"}})

    try:
        import torch_sim.models.fairchem as _  # noqa: F401
    except ImportError:
        pytest.skip("torch_sim not installed")

    with patch("torch_sim.models.fairchem.FairChemModel") as mock_model_cls:
        mock_model_cls.return_value = MagicMock()

        model = load_model_torchsim(config)

        mock_model_cls.assert_called()
        assert model is mock_model_cls.return_value
