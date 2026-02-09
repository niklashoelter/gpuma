from unittest.mock import MagicMock, patch, ANY
import os

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

    # Test invalid device fallback
    with patch("torch.cuda.is_available", return_value=False):
        # Even if we ask for cuda, if not available it might raise or fallback depending on implementation
        # _device_for_torch calls _parse_device_string which checks availability.
        # If _parse_device_string returns cpu, _device_for_torch returns cpu device.
        dev = _device_for_torch("cuda")
        assert dev.type == "cpu"

def test_load_model_fairchem_logic(mock_hf_token):
    config = Config({"optimization": {"model_name": "test_model"}})

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

def test_load_model_fairchem_path(mock_hf_token, tmp_path):
    model_file = tmp_path / "model.pt"
    model_file.touch()

    config = Config({"optimization": {"model_path": str(model_file)}})

    try:
        import fairchem.core as _
    except ImportError:
        pytest.skip("fairchem.core not installed")

    with patch("fairchem.core.pretrained_mlip.load_predict_unit") as mock_load_unit, \
         patch("fairchem.core.FAIRChemCalculator") as mock_calc_cls:

        mock_load_unit.return_value = MagicMock()
        mock_calc_cls.return_value = MagicMock()

        calc = load_model_fairchem(config)

        mock_load_unit.assert_called_with(path=model_file, device=ANY)
        mock_calc_cls.assert_called()

def test_load_model_torchsim_path(mock_hf_token, tmp_path):
    model_file = tmp_path / "model.pt"
    model_file.touch()

    config = Config({"optimization": {"model_path": str(model_file)}})

    try:
        import torch_sim.models.fairchem as _
    except ImportError:
        pytest.skip("torch_sim not installed")

    with patch("torch_sim.models.fairchem.FairChemModel") as mock_model_cls:
        mock_model_cls.return_value = MagicMock()

        model = load_model_torchsim(config)

        call_kwargs = mock_model_cls.call_args.kwargs
        assert call_kwargs["model"] == model_file

def test_model_cache_creation_failure(mock_hf_token, caplog):
    # Test fails to create cache dir
    config = Config({"optimization": {"model_cache_dir": "/invalid/path/cache", "model_name": "uma"}})

    try:
        import fairchem.core as _
    except ImportError:
        pytest.skip("fairchem.core not installed")

    with patch("os.makedirs", side_effect=OSError("Permission denied")), \
         patch("fairchem.core.pretrained_mlip.get_predict_unit") as mock_get_unit, \
         patch("fairchem.core.FAIRChemCalculator"):

        load_model_fairchem(config)

        # Verify get_predict_unit called without cache_dir
        call_kwargs = mock_get_unit.call_args.kwargs
        assert "cache_dir" not in call_kwargs

        # Check that warning was logged
        assert "Could not create model cache directory" in caplog.text

def test_model_path_not_exists(mock_hf_token):
    # Path doesn't exist, should fall back to name
    config = Config({"optimization": {"model_path": "/non/existent/path", "model_name": "fallback"}})

    try:
        import fairchem.core as _
    except ImportError:
        pytest.skip("fairchem.core not installed")

    with patch("fairchem.core.pretrained_mlip.get_predict_unit") as mock_get_unit, \
         patch("fairchem.core.FAIRChemCalculator"):

         load_model_fairchem(config)

         # Should call get_predict_unit (name based)
         mock_get_unit.assert_called()
         args, _ = mock_get_unit.call_args
         assert args[0] == "fallback"

def test_missing_model_name(mock_hf_token):
    # Config without model_name (and no valid path) should raise ValueError
    # Config default has model_name, so we must explicitly set it to empty
    config = Config({"optimization": {"model_name": ""}})

    with pytest.raises(ValueError, match="Model name must be specified"):
        load_model_fairchem(config)

def test_hf_token_set_from_config(monkeypatch):
    token = "my_token"
    config = Config({"optimization": {"huggingface_token": token, "model_name": "test"}})

    # Ensure env is clean
    monkeypatch.delenv("HF_TOKEN", raising=False)

    from gpuma.models import _load_hf_token_to_env
    _load_hf_token_to_env(config)
    assert os.environ["HF_TOKEN"] == token
