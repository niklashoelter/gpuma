import os
from unittest.mock import MagicMock, patch

import pytest

from gpuma import models as model_utils
from gpuma.config import Config

# Note: Tests rely on mocked fairchem/torch_sim from conftest.py or internal patches


def test_models_module_exports():
    assert hasattr(model_utils, "load_model_fairchem")
    assert hasattr(model_utils, "load_model_torchsim")


def test_load_model_fairchem_empty_name_raises():
    cfg = Config.from_dict({"optimization": {"model_name": "", "device": "cpu"}})
    with pytest.raises(ValueError):
        model_utils.load_model_fairchem(cfg)


def test_load_model_fairchem_uses_cache_dir(tmp_path):
    cfg = Config()
    cfg.optimization.model_name = "test_model"
    cfg.optimization.model_cache_dir = str(tmp_path)
    cfg.optimization.device = "cpu"

    # fairchem.core might be a mock from conftest.
    # We patch the attributes we need.

    with patch("fairchem.core.pretrained_mlip.get_predict_unit") as mock_get:
        mock_get.return_value = MagicMock()
        with patch("fairchem.core.FAIRChemCalculator") as MockCalc:
            model_utils.load_model_fairchem(cfg)

            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert args[0] == "test_model"
            assert kwargs.get("cache_dir") == str(tmp_path)

            MockCalc.assert_called_once()


def test_load_model_fairchem_from_path(tmp_path):
    # Create dummy model file
    model_file = tmp_path / "model.pt"
    model_file.touch()

    cfg = Config()
    cfg.optimization.model_name = "ignored"
    cfg.optimization.model_path = str(model_file)
    cfg.optimization.device = "cpu"

    with patch("fairchem.core.pretrained_mlip.load_predict_unit") as mock_load:
        mock_load.return_value = MagicMock()
        with patch("fairchem.core.FAIRChemCalculator"):
            model_utils.load_model_fairchem(cfg)

            mock_load.assert_called_once()
            _, kwargs = mock_load.call_args
            # Verify path argument (might be Path object or string)
            assert str(kwargs.get("path")) == str(model_file)


def test_load_model_torchsim_from_path(tmp_path):
    model_file = tmp_path / "model.pt"
    model_file.touch()

    cfg = Config()
    cfg.optimization.model_path = str(model_file)
    cfg.optimization.device = "cpu"

    # We need to mock torch_sim.models.fairchem.FairChemModel
    # Since torch_sim is mocked in conftest, we patch the mock in sys.modules
    # or use patch string "torch_sim.models.fairchem.FairChemModel" should work
    # if sys.modules has torch_sim.models.fairchem

    with patch("torch_sim.models.fairchem.FairChemModel") as MockModel:
        model_utils.load_model_torchsim(cfg)

        MockModel.assert_called_once()
        _, kwargs = MockModel.call_args
        assert str(kwargs.get("model")) == str(model_file)


def test_hf_token_is_set_in_env(monkeypatch):
    cfg = Config()
    cfg.optimization.huggingface_token = "my_token"
    cfg.optimization.device = "cpu"

    # Ensure env var is clear initially
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with patch("fairchem.core.pretrained_mlip.get_predict_unit"), patch(
        "fairchem.core.FAIRChemCalculator"
    ):
        model_utils.load_model_fairchem(cfg)

    assert os.environ.get("HF_TOKEN") == "my_token"


def test_device_fallback_logic():
    # Test that cuda fallback to cpu if not available happens inside
    # _parse_device_string. We can mock torch.cuda.is_available

    with patch("torch.cuda.is_available", return_value=False):
        dev = model_utils._parse_device_string("cuda:0")
        assert dev == "cpu"

    with patch("torch.cuda.is_available", return_value=True):
        dev = model_utils._parse_device_string("cuda:0")
        assert dev == "cuda:0"

        backend = model_utils._backend_device_for_fairchem("cuda:0")
        assert backend == "cuda"

        backend_cpu = model_utils._backend_device_for_fairchem("cpu")
        assert backend_cpu == "cpu"
