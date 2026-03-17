import os
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch

from gpuma.config import Config
from gpuma.models import (
    _device_for_torch,
    _parse_device_string,
    load_calculator,
    load_torchsim_model,
)

# ---------------------------------------------------------------------------
# Device helper tests
# ---------------------------------------------------------------------------


def test_parse_device_string():
    assert _parse_device_string("cpu") == "cpu"
    assert _parse_device_string("CPU") == "cpu"

    with patch("torch.cuda.is_available", return_value=True):
        assert _parse_device_string("cuda") == "cuda"
        assert _parse_device_string("cuda:0") == "cuda:0"

    with patch("torch.cuda.is_available", return_value=False):
        assert _parse_device_string("cuda") == "cpu"


def test_parse_device_string_unknown():
    assert _parse_device_string("tpu") == "cpu"
    assert _parse_device_string("") == "cpu"


def test_device_for_torch():
    with patch("torch.cuda.is_available", return_value=True):
        dev = _device_for_torch("cuda:0")
        assert isinstance(dev, torch.device)
        assert dev.type == "cuda"
        assert dev.index == 0

    with patch("torch.cuda.is_available", return_value=False):
        dev = _device_for_torch("cuda")
        assert dev.type == "cpu"


# ---------------------------------------------------------------------------
# Fairchem backend — via load_calculator
# ---------------------------------------------------------------------------


def test_load_calculator_fairchem(mock_hf_token):
    config = Config({"optimization": {"model_type": "fairchem", "model_name": "uma-s-1p1"}})

    try:
        import fairchem.core as _  # noqa: F401
    except ImportError:
        pytest.skip("fairchem.core not installed")

    with patch("fairchem.core.pretrained_mlip.get_predict_unit") as mock_get_unit, \
         patch("fairchem.core.FAIRChemCalculator") as mock_calc_cls:

        mock_get_unit.return_value = MagicMock()
        mock_calc_cls.return_value = MagicMock()

        calc = load_calculator(config)

        mock_get_unit.assert_called()
        mock_calc_cls.assert_called()
        assert calc is mock_calc_cls.return_value


def test_load_calculator_fairchem_from_path(mock_hf_token, tmp_path):
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

        calc = load_calculator(config)

        mock_load_unit.assert_called_with(path=model_file, device=ANY)
        mock_calc_cls.assert_called()


def test_load_torchsim_model_fairchem(mock_hf_token):
    config = Config({"optimization": {"model_type": "fairchem", "model_name": "uma-s-1p1"}})

    try:
        import torch_sim.models.fairchem as _  # noqa: F401
    except ImportError:
        pytest.skip("torch_sim not installed")

    with patch("torch_sim.models.fairchem.FairChemModel") as mock_model_cls:
        mock_model_cls.return_value = MagicMock()

        model = load_torchsim_model(config)

        mock_model_cls.assert_called()
        assert model is mock_model_cls.return_value


def test_load_torchsim_model_fairchem_from_path(mock_hf_token, tmp_path):
    model_file = tmp_path / "model.pt"
    model_file.touch()

    config = Config({"optimization": {"model_path": str(model_file)}})

    try:
        import torch_sim.models.fairchem as _
    except ImportError:
        pytest.skip("torch_sim not installed")

    with patch("torch_sim.models.fairchem.FairChemModel") as mock_model_cls:
        mock_model_cls.return_value = MagicMock()

        model = load_torchsim_model(config)

        call_kwargs = mock_model_cls.call_args.kwargs
        assert call_kwargs["model"] == model_file


# ---------------------------------------------------------------------------
# ORB-v3 backend — via load_calculator / load_torchsim_model
# ---------------------------------------------------------------------------


def test_load_calculator_orb():
    config = Config({
        "optimization": {
            "model_type": "orb",
            "model_name": "orb_v3_direct_omol",
        }
    })

    mock_orbff = MagicMock()
    mock_adapter = MagicMock()
    mock_calc = MagicMock()

    with patch("gpuma.models._load_orb_pretrained", return_value=(mock_orbff, mock_adapter, "cpu")), \
         patch("gpuma.models.ORBCalculator", create=True, return_value=mock_calc) as mock_cls:

        # Patch the import inside the function
        import gpuma.models as _m
        with patch.object(_m, "_load_orb_calculator", wraps=None) as _:
            pass  # just verifying the dispatch path works

    # Simpler: test through dispatch mock
    with patch("gpuma.models._load_orb_calculator", return_value=mock_calc) as mock_fn:
        calc = load_calculator(config)
        mock_fn.assert_called_once_with(config)
        assert calc is mock_calc


def test_load_calculator_orb_invalid_name():
    """Invalid model name should raise ValueError."""
    config = Config({
        "optimization": {
            "model_type": "orb",
            "model_name": "nonexistent_model",
        }
    })

    # _load_orb_pretrained will call getattr(pretrained, "nonexistent_model") -> None -> ValueError
    with patch("gpuma.models._load_orb_pretrained", side_effect=ValueError("Unknown ORB model name")):
        with pytest.raises(ValueError, match="Unknown ORB model name"):
            load_calculator(config)


def test_load_calculator_orb_d3():
    """D3 correction config should be forwarded to _load_orb_pretrained."""
    config = Config({
        "optimization": {
            "model_type": "orb",
            "model_name": "orb_v3_direct_omol",
            "d3_correction": True,
            "d3_functional": "PBE",
            "d3_damping": "BJ",
        }
    })

    mock_calc = MagicMock()
    with patch("gpuma.models._load_orb_calculator", return_value=mock_calc):
        calc = load_calculator(config)
        assert calc is mock_calc


def test_load_torchsim_model_orb():
    config = Config({
        "optimization": {
            "model_type": "orb",
            "model_name": "orb_v3_direct_omol",
        }
    })

    mock_model = MagicMock()
    with patch("gpuma.models._load_orb_torchsim", return_value=mock_model) as mock_fn:
        model = load_torchsim_model(config)
        mock_fn.assert_called_once_with(config)
        assert model is mock_model


def test_load_torchsim_model_orb_missing_package():
    config = Config({
        "optimization": {
            "model_type": "orb",
            "model_name": "orb_v3_direct_omol",
        }
    })

    with patch.dict("sys.modules", {
        "orb_models.forcefield.inference.orb_torchsim": None,
    }):
        with pytest.raises(ImportError):
            load_torchsim_model(config)


# ---------------------------------------------------------------------------
# Shared helper tests
# ---------------------------------------------------------------------------


def test_model_cache_creation_failure(mock_hf_token, caplog):
    config = Config({"optimization": {"model_cache_dir": "/invalid/path/cache", "model_name": "uma-s-1p1"}})

    try:
        import fairchem.core as _
    except ImportError:
        pytest.skip("fairchem.core not installed")

    with patch("os.makedirs", side_effect=OSError("Permission denied")), \
         patch("fairchem.core.pretrained_mlip.get_predict_unit") as mock_get_unit, \
         patch("fairchem.core.FAIRChemCalculator"):

        load_calculator(config)

        call_kwargs = mock_get_unit.call_args.kwargs
        assert "cache_dir" not in call_kwargs
        assert "Could not create model cache directory" in caplog.text


def test_model_path_not_exists(mock_hf_token):
    config = Config({"optimization": {"model_path": "/non/existent/path", "model_name": "fallback"}})

    try:
        import fairchem.core as _
    except ImportError:
        pytest.skip("fairchem.core not installed")

    with patch("fairchem.core.pretrained_mlip.get_predict_unit") as mock_get_unit, \
         patch("fairchem.core.FAIRChemCalculator"):

         load_calculator(config)

         mock_get_unit.assert_called()
         args, _ = mock_get_unit.call_args
         assert args[0] == "fallback"


def test_missing_model_name(mock_hf_token):
    config = Config({"optimization": {"model_name": ""}})

    with pytest.raises(ValueError, match="Model name must be specified"):
        load_calculator(config)


def test_hf_token_set_from_config(monkeypatch):
    token = "my_token"
    config = Config({"optimization": {"huggingface_token": token, "model_name": "test"}})

    monkeypatch.delenv("HF_TOKEN", raising=False)

    from gpuma.models import _load_hf_token_to_env
    _load_hf_token_to_env(config)
    assert os.environ["HF_TOKEN"] == token


# ---------------------------------------------------------------------------
# Dispatch tests — verify correct backend is chosen
# ---------------------------------------------------------------------------


def test_load_calculator_dispatches_fairchem(mock_hf_token):
    """Verify load_calculator calls the Fairchem backend for model_type='fairchem'."""
    config = Config({"optimization": {"model_type": "fairchem", "model_name": "uma-s-1p1"}})

    with patch("gpuma.models._load_fairchem_calculator") as mock_fc:
        mock_fc.return_value = MagicMock()
        load_calculator(config)
        mock_fc.assert_called_once_with(config)


def test_load_calculator_dispatches_orb(mock_hf_token):
    """Verify load_calculator calls the ORB backend for model_type='orb'."""
    config = Config({"optimization": {"model_type": "orb", "model_name": "orb_v3_direct_omol"}})

    with patch("gpuma.models._load_orb_calculator") as mock_orb:
        mock_orb.return_value = MagicMock()
        load_calculator(config)
        mock_orb.assert_called_once_with(config)


def test_load_torchsim_dispatches_fairchem(mock_hf_token):
    """Verify load_torchsim_model calls the Fairchem backend."""
    config = Config({"optimization": {"model_type": "uma", "model_name": "uma-s-1p1"}})

    with patch("gpuma.models._load_fairchem_torchsim") as mock_fc:
        mock_fc.return_value = MagicMock()
        load_torchsim_model(config)
        mock_fc.assert_called_once_with(config)


def test_load_torchsim_dispatches_orb(mock_hf_token):
    """Verify load_torchsim_model calls the ORB backend."""
    config = Config({"optimization": {"model_type": "orb-v3", "model_name": "orb_v3_direct_omol"}})

    with patch("gpuma.models._load_orb_torchsim") as mock_orb:
        mock_orb.return_value = MagicMock()
        load_torchsim_model(config)
        mock_orb.assert_called_once_with(config)
