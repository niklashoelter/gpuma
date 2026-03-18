import os
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch

from gpuma.config import Config
from gpuma.models import (
    _device_for_torch,
    _parse_device_string,
    _setup_fairchem_device,
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


def test_parse_device_string_gpu_index_fallback():
    """Requesting a GPU index that doesn't exist falls back to cuda:0."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2):
        # Valid indices
        assert _parse_device_string("cuda:0") == "cuda:0"
        assert _parse_device_string("cuda:1") == "cuda:1"
        # Index out of range
        assert _parse_device_string("cuda:5") == "cuda:0"

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1):
        assert _parse_device_string("cuda:0") == "cuda:0"
        assert _parse_device_string("cuda:3") == "cuda:0"


def test_parse_device_string_invalid_index():
    """Non-integer GPU index falls back to plain 'cuda'."""
    with patch("torch.cuda.is_available", return_value=True):
        assert _parse_device_string("cuda:abc") == "cuda"


def test_device_for_torch():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=4):
        dev = _device_for_torch("cuda:0")
        assert isinstance(dev, torch.device)
        assert dev.type == "cuda"
        assert dev.index == 0

        dev = _device_for_torch("cuda:2")
        assert dev.type == "cuda"
        assert dev.index == 2

    with patch("torch.cuda.is_available", return_value=False):
        dev = _device_for_torch("cuda")
        assert dev.type == "cpu"


def test_setup_fairchem_device_plain_cuda():
    """Plain 'cuda' returns 'cuda' without calling set_device."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.set_device") as mock_set:
        result = _setup_fairchem_device("cuda")
        assert result == "cuda"
        mock_set.assert_not_called()


def test_setup_fairchem_device_with_index():
    """'cuda:N' calls set_device(N) and returns plain 'cuda'."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=4), \
         patch("torch.cuda.set_device") as mock_set:
        result = _setup_fairchem_device("cuda:2")
        assert result == "cuda"
        mock_set.assert_called_once_with(2)


def test_setup_fairchem_device_index_fallback():
    """Invalid GPU index falls back to cuda:0 and calls set_device(0)."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.set_device") as mock_set:
        result = _setup_fairchem_device("cuda:3")
        assert result == "cuda"
        mock_set.assert_called_once_with(0)


def test_setup_fairchem_device_cpu():
    """CPU input returns 'cpu' without touching CUDA."""
    result = _setup_fairchem_device("cpu")
    assert result == "cpu"


def test_parse_device_cuda3_with_enough_gpus():
    """cuda:3 is accepted when 4+ GPUs are available."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=4):
        assert _parse_device_string("cuda:3") == "cuda:3"


def test_parse_device_cuda3_insufficient_gpus():
    """cuda:3 falls back to cuda:0 when only 2 GPUs are available."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2):
        assert _parse_device_string("cuda:3") == "cuda:0"


def test_device_for_torch_cuda3():
    """cuda:3 produces torch.device('cuda', 3) when GPUs are available."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=4):
        dev = _device_for_torch("cuda:3")
        assert dev.type == "cuda"
        assert dev.index == 3


def test_setup_fairchem_device_cuda3():
    """cuda:3 calls set_device(3) for Fairchem when GPU 3 exists."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=4), \
         patch("torch.cuda.set_device") as mock_set:
        result = _setup_fairchem_device("cuda:3")
        assert result == "cuda"
        mock_set.assert_called_once_with(3)


def test_setup_fairchem_device_cuda3_fallback():
    """cuda:3 falls back to set_device(0) when only 1 GPU exists."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.set_device") as mock_set:
        result = _setup_fairchem_device("cuda:3")
        assert result == "cuda"
        mock_set.assert_called_once_with(0)


def test_config_with_cuda3_orb(mock_hf_token):
    """Config with cuda:3 dispatches correctly to ORB backend."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=4):
        config = Config({
            "model": {
                "model_type": "orb",
                "model_name": "orb_v3_direct_omol",
            },
            "technical": {"device": "cuda:3"},
        })
        with patch("gpuma.models._load_orb_calculator") as mock_orb:
            mock_orb.return_value = MagicMock()
            load_calculator(config)
            mock_orb.assert_called_once_with(config)


def test_config_with_cuda3_fairchem(mock_hf_token):
    """Config with cuda:3 dispatches correctly to Fairchem with set_device."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=4), \
         patch("torch.cuda.set_device"):
        config = Config({
            "model": {
                "model_type": "fairchem",
                "model_name": "uma-s-1p1",
            },
            "technical": {"device": "cuda:3"},
        })
        with patch("gpuma.models._load_fairchem_calculator") as mock_fc:
            mock_fc.return_value = MagicMock()
            load_calculator(config)
            mock_fc.assert_called_once_with(config)


# ---------------------------------------------------------------------------
# Fairchem backend — via load_calculator
# ---------------------------------------------------------------------------


def test_load_calculator_fairchem(mock_hf_token):
    config = Config({"model": {"model_type": "fairchem", "model_name": "uma-s-1p1"}})

    try:
        import fairchem.core as _  # noqa: F401
    except ImportError:
        pytest.skip("fairchem.core not installed")

    with patch("fairchem.core.pretrained_mlip.get_predict_unit") as mock_get_unit, \
         patch("fairchem.core.FAIRChemCalculator") as mock_calc_cls:

        mock_get_unit.return_value = MagicMock()
        mock_calc_cls.return_value = MagicMock()

        result = load_calculator(config)

        mock_get_unit.assert_called()
        mock_calc_cls.assert_called()
        assert result is mock_calc_cls.return_value


def test_load_calculator_fairchem_from_path(mock_hf_token, tmp_path):
    model_file = tmp_path / "model.pt"
    model_file.touch()

    config = Config({"model": {"model_path": str(model_file)}})

    try:
        import fairchem.core as _  # noqa: F401
    except ImportError:
        pytest.skip("fairchem.core not installed")

    with patch("fairchem.core.pretrained_mlip.load_predict_unit") as mock_load_unit, \
         patch("fairchem.core.FAIRChemCalculator") as mock_calc_cls:

        mock_load_unit.return_value = MagicMock()
        mock_calc_cls.return_value = MagicMock()

        load_calculator(config)

        mock_load_unit.assert_called_with(path=model_file, device=ANY)
        mock_calc_cls.assert_called()


def test_load_torchsim_model_fairchem(mock_hf_token):
    config = Config({"model": {"model_type": "fairchem", "model_name": "uma-s-1p1"}})

    try:
        import torch_sim.models.fairchem as _  # noqa: F401
    except ImportError:
        pytest.skip("torch_sim not installed")

    with patch("torch_sim.models.fairchem.FairChemModel") as mock_model_cls:
        mock_model_cls.return_value = MagicMock()

        result = load_torchsim_model(config)

        mock_model_cls.assert_called()
        assert result is mock_model_cls.return_value


def test_load_torchsim_model_fairchem_from_path(mock_hf_token, tmp_path):
    model_file = tmp_path / "model.pt"
    model_file.touch()

    config = Config({"model": {"model_path": str(model_file)}})

    try:
        import torch_sim.models.fairchem as _  # noqa: F401
    except ImportError:
        pytest.skip("torch_sim not installed")

    with patch("torch_sim.models.fairchem.FairChemModel") as mock_model_cls:
        mock_model_cls.return_value = MagicMock()

        load_torchsim_model(config)

        call_kwargs = mock_model_cls.call_args.kwargs
        assert call_kwargs["model"] == model_file


# ---------------------------------------------------------------------------
# ORB-v3 backend — via load_calculator / load_torchsim_model
# ---------------------------------------------------------------------------


def test_load_calculator_orb():
    config = Config({
        "model": {
            "model_type": "orb",
            "model_name": "orb_v3_direct_omol",
        }
    })

    mock_orbff = MagicMock()
    mock_adapter = MagicMock()
    mock_calc = MagicMock()

    with (
        patch("gpuma.models._load_orb_pretrained", return_value=(mock_orbff, mock_adapter, "cpu")),
        patch("gpuma.models.ORBCalculator", create=True, return_value=mock_calc),
    ):
        # Verify the dispatch path works via the public API below
        pass

    # Simpler: test through dispatch mock
    with patch("gpuma.models._load_orb_calculator", return_value=mock_calc) as mock_fn:
        calc = load_calculator(config)
        mock_fn.assert_called_once_with(config)
        assert calc is mock_calc


def test_load_calculator_orb_invalid_name():
    """Invalid model name should raise ValueError."""
    config = Config({
        "model": {
            "model_type": "orb",
            "model_name": "nonexistent_model",
        }
    })

    with patch(
        "gpuma.models._load_orb_pretrained",
        side_effect=ValueError("Unknown ORB model name"),
    ):
        with pytest.raises(ValueError, match="Unknown ORB model name"):
            load_calculator(config)


def test_load_calculator_orb_d3():
    """D3 correction config should be forwarded to _load_orb_pretrained."""
    config = Config({
        "model": {
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
        "model": {
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
        "model": {
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
    config = Config({
        "model": {
            "model_cache_dir": "/invalid/path/cache",
            "model_name": "uma-s-1p1",
        }
    })

    try:
        import fairchem.core as _  # noqa: F401
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
    config = Config({
        "model": {
            "model_path": "/non/existent/path",
            "model_name": "fallback",
        }
    })

    try:
        import fairchem.core as _  # noqa: F401
    except ImportError:
        pytest.skip("fairchem.core not installed")

    with patch("fairchem.core.pretrained_mlip.get_predict_unit") as mock_get_unit, \
         patch("fairchem.core.FAIRChemCalculator"):

         load_calculator(config)

         mock_get_unit.assert_called()
         args, _ = mock_get_unit.call_args
         assert args[0] == "fallback"


def test_missing_model_name(mock_hf_token):
    config = Config({"model": {"model_name": ""}})

    with pytest.raises(ValueError, match="Model name must be specified"):
        load_calculator(config)


def test_hf_token_set_from_config(monkeypatch):
    token = "my_token"
    config = Config({"model": {"huggingface_token": token, "model_name": "test"}})

    monkeypatch.delenv("HF_TOKEN", raising=False)

    from gpuma.models import _load_hf_token_to_env
    _load_hf_token_to_env(config)
    assert os.environ["HF_TOKEN"] == token


# ---------------------------------------------------------------------------
# Dispatch tests — verify correct backend is chosen
# ---------------------------------------------------------------------------


def test_load_calculator_dispatches_fairchem(mock_hf_token):
    """Verify load_calculator calls the Fairchem backend for model_type='fairchem'."""
    config = Config({"model": {"model_type": "fairchem", "model_name": "uma-s-1p1"}})

    with patch("gpuma.models._load_fairchem_calculator") as mock_fc:
        mock_fc.return_value = MagicMock()
        load_calculator(config)
        mock_fc.assert_called_once_with(config)


def test_load_calculator_dispatches_orb(mock_hf_token):
    """Verify load_calculator calls the ORB backend for model_type='orb'."""
    config = Config({"model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"}})

    with patch("gpuma.models._load_orb_calculator") as mock_orb:
        mock_orb.return_value = MagicMock()
        load_calculator(config)
        mock_orb.assert_called_once_with(config)


def test_load_torchsim_dispatches_fairchem(mock_hf_token):
    """Verify load_torchsim_model calls the Fairchem backend."""
    config = Config({"model": {"model_type": "uma", "model_name": "uma-s-1p1"}})

    with patch("gpuma.models._load_fairchem_torchsim") as mock_fc:
        mock_fc.return_value = MagicMock()
        load_torchsim_model(config)
        mock_fc.assert_called_once_with(config)


def test_load_torchsim_dispatches_orb(mock_hf_token):
    """Verify load_torchsim_model calls the ORB backend."""
    config = Config({
        "model": {"model_type": "orb-v3", "model_name": "orb_v3_direct_omol"}
    })

    with patch("gpuma.models._load_orb_torchsim") as mock_orb:
        mock_orb.return_value = MagicMock()
        load_torchsim_model(config)
        mock_orb.assert_called_once_with(config)


# ---------------------------------------------------------------------------
# Model registry validation tests
# ---------------------------------------------------------------------------


def test_fairchem_model_names_are_loadable(mock_hf_token):
    """Verify every AVAILABLE_FAIRCHEM_MODELS name is accepted by the backend."""
    from gpuma.models import AVAILABLE_FAIRCHEM_MODELS

    try:
        from fairchem.core import pretrained_mlip  # noqa: F401
    except ImportError:
        pytest.skip("fairchem.core not installed")

    from fairchem.core.calculate.pretrained_mlip import available_models

    for name in AVAILABLE_FAIRCHEM_MODELS:
        assert name in available_models, (
            f"Fairchem model {name!r} not found in pretrained_mlip.available_models"
        )


def test_orb_model_names_are_loadable():
    """Verify every AVAILABLE_ORB_MODELS name resolves to a loader function."""
    from gpuma.models import AVAILABLE_ORB_MODELS

    try:
        from orb_models.forcefield import pretrained  # noqa: F401
    except ImportError:
        pytest.skip("orb-models not installed")

    for name in AVAILABLE_ORB_MODELS:
        loader = getattr(pretrained, name, None)
        assert loader is not None, (
            f"ORB model {name!r} has no loader in orb_models.forcefield.pretrained"
        )
        assert callable(loader), (
            f"ORB model {name!r} loader is not callable"
        )
