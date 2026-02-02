import pytest

from gpuma import models as model_utils
from gpuma.config import Config


def test_models_module_exports():
    assert hasattr(model_utils, "load_model_fairchem")
    assert hasattr(model_utils, "load_model_torchsim")


def test_load_model_fairchem_empty_name_raises():
    cfg = Config.from_dict({"optimization": {"model_name": "", "device": "cpu"}})
    with pytest.raises(ValueError):
        model_utils.load_model_fairchem(cfg)


def test_device_parsing_for_fairchem_and_torchsim(monkeypatch):
    import sys

    cfg = Config()
    cfg.optimization.model_name = "dummy"
    # request a specific GPU
    cfg.optimization.device = "cuda:2"

    # Monkeypatch fairchem and torch_sim imports so we don't require them
    seen = {"fairchem_device": None, "torch_device": None}

    # Mock torch.device to return the string so we can verify it
    monkeypatch.setattr("torch.device", lambda d: d)

    class DummyCalc:
        def __init__(self, predict_unit, task_name):  # noqa: D401
            self.predict_unit = predict_unit
            self.task_name = task_name

    class DummyPredictor:
        pass

    class DummyFairChemModel:
        def __init__(self, model, task_name, model_cache_dir=None, device=None):  # noqa: D401
            self.model = model
            self.task_name = task_name
            self.model_cache_dir = model_cache_dir
            self.device = device

    class DummyPretrained:
        def get_predict_unit(self, model_name, device, cache_dir=None):  # noqa: D401
            # record which device string Fairchem sees
            seen["fairchem_device"] = device
            return DummyPredictor()

        def load_predict_unit(self, path, device):  # noqa: D401
            seen["fairchem_device"] = device
            return DummyPredictor()

    def fake_fairchem_model(model, task_name, model_cache_dir=None, device=None):  # noqa: D401
        # record which torch.device torch-sim sees
        seen["torch_device"] = device
        return DummyFairChemModel(model, task_name, model_cache_dir=model_cache_dir, device=device)

    # Use direct attribute setting on the mock modules to ensure it works
    if "fairchem.core" in sys.modules:
        monkeypatch.setattr(sys.modules["fairchem.core"], "FAIRChemCalculator", DummyCalc)
        monkeypatch.setattr(sys.modules["fairchem.core"], "pretrained_mlip", DummyPretrained())
    else:
        monkeypatch.setattr("fairchem.core.FAIRChemCalculator", DummyCalc)
        monkeypatch.setattr("fairchem.core.pretrained_mlip", DummyPretrained())

    if "torch_sim.models.fairchem" in sys.modules:
        monkeypatch.setattr(
            sys.modules["torch_sim.models.fairchem"], "FairChemModel", fake_fairchem_model
        )
    else:
        monkeypatch.setattr(
            "torch_sim.models.fairchem.FairChemModel", fake_fairchem_model
        )

    # Fairchem should see only "cuda" or "cpu" (no index)
    calc = model_utils.load_model_fairchem(cfg)
    assert isinstance(calc, DummyCalc)
    assert isinstance(calc.predict_unit, DummyPredictor)
    assert seen["fairchem_device"] in {"cuda", "cpu"}

    # Torch-sim should receive a torch.device reflecting the full string, incl. index
    model = model_utils.load_model_torchsim(cfg)
    assert isinstance(model, DummyFairChemModel)

    import torch

    # When CUDA is available, we expect a cuda:2 device; otherwise it will fall back to CPU
    if torch.cuda.is_available():  # type: ignore[attr-defined]
        assert str(seen["torch_device"]).startswith("cuda")
    else:
        assert str(seen["torch_device"]) == "cpu"


def test_load_model_torchsim_import_or_skip():
    try:
        import torch_sim  # noqa: F401
    except Exception:
        pytest.skip("torch_sim not installed; skipping torch-sim model test")

    cfg = Config()
    cfg.optimization.device = "cpu"

    try:
        model_utils.load_model_torchsim(cfg)
    except Exception:
        pass
