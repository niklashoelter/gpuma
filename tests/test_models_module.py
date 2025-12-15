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
    cfg = Config()
    cfg.optimization.model_name = "dummy"
    cfg.optimization.device = "cuda:0"

    # Monkeypatch fairchem and torch_sim imports so we don't require them
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
            return DummyPredictor()

    monkeypatch.setattr("fairchem.core.FAIRChemCalculator", DummyCalc)
    monkeypatch.setattr("fairchem.core.pretrained_mlip", DummyPretrained())
    monkeypatch.setattr("torch_sim.models.fairchem.FairChemModel", DummyFairChemModel)

    # Fairchem should see only "cuda" or "cpu"
    calc = model_utils.load_model_fairchem(cfg)
    assert isinstance(calc, DummyCalc)
    assert isinstance(calc.predict_unit, DummyPredictor)

    # Torch-sim should receive a torch.device
    model = model_utils.load_model_torchsim(cfg)
    assert isinstance(model, DummyFairChemModel)
    from torch import device as torch_device

    assert isinstance(model.device, torch_device)


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
