"""Fairchem UMA backend loaders.

Provides the ASE-calculator and torch-sim model loaders for the Fairchem
UMA family, optionally layered with DFT-D3(BJ) dispersion.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..config import Config
from .base import (
    _load_hf_token_to_env,
    _verify_model_name_and_cache_dir,
    _verify_model_path,
)
from .device import _device_for_torch, _setup_fairchem_device
from .dispersion import _FairchemD3Calculator, _build_d3_dispersion_model

logger = logging.getLogger(__name__)

# fairchem's default compiles for one composition and falls back at the next molecule.
_INFERENCE_SETTINGS = "batch"


def _load_fairchem_calculator(config: Config) -> Any:
    """Load a ``FAIRChemCalculator`` from a pretrained UMA model.

    When ``config.model.d3_correction`` is True the calculator is wrapped
    with :class:`_FairchemD3Calculator`, which adds DFT-D3(BJ) energy and
    force contributions on top of every prediction.
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore

    _load_hf_token_to_env(config)
    backend_device = _setup_fairchem_device(str(config.technical.device))

    model_path = _verify_model_path(config)
    if model_path:
        predictor = pretrained_mlip.load_predict_unit(
            path=model_path, device=backend_device, inference_settings=_INFERENCE_SETTINGS
        )
        calc = FAIRChemCalculator(predict_unit=predictor, task_name="omol")
    else:
        model_name, model_cache_dir = _verify_model_name_and_cache_dir(config)
        kwargs: dict = {"device": backend_device, "inference_settings": _INFERENCE_SETTINGS}
        if model_cache_dir:
            kwargs["cache_dir"] = str(model_cache_dir)
        predictor = pretrained_mlip.get_predict_unit(model_name, **kwargs)
        calc = FAIRChemCalculator(predict_unit=predictor, task_name="omol")

    if config.model.d3_correction:
        torch_device = _device_for_torch(str(config.technical.device))
        d3_model = _build_d3_dispersion_model(
            config, torch_device, torch.float64, compute_stress=False
        )
        return _FairchemD3Calculator(calc, d3_model, torch_device)
    return calc


def _load_fairchem_torchsim(config: Config) -> Any:
    """Load a ``FairChemModel`` for torch-sim batch optimization.

    When ``config.model.d3_correction`` is True the model is wrapped with
    :class:`torch_sim.models.interface.SumModel` to add DFT-D3(BJ)
    contributions on top of UMA predictions.
    """
    from torch_sim.models.fairchem import FairChemModel  # type: ignore

    _load_hf_token_to_env(config)
    model_path = _verify_model_path(config)
    # Fairchem internally only accepts "cuda" or "cpu"; _setup_fairchem_device
    # calls torch.cuda.set_device(N) when a specific GPU is requested so that
    # Fairchem's internal device resolution picks the correct GPU.
    backend_device = _setup_fairchem_device(str(config.technical.device))
    torch_device = torch.device(backend_device)

    if model_path:
        # model_name is deliberately not resolved on this branch.
        # _verify_model_name_and_cache_dir raises when model_name is empty, so
        # calling it unconditionally rejected a config that supplies only a
        # local checkpoint -- which _load_fairchem_calculator accepts. The same
        # config then worked in sequential mode and failed in batch mode.
        uma_model = FairChemModel(model=model_path, task_name="omol", device=torch_device)
    else:
        model_name, model_cache_dir = _verify_model_name_and_cache_dir(config)
        uma_model = FairChemModel(
            model=model_name,
            model_cache_dir=model_cache_dir,
            task_name="omol",
            device=torch_device,
        )

    if config.model.d3_correction:
        from torch_sim.models.interface import SumModel  # type: ignore

        d3_model = _build_d3_dispersion_model(
            config, torch_device, uma_model.dtype, compute_stress=uma_model.compute_stress
        )
        return SumModel(uma_model, d3_model)
    return uma_model
