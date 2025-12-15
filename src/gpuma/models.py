"""Model loading utilities for Fairchem UMA and torch-sim models."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import torch

from .config import Config
from .decorators import time_it


def _load_hf_token_to_env(config: Config) -> None:
    """Load Huggingface token from config to environment variable."""
    hf_token = config.optimization.get_huggingface_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token


def _parse_device_string(device: str) -> str:
    """Normalize a device string.

    Accepts "cpu", "cuda" or "cuda:N" (N integer), case-insensitive.
    Falls back to "cpu" if CUDA is not available.
    """
    dev = (device or "").strip().lower()
    if dev == "cpu":
        return "cpu"
    if dev.startswith("cuda"):
        # allow "cuda" or "cuda:N"
        if not torch.cuda.is_available():
            logging.warning(
                "CUDA device requested (%s) but CUDA is not available; falling back to 'cpu'.",
                device,
            )
            return "cpu"
        # torch will later validate indices; we just normalize spelling
        return dev
    # unknown device string -> fallback and warn
    logging.warning("Unknown device '%s'; falling back to 'cpu'.", device)
    return "cpu"


def _check_device(device: str) -> Literal["cuda", "cpu"]:
    """Backwards-compatible check returning only "cuda" or "cpu".

    This is used by Fairchem, which only distinguishes between CPU and CUDA.
    """
    normalized = _parse_device_string(device)
    return "cuda" if normalized.startswith("cuda") else "cpu"


def _device_for_torch(device: str) -> torch.device:
    """Return a :class:`torch.device` for torch-sim based on config string.

    - "cpu" -> torch.device("cpu")
    - "cuda" -> torch.device("cuda")
    - "cuda:N" -> torch.device("cuda:N")
    Any invalid or unavailable CUDA device falls back to CPU.
    """
    normalized = _parse_device_string(device)
    if normalized == "cpu":
        return torch.device("cpu")
    try:
        return torch.device(normalized)
    except Exception:
        logging.warning("Invalid CUDA device '%s'; falling back to 'cpu'.", device)
        return torch.device("cpu")


def _verify_model_name_and_cache_dir(config: Config) -> tuple[str, Path | None]:
    """Verify that model name is provided and return model name and cache directory."""
    opt = config.optimization
    model_name = opt.model_name
    if not model_name:
        raise ValueError("Model name must be specified in the configuration")
    model_cache_dir = Path(opt.model_cache_dir) if opt.model_cache_dir else None
    if model_cache_dir and not model_cache_dir.exists():
        try:
            os.makedirs(model_cache_dir, exist_ok=True)
        except OSError as e:
            logging.warning(f"Could not create model cache directory at {model_cache_dir}: {e}")
            model_cache_dir = None

    if model_cache_dir is not None:
        if not model_cache_dir.exists():
            model_cache_dir = None

    return model_name, model_cache_dir

def _verify_model_path(config: Config) -> Path | None:
    """Verify that model path is provided and return model name and cache directory."""
    opt = config.optimization
    model_path = opt.model_path
    if model_path:
        model_path = Path(model_path)
        if not model_path.exists():
            return None
        return model_path
    return None

@time_it
def load_model_fairchem(config: Config):
    """Load a FAIRChemCalculator using fairchem's pretrained_mlip helper."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore

    opt = config.optimization
    _load_hf_token_to_env(config)
    device = _check_device(str(opt.device).lower())
    model_path = _verify_model_path(config)

    if model_path:
        predictor = pretrained_mlip.load_predict_unit(
            path=model_path,
            device=device
        )
        calculator = FAIRChemCalculator(predict_unit=predictor, task_name="omol")
        return calculator
    model_name, model_dir = _verify_model_name_and_cache_dir(config)

    if model_dir:
        predictor = pretrained_mlip.get_predict_unit(
            model_name,
            device=device,
            cache_dir=str(model_dir),
        )
    else:
        predictor = pretrained_mlip.get_predict_unit(
            model_name,
            device=device
        )

    calculator = FAIRChemCalculator(predict_unit=predictor, task_name="omol")
    return calculator

@time_it
def load_model_torchsim(config: Config):
    """Load a torch-sim FairChemModel from name or checkpoint path."""
    from torch_sim.models.fairchem import FairChemModel  # type: ignore

    model_path = _verify_model_path(config)
    model_name, model_cache_dir = _verify_model_name_and_cache_dir(config)
    torch_device = _device_for_torch(str(config.optimization.device).lower())
    _load_hf_token_to_env(config)

    if model_path:
        model = FairChemModel(model=model_path, task_name="omol", device=torch_device)
        return model

    model = FairChemModel(
        model=model_name,
        model_cache_dir=model_cache_dir,
        task_name="omol",
        device=torch_device,
    )
    return model

