"""Model loading utilities for GPUMA.

This module provides two public entry points for loading machine-learning
interatomic potentials:

- :func:`load_calculator` returns an ASE-compatible calculator for
  single-structure optimization (ASE/BFGS).
- :func:`load_torchsim_model` returns a torch-sim model wrapper for
  GPU-accelerated batch optimization.

Both functions inspect ``config.optimization.model_type`` and dispatch to
the appropriate backend (Fairchem UMA or ORB-v3).

Supported backends
------------------
- **Fairchem** (``model_type="fairchem"`` or ``"uma"``): Uses
  ``fairchem-core`` and ``torch-sim-atomistic``.
- **ORB-v3** (``model_type="orb"`` or ``"orb-v3"``): Uses the
  ``orb-models`` package.  Optional D3 dispersion correction can be
  enabled via ``config.optimization.d3_correction = True``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from .config import Config, resolve_model_type
from .decorators import time_it

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _parse_device_string(device: str) -> str:
    """Normalize a device string to ``"cpu"`` or ``"cuda[:N]"``.

    Falls back to ``"cpu"`` when CUDA is requested but unavailable.
    """
    dev = (device or "").strip().lower()
    if dev == "cpu":
        return "cpu"
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA device requested (%s) but CUDA is not available; falling back to 'cpu'.",
                device,
            )
            return "cpu"
        return dev
    logger.warning("Unknown device '%s'; falling back to 'cpu'.", device)
    return "cpu"


def _device_for_torch(device: str) -> torch.device:
    """Convert a config device string to a :class:`torch.device`.

    Any invalid or unavailable CUDA specification falls back to CPU.
    """
    normalized = _parse_device_string(device)
    if normalized == "cpu":
        return torch.device("cpu")
    try:
        return torch.device(normalized)
    except Exception:
        logger.warning("Invalid CUDA device '%s'; falling back to 'cpu'.", device)
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_hf_token_to_env(config: Config) -> None:
    """Set the ``HF_TOKEN`` environment variable from config if available."""
    hf_token = config.optimization.get_huggingface_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token


def _verify_model_name_and_cache_dir(config: Config) -> tuple[str, Path | None]:
    """Return ``(model_name, cache_dir)`` after validating the config values.

    Raises :class:`ValueError` if ``model_name`` is empty or missing.
    """
    opt = config.optimization
    model_name = opt.model_name
    if not model_name:
        raise ValueError("Model name must be specified in the configuration")
    model_cache_dir = Path(opt.model_cache_dir) if opt.model_cache_dir else None
    if model_cache_dir and not model_cache_dir.exists():
        try:
            os.makedirs(model_cache_dir, exist_ok=True)
        except OSError as e:
            logger.warning("Could not create model cache directory at %s: %s", model_cache_dir, e)
            model_cache_dir = None
    if model_cache_dir is not None and not model_cache_dir.exists():
        model_cache_dir = None
    return model_name, model_cache_dir


def _verify_model_path(config: Config) -> Path | None:
    """Return the model checkpoint path if it exists, else ``None``."""
    opt = config.optimization
    if opt.model_path:
        p = Path(opt.model_path)
        return p if p.exists() else None
    return None


# ---------------------------------------------------------------------------
# Public API — two dispatcher functions
# ---------------------------------------------------------------------------


@time_it
def load_calculator(config: Config):
    """Load an ASE-compatible calculator for single-structure optimization.

    Dispatches to the Fairchem or ORB-v3 backend based on
    ``config.optimization.model_type``.

    Parameters
    ----------
    config : Config
        GPUMA configuration object.

    Returns
    -------
    calculator
        An ASE calculator (``FAIRChemCalculator`` or ``ORBCalculator``).

    Raises
    ------
    ImportError
        If the required backend package is not installed.
    ValueError
        If the model name is unknown or missing.
    """
    model_type = resolve_model_type(config)
    if model_type == "orb":
        return _load_orb_calculator(config)
    return _load_fairchem_calculator(config)


@time_it
def load_torchsim_model(config: Config):
    """Load a torch-sim model wrapper for GPU-accelerated batch optimization.

    Dispatches to the Fairchem or ORB-v3 backend based on
    ``config.optimization.model_type``.

    Parameters
    ----------
    config : Config
        GPUMA configuration object.

    Returns
    -------
    model
        A torch-sim model (``FairChemModel`` or ``OrbTorchSimModel``).

    Raises
    ------
    ImportError
        If the required backend package is not installed.
    ValueError
        If the model name is unknown or missing.
    """
    model_type = resolve_model_type(config)
    if model_type == "orb":
        return _load_orb_torchsim(config)
    return _load_fairchem_torchsim(config)


# ---------------------------------------------------------------------------
# Fairchem backend
# ---------------------------------------------------------------------------


def _load_fairchem_calculator(config: Config):
    """Load a ``FAIRChemCalculator`` from a pretrained UMA model."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore

    _load_hf_token_to_env(config)
    device_str = _parse_device_string(str(config.optimization.device))
    # Fairchem only accepts "cuda" or "cpu" (no index).
    backend_device = "cuda" if device_str.startswith("cuda") else "cpu"

    model_path = _verify_model_path(config)
    if model_path:
        predictor = pretrained_mlip.load_predict_unit(path=model_path, device=backend_device)
        return FAIRChemCalculator(predict_unit=predictor, task_name="omol")

    model_name, model_cache_dir = _verify_model_name_and_cache_dir(config)
    kwargs: dict = {"device": backend_device}
    if model_cache_dir:
        kwargs["cache_dir"] = str(model_cache_dir)
    predictor = pretrained_mlip.get_predict_unit(model_name, **kwargs)
    return FAIRChemCalculator(predict_unit=predictor, task_name="omol")


def _load_fairchem_torchsim(config: Config):
    """Load a ``FairChemModel`` for torch-sim batch optimization."""
    from torch_sim.models.fairchem import FairChemModel  # type: ignore

    _load_hf_token_to_env(config)
    model_path = _verify_model_path(config)
    model_name, model_cache_dir = _verify_model_name_and_cache_dir(config)
    torch_device = _device_for_torch(str(config.optimization.device))

    if model_path:
        return FairChemModel(model=model_path, task_name="omol", device=torch_device)
    return FairChemModel(
        model=model_name,
        model_cache_dir=model_cache_dir,
        task_name="omol",
        device=torch_device,
    )


# ---------------------------------------------------------------------------
# ORB-v3 backend
# ---------------------------------------------------------------------------


def _load_orb_pretrained(config: Config):
    """Load a pretrained ORB model and return ``(orbff, atoms_adapter)``.

    If ``config.optimization.d3_correction`` is ``True``, the model is
    wrapped with D3 dispersion correction via ``D3SumModel``.
    """
    from orb_models.forcefield import pretrained  # type: ignore

    model_name, _ = _verify_model_name_and_cache_dir(config)
    device = _parse_device_string(str(config.optimization.device))

    loader = getattr(pretrained, model_name, None)
    if loader is None:
        raise ValueError(
            f"Unknown ORB model name {model_name!r}. "
            "Check orb_models.forcefield.pretrained for available models."
        )
    orbff, atoms_adapter = loader(device=device)

    # Optionally wrap with D3 dispersion correction
    use_d3 = getattr(config.optimization, "d3_correction", False)
    if use_d3:
        from orb_models.forcefield.inference.d3_model import (  # type: ignore
            AlchemiDFTD3,
            D3SumModel,
        )

        functional = str(getattr(config.optimization, "d3_functional", "PBE"))
        damping = str(getattr(config.optimization, "d3_damping", "BJ"))
        logger.info("Applying D3 dispersion correction (functional=%s, damping=%s)", functional, damping)
        orbff = D3SumModel(orbff, AlchemiDFTD3(functional=functional, damping=damping))

    return orbff, atoms_adapter, device


def _load_orb_calculator(config: Config):
    """Load an ``ORBCalculator`` from a pretrained ORB-v3 model."""
    try:
        from orb_models.forcefield.inference.calculator import ORBCalculator  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "orb-models>=0.6.0 is required for ORB model support. "
            "Install it with: pip install gpuma"
        ) from exc

    orbff, atoms_adapter, device = _load_orb_pretrained(config)
    return ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=device)


def _load_orb_torchsim(config: Config):
    """Load an ``OrbTorchSimModel`` for torch-sim batch optimization."""
    try:
        from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "orb-models>=0.6.0 is required for ORB model support. "
            "Install it with: pip install gpuma"
        ) from exc

    orbff, atoms_adapter, _ = _load_orb_pretrained(config)
    return OrbTorchSimModel(orbff, atoms_adapter)
