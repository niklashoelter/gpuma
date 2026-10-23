"""ORB-v3 backend loaders.

Provides the ASE-calculator and torch-sim model loaders for the ORB-v3
family, optionally wrapped with orb-models' native ``D3SumModel``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import Config, warn_d3_double_counting
from .base import _verify_model_name_and_cache_dir
from .device import _device_for_torch, _parse_device_string, _setup_orb_device

logger = logging.getLogger(__name__)


def _load_orb_pretrained(config: Config) -> tuple[Any, Any, str]:
    """Load a pretrained ORB model and return ``(orbff, atoms_adapter, device)``.

    If ``config.model.d3_correction`` is ``True``, the model is
    wrapped with D3 dispersion correction via ``D3SumModel``.
    """
    from orb_models.forcefield import pretrained  # type: ignore

    model_name, _ = _verify_model_name_and_cache_dir(config)
    device = _parse_device_string(str(config.technical.device))
    # orb-models default-resolves bare "cuda" to cuda:0 inside its loaders
    # and inside OrbTorchSimModel.__init__; pin the active CUDA device so
    # those internal .to("cuda") calls pick the correct GPU index.
    _setup_orb_device(device)

    loader = getattr(pretrained, model_name, None)
    if loader is None:
        raise ValueError(
            f"Unknown ORB model name {model_name!r}. "
            "Check orb_models.forcefield.pretrained for available models."
        )
    orbff, atoms_adapter = loader(device=device)

    # Optionally wrap with D3 dispersion correction
    if config.model.d3_correction:
        from orb_models.forcefield.inference.d3_model import (  # type: ignore
            AlchemiDFTD3,
            D3SumModel,
        )

        warn_d3_double_counting(config)

        functional = str(config.model.d3_functional)
        damping = str(config.model.d3_damping)
        logger.info(
            "Applying D3 dispersion correction (functional=%s, damping=%s)",
            functional,
            damping,
        )
        orbff = D3SumModel(
            orbff,
            AlchemiDFTD3(functional=functional, damping=damping).to(
                _device_for_torch(device)
            ),
        )

    return orbff, atoms_adapter, device


def _load_orb_calculator(config: Config) -> Any:
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


def _load_orb_torchsim(config: Config) -> Any:
    """Load an ``OrbTorchSimModel`` for torch-sim batch optimization."""
    try:
        from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "orb-models>=0.6.0 is required for ORB model support. "
            "Install it with: pip install gpuma"
        ) from exc

    orbff, atoms_adapter, device = _load_orb_pretrained(config)
    return OrbTorchSimModel(orbff, atoms_adapter, device=device)
