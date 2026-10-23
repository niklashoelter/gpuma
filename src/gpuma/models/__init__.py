"""Model loading utilities for GPUMA.

This package provides two public entry points for loading machine-learning
interatomic potentials:

- :func:`load_calculator` returns an ASE-compatible calculator for
  single-structure optimization (ASE).
- :func:`load_torchsim_model` returns a torch-sim model wrapper for
  GPU-accelerated batch optimization.

Both functions inspect ``config.model.model_type`` and dispatch to
the appropriate backend (Fairchem UMA, ORB-v3, or SevenNet).

Supported backends
------------------
- **Fairchem** (``model_type="fairchem"`` or ``"uma"``): Uses
  ``fairchem-core`` and ``torch-sim-atomistic`` (:mod:`gpuma.models.fairchem`).
- **ORB-v3** (``model_type="orb"`` or ``"orb-v3"``): Uses the
  ``orb-models`` package (:mod:`gpuma.models.orb`).
- **SevenNet** (``model_type="sevennet"`` or ``"7net"``): Uses the
  ``sevenn`` package's ``sevenn.torchsim``/``sevenn.calculator``
  integrations (:mod:`gpuma.models.sevennet`).

DFT-D3(BJ) dispersion correction can be enabled for all backends via
``config.model.d3_correction = True``, except for ``uma-s-1p2p1-ft-d4rev``,
which already includes dispersion.  ORB models use
orb-models' native ``D3SumModel``; Fairchem/UMA models are layered with
torch-sim's ``D3DispersionModel`` (added in torch-sim 0.6.0) via
``SumModel`` for the batch path and via a thin ASE wrapper for the
single-structure path (:mod:`gpuma.models.dispersion`). Both share the same
``nvalchemiops`` GPU kernel underneath. SevenNet ships its own D3
implementation (``SevenNetD3Model`` for the batch path,
``SevenNetD3Calculator`` for the single-structure path), which gpuma reuses
when ``d3_correction`` is enabled.

Notes on SevenNet
-----------------
SevenNet is primarily a materials (periodic) potential.  It has no
charge/spin channel, so a :class:`~gpuma.structure.Structure`'s ``charge``
and ``multiplicity`` are ignored by SevenNet predictions.  Multi-modal
checkpoints (``7net-mf-ompa``, ``7net-omni``) expose molecular fidelities
(e.g. ``omol25_high``, ``spice``) selected via ``config.model.model_modal``.
SevenNet only supports ``float32``; because gpuma's batch pipeline runs in
``float64``, the torch-sim model is wrapped in ``sevenn``'s
``Float64Wrapper`` so state tensors are cast around the model.
"""

from __future__ import annotations

import sys as _sys

# nvalchemiops 0.3.x split torch-dependent symbols out of the warp-only modules
# but orb-models 0.6.x still imports from the old paths.  Apply two shims so
# the legacy imports succeed until orb-models is updated.  This must run at
# package import time, before any orb-models import, so it lives here in the
# package __init__ rather than in a lazily-imported backend module.
import nvalchemiops.neighbors.neighbor_utils as _warp_nu

if not hasattr(_warp_nu, "get_neighbor_list_from_neighbor_matrix"):
    from nvalchemiops.torch.neighbors.neighbor_utils import (
        get_neighbor_list_from_neighbor_matrix,
    )

    _warp_nu.get_neighbor_list_from_neighbor_matrix = get_neighbor_list_from_neighbor_matrix

if "nvalchemiops.interactions.dispersion.dftd3" not in _sys.modules:
    from nvalchemiops.torch.interactions.dispersion import _dftd3

    _sys.modules["nvalchemiops.interactions.dispersion.dftd3"] = _dftd3

from ..config import Config, resolve_model_type
from ..utils.decorators import time_it
from .base import (
    AVAILABLE_FAIRCHEM_MODELS,
    AVAILABLE_ORB_MODELS,
    AVAILABLE_SEVENNET_MODELS,
)
# Re-exported for backward compatibility: these device helpers were public
# attributes of the old flat ``gpuma.models`` module and are imported directly
# by the test-suite and by ``gpuma.optimizer``.
from .device import (  # noqa: F401
    _device_for_torch,
    _parse_device_string,
    _setup_fairchem_device,
    _setup_orb_device,
    _setup_sevennet_device,
)
from .fairchem import _load_fairchem_calculator, _load_fairchem_torchsim
from .orb import _load_orb_calculator, _load_orb_torchsim
from .sevennet import _load_sevennet_calculator, _load_sevennet_torchsim

__all__ = [
    "AVAILABLE_FAIRCHEM_MODELS",
    "AVAILABLE_ORB_MODELS",
    "AVAILABLE_SEVENNET_MODELS",
    "load_calculator",
    "load_torchsim_model",
]


@time_it
def load_calculator(config: Config):
    """Load an ASE-compatible calculator for single-structure optimization.

    Dispatches to the Fairchem, ORB-v3, or SevenNet backend based on
    ``config.model.model_type``.

    Parameters
    ----------
    config : Config
        GPUMA configuration object.

    Returns
    -------
    calculator
        An ASE calculator (``FAIRChemCalculator``, ``ORBCalculator``, or
        ``SevenNetCalculator``).

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
    if model_type == "sevennet":
        return _load_sevennet_calculator(config)
    return _load_fairchem_calculator(config)


@time_it
def load_torchsim_model(config: Config):
    """Load a torch-sim model wrapper for GPU-accelerated batch optimization.

    Dispatches to the Fairchem, ORB-v3, or SevenNet backend based on
    ``config.model.model_type``.

    Parameters
    ----------
    config : Config
        GPUMA configuration object.

    Returns
    -------
    model
        A torch-sim model (``FairChemModel``, ``OrbTorchSimModel``, or a
        ``Float64Wrapper``-wrapped SevenNet model).

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
    if model_type == "sevennet":
        return _load_sevennet_torchsim(config)
    return _load_fairchem_torchsim(config)
