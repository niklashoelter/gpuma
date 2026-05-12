"""Model loading utilities for GPUMA.

This module provides two public entry points for loading machine-learning
interatomic potentials:

- :func:`load_calculator` returns an ASE-compatible calculator for
  single-structure optimization (ASE).
- :func:`load_torchsim_model` returns a torch-sim model wrapper for
  GPU-accelerated batch optimization.

Both functions inspect ``config.model.model_type`` and dispatch to
the appropriate backend (Fairchem UMA or ORB-v3).

Supported backends
------------------
- **Fairchem** (``model_type="fairchem"`` or ``"uma"``): Uses
  ``fairchem-core`` and ``torch-sim-atomistic``.
- **ORB-v3** (``model_type="orb"`` or ``"orb-v3"``): Uses the
  ``orb-models`` package.

DFT-D3(BJ) dispersion correction can be enabled for both backends via
``config.model.d3_correction = True``.  ORB models use orb-models'
native ``D3SumModel``; Fairchem/UMA models are layered with torch-sim's
``D3DispersionModel`` (added in torch-sim 0.6.0) via ``SumModel`` for
the batch path and via a thin ASE wrapper for the single-structure path.
Both share the same ``nvalchemiops`` GPU kernel underneath.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

# nvalchemiops 0.3.x split torch-dependent symbols out of the warp-only modules
# but orb-models 0.6.x still imports from the old paths.  Apply two shims so
# the legacy imports succeed until orb-models is updated.
import sys as _sys

import nvalchemiops.neighbors.neighbor_utils as _warp_nu

if not hasattr(_warp_nu, "get_neighbor_list_from_neighbor_matrix"):
    from nvalchemiops.torch.neighbors.neighbor_utils import (
        get_neighbor_list_from_neighbor_matrix,
    )

    _warp_nu.get_neighbor_list_from_neighbor_matrix = get_neighbor_list_from_neighbor_matrix

if "nvalchemiops.interactions.dispersion.dftd3" not in _sys.modules:
    from nvalchemiops.torch.interactions.dispersion import _dftd3

    _sys.modules["nvalchemiops.interactions.dispersion.dftd3"] = _dftd3

from .config import Config, resolve_model_type
from .decorators import time_it

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Available model names
# ---------------------------------------------------------------------------

#: Fairchem UMA model names accepted by ``fairchem.core.pretrained_mlip``.
AVAILABLE_FAIRCHEM_MODELS: tuple[str, ...] = (
    "uma-s-1p2",
    "uma-s-1p1",
    "uma-m-1p1",
)

#: ORB-v3 model names accepted by ``orb_models.forcefield.pretrained``.
#: Use the **underscored** form (e.g. ``orb_v3_direct_omol``) as
#: ``model_name`` in the configuration.
AVAILABLE_ORB_MODELS: tuple[str, ...] = (
    # ORB-v3 — omol
    "orb_v3_conservative_omol",
    "orb_v3_direct_omol",
    # ORB-v3 — omat
    "orb_v3_conservative_20_omat",
    "orb_v3_conservative_inf_omat",
    "orb_v3_direct_20_omat",
    "orb_v3_direct_inf_omat",
    # ORB-v3 — mpa
    "orb_v3_conservative_20_mpa",
    "orb_v3_conservative_inf_mpa",
    "orb_v3_direct_20_mpa",
    "orb_v3_direct_inf_mpa",
)

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _parse_device_string(device: str) -> str:
    """Normalize a device string to ``"cpu"`` or ``"cuda[:N]"``.

    Falls back to ``"cpu"`` when CUDA is requested but unavailable.
    When a specific GPU index is requested but does not exist, falls
    back to ``"cuda:0"`` with a warning.
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
        # Validate GPU index if specified
        if ":" in dev:
            try:
                idx = int(dev.split(":")[1])
            except (ValueError, IndexError):
                logger.warning(
                    "Invalid CUDA device index in '%s'; using default GPU.",
                    device,
                )
                return "cuda"
            num_gpus = torch.cuda.device_count()
            if idx >= num_gpus:
                logger.warning(
                    "Requested GPU %d (via '%s') but only %d GPU(s) available. "
                    "Falling back to cuda:0.",
                    idx,
                    device,
                    num_gpus,
                )
                return "cuda:0"
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
    except (RuntimeError, ValueError):
        logger.warning("Invalid CUDA device '%s'; falling back to 'cpu'.", device)
        return torch.device("cpu")


def _setup_fairchem_device(device: str) -> str:
    """Prepare the CUDA device for the Fairchem backend.

    Fairchem only accepts ``"cuda"`` or ``"cpu"`` — not ``"cuda:N"``.
    When a specific GPU index is requested (e.g. ``"cuda:1"``), this
    function calls :func:`torch.cuda.set_device` so that Fairchem's
    internal device resolution picks the correct GPU.

    Returns
    -------
    str
        ``"cuda"`` or ``"cpu"`` — safe to pass to Fairchem APIs.
    """
    normalized = _parse_device_string(device)
    if not normalized.startswith("cuda"):
        return "cpu"
    if ":" in normalized:
        idx = int(normalized.split(":")[1])
        torch.cuda.set_device(idx)
        logger.info("Selected GPU %d for Fairchem backend.", idx)
    return "cuda"


def _setup_orb_device(device: str) -> None:
    """Prepare the CUDA device for the ORB backend.

    ORB's pretrained loaders and ``OrbTorchSimModel`` default-resolve a
    bare ``"cuda"`` string (or no device at all) to ``cuda:0`` via
    :func:`torch.device`, regardless of what GPU index the caller
    requested. When a specific GPU index is requested (e.g. ``"cuda:1"``),
    this function calls :func:`torch.cuda.set_device` so that any later
    internal ``.to("cuda")`` calls inside orb-models pick the correct GPU.

    No-op for CPU / non-CUDA targets.
    """
    normalized = _parse_device_string(device)
    if not normalized.startswith("cuda"):
        return
    if ":" in normalized:
        idx = int(normalized.split(":")[1])
        torch.cuda.set_device(idx)
        logger.info("Selected GPU %d for ORB backend.", idx)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_hf_token_to_env(config: Config) -> None:
    """Set the ``HF_TOKEN`` environment variable from config if available."""
    hf_token = config.model.get_huggingface_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token


def _verify_model_name_and_cache_dir(config: Config) -> tuple[str, Path | None]:
    """Return ``(model_name, cache_dir)`` after validating the config values.

    Raises :class:`ValueError` if ``model_name`` is empty or missing.
    """
    opt = config.model
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
    opt = config.model
    if opt.model_path:
        p = Path(opt.model_path)
        return p if p.exists() else None
    return None


# ---------------------------------------------------------------------------
# D3 dispersion correction (Fairchem)
# ---------------------------------------------------------------------------


def _build_d3_dispersion_model(
    config: Config,
    device: torch.device,
    dtype: torch.dtype,
    *,
    compute_stress: bool,
):
    """Construct a torch-sim ``D3DispersionModel`` from the config.

    Reuses orb-models' bundled D3 reference-data file and BJ damping
    parameter table so we don't duplicate them inside gpuma.
    """
    from orb_models.forcefield.inference.d3_model import AlchemiDFTD3  # type: ignore
    from torch_sim.models.dispersion import D3DispersionModel  # type: ignore

    functional = str(config.model.d3_functional)
    damping = str(config.model.d3_damping)
    coeffs = AlchemiDFTD3.get_d3_coefficients(functional, damping)
    d3_params = AlchemiDFTD3.load_d3_parameters().to(device=device, dtype=dtype)
    logger.info(
        "Applying D3 dispersion correction to Fairchem (functional=%s, damping=%s)",
        functional,
        damping,
    )
    return D3DispersionModel(
        a1=coeffs["a1"],
        a2=coeffs["a2"],
        s8=coeffs["s8"],
        s6=coeffs["s6"],
        d3_params=d3_params,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=compute_stress,
    )


class _FairchemD3Calculator:
    """Thin ASE-style wrapper that adds D3 corrections to a Fairchem calculator.

    We delegate to the underlying ``FAIRChemCalculator`` for the ML
    energy/forces and add the ``D3DispersionModel`` contributions on top.
    Behaves like an ASE calculator for single-structure use.
    """

    def __init__(self, fairchem_calc: Any, d3_model: Any, device: torch.device) -> None:
        self._fairchem = fairchem_calc
        self._d3_model = d3_model
        self._device = device
        self.implemented_properties = ("energy", "forces")
        self.results: dict[str, Any] = {}

    def calculate(
        self,
        atoms: Any = None,
        properties: tuple[str, ...] = ("energy", "forces"),
        system_changes: Any = None,
    ) -> None:
        """Run Fairchem then add D3 contributions to energy and forces."""
        import numpy as np
        from ase.calculators.calculator import all_changes
        from torch_sim.io import atoms_to_state  # type: ignore

        target = atoms if atoms is not None else getattr(self._fairchem, "atoms", None)
        if target is None:
            raise ValueError("FairchemD3Calculator.calculate requires an Atoms object")

        self._fairchem.calculate(
            atoms=target,
            properties=list(properties),
            system_changes=system_changes or all_changes,
        )
        e_ml = float(self._fairchem.results["energy"])
        f_ml = np.asarray(self._fairchem.results["forces"], dtype=float)

        state = atoms_to_state(target, device=self._device, dtype=self._d3_model.dtype)
        with torch.no_grad():
            d3_out = self._d3_model.forward(state)
        e_d3 = float(d3_out["energy"][0].item())
        f_d3 = d3_out["forces"].detach().cpu().numpy()

        self.results = {"energy": e_ml + e_d3, "forces": f_ml + f_d3}

    def get_potential_energy(
        self, atoms: Any = None, force_consistent: bool = False
    ) -> float:
        """ASE-style accessor: trigger calculation and return the energy."""
        self.calculate(atoms=atoms, properties=("energy", "forces"))
        return float(self.results["energy"])

    def get_forces(self, atoms: Any = None):
        """ASE-style accessor: trigger calculation and return forces."""
        self.calculate(atoms=atoms, properties=("energy", "forces"))
        return self.results["forces"]

    def get_property(self, name: str, atoms: Any = None, allow_calculation: bool = True):
        """ASE-style accessor for a single property."""
        if not allow_calculation and name not in self.results:
            return None
        self.calculate(atoms=atoms, properties=("energy", "forces"))
        return self.results.get(name)

    def calculation_required(self, atoms: Any, properties) -> bool:
        """Always recompute; we don't cache against atoms identity here."""
        return True


# ---------------------------------------------------------------------------
# Public API — two dispatcher functions
# ---------------------------------------------------------------------------


@time_it
def load_calculator(config: Config):
    """Load an ASE-compatible calculator for single-structure optimization.

    Dispatches to the Fairchem or ORB-v3 backend based on
    ``config.model.model_type``.

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
    ``config.model.model_type``.

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
        predictor = pretrained_mlip.load_predict_unit(path=model_path, device=backend_device)
        calc = FAIRChemCalculator(predict_unit=predictor, task_name="omol")
    else:
        model_name, model_cache_dir = _verify_model_name_and_cache_dir(config)
        kwargs: dict = {"device": backend_device}
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
    model_name, model_cache_dir = _verify_model_name_and_cache_dir(config)
    # Fairchem internally only accepts "cuda" or "cpu"; _setup_fairchem_device
    # calls torch.cuda.set_device(N) when a specific GPU is requested so that
    # Fairchem's internal device resolution picks the correct GPU.
    backend_device = _setup_fairchem_device(str(config.technical.device))
    torch_device = torch.device(backend_device)

    if model_path:
        uma_model = FairChemModel(model=model_path, task_name="omol", device=torch_device)
    else:
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


# ---------------------------------------------------------------------------
# ORB-v3 backend
# ---------------------------------------------------------------------------


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
