"""Core geometry optimization module for GPUMA.

Provides two public functions:

- :func:`optimize_single_structure` — optimize a single :class:`Structure`
  using an ASE optimizer (FIRE, BFGS, or LBFGS) with an ML calculator.
- :func:`optimize_structure_batch` — optimize a list of structures, either
  sequentially or via GPU-accelerated torch-sim batch optimization.

The Fairchem UMA, ORB-v3 and SevenNet backends are supported; the backend is
selected automatically from the configuration.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

from ase import Atoms
from ase.optimize import BFGS, FIRE, LBFGS

from .config import (
    DEFAULT_CONFIG,
    VALID_MEMORY_SCALES_WITH,
    Config,
    load_config_from_file,
    resolve_model_type,
)
from .models import _parse_device_string, load_calculator, load_torchsim_model
from .structure import Structure
from .utils import reporting as _reporting
from .utils.decorators import timed_block
from .utils.logging_utils import log_optimization_summary

logger = logging.getLogger(__name__)

_DEFAULT_FORCE_CRITERION: float = float(
    DEFAULT_CONFIG["optimization"]["force_convergence_criterion"]
)

# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------


def _cache_key(config: Config) -> str:
    """Return a hashable key covering everything the model loaders read.

    Serializes the whole ``model`` section plus the device rather than listing
    fields one by one. The enumerated version silently omitted ``model_modal``,
    so every SevenNet run through this module loaded the checkpoint's default
    fidelity instead of the configured one, and two configs differing only in
    ``model_modal`` collided on the same cache entry. Any field added to the
    model section in future is covered here automatically.

    ``model_type`` is stored in canonical form so that aliases which resolve to
    the same backend (``"7net"``/``"sevennet"``) share one cache entry.
    """
    payload = dict(config.model.to_dict())
    payload["model_type"] = resolve_model_type(config)
    payload["__device__"] = str(config.technical.device)
    # sort_keys so key ordering in the config file cannot split the cache;
    # default=str so an unexpected non-JSON value degrades to its repr rather
    # than raising and taking the whole run down.
    return json.dumps(payload, sort_keys=True, default=str)


def _config_from_key(key: str) -> Config:
    """Rebuild the :class:`Config` that :func:`_cache_key` was built from."""
    payload = json.loads(key)
    device = payload.pop("__device__")
    return Config({"model": payload, "technical": {"device": device}})


@functools.lru_cache(maxsize=2)
def _load_calculator_cached(key: str) -> Any:
    """Cached calculator loading (hashable arg required by lru_cache)."""
    return load_calculator(_config_from_key(key))


@functools.lru_cache(maxsize=2)
def _load_torchsim_cached(key: str) -> Any:
    """Cached torch-sim model loading (hashable arg required by lru_cache)."""
    return load_torchsim_model(_config_from_key(key))


def _get_cached_calculator(config: Config) -> Any:
    """Return a cached ASE calculator for the given configuration."""
    return _load_calculator_cached(_cache_key(config))


def _get_cached_torchsim_model(config: Config) -> Any:
    """Return a cached torch-sim model for the given configuration."""
    return _load_torchsim_cached(_cache_key(config))


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------


def _resolve_force_criterion(config: Config) -> float:
    """Determine the force convergence threshold for single-structure optimization.

    Single-structure optimization only supports force convergence. If only an
    energy criterion is set, a warning is logged and the package default force
    threshold (``DEFAULT_CONFIG["optimization"]["force_convergence_criterion"]``)
    is used.
    """
    force_crit = config.optimization.force_convergence_criterion
    energy_crit = config.optimization.energy_convergence_criterion

    if force_crit is not None and energy_crit is not None:
        logger.warning(
            "Both force and energy convergence criteria given. "
            "For single structure optimization, the force criterion is used."
        )
        return float(force_crit)
    if force_crit is not None:
        return float(force_crit)
    if energy_crit is not None:
        logger.warning(
            "Energy convergence criterion requested but only force convergence "
            "is supported for single structure optimization. "
            "Falling back to default force criterion (%s).",
            _DEFAULT_FORCE_CRITERION,
        )
    return _DEFAULT_FORCE_CRITERION


def _resolve_batch_convergence(config: Config):
    """Build a torch-sim convergence function for batch optimization.

    Supports both force and energy convergence criteria. When both are
    set, force takes precedence.
    """
    import torch_sim

    force_crit = config.optimization.force_convergence_criterion
    energy_crit = config.optimization.energy_convergence_criterion

    if force_crit is not None and energy_crit is not None:
        logger.warning(
            "Both force and energy convergence criteria given. "
            "Using force convergence criterion for batch optimization."
        )
        return torch_sim.generate_force_convergence_fn(force_tol=force_crit)
    if force_crit is not None:
        return torch_sim.generate_force_convergence_fn(force_tol=force_crit)
    if energy_crit is not None:
        return torch_sim.generate_energy_convergence_fn(energy_tol=energy_crit)
    return torch_sim.generate_force_convergence_fn(force_tol=_DEFAULT_FORCE_CRITERION)


_ASE_OPTIMIZER_MAP = {
    "fire": FIRE,
    "bfgs": BFGS,
    "lbfgs": LBFGS,
}


def _resolve_ase_optimizer(config: Config) -> tuple[type, str]:
    """Return the ASE optimizer class and effective name from config.

    ``gradient_descent`` has no ASE equivalent and falls back to FIRE
    with a warning.
    """
    name = str(config.optimization.batch_optimizer).strip().lower()
    cls = _ASE_OPTIMIZER_MAP.get(name)
    if cls is not None:
        return cls, name
    if name == "gradient_descent":
        logger.warning(
            "ASE has no gradient_descent optimizer; falling back to FIRE "
            "for single-structure optimization."
        )
        return FIRE, "fire"
    logger.warning("Unknown optimizer %r; falling back to FIRE.", name)
    return FIRE, "fire"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@_reporting._instrument("mlip_optimize", "ase", share=True)
def optimize_single_structure(
    structure: Structure,
    config: Config | None = None,
    calculator: Any | None = None,
) -> Structure:
    """Optimize a single :class:`Structure` using an ASE optimizer.

    The same ``structure`` instance is returned with updated coordinates and
    energy.

    Parameters
    ----------
    structure : Structure
        Molecular structure to optimize.
    config : Config, optional
        Configuration controlling the model and convergence settings.
        Defaults to :func:`load_config_from_file` if not provided.
    calculator : optional
        Pre-loaded ASE calculator. If ``None``, one is loaded (and cached)
        from the configuration.

    Returns
    -------
    Structure
        The input structure with optimized coordinates and energy set.

    Raises
    ------
    RuntimeError
        If the optimization fails for any reason.
    """
    if config is None:
        config = load_config_from_file()

    try:
        if calculator is None:
            calculator = _get_cached_calculator(config)

        atoms = Atoms(symbols=structure.symbols, positions=structure.coordinates)
        atoms.calc = calculator
        atoms.info = {"charge": structure.charge, "spin": structure.multiplicity}

        fmax = _resolve_force_criterion(config)
        opt_cls, opt_name = _resolve_ase_optimizer(config)

        logger.info(
            "Starting single geometry optimization for structure with %d atoms "
            "(optimizer=%s)",
            structure.n_atoms,
            opt_name,
        )
        optimizer = opt_cls(atoms, logfile=None)
        converged = optimizer.run(fmax=fmax)
        steps = optimizer.get_number_of_steps()
        logger.info("Optimization completed after %d steps", steps)
        _reporting._ase_result(converged, steps, opt_name, fmax)

        structure.coordinates = atoms.get_positions().tolist()
        structure.energy = float(atoms.get_potential_energy())
        _reporting._ase_returned()
        return structure

    except Exception as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(f"Optimization failed: {exc}") from exc


@_reporting._instrument("mlip_optimize")
def optimize_structure_batch(
    structures: list[Structure],
    config: Config | None = None,
) -> list[Structure | None]:
    """Optimize a list of structures and return them with updated coordinates.

    The optimization mode is controlled by
    ``config.optimization.batch_optimization_mode``:

    - ``"sequential"``: Each structure is optimized individually with
      ASE using a shared calculator.
    - ``"batch"``: All structures are optimized together using torch-sim
      GPU-accelerated batch optimization (requires GPU).

    Parameters
    ----------
    structures : list[Structure]
        Structures to optimize.
    config : Config, optional
        Configuration object. Defaults to :func:`load_config_from_file`.

    Returns
    -------
    list[Structure | None]
        One entry per input, in input order. ``None`` marks a structure whose
        optimization failed, so a single bad geometry cannot abort the run.

        .. versionchanged::
            Previously the failures were simply left out, which silently
            shortened the list and shifted every later result onto the wrong
            input. Callers that index results against inputs, or label output
            by position, were quietly misattributing geometries. Iterating
            callers now need to skip ``None``; this matches
            :func:`gpuma.conformer_generation.embed.generate_structures`,
            which already reports per-molecule failure this way.

    Raises
    ------
    ValueError
        If structures have mismatched symbols/coordinates or are empty,
        or if the optimization mode is unknown.
    """
    if config is None:
        config = load_config_from_file()
    if not structures:
        return []

    for i, struct in enumerate(structures):
        if struct.n_atoms != len(struct.coordinates):
            raise ValueError(f"Structure {i}: symbols/coords length mismatch")
        if struct.n_atoms == 0:
            raise ValueError(f"Structure {i}: empty structure")

    on_cpu = _parse_device_string(config.technical.device) == "cpu"
    mode = str(config.optimization.batch_optimization_mode).lower()

    logger.info("Optimization device: %s", "CPU" if on_cpu else "GPU")

    with timed_block("Total optimization") as tb:
        if mode == "sequential" or on_cpu:
            if not on_cpu and mode == "batch":
                logger.warning(
                    "Batch optimization mode requires GPU, falling back to sequential mode on CPU.",
                )
            results = _optimize_sequential(structures, config)
        elif mode == "batch":
            results = _optimize_batch(structures, config)
        else:
            raise ValueError(
                f"Unknown optimization mode: {mode!r}. Use 'sequential' or 'batch' "
                "(batch requires GPU)."
            )

    log_optimization_summary(structures, results, tb.elapsed, mode, config)
    return results


# ---------------------------------------------------------------------------
# Internal optimization implementations
# ---------------------------------------------------------------------------


def _optimize_sequential(
    structures: list[Structure],
    config: Config,
) -> list[Structure | None]:
    """Optimize structures one-by-one using ASE with a shared calculator.

    Returns one entry per input, in input order, with ``None`` where the
    optimization raised. Appending only the successes -- as this used to --
    made the returned list shorter than the input and shifted every result
    after a failure onto the wrong structure, with nothing to signal it.
    """
    calculator = _get_cached_calculator(config)

    logger.info("Starting sequential optimization of %d structures", len(structures))
    results: list[Structure | None] = [None] * len(structures)
    n_ok = 0
    for i, struct in enumerate(structures):
        try:
            with _reporting._input(i):
                results[i] = optimize_single_structure(struct, config, calculator)
            n_ok += 1
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Structure %d optimization failed: %s", i + 1, exc)

    logger.info(
        "Sequential optimization completed. %d/%d successful",
        n_ok,
        len(structures),
    )
    return results


def _whole_job_memory_scaler(state: Any, model: Any, batcher: Any) -> float | None:
    """Return the job's total memory metric if one forward pass over it fits, else None."""
    import torch
    from torch_sim import autobatching as ts_ab

    if state.n_atoms > batcher.max_atoms_to_try:
        return None
    try:
        peak_gb = ts_ab.measure_model_memory_forward(state, model)
    except Exception as exc:
        if not any(msg in str(exc) for msg in batcher.oom_error_message):
            raise
        peak_gb = None
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if peak_gb is None:
        logger.info("Job does not fit in one forward pass; probing GPU memory")
        return None

    free, _ = torch.cuda.mem_get_info()
    budget_gb = (free + torch.cuda.memory_reserved()) / 1024**3
    # Keep the probe's margin: its batch sits one growth step below a size that fit.
    if peak_gb * batcher.memory_scaling_factor > batcher.max_memory_padding * budget_gb:
        logger.info(
            "Job peaks at %.2f of %.2f GB, too close to the limit for one batch; "
            "probing GPU memory",
            peak_gb,
            budget_gb,
        )
        return None
    logger.info(
        "Job fits in one batch (%.2f of %.2f GB); skipping memory probe",
        peak_gb,
        budget_gb,
    )
    return float(sum(ts_ab.calculate_memory_scalers(
        state, batcher.memory_scales_with, batcher.cutoff
    )))


def _memory_scales_with(config: Config) -> str:
    """Return the autobatcher memory metric: ``n_atoms`` for UMA, else ``n_edges``, unless set."""
    choice = str(config.technical.get("memory_scales_with", "auto") or "auto").strip().lower()
    if choice not in VALID_MEMORY_SCALES_WITH:
        raise ValueError(
            f"memory_scales_with must be one of {sorted(VALID_MEMORY_SCALES_WITH)}, "
            f"got {choice!r}"
        )
    if choice != "auto":
        return choice
    # UMA's large per-system memory makes edge counts overcharge small molecules.
    return "n_atoms" if resolve_model_type(config) == "fairchem" else "n_edges"


@_reporting._instrument("mlip_optimize", "torchsim")
def _optimize_batch(
    structures: list[Structure],
    config: Config,
) -> list[Structure | None]:
    """Optimize structures in parallel using torch-sim batch inference."""
    import torch
    import torch._inductor.config as inductor_config
    import torch_sim
    from torch_sim.autobatching import InFlightAutoBatcher
    from torch_sim.typing import SystemExtras

    from .autobatching import install_chunked_edge_scalers
    from .models import _device_for_torch

    logger.info("Starting batch optimization of %d structures", len(structures))

    device = _device_for_torch(config.technical.device)
    with timed_block("Model loading"):
        model = _get_cached_torchsim_model(config)

    # Select optimizer (validated/defaulted to "fire" by config validation)
    optimizer_name = str(config.optimization.batch_optimizer).strip().lower()
    _batch_optimizer_map = {
        "fire": torch_sim.Optimizer.fire,
        "gradient_descent": torch_sim.Optimizer.gradient_descent,
        "lbfgs": torch_sim.Optimizer.lbfgs,
        "bfgs": torch_sim.Optimizer.bfgs,
    }
    optimizer = _batch_optimizer_map.get(optimizer_name)
    if optimizer is None:
        logger.warning(
            "Unknown batch optimizer %r; falling back to FIRE.", optimizer_name
        )
        optimizer = torch_sim.Optimizer.fire
        optimizer_name = "fire"
    logger.info("Batch optimizer: %s", optimizer_name)

    convergence_fn = _resolve_batch_convergence(config)

    ase_structures = [
        Atoms(
            symbols=s.symbols,
            positions=s.coordinates,
            info={"charge": s.charge, "spin": s.multiplicity},
        )
        for s in structures
    ]

    batched_state = torch_sim.io.atoms_to_state(
        ase_structures,
        device=torch.device(device),
        dtype=torch.float64,
        system_extras_map={
            SystemExtras.CHARGE: "charge",
            SystemExtras.SPIN: "spin",
        },
    )

    max_memory_padding = float(config.technical.max_memory_padding)
    memory_scaling_factor = float(config.technical.memory_scaling_factor)
    max_atoms_to_try = int(config.technical.max_atoms_to_try)
    steps_between_swaps = int(config.technical.steps_between_swaps)

    # torch-sim measures every system's edge count over the whole state at once;
    # the chunked pass keeps that under the device's free memory.
    install_chunked_edge_scalers()

    # max_atoms_to_try bounds the device probe, not the job: capping it at the
    # job size put the ceiling below the job's own largest structure.
    memory_scales_with = _memory_scales_with(config)
    batcher = InFlightAutoBatcher(
        model,
        memory_scales_with=memory_scales_with,
        memory_scaling_factor=memory_scaling_factor,
        max_memory_padding=max_memory_padding,
        max_atoms_to_try=max_atoms_to_try,
    )
    logger.debug(
        "Autobatcher params: memory_scales_with=%s, "
        "max_memory_padding=%.2f, max_atoms_to_try=%d, steps_between_swaps=%d",
        memory_scales_with,
        max_memory_padding,
        max_atoms_to_try,
        steps_between_swaps,
    )

    # torch 2.13 tunes pointwise kernels at run time and, on a nearly full GPU, copies
    # their buffers to pinned host memory; that allocation can fail and crash the probe.
    with inductor_config.patch({"triton.autotune_pointwise": False}):
        # A job that fits runs as one batch, so the probe is skipped. Otherwise
        # torch_sim.optimize probes on the OptimState itself; no load_states() here,
        # which would pin the ceiling to the bare SimState.
        with timed_block("Memory estimation"):
            batcher.max_memory_scaler = _whole_job_memory_scaler(batched_state, model, batcher)

        with timed_block("Optimization"):
            observation = _reporting._observe_criterion(batcher, len(structures), convergence_fn)
            observer = None
            if observation is not None:
                observer, convergence_fn = observation
            try:
                final_state = torch_sim.optimize(
                    system=batched_state,
                    model=model,
                    optimizer=optimizer,
                    convergence_fn=convergence_fn,
                    autobatcher=batcher,
                    steps_between_swaps=steps_between_swaps,
                )
            except BaseException:
                _reporting._finish_criterion(
                    observer, False, config, optimizer_name, _DEFAULT_FORCE_CRITERION
                )
                raise
            _reporting._finish_criterion(
                observer, True, config, optimizer_name, _DEFAULT_FORCE_CRITERION
            )

    # Extract results
    final_atoms = final_state.to_atoms()
    # The torch-sim model wrapper does not carry a ``model_name`` attribute, so
    # resolve the name from the configuration (the source of truth used by the
    # run summary) instead of the model object, which always read "unknown".
    model_name = getattr(config.model, "model_name", None) or "unknown"
    results: list[Structure] = []
    for i, atoms in enumerate(final_atoms):
        struct = Structure(
            symbols=atoms.get_chemical_symbols(),
            coordinates=atoms.get_positions().tolist(),
            energy=float(final_state.energy[i].item()),
            charge=int(final_state.charge[i].item()),
            multiplicity=int(final_state.spin[i].item()),
            comment=f"Optimized with model {model_name} in batch mode",
        )
        results.append(struct)
        _reporting._emit("mlip_optimize", "optimization_returned", input_index=i,
                         output_index=i, optimized=True)
    return results
