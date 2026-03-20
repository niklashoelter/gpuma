"""Core geometry optimization module for GPUMA.

Provides two public functions:

- :func:`optimize_single_structure` — optimize a single :class:`Structure`
  using ASE/BFGS with an ML calculator.
- :func:`optimize_structure_batch` — optimize a list of structures, either
  sequentially or via GPU-accelerated torch-sim batch optimization.

Both Fairchem UMA and ORB-v3 backends are supported; the backend is selected
automatically from the configuration.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from ase import Atoms
from ase.optimize import BFGS

from .config import Config, load_config_from_file, resolve_model_type
from .decorators import timed_block
from .logging_utils import log_optimization_summary
from .models import _parse_device_string, load_calculator, load_torchsim_model
from .structure import Structure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------


def _cache_key(config: Config) -> tuple:
    """Extract a hashable cache key from configuration parameters."""
    mdl = config.model
    tech = config.technical
    return (
        resolve_model_type(config),
        str(tech.device),
        str(mdl.model_name),
        str(mdl.model_path) if mdl.model_path else None,
        str(mdl.model_cache_dir) if mdl.model_cache_dir else None,
        str(mdl.huggingface_token) if mdl.huggingface_token else None,
        str(mdl.huggingface_token_file) if mdl.huggingface_token_file else None,
    )


def _config_from_key(key: tuple) -> Config:
    """Reconstruct a minimal :class:`Config` from a cache key tuple."""
    model_type, device, model_name, model_path, cache_dir, hf_token, hf_token_file = key
    return Config(
        {
            "model": {
                "model_type": model_type,
                "model_name": model_name,
                "model_path": model_path,
                "model_cache_dir": cache_dir,
                "huggingface_token": hf_token,
                "huggingface_token_file": hf_token_file,
            },
            "technical": {
                "device": device,
            },
        }
    )


@functools.lru_cache(maxsize=2)
def _load_calculator_cached(
    model_type: str,
    device: str,
    model_name: str,
    model_path: str | None,
    model_cache_dir: str | None,
    hf_token: str | None,
    hf_token_file: str | None,
) -> Any:
    """Cached calculator loading (hashable args required by lru_cache)."""
    cfg = _config_from_key(
        (model_type, device, model_name, model_path, model_cache_dir, hf_token, hf_token_file)
    )
    return load_calculator(cfg)


@functools.lru_cache(maxsize=2)
def _load_torchsim_cached(
    model_type: str,
    device: str,
    model_name: str,
    model_path: str | None,
    model_cache_dir: str | None,
    hf_token: str | None,
    hf_token_file: str | None,
) -> Any:
    """Cached torch-sim model loading (hashable args required by lru_cache)."""
    cfg = _config_from_key(
        (model_type, device, model_name, model_path, model_cache_dir, hf_token, hf_token_file)
    )
    return load_torchsim_model(cfg)


def _get_cached_calculator(config: Config) -> Any:
    """Return a cached ASE calculator for the given configuration."""
    return _load_calculator_cached(*_cache_key(config))


def _get_cached_torchsim_model(config: Config) -> Any:
    """Return a cached torch-sim model for the given configuration."""
    return _load_torchsim_cached(*_cache_key(config))


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------


def _resolve_force_criterion(config: Config) -> float:
    """Determine the force convergence threshold for single-structure optimization.

    Single-structure optimization only supports force convergence. If only an
    energy criterion is set, a warning is logged and the default force threshold
    (0.05 eV/A) is used.
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
            "Falling back to default force criterion (0.05)."
        )
    return 0.05


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
    return torch_sim.generate_force_convergence_fn(force_tol=0.05)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def optimize_single_structure(
    structure: Structure,
    config: Config | None = None,
    calculator: Any | None = None,
) -> Structure:
    """Optimize a single :class:`Structure` using ASE/BFGS.

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

        logger.info(
            "Starting single geometry optimization for structure with %d atoms",
            structure.n_atoms,
        )
        optimizer = BFGS(atoms, logfile=None)
        optimizer.run(fmax=fmax)
        logger.info("Optimization completed after %d steps", optimizer.get_number_of_steps())

        structure.coordinates = atoms.get_positions().tolist()
        structure.energy = float(atoms.get_potential_energy())
        return structure

    except Exception as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(f"Optimization failed: {exc}") from exc


def optimize_structure_batch(
    structures: list[Structure],
    config: Config | None = None,
) -> list[Structure]:
    """Optimize a list of structures and return them with updated coordinates.

    The optimization mode is controlled by
    ``config.optimization.batch_optimization_mode``:

    - ``"sequential"``: Each structure is optimized individually with
      ASE/BFGS using a shared calculator.
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
    list[Structure]
        Optimized structures with coordinates and energies set.

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
) -> list[Structure]:
    """Optimize structures one-by-one using ASE/BFGS with a shared calculator."""
    calculator = _get_cached_calculator(config)

    logger.info("Starting sequential optimization of %d structures", len(structures))
    results: list[Structure] = []
    for i, struct in enumerate(structures):
        try:
            optimized = optimize_single_structure(struct, config, calculator)
            results.append(optimized)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Structure %d optimization failed: %s", i + 1, exc)
            continue

    logger.info(
        "Sequential optimization completed. %d/%d successful",
        len(results),
        len(structures),
    )
    return results


def _optimize_batch(
    structures: list[Structure],
    config: Config,
) -> list[Structure]:
    """Optimize structures in parallel using torch-sim batch inference."""
    import torch
    import torch_sim
    from torch_sim.autobatching import InFlightAutoBatcher

    from .models import _device_for_torch

    logger.info("Starting batch optimization of %d structures", len(structures))

    device = _device_for_torch(config.technical.device)
    model = _get_cached_torchsim_model(config)

    # Select optimizer
    optimizer_name = str(config.optimization.batch_optimizer).strip().lower()
    optimizer = (
        torch_sim.Optimizer.fire
        if optimizer_name == "fire"
        else torch_sim.Optimizer.gradient_descent
    )

    convergence_fn = _resolve_batch_convergence(config)

    # Convert structures to ASE Atoms.  A non-zero bounding-box cell is set
    # because ORB's nvalchemiops cell-list neighbor list overflows when the
    # cell matrix is all-zeros.  PBC remains False.
    import numpy as np

    ase_structures = []
    for s in structures:
        atoms = Atoms(
            symbols=s.symbols,
            positions=s.coordinates,
            info={"charge": s.charge, "spin": s.multiplicity},
        )
        pos = np.array(s.coordinates)
        atoms.cell = np.diag(pos.max(axis=0) - pos.min(axis=0) + 20.0)
        atoms.center()
        ase_structures.append(atoms)

    batched_state = torch_sim.io.atoms_to_state(
        ase_structures,
        device=torch.device(device),
        dtype=torch.float64,
    )

    max_memory_padding = float(config.technical.max_memory_padding)
    memory_scaling_factor = float(config.technical.memory_scaling_factor)
    max_atoms_to_try = int(config.technical.max_atoms_to_try)
    steps_between_swaps = int(config.optimization.steps_between_swaps)

    # Use the model's own recommendation when it declares one (e.g. Fairchem
    # sets "n_atoms" which is accurate for fixed-neighbor models).  For models
    # that default to "n_atoms_x_density" (e.g. ORB), override to "n_edges"
    # which gives accurate estimates for diverse molecular batches.
    model_metric = getattr(model, "memory_scales_with", "n_atoms_x_density")
    memory_scales_with = "n_edges" if model_metric == "n_atoms_x_density" else model_metric

    effective_max_atoms = min(batched_state.n_atoms, max_atoms_to_try)
    with timed_block("Autobatcher setup") as t_batcher:
        batcher = InFlightAutoBatcher(
            model,
            memory_scales_with=memory_scales_with,
            memory_scaling_factor=memory_scaling_factor,
            max_memory_padding=max_memory_padding,
            max_atoms_to_try=effective_max_atoms,
        )
        # Pre-calibrate: estimate max_memory_scaler once so that both the
        # BinningAutoBatcher (FIRE init) and InFlightAutoBatcher (optimization)
        # reuse the same value without redundant GPU probing.
        batcher.load_states(batched_state)
        logger.debug(
            "Autobatcher params: memory_scales_with=%s, max_memory_scaler=%.0f, "
            "max_memory_padding=%.2f, max_atoms=%d, steps_between_swaps=%d",
            memory_scales_with,
            batcher.max_memory_scaler,
            max_memory_padding,
            effective_max_atoms,
            steps_between_swaps,
        )

    with timed_block("Batch optimization") as t_opt:
        final_state = torch_sim.optimize(
            system=batched_state,
            model=model,
            optimizer=optimizer,
            convergence_fn=convergence_fn,
            autobatcher=batcher,
            steps_between_swaps=steps_between_swaps,
        )
    logger.debug(
        "Timing breakdown — autobatcher: %.2f sec, optimization: %.2f sec",
        t_batcher.elapsed,
        t_opt.elapsed,
    )

    # Extract results
    final_atoms = final_state.to_atoms()
    results: list[Structure] = []
    for i, atoms in enumerate(final_atoms):
        struct = Structure(
            symbols=atoms.get_chemical_symbols(),
            coordinates=atoms.get_positions().tolist(),
            energy=float(final_state.energy[i].item()),
            charge=int(final_state.charge[i].item()),
            multiplicity=int(final_state.spin[i].item()),
            comment=(
                f"Optimized with model {getattr(model, 'model_name', None) or ''} in batch mode"
            ),
        )
        results.append(struct)
    return results
