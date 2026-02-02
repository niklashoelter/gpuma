"""Core geometry optimization module using Fairchem UMA models for GPUMA.

This module contains optimization logic for single structures and batches of
structures (e.g., conformer ensembles).
"""

from __future__ import annotations

import functools
import logging

from ase import Atoms
from ase.optimize import BFGS
from fairchem.core import FAIRChemCalculator

from .config import Config, load_config_from_file
from .decorators import time_it
from .models import (
    _device_for_torch,
    _parse_device_string,
    load_model_fairchem,
    load_model_torchsim,
)
from .structure import Structure

logger = logging.getLogger(__name__)

_CALCULATOR_CACHE: dict[tuple, FAIRChemCalculator] = {}


@functools.lru_cache(maxsize=1)
def _load_calculator_impl(
    device: str,
    model_name: str,
    model_path: str | None,
    model_cache_dir: str | None,
    huggingface_token: str | None,
    huggingface_token_file: str | None,
) -> FAIRChemCalculator:
    """Cached implementation of calculator loading."""
    # Reconstruct a temporary config for load_model_fairchem
    temp_config = Config(
        {
            "optimization": {
                "device": device,
                "model_name": model_name,
                "model_path": model_path,
                "model_cache_dir": model_cache_dir,
                "huggingface_token": huggingface_token,
                "huggingface_token_file": huggingface_token_file,
            }
        }
    )
    return load_model_fairchem(temp_config)


def _get_cached_calculator(config: Config) -> FAIRChemCalculator:
    """Retrieve or load a calculator based on configuration parameters."""
    opt = config.optimization
    # Cache key based on parameters that affect model loading
    return _load_calculator_impl(
        str(opt.device),
        str(opt.model_name),
        str(opt.model_path) if opt.model_path else None,
        str(opt.model_cache_dir) if opt.model_cache_dir else None,
        str(opt.huggingface_token) if opt.huggingface_token else None,
        str(opt.huggingface_token_file) if opt.huggingface_token_file else None,
    )


@functools.lru_cache(maxsize=1)
def _load_model_torchsim_impl(
    device: str,
    model_name: str,
    model_path: str | None,
    model_cache_dir: str | None,
    hf_token: str | None,
    hf_token_file: str | None,
):
    """Load a torch-sim model with caching support."""
    # Reconstruct minimal config for loading
    config_data = {
        "optimization": {
            "device": device,
            "model_name": model_name,
            "model_path": model_path,
            "model_cache_dir": model_cache_dir,
            "huggingface_token": hf_token,
            "huggingface_token_file": hf_token_file,
        }
    }
    cfg = Config(config_data)
    return load_model_torchsim(cfg)


def _get_cached_torchsim_model(config: Config):
    """Retrieve or load a torch-sim model based on configuration parameters."""
    opt = config.optimization
    return _load_model_torchsim_impl(
        str(opt.device),
        str(opt.model_name),
        str(opt.model_path) if opt.model_path else None,
        str(opt.model_cache_dir) if opt.model_cache_dir else None,
        str(opt.huggingface_token) if opt.huggingface_token else None,
        str(opt.huggingface_token_file) if opt.huggingface_token_file else None,
    )


@time_it
def optimize_single_structure(
    structure: Structure,
    config: Config | None = None,
    calculator: FAIRChemCalculator | None = None,
) -> Structure:
    """Optimize a single :class:`Structure` using a Fairchem UMA model.

    The same :class:`Structure` instance is returned with optimized coordinates
    and energy set.
    """
    if config is None:
        config = load_config_from_file()

    symbols = structure.symbols
    coordinates = structure.coordinates
    charge = structure.charge
    multiplicity = structure.multiplicity

    try:
        if calculator is None:
            calculator = _get_cached_calculator(config)

        atoms = Atoms(symbols=symbols, positions=coordinates)
        atoms.calc = calculator
        atoms.info = {"charge": charge, "spin": multiplicity}

        logger.info(
            "Starting single geometry optimization for structure with %d atoms",
            len(symbols),
        )
        optimizer = BFGS(atoms, logfile=None)
        optimizer.run()
        logger.info(
            "Optimization completed after %d steps",
            optimizer.get_number_of_steps(),
        )

        optimized_coords = atoms.get_positions().tolist()
        potential_energy = atoms.get_potential_energy()

        structure.coordinates = optimized_coords
        structure.energy = float(potential_energy)
        return structure

    except Exception as exc:  # pragma: no cover - defensive logging
        raise RuntimeError(f"Optimization failed: {exc}") from exc


def optimize_structure_batch(
    structures: list[Structure],
    config: Config | None = None,
) -> list[Structure]:
    """Optimize a list of structures and return optimized structures with energies.

    The behavior is controlled by
    ``config.optimization.batch_optimization_mode``:

    - ``"sequential"``: Optimize each structure with ASE/BFGS and a shared
      calculator.
    - ``"batch"``: Use torch-sim batch optimization with a FairChem model
      wrapper (requires GPU).
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

    force_cpu = _parse_device_string(config.optimization.device) == "cpu"

    logger.info("Optimization device: %s", "CPU" if force_cpu else "GPU")
    mode = str(config.optimization.batch_optimization_mode).lower()
    if mode == "sequential" or force_cpu:
        if not force_cpu and mode == "batch":
            logger.warning(
                "Batch optimization mode requires GPU, falling back to sequential mode on CPU.",
            )
        return _optimize_batch_sequential(structures, config)
    if mode == "batch" and not force_cpu:
        return _optimize_batch_structures(structures, config)
    raise ValueError(
        "Unknown optimization mode: {mode} on {device}. Use 'sequential' or "
        "'batch', where batch is only supported on GPU."
    )


@time_it
def _optimize_batch_sequential(
    structures: list[Structure],
    config: Config,
) -> list[Structure]:
    """Optimize structures sequentially using single-structure optimization."""
    calculator = _get_cached_calculator(config)

    optimized_results: list[Structure] = []
    logger.info("Starting sequential optimization of %d structures", len(structures))

    for i, struct in enumerate(structures):
        try:
            optimized = optimize_single_structure(struct, config, calculator)
            optimized_results.append(optimized)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Structure %d optimization failed: %s", i + 1, exc)
            continue

    logger.info(
        "Sequential optimization completed. %d/%d successful",
        len(optimized_results),
        len(structures),
    )
    return optimized_results


@time_it
def _optimize_batch_structures(
    structures: list[Structure],
    config: Config,
) -> list[Structure]:
    """Optimize structures in batches using Fairchem's batch prediction."""
    import torch
    import torch_sim
    from torch_sim.autobatching import InFlightAutoBatcher

    logger.info("Starting batch optimization of %d structures", len(structures))

    device = _device_for_torch(config.optimization.device)
    model = _get_cached_torchsim_model(config)

    optimizer_name = getattr(config.optimization, "batch_optimizer", "fire") or "fire"
    optimizer_name = str(optimizer_name).strip().lower()
    if optimizer_name == "fire":
        optimizer = torch_sim.Optimizer.fire
    else:
        optimizer = torch_sim.Optimizer.gradient_descent
    convergence_fn = torch_sim.generate_energy_convergence_fn(energy_tol=1e-6)
    convergence_fn = torch_sim.generate_force_convergence_fn(force_tol=1e-2)

    ase_structures = [
        Atoms(
            symbols=struct.symbols,
            positions=struct.coordinates,
            info={"charge": struct.charge, "spin": struct.multiplicity},
        )
        for struct in structures
    ]

    batched_state = torch_sim.io.atoms_to_state(
        ase_structures,
        device=torch.device(device),
        dtype=torch.float64,
    )

    batcher = InFlightAutoBatcher(
        model,
        memory_scales_with="n_atoms",
        max_memory_padding=0.95,
        max_atoms_to_try=min(batched_state.n_atoms, 500_000),
    )

    final_state = torch_sim.optimize(
        system=batched_state,
        model=model,
        optimizer=optimizer,
        convergence_fn=convergence_fn,
        autobatcher=batcher,
        steps_between_swaps=5,
    )

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
