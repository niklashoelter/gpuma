"""Core geometry optimization module using Fairchem UMA models for GPUMA.

This module contains optimization logic for single structures and batches of
structures (e.g., conformer ensembles).
"""
from __future__ import annotations

import logging

from ase import Atoms
from ase.optimize import BFGS
from fairchem.core import FAIRChemCalculator

from .config import Config, load_config_from_file
from .decorators import time_it
from .models import _check_device, load_model_fairchem, load_model_torchsim
from .structure import Structure

logger = logging.getLogger(__name__)


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
            calculator = load_model_fairchem(config)

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

    device = _check_device(config.optimization.device)
    force_cpu = device == "cpu"

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
    calculator = load_model_fairchem(config)

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

    device = _check_device(config.optimization.device)
    model = load_model_torchsim(config)

    optimizer_name = getattr(config.optimization, "batch_optimizer", "fire") or "fire"
    optimizer_name = str(optimizer_name).strip().lower()
    if optimizer_name == "fire":
        optimizer = torch_sim.Optimizer.fire
    else:
        optimizer = torch_sim.Optimizer.gradient_descent
    convergence_fn = torch_sim.generate_energy_convergence_fn(energy_tol=1e-6)

    ase_structures = [
        Atoms(
            symbols=struct.symbols,
            positions=[tuple(coord) for coord in struct.coordinates],
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
        steps_between_swaps=3,
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
