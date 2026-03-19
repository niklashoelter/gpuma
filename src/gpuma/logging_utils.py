"""Logging utilities for GPUMA."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config
    from .structure import Structure


def configure_logging(level: int = logging.INFO, logger_name: str | None = None) -> None:
    """Configure the root logger or a named logger for GPUMA.

    Parameters
    ----------
    level:
        Logging level from :mod:`logging` (defaults to ``logging.INFO``).
    logger_name:
        Optional name of the logger to configure. If omitted, the root logger
        is configured.

    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


logger = logging.getLogger(__name__)


def log_optimization_summary(
    input_structures: list[Structure],
    results: list[Structure],
    total_time: float,
    mode: str,
    config: Config,
) -> None:
    """Log a summary of the optimization run.

    Parameters
    ----------
    input_structures:
        Structures that were submitted for optimization.
    results:
        Successfully optimized structures.
    total_time:
        Wall-clock time in seconds.
    mode:
        Optimization mode (``"batch"`` or ``"sequential"``).
    config:
        Configuration used for the run.

    """
    from .config import resolve_model_type

    n_input = len(input_structures)
    n_output = len(results)
    energies = [s.energy for s in results if s.energy is not None]
    atom_counts = [s.n_atoms for s in results]

    model_name = getattr(config.model, "model_name", "unknown")
    model_type = resolve_model_type(config)
    device = str(config.technical.device)

    lines = [
        "",
        "=" * 60,
        "  GPUMA Optimization Summary",
        "=" * 60,
        f"  Model:               {model_type} / {model_name}",
        f"  Device:              {device}",
        f"  Mode:                {mode}",
        "-" * 60,
        f"  Structures input:    {n_input}",
        f"  Structures output:   {n_output}",
    ]
    if n_input > 0:
        lines.append(
            f"  Success rate:        {n_output}/{n_input}"
            f" ({100 * n_output / n_input:.1f}%)"
        )
    lines.append(f"  Total time:          {total_time:.2f} sec")
    if n_output > 0:
        lines.append(f"  Avg time/structure:  {total_time / n_output:.3f} sec")
        lines.append(f"  Throughput:          {n_output / total_time:.1f} structures/sec")
    if atom_counts:
        lines.append(
            f"  Atoms per structure: {min(atom_counts)}-{max(atom_counts)}"
            f" (avg {sum(atom_counts) / len(atom_counts):.1f})"
        )
        total_atoms = sum(atom_counts)
        lines.append(f"  Total atoms:         {total_atoms}")
        if total_time > 0:
            lines.append(f"  Atom throughput:     {total_atoms / total_time:.0f} atoms/sec")
    if energies:
        mean_e = sum(energies) / len(energies)
        lines.extend([
            "-" * 60,
            f"  Energy min:          {min(energies):.4f} eV",
            f"  Energy max:          {max(energies):.4f} eV",
            f"  Energy mean:         {mean_e:.4f} eV",
            f"  Energy spread:       {max(energies) - min(energies):.4f} eV",
        ])
    lines.extend(["=" * 60, ""])

    logger.info("\n".join(lines))
