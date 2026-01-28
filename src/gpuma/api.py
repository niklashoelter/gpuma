"""Public high-level API for common geometry optimization workflows.

This module provides convenience functions built on top of the lower-level
I/O and optimization utilities. It is deliberately kept separate from the
package's ``__init__`` to keep the top-level namespace lightweight.
"""

from __future__ import annotations

from .config import Config, load_config_from_file
from .io_handler import (
    read_xyz,
    save_multi_xyz,
    save_xyz_file,
    smiles_to_ensemble,
    smiles_to_xyz,
)
from .optimizer import optimize_single_structure, optimize_structure_batch
from .structure import Structure


def optimize_single_smiles(
    smiles: str,
    output_file: str | None = None,
    config: Config | None = None,
) -> Structure:
    """Optimize a single molecule from a SMILES string.

    Parameters
    ----------
    smiles:
        SMILES string of the molecule to optimize.
    output_file:
        Optional path to an output XYZ file where the optimized structure
        will be written.
    config:
        Optional :class:`~gpuma.config.Config` object to
        control the optimization pipeline.

    Returns
    -------
    Structure
        The optimized molecular structure.
    """
    if config is None:
        config = load_config_from_file()
    multiplicity = getattr(config.optimization, "multiplicity", 1)
    structure = smiles_to_xyz(smiles, multiplicity=multiplicity)
    if not isinstance(structure, Structure):
        raise ValueError("smiles_to_xyz did not return a Structure")
    structure.comment = f"Optimized from SMILES: {smiles}"
    result = optimize_single_structure(structure, config)

    if output_file:
        save_xyz_file(result, output_file)

    return result


def optimize_single_xyz_file(
    input_file: str,
    output_file: str | None = None,
    config: Config | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
) -> Structure:
    """Optimize a single structure from an XYZ file.

    Parameters
    ----------
    input_file:
        Path to the input XYZ file.
    output_file:
        Optional path to an output XYZ file where the optimized structure
        will be written.
    config:
        Optional :class:`~gpuma.config.Config` object to
        control the optimization pipeline.
    charge:
        Total charge of the system. Defaults to ``0``.
    multiplicity:
        Spin multiplicity of the system. Defaults to ``1``.

    Returns
    -------
    Structure
        The optimized molecular structure.
    """
    if config is None:
        config = load_config_from_file()
    eff_charge = int(charge if charge is not None else getattr(config.optimization, "charge", 0))
    eff_mult = int(
        multiplicity
        if multiplicity is not None
        else getattr(config.optimization, "multiplicity", 1)
    )
    structure = read_xyz(input_file, charge=eff_charge, multiplicity=eff_mult)
    if not isinstance(structure, Structure):
        raise ValueError("read_xyz did not return a Structure")
    structure.comment = f"Optimized from: {input_file}"
    result = optimize_single_structure(structure, config)

    if output_file:
        save_xyz_file(result, output_file)

    return result


def optimize_smiles_ensemble(
    smiles: str,
    num_conformers: int = 10,
    output_file: str | None = None,
    config: Config | None = None,
) -> list[Structure]:
    """Optimize a conformer ensemble generated from a SMILES string.

    Parameters
    ----------
    smiles:
        SMILES string of the molecule.
    num_conformers:
        Number of conformers to generate and optimize.
    output_file:
        Optional path to a multi-structure XYZ file where the optimized
        ensemble will be written.
    config:
        Optional :class:`~gpuma.config.Config` object to
        control the optimization pipeline.

    Returns
    -------
    list[Structure]
        The list of optimized structures.
    """
    if config is None:
        config = load_config_from_file()
    multiplicity = int(getattr(config.optimization, "multiplicity", 1))
    conformers = smiles_to_ensemble(smiles, num_conformers)
    if not isinstance(conformers, list) or (
        len(conformers) and not isinstance(conformers[0], Structure)
    ):
        raise ValueError("smiles_to_ensemble did not return a list of Structure")
    for s in conformers:
        s.multiplicity = multiplicity
    results = optimize_structure_batch(conformers, config)

    if output_file:
        comments = [
            f"Optimized conformer {i + 1} from SMILES: {smiles}" for i in range(len(results))
        ]
        save_multi_xyz(results, output_file, comments)

    return results


__all__ = [
    "optimize_single_smiles",
    "optimize_single_xyz_file",
    "optimize_smiles_ensemble",
]
