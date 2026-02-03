"""Public high-level Python API for common geometry optimization workflows.

This module provides convenience functions built on top of the lower-level
I/O and optimization utilities. It allows users to easily optimize molecular
structures starting from SMILES strings or XYZ files, as well as optimizing
ensembles of conformers.

"""

from __future__ import annotations

from typing import cast

from .config import Config, load_config_from_file
from .io_handler import (
    file_exists,
    read_multi_xyz,
    read_xyz,
    read_xyz_directory,
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

    This function uses the provided SMILES string to generate an initial 3D structure
    using the Morfeus library. It then optimizes the structure using the specified
    optimization pipeline.

    Args:
        smiles (str): SMILES string of the molecule to optimize.
        output_file (str): Path to an output XYZ file where the optimized structure
            will be written. If None, the optimized structure is not saved to a file.
        config (Config, optional): Config object to control the optimization pipeline.
            Highly recommended to specify. If None, the configuration will be loaded
            from the default file.

    Returns:
        Structure: The optimized molecular structure as a Structure object.

    Raises:
        ValueError: If the generated structure is not valid.

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
) -> Structure:
    """Optimize a single structure from an XYZ file.

    This function reads a molecular structure from the specified XYZ file,
    optimizes it using the provided optimization pipeline, and optionally
    writes the optimized structure to an output XYZ file.

    Args:
        input_file (str): Path to an input XYZ file from which to read the initial structure.
        output_file (str): Path to an output XYZ file where the optimized structure will be written.
            If None, the optimized structure will not be saved to a file.
        config (Config, optional): Config object to control the optimization pipeline.
            Highly recommended to specify. If None, the configuration will be loaded from
            the default file.

    Returns:
        Structure: The optimized molecular structure as a Structure object.

    Raises:
        ValueError: If the input file does not exist or if the read structure is not valid.

    """
    if not file_exists(input_file):
        raise ValueError(f"Input file {input_file} does not exist.")
    if config is None:
        config = load_config_from_file()

    eff_charge = int(getattr(config.optimization, "charge", 0))
    eff_mult = int(getattr(config.optimization, "multiplicity", 1))
    structure = read_xyz(input_file, charge=eff_charge, multiplicity=eff_mult)
    if not isinstance(structure, Structure):
        raise ValueError("read_xyz did not return a Structure")
    structure.comment = f"Optimized from: {input_file}"
    result = optimize_single_structure(structure, config)

    if output_file:
        save_xyz_file(result, output_file)

    return result


def optimize_ensemble_smiles(
    smiles: str,
    output_file: str | None = None,
    config: Config | None = None,
) -> list[Structure]:
    """Optimize a conformer ensemble generated from a SMILES string.

    This function generates a specified number of conformers from the provided
    SMILES string using the Morfeus library. It then optimizes each conformer
    using the specified optimization pipeline. Optionally, the optimized ensemble
    can be saved to a multi-structure XYZ file.

    Args:
        smiles (str): SMILES string of the molecule for which to generate conformers.
        output_file (str, optional): Path to an output multi-structure XYZ file where
            the optimized ensemble will be written. If None, the ensemble is not saved to a file.
        config (Config, optional): Config object to control the optimization pipeline.
            Highly recommended to specify. If None, the configuration will be loaded from
            the default file.

    Returns:
        list[Structure]: A list of optimized molecular structures as Structure objects.

    Raises:
        ValueError: If output_file is not specified when required or if the generated
            conformers are not valid.

    """
    if config is None:
        config = load_config_from_file()
    multiplicity = int(getattr(config.optimization, "multiplicity", 1))
    num_conformers = int(getattr(config.optimization, "max_num_conformers", 10))
    conformers = smiles_to_ensemble(smiles, num_conformers, multiplicity)
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

def optimize_batch_multi_xyz_file(
    input_file: str,
    output_file: str | None = None,
    config: Config | None = None,
) -> list[Structure]:
    """Optimize a batch of structures from a multi-structure XYZ file.

    This function reads multiple molecular structures from the specified multi-structure
    XYZ file, optimizes each structure using the provided optimization pipeline, and writes
    the optimized structures to an output multi-structure XYZ file.

    Args:
        input_file (str): Path to an input multi-structure XYZ file from which to
            read the initial structures.
        output_file (str): Path to an output multi-structure XYZ file where the optimized
            structures will be written.
            If None, the optimized structures will not be saved to a file.
        config (Config, optional): Config object to control the optimization pipeline.
            Highly recommended to specify. If None, the configuration will be loaded from
            the default file.

    Returns:
        list[Structure]: A list of optimized molecular structures as Structure objects.

    Raises:
        ValueError: If the input file does not exist or if the read structures are not valid.

    """
    if not file_exists(input_file):
        raise ValueError(f"Input file {input_file} does not exist.")

    if config is None:
        config = load_config_from_file()

    eff_charge = int(getattr(config.optimization, "charge", 0))
    eff_mult = int(getattr(config.optimization, "multiplicity", 1))

    structures = cast(
        list[Structure],
        read_multi_xyz(input_file, charge=eff_charge, multiplicity=eff_mult),
    )
    if not isinstance(structures, list) or (
        len(structures) and not isinstance(structures[0], Structure)
    ):
        raise ValueError("read_multi_xyz did not return a list of Structure")

    results = optimize_structure_batch(structures, config)

    if output_file:
        comments = [
            f"Optimized structure {i + 1} from: {input_file}" for i in range(len(results))
        ]
        save_multi_xyz(results, output_file, comments)

    return results

def optimize_batch_xyz_directory(
    input_directory: str,
    output_file: str,
    config: Config | None = None,
) -> list[Structure]:
    """Optimize a batch of structures from XYZ files in a directory.

    This function reads multiple molecular structures from XYZ files in the specified input
    directory, optimizes each structure using the provided optimization pipeline,
    and writes the optimized structures to XYZ files in the specified output directory.

    Args:
        input_directory (str): Path to an input directory containing XYZ files.
        output_file (str): Path to an output multi-structure XYZ file where the
            optimized structures will be written.
        config (Config, optional): Config object to control the optimization pipeline.
            Highly recommended to specify. If None, the configuration will be loaded from the
            default file.

    Returns:
        list[Structure]: A list of optimized molecular structures as Structure objects.

    Raises:
        ValueError: If the input directory does not exist or contains no valid XYZ files.

    """
    if config is None:
        config = load_config_from_file()

    eff_charge = int(getattr(config.optimization, "charge", 0))
    eff_mult = int(getattr(config.optimization, "multiplicity", 1))

    structures = cast(
        list[Structure],
        read_xyz_directory(
            input_directory,
            charge=eff_charge,
            multiplicity=eff_mult,
        ),
    )

    results = optimize_structure_batch(structures, config)

    if output_file:
        comments = [
            f"Optimized structure {i + 1} from batch input"
            for i in range(len(results))
        ]
        save_multi_xyz(results, output_file, comments)

    return results

__all__ = [
    "optimize_single_smiles",
    "optimize_single_xyz_file",
    "optimize_ensemble_smiles",
    "optimize_batch_multi_xyz_file",
    "optimize_batch_xyz_directory",
]
