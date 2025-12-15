"""Input/Output handler for molecular structures in GPUMA.

This module provides functions for reading molecular structures from various
formats and converting between different representations.
"""
from __future__ import annotations

import glob
import logging
import os

from .mol_utils import (
    smiles_to_conformer_ensemble as _smiles_to_ensemble_util,
)
from .mol_utils import (
    smiles_to_structure as _smiles_to_structure_util,
)
from .structure import Structure

logger = logging.getLogger(__name__)


def read_xyz(file_path: str, charge: int = 0, multiplicity: int = 1) -> Structure:
    """Read an XYZ file and return a :class:`Structure` instance.

    Parameters
    ----------
    file_path:
        Path to the XYZ file to read.
    charge:
        Optional total charge to set on the structure (default: ``0``).
    multiplicity:
        Optional spin multiplicity to set (default: ``1``).

    Returns
    -------
    Structure
        Object with symbols, coordinates, and an optional comment.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is invalid.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    symbols: list[str] = []
    coordinates: list[tuple[float, float, float]] = []

    try:
        with open(file_path, encoding="utf-8") as infile:
            lines = [line.rstrip("\n") for line in infile.readlines()]

        try:
            num_atoms = int(lines[0])
        except Exception as exc:
            raise ValueError("First line must contain the number of atoms as an integer") from exc

        if len(lines) < 2 + num_atoms:
            raise ValueError(f"Expected {num_atoms} atom lines, but found {max(0, len(lines) - 2)}")
        comment = lines[1] if len(lines) > 1 else ""

        for i in range(num_atoms):
            parts = lines[2 + i].split()
            if len(parts) < 4:
                raise ValueError(f"Line {i + 3} must contain at least 4 elements: symbol x y z")
            symbol = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError as exc:
                raise ValueError(f"Invalid coordinates in line {i + 3}: {parts[1:4]}") from exc
            symbols.append(symbol)
            coordinates.append((x, y, z))

    except Exception as exc:
        if isinstance(exc, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error reading XYZ file: {exc}") from exc

    return Structure(
        symbols=symbols,
        coordinates=coordinates,
        comment=comment,
        charge=charge,
        multiplicity=multiplicity,
    )


def read_multi_xyz(file_path: str, charge: int = 0, multiplicity: int = 1) -> list[Structure]:
    """Read an XYZ file containing multiple structures.

    Parameters
    ----------
    file_path:
        Path to the multi-structure XYZ file.
    charge:
        Optional total charge to set on all returned structures (default: ``0``).
    multiplicity:
        Optional spin multiplicity to set (default: ``1``).

    Returns
    -------
    list[Structure]
        List of structures read from the file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is invalid.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    structures: list[Structure] = []

    try:
        with open(file_path, encoding="utf-8") as infile:
            lines = [line.rstrip("\n") for line in infile.readlines()]

        i = 0
        while i < len(lines):
            if lines[i].strip() == "":
                i += 1
                continue

            try:
                num_atoms = int(lines[i].strip())
            except ValueError:
                i += 1
                continue

            if i + 1 + num_atoms >= len(lines):
                break

            comment = lines[i + 1] if (i + 1) < len(lines) else ""

            symbols: list[str] = []
            coordinates: list[tuple[float, float, float]] = []

            valid = True
            for j in range(num_atoms):
                parts = lines[i + 2 + j].split()
                if len(parts) < 4:
                    valid = False
                    break
                symbol = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    valid = False
                    break
                symbols.append(symbol)
                coordinates.append((x, y, z))

            if valid and len(symbols) == num_atoms:
                structures.append(
                    Structure(
                        symbols=symbols,
                        coordinates=coordinates,
                        comment=comment,
                        charge=charge,
                        multiplicity=multiplicity,
                    )
                )

            i = i + 2 + num_atoms

    except Exception as exc:
        raise ValueError(f"Error reading multi-XYZ file: {exc}") from exc

    return structures


def read_xyz_directory(
    directory_path: str, charge: int = 0, multiplicity: int = 1
) -> list[Structure]:
    """Read all XYZ files from a directory.

    Parameters
    ----------
    directory_path:
        Path to directory containing XYZ files.
    charge:
        Optional total charge to set on all returned structures (default: ``0``).
    multiplicity:
        Optional spin multiplicity to set (default: ``1``).

    Returns
    -------
    list[Structure]
        List of structures from all XYZ files in the directory.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If no valid XYZ files are found.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} not found")

    xyz_files = glob.glob(os.path.join(directory_path, "*.xyz"))

    if not xyz_files:
        raise ValueError(f"No XYZ files found in directory {directory_path}")

    structures: list[Structure] = []

    for xyz_file in xyz_files:
        try:
            structures.append(read_xyz(xyz_file, charge=charge, multiplicity=multiplicity))
        except Exception as exc:  # pragma: no cover - logged and skipped
            logger.warning("Failed to read %s: %s", xyz_file, exc)

    if not structures:
        raise ValueError("No valid structures could be read from any XYZ files")

    return structures


def smiles_to_xyz(smiles_string: str,
                  return_full_xyz_str: bool = False,
                  multiplicity: int | None = None) -> Structure | str:
    """Convert a SMILES string to a :class:`Structure` or an XYZ string.

    Parameters
    ----------
    smiles_string:
        Valid SMILES string representing the molecular structure.
    return_full_xyz_str:
        If ``True``, return an XYZ-format string instead of a
        :class:`Structure` instance.
    multiplicity:
        Optional spin multiplicity to set on the structure (default: ``None``).

    Returns
    -------
    Structure | str
        Either a :class:`Structure` or an XYZ string depending on
        ``return_full_xyz_str``.
    """
    if not smiles_string or not smiles_string.strip():
        raise ValueError("SMILES string cannot be empty or None")

    struct = _smiles_to_structure_util(smiles_string.strip())
    if multiplicity is not None:
        struct.multiplicity = int(multiplicity)

    if return_full_xyz_str:
        xyz_lines = [str(struct.n_atoms)]
        xyz_lines.append(
            f"Generated from SMILES using MORFEUS | "
            f"Charge: {struct.charge} | "
            f"Multiplicity: {struct.multiplicity}"
        )
        for atom, coord in zip(struct.symbols, struct.coordinates, strict=True):
            xyz_lines.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
        return "\n".join(xyz_lines)

    struct.comment = (f"Generated from SMILES: {smiles_string} | "
                      f"Charge: {struct.charge} | "
                      f"Multiplicity: {struct.multiplicity}")
    return struct


def smiles_to_ensemble(
    smiles_string: str,
    max_num_confs: int,
    multiplicity: int | None = None,
) -> list[Structure]:
    """Generate conformer ensemble from SMILES.

    Parameters
    ----------
    smiles_string:
        Valid SMILES string representing the molecular structure.
    max_num_confs:
        Maximum number of conformers to generate.
    multiplicity:
        Optional spin multiplicity to set on the structures (default: ``None``).

    Returns
    -------
    list[Structure]
        A list of :class:`Structure` instances representing the conformers.
    """
    if not smiles_string or not smiles_string.strip():
        raise ValueError("SMILES string cannot be empty or None")

    mult = int(multiplicity) if multiplicity is not None else 1
    structs = _smiles_to_ensemble_util(smiles_string.strip(), max_num_confs, multiplicity=mult)
    return structs


def save_xyz_file(structure: Structure, file_path: str) -> None:
    """Save a single Structure to XYZ, including energy and electronic state."""
    lines: list[str] = [str(structure.n_atoms)]
    # include existing comment and ensure energy/charge/multiplicity are visible
    base_comment = structure.comment or ""
    energy_part = ""
    if structure.energy is not None:
        energy_part = f" | Energy: {structure.energy:.6f} eV"
    state_part = f" | Charge: {structure.charge} | Multiplicity: {structure.multiplicity}"
    comment = (base_comment + energy_part + state_part).strip() or "Structure"
    lines.append(comment)
    for symbol, coord in zip(structure.symbols, structure.coordinates, strict=True):
        lines.append(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def save_multi_xyz(structures: list[Structure],
                   file_path: str,
                   comments: list[str] | None = None) -> None:
    """Save multiple structures to a multi-XYZ file, including energy and state."""
    lines: list[str] = []
    for idx, struct in enumerate(structures):
        lines.append(str(struct.n_atoms))
        base_comment = ""
        if comments and idx < len(comments):
            base_comment = comments[idx]
        elif struct.comment:
            base_comment = struct.comment
        energy_part = ""
        if struct.energy is not None:
            energy_part = f" | Energy: {struct.energy:.6f} eV"
        state_part = f" | Charge: {struct.charge} | Multiplicity: {struct.multiplicity}"
        comment = (base_comment + energy_part + state_part).strip() or f"Structure {idx + 1}"
        lines.append(comment)
        for symbol, coord in zip(struct.symbols, struct.coordinates, strict=True):
            lines.append(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
