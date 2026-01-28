"""Molecular utilities for SMILES processing and structure generation in GPUMA.

This module provides functions for converting SMILES strings to 3D molecular
structures and generating conformer ensembles using the :mod:`morfeus` library
with RDKit.
"""

from numbers import Integral

from .decorators import time_it
from .structure import Structure


def _to_symbol_list(elements) -> list[str]:
    """Convert a sequence of element descriptors to a list of atomic symbols.

    The function accepts element symbols as strings or atomic numbers
    (:class:`int` or other :class:`numbers.Integral` types) and converts them
    to a list of string symbols. Numpy arrays are supported transparently.
    """
    from ase.data import chemical_symbols

    try:
        if hasattr(elements, "tolist"):
            elements = elements.tolist()
    except Exception:  # pragma: no cover - defensive
        pass

    symbols: list[str] = []
    for elem in elements:
        if isinstance(elem, str):
            symbols.append(elem)
        elif isinstance(elem, Integral):
            try:
                symbols.append(chemical_symbols[int(elem)])
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid atomic number: {elem}") from exc
        else:
            symbols.append(str(elem))
    return symbols


def _to_coord_list(coords) -> list[tuple[float, float, float]]:
    """Convert coordinates to a nested Python list of float tuples."""
    try:
        if hasattr(coords, "tolist"):
            coords = coords.tolist()
    except Exception:  # pragma: no cover - defensive
        pass
    return [(float(row[0]), float(row[1]), float(row[2])) for row in coords]


@time_it
def smiles_to_conformer_ensemble(
    smiles: str, max_num_confs: int = 5, multiplicity: int = 1
) -> list[Structure]:
    """Generate multiple conformers from a SMILES string.

    This function uses the :mod:`morfeus` library to generate conformers from a
    SMILES string. The resulting conformers are automatically pruned based on
    RMSD and sorted by energy.

    Parameters
    ----------
    smiles:
        Valid SMILES string representing the molecular structure.
    max_num_confs:
        Maximum number of conformers to return (default: ``10``).

    Returns
    -------
    list[Structure]
        List of conformer structures.

    Raises
    ------
    ValueError
        If the SMILES string is invalid or conformer generation fails.
    ImportError
        If :mod:`morfeus` or RDKit dependencies are not available.

    Notes
    -----
    Conformers are sorted by energy (lowest first) and pruned by RMSD to remove
    duplicates. The actual number returned may be less than ``max_num_confs``.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string cannot be empty")

    if max_num_confs <= 0:
        raise ValueError("max_num_confs must be positive")

    try:
        from morfeus.conformer import ConformerEnsemble  # type: ignore
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        charge = Chem.GetFormalCharge(mol)
        mol = Chem.AddHs(mol)
        ensemble = ConformerEnsemble.from_rdkit(mol)
        ensemble.prune_rmsd()
        ensemble.multiplicity = multiplicity
        ensemble.sort()

        structures: list[Structure] = []
        for i, conformer in enumerate(ensemble):
            if i >= max_num_confs:
                break

            atoms = _to_symbol_list(getattr(conformer, "elements", []))
            coordinates = _to_coord_list(getattr(conformer, "coordinates", []))

            if len(atoms) != len(coordinates):
                continue

            structures.append(
                Structure(
                    symbols=atoms,
                    coordinates=coordinates,
                    charge=charge,
                    multiplicity=ensemble.multiplicity,
                )
            )

        if not structures:
            raise ValueError("No valid conformers could be generated from SMILES")

        return structures

    except ImportError as exc:  # pragma: no cover - dependency error
        raise ImportError(
            "Required dependencies not found. Please install with: "
            "uv pip install 'gpuma' or install 'morfeus-ml rdkit'"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to generate conformers from SMILES '{smiles}': {exc}") from exc


def smiles_to_structure(smiles: str) -> Structure:
    """Convert a SMILES string to a single 3D molecular structure.

    This function generates the lowest-energy conformer from a SMILES string
    by calling :func:`smiles_to_conformer_ensemble` and returning only the
    first structure.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string cannot be empty")

    try:
        ensemble = smiles_to_conformer_ensemble(smiles.strip(), max_num_confs=1)

        if not ensemble:
            raise ValueError("No conformers generated from SMILES")

        return ensemble[0]

    except Exception as exc:
        if isinstance(exc, (ValueError, ImportError)):
            raise
        raise ValueError(f"Failed to generate structure from SMILES '{smiles}': {exc}") from exc
