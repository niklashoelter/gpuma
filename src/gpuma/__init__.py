"""GPUMA: A minimal package for molecular geometry optimization using Fairchem's UMA models.

This package provides essential tools for:
- Single molecule optimization from SMILES strings or XYZ files
- Batch optimization of multiple structures (e.g., conformer ensembles)
- Format conversion between SMILES and XYZ coordinates
- Configurable optimization parameters through JSON/YAML configuration files

The package is designed to be simple and focused, providing only the functionality
that is actually implemented and tested.
"""

from .api import (
    optimize_batch_multi_xyz_file,
    optimize_batch_xyz_directory,
    optimize_ensemble_smiles,
    optimize_single_smiles,
    optimize_single_xyz_file,
)
from .config import Config, default_config, load_config_from_file, save_config_to_file
from .decorators import time_it
from .io_handler import (
    read_multi_xyz,
    read_xyz,
    read_xyz_directory,
    save_multi_xyz,
    save_xyz_file,
    smiles_to_ensemble,
    smiles_to_xyz,
)
from .models import load_model_fairchem, load_model_torchsim
from .optimizer import optimize_single_structure, optimize_structure_batch
from .structure import Structure

__all__ = [
    # Data types
    "Structure",
    # I/O functions
    "read_xyz",
    "read_multi_xyz",
    "read_xyz_directory",
    "smiles_to_xyz",
    "smiles_to_ensemble",
    "save_xyz_file",
    "save_multi_xyz",
    # Optimization functions
    "optimize_single_structure",
    "optimize_structure_batch",
    # Convenience functions (re-exported from ``api``)
    "optimize_single_smiles",
    "optimize_single_xyz_file",
    "optimize_ensemble_smiles",
    "optimize_batch_multi_xyz_file",
    "optimize_batch_xyz_directory",
    # Model functions
    "load_model_torchsim",
    "load_model_fairchem",
    # Configuration
    "Config",
    "default_config",
    "load_config_from_file",
    "save_config_to_file",
    # Decorators
    "time_it",
]
