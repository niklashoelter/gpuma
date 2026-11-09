# Python API

Documentation for the core functions in `gpuma`.

!!! note "Package layout"
    Everything documented here is re-exported from the top-level `gpuma`
    namespace, so the public API is flat: `from gpuma import optimize_single_smiles,
    Config, read_multi_xyz` all work regardless of where a symbol lives internally.
    Internally the package is organised into subpackages:

    - `gpuma.models` — model dispatchers (`load_calculator`, `load_torchsim_model`)
      and the `AVAILABLE_*_MODELS` registries, with backend-specific loaders under
      `models/fairchem.py`, `models/orb.py`, `models/sevennet.py`, `models/aimnet.py`.
    - `gpuma.conformer_generation` — SMILES → 3D structure/ensemble embedding
      (`embed.py`, `mol_utils.py`).
    - `gpuma.utils` — I/O (`io_handler.py`), logging (`logging_utils.py`),
      timing helpers (`decorators.py`) and opt-in diagnostic reporting (`reporting.py`).
    - Top-level modules `structure.py`, `config.py`, `optimizer.py`, `api.py`,
      `cli.py`.

    The fully-qualified paths below (e.g. `gpuma.utils.io_handler.read_xyz`) are
    the canonical source locations; the short top-level aliases (`gpuma.read_xyz`)
    are equivalent.

::: gpuma.api
    options:
      members: []
      show_root_heading: false
      show_source: false

## Data Structures
The core container passed between I/O, conformer generation and optimization.

::: gpuma.structure.Structure
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Single Structure Optimization
Methods for optimizing individual molecules provided as SMILES or XYZ files.

::: gpuma.api.optimize_single_smiles
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.api.optimize_single_xyz_file
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Batch & Ensemble Optimization
Methods for processing multiple structures, ensembles, or entire directories.

::: gpuma.api.optimize_ensemble_smiles
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.api.optimize_batch_multi_xyz_file
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.api.optimize_batch_xyz_directory
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Model Loading
Functions for directly loading model calculators and torch-sim wrappers.

::: gpuma.models.load_calculator
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.models.load_torchsim_model
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Configuration
The `Config` object controls model selection, optimization settings, conformer
generation and technical/device options. It is accepted by every high- and
low-level optimization function. A ready-to-use instance carrying the built-in
defaults is available as `gpuma.default_config` (equivalent to `Config()`). See
the [Configuration](config.md) page for the full list of keys and defaults.

::: gpuma.config.Config
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.config.load_config_from_file
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.config.save_config_to_file
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.config.resolve_model_type
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## I/O & Structure Conversion
Functions for reading, writing, and converting molecular structures.

::: gpuma.utils.io_handler.read_xyz
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.utils.io_handler.read_multi_xyz
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.utils.io_handler.read_xyz_directory
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.utils.io_handler.smiles_to_xyz
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.utils.io_handler.smiles_to_ensemble
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.utils.io_handler.save_xyz_file
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.utils.io_handler.save_multi_xyz
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.utils.io_handler.save_as_single_xyz_files
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Batched Conformer Generation
Convert many SMILES to 3D structures in one call. These batch across molecules,
which is what makes the optional GPU backend worthwhile; the per-molecule
helpers above call into them. The backend follows `technical.device` in the
configuration and falls back to CPU when no GPU is usable.

::: gpuma.conformer_generation.embed.generate_structures
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.conformer_generation.embed.generate_ensembles
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Low-Level Optimization
Lower-level functions used by the high-level API.

::: gpuma.optimizer.optimize_single_structure
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.optimizer.optimize_structure_batch
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

## Utilities

Timing decorators and context managers for profiling.

::: gpuma.utils.decorators.time_it
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

::: gpuma.utils.decorators.timed_block
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3
