# Python API

Documentation for the core functions in `gpuma`.

::: gpuma.api
    options:
      members: []
      show_root_heading: false
      show_source: false

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
