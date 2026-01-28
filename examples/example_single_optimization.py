#!/usr/bin/env python3
"""Example: Single Structure Optimization with GPUMA.

This example demonstrates how to optimize single molecular structures
using different input methods (SMILES and XYZ files) via the GPUMA API.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gpuma
from gpuma import Structure
from gpuma.config import load_config_from_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "example_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def example_optimize_from_smiles():
    """Example 1: Optimize a molecule from a SMILES string."""
    print("=== Example 1: Single optimization from SMILES ===")

    smiles = "C1=C[O+]=CC=C1CCCCCC"
    print(f"Optimizing {smiles} ...")

    cfg = load_config_from_file("config.json")
    cfg.optimization.multiplicity = 1

    output_file = os.path.join(OUTPUT_DIR, "python_single_smiles_basic.xyz")

    struct: Structure = gpuma.optimize_single_smiles(
        smiles=smiles,
        output_file=output_file,
        config=cfg,
    )

    print("✓ Optimization successful!")
    print(f"  Atoms: {struct.n_atoms}")
    print(f"  Final energy: {struct.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")

    print("\n--- Alternative step-by-step approach ---")
    struct2: Structure = gpuma.smiles_to_xyz(smiles)
    struct2.comment = f"Optimized from SMILES: {smiles}"
    struct2 = gpuma.optimize_single_structure(struct2)

    step_output = os.path.join(OUTPUT_DIR, "python_single_smiles_stepwise.xyz")
    gpuma.save_xyz_file(struct2, step_output)

    print("✓ Step-by-step optimization successful!")
    print(f"  Final energy: {struct2.energy:.6f} eV")
    print(f"  Output saved to: {step_output}")


def example_optimize_from_xyz():
    """Example 2: Optimize a molecule from an XYZ file."""
    print("\n=== Example 2: Single optimization from XYZ file ===")

    input_file = "example_input_xyzs/multi_xyz_dir/input_1.xyz"
    output_file = os.path.join(OUTPUT_DIR, "python_single_xyz_basic.xyz")

    if not os.path.exists(input_file):
        print(f"✗ Input file {input_file} not found")
        return

    cfg = load_config_from_file("config.json")
    cfg.optimization.charge = 0
    cfg.optimization.multiplicity = 1

    struct: Structure = gpuma.optimize_single_xyz_file(
        input_file=input_file,
        output_file=output_file,
        config=cfg,
    )

    print("✓ Optimization successful!")
    print(f"  Input: {input_file}")
    print(f"  Atoms: {struct.n_atoms}")
    print(f"  Final energy: {struct.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_optimize_with_custom_config():
    """Example 3: Optimize with a custom configuration file."""
    print("\n=== Example 3: Optimization with custom configuration ===")

    config = load_config_from_file("config.json")

    smiles = "C1=C[O+]=CC=C1"
    print(f"Optimizing {smiles} with custom config...")

    output_file = os.path.join(OUTPUT_DIR, "python_single_smiles_custom_config.xyz")

    struct: Structure = gpuma.optimize_single_smiles(
        smiles=smiles,
        output_file=output_file,
        config=config,
    )

    print("✓ Optimization successful!")
    print(f"  Final energy: {struct.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    print("GPUMA - Single Structure Optimization Examples")
    print("=" * 70)

    example_optimize_from_smiles()
    example_optimize_from_xyz()
    example_optimize_with_custom_config()

    print("\n" + "=" * 70)
    print("Examples completed! Check the generated XYZ files.")
