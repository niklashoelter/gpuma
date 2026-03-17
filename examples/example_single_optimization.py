#!/usr/bin/env python3
"""Example: Single Structure Optimization with GPUMA.

Demonstrates how to optimize single molecular structures using different
input methods (SMILES and XYZ files) and model backends (Fairchem UMA
and ORB-v3) via the GPUMA API.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gpuma
from gpuma.config import load_config_from_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "example_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def example_optimize_from_smiles():
    """Example 1: Optimize a molecule from a SMILES string (Fairchem UMA)."""
    print("=== Example 1: Single optimization from SMILES (Fairchem UMA) ===")

    smiles = "C1=C[O+]=CC=C1CCCCCC"
    print(f"Optimizing {smiles} ...")

    cfg = load_config_from_file("config.json")
    cfg.optimization.multiplicity = 1

    output_file = os.path.join(OUTPUT_DIR, "python_single_smiles_basic.xyz")
    structure = gpuma.optimize_single_smiles(
        smiles=smiles,
        output_file=output_file,
        config=cfg,
    )

    print("  Optimization successful!")
    print(f"  Atoms: {structure.n_atoms}")
    print(f"  Final energy: {structure.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")

    # Alternative step-by-step approach
    print("\n--- Alternative step-by-step approach ---")
    struct2 = gpuma.smiles_to_xyz(smiles)
    struct2.comment = f"Optimized from SMILES: {smiles}"
    struct2 = gpuma.optimize_single_structure(struct2)

    step_output = os.path.join(OUTPUT_DIR, "python_single_smiles_stepwise.xyz")
    gpuma.save_xyz_file(struct2, step_output)

    print("  Step-by-step optimization successful!")
    print(f"  Final energy: {struct2.energy:.6f} eV")
    print(f"  Output saved to: {step_output}")


def example_optimize_from_xyz():
    """Example 2: Optimize a molecule from an XYZ file."""
    print("\n=== Example 2: Single optimization from XYZ file ===")

    input_file = "example_input_xyzs/butene_triplet.xyz"
    output_file = os.path.join(OUTPUT_DIR, "python_single_xyz_basic.xyz")

    if not os.path.exists(input_file):
        print(f"  Input file {input_file} not found")
        return

    cfg = load_config_from_file("config.json")
    cfg.optimization.charge = 0
    cfg.optimization.multiplicity = 3

    structure = gpuma.optimize_single_xyz_file(
        input_file=input_file,
        output_file=output_file,
        config=cfg,
    )

    print("  Optimization successful!")
    print(f"  Input: {input_file}")
    print(f"  Atoms: {structure.n_atoms}")
    print(f"  Final energy: {structure.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_optimize_from_smiles_orb():
    """Example 3: Optimize a molecule from SMILES using ORB-v3."""
    print("\n=== Example 3: Single optimization from SMILES (ORB-v3) ===")

    smiles = "C1=C[O+]=CC=C1CCCCCC"
    print(f"Optimizing {smiles} with ORB-v3 ...")

    # Load the ORB-specific config (sets model_type="orb",
    # model_name="orb_v3_direct_omol")
    cfg = load_config_from_file("config_orb.json")
    cfg.optimization.multiplicity = 1

    output_file = os.path.join(OUTPUT_DIR, "python_single_smiles_orb.xyz")
    structure = gpuma.optimize_single_smiles(
        smiles=smiles,
        output_file=output_file,
        config=cfg,
    )

    print("  ORB-v3 optimization successful!")
    print(f"  Atoms: {structure.n_atoms}")
    print(f"  Final energy: {structure.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    print("GPUMA - Single Structure Optimization Examples")
    print("=" * 70)

    example_optimize_from_smiles()
    example_optimize_from_xyz()
    example_optimize_from_smiles_orb()

    print("\n" + "=" * 70)
    print("Examples completed! Check the generated XYZ files.")
