#!/usr/bin/env python3
"""Example: Ensemble and Batch Optimization with GPUMA.

This example demonstrates how to optimize SMILES-generated conformer ensembles
and how to batch-optimize general structures from multi-XYZ files or directories
using the GPUMA API.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gpuma
from gpuma.config import default_config, load_config_from_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "example_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def example_ensemble_from_smiles():
    """Example 1: Generate and optimize a conformer ensemble from SMILES."""
    print("=== Example 1: Ensemble optimization from SMILES ===")

    smiles = "CCC(CC)CCOOC(CC)CCOC"  # Example SMILES
    config = default_config
    config.optimization.max_num_conformers = 50
    config.optimization.force_convergence_criterion = 5e-1
    print(f"Generating conformers for {smiles} and optimizing...")

    output_file = os.path.join(OUTPUT_DIR, "python_ensemble_from_smiles_optimized.xyz")

    results = gpuma.optimize_ensemble_smiles(
        smiles=smiles,
        output_file=output_file,
        config=config
    )

    print("✓ Ensemble optimization successful!")
    print(f"  Generated conformers: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Conformer {i + 1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_batch_from_multi_xyz():
    """Example 2: Batch optimize structures from a multi-structure XYZ file."""
    print("\n=== Example 2: Batch optimization from multi-XYZ file ===")

    input_file = "example_input_xyzs/butene_triplet_multi.xyz"
    output_file = os.path.join(OUTPUT_DIR, "python_batch_from_multi_xyz_optimized.xyz")

    config = load_config_from_file("config.json")
    config.optimization.multiplicity = 3
    results = gpuma.optimize_batch_multi_xyz_file(
        input_file=input_file,
        output_file=output_file,
        config=config
    )

    print("✓ Batch optimization successful!")
    print(f"  Successfully optimized: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Structure {i + 1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_batch_from_xyz_directory():
    """Example 3: Batch optimize structures from a directory of XYZ files."""
    print("\n=== Example 3: Batch optimization from XYZ directory ===")

    input_dir = "example_input_xyzs/multi_xyz_dir"
    output_file = os.path.join(OUTPUT_DIR, "python_batch_from_directory_optimized.xyz")

    results = gpuma.optimize_batch_xyz_directory(
        input_directory=input_dir,
        output_file=output_file
    )

    comments = [f"Optimized structure {i + 1} from directory" for i in range(len(results))]
    gpuma.save_multi_xyz(results, output_file, comments)

    print("✓ Batch optimization successful!")
    print(f"  Successfully optimized: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Structure {i + 1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    print("GPUMA - Ensemble and Batch Optimization Examples")
    print("=" * 70)

    example_ensemble_from_smiles()
    example_batch_from_multi_xyz()
    example_batch_from_xyz_directory()

    print("\n" + "=" * 70)
    print("Examples completed! Check the generated XYZ files.")
