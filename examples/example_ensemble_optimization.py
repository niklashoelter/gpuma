#!/usr/bin/env python3
"""Example: Ensemble and Batch Optimization with GPUMA.

This example demonstrates how to optimize SMILES-generated conformer ensembles
and how to batch-optimize general structures from multi-XYZ files or directories
using the GPUMA API.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gpuma
from gpuma.config import load_config_from_file


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "example_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def example_ensemble_from_smiles():
    """Example 1: Generate and optimize a conformer ensemble from SMILES."""
    print("=== Example 1: Ensemble optimization from SMILES ===")

    smiles = "CCC(C)COCCCCOCOCCCCOCCC(CC)CCCC(CCCC)OOCCC(CCC)CCC(CC)CC(C)CC"  # Example SMILES
    num_conformers = 100
    print(f"Generating {num_conformers} conformers for {smiles} and optimizing...")

    output_file = os.path.join(OUTPUT_DIR, "ensemble_from_smiles_optimized.xyz")

    results = gpuma.optimize_smiles_ensemble(
        smiles=smiles,
        num_conformers=num_conformers,
        output_file=output_file,
    )

    print("✓ Ensemble optimization successful!")
    print(f"  Generated conformers: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Conformer {i+1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_batch_from_multi_xyz():
    """Example 2: Batch optimize structures from a multi-structure XYZ file."""
    print("\n=== Example 2: Batch optimization from multi-XYZ file ===")

    input_file = "example_input_xyzs/multi_xyz_file.xyz"
    output_file = os.path.join(OUTPUT_DIR, "batch_from_multi_xyz_optimized.xyz")

    if not os.path.exists(input_file):
        print(f"✗ Input file {input_file} not found")
        return

    structures = gpuma.read_multi_xyz(input_file)
    print(f"Read {len(structures)} structures from {input_file}")

    results = gpuma.optimize_structure_batch(structures)

    comments = [f"Optimized structure {i+1} from multi-XYZ" for i in range(len(results))]
    gpuma.save_multi_xyz(results, output_file, comments)

    print("✓ Batch optimization successful!")
    print(f"  Input structures: {len(structures)}")
    print(f"  Successfully optimized: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Structure {i+1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_batch_from_xyz_directory():
    """Example 3: Batch optimize structures from a directory of XYZ files."""
    print("\n=== Example 3: Batch optimization from XYZ directory ===")

    input_dir = "example_input_xyzs/multi_xyz_dir"
    output_file = os.path.join(OUTPUT_DIR, "batch_from_directory_optimized.xyz")

    if not os.path.exists(input_dir):
        print(f"✗ Input directory {input_dir} not found")
        return

    structures = gpuma.read_xyz_directory(input_dir)
    print(f"Read {len(structures)} XYZ files from {input_dir}/")

    import glob

    xyz_files = glob.glob(os.path.join(input_dir, "*.xyz"))
    for xyz_file in xyz_files:
        print(f"  - {os.path.basename(xyz_file)}")

    results = gpuma.optimize_structure_batch(structures)

    comments = [f"Optimized structure {i+1} from directory" for i in range(len(results))]
    gpuma.save_multi_xyz(results, output_file, comments)

    print("✓ Batch optimization successful!")
    print(f"  Input structures: {len(structures)}")
    print(f"  Successfully optimized: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Structure {i+1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_ensemble_with_config():
    """Example 4: Ensemble optimization with a custom configuration."""
    print("\n=== Example 4: Ensemble optimization with custom config ===")

    config = load_config_from_file("config.json")

    smiles = "c1ccccc1CCCCC"
    print(f"Optimizing {smiles} ensemble with custom config...")

    conformers = gpuma.smiles_to_ensemble(smiles, config.optimization.max_num_conformers)
    results = gpuma.optimize_structure_batch(conformers, config)

    output_file = os.path.join(OUTPUT_DIR, "ensemble_custom_config_optimized.xyz")

    gpuma.save_multi_xyz(
        results,
        output_file,
        [f"Optimized conformer {i+1} from SMILES: {smiles}" for i in range(len(results))],
    )

    print("✓ Custom ensemble optimization successful!")
    print(f"  Conformers: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Conformer {i+1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    print("GPUMA - Ensemble and Batch Optimization Examples")
    print("=" * 70)

    example_ensemble_from_smiles()
    example_batch_from_multi_xyz()
    example_batch_from_xyz_directory()
    example_ensemble_with_config()

    print("\n" + "=" * 70)
    print("Examples completed! Check the generated XYZ files.")
