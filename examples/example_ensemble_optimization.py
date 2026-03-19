#!/usr/bin/env python3
"""Example: Ensemble and Batch Optimization with GPUMA.

Demonstrates how to optimize SMILES-generated conformer ensembles and how to
batch-optimize structures from multi-XYZ files or directories.  Examples cover
both the Fairchem UMA and ORB-v3 backends.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gpuma
from gpuma.config import Config, load_config_from_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "example_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fairchem UMA examples
# ---------------------------------------------------------------------------


def example_ensemble_from_smiles():
    """Example 1: Conformer ensemble optimization (Fairchem UMA, batch)."""
    print("=== Example 1: Ensemble optimization from SMILES (Fairchem UMA) ===")

    smiles = "CCC(CC)CCOOC(CC)CCOC"
    config = Config()
    config.conformer_generation.max_num_conformers = 50
    config.optimization.force_convergence_criterion = 5e-1
    print(f"Generating conformers for {smiles} and optimizing...")

    output_file = os.path.join(OUTPUT_DIR, "python_ensemble_from_smiles_optimized.xyz")

    results = gpuma.optimize_ensemble_smiles(
        smiles=smiles,
        output_file=output_file,
        config=config,
    )

    print("  Ensemble optimization successful!")
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
        config=config,
    )

    print("  Batch optimization successful!")
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
        output_file=output_file,
    )

    comments = [f"Optimized structure {i + 1} from directory" for i in range(len(results))]
    gpuma.save_multi_xyz(results, output_file, comments)

    print("  Batch optimization successful!")
    print(f"  Successfully optimized: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Structure {i + 1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


# ---------------------------------------------------------------------------
# ORB-v3 examples
# ---------------------------------------------------------------------------


def example_ensemble_from_smiles_orb():
    """Example 4: Conformer ensemble optimization (ORB-v3, batch mode).

    Uses GPU-accelerated torch-sim batch optimization with the ORB-v3 model.
    """
    print("\n=== Example 4: Ensemble optimization from SMILES (ORB-v3, batch) ===")

    smiles = "CCC(CC)CCOOC(CC)CCOC"
    config = load_config_from_file("config_orb.json")
    config.conformer_generation.max_num_conformers = 50
    config.optimization.force_convergence_criterion = 5e-1
    # batch mode is the default in config_orb.json
    print(f"Generating conformers for {smiles} and optimizing with ORB-v3 (batch)...")

    output_file = os.path.join(OUTPUT_DIR, "python_ensemble_orb_batch.xyz")

    results = gpuma.optimize_ensemble_smiles(
        smiles=smiles,
        output_file=output_file,
        config=config,
    )

    print("  ORB-v3 batch ensemble optimization successful!")
    print(f"  Generated conformers: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Conformer {i + 1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_batch_from_multi_xyz_orb():
    """Example 6: Batch optimize structures from a multi-XYZ file (ORB-v3)."""
    print("\n=== Example 6: Batch optimization from multi-XYZ file (ORB-v3) ===")

    input_file = "example_input_xyzs/butene_triplet_multi.xyz"
    output_file = os.path.join(OUTPUT_DIR, "python_batch_from_multi_xyz_orb.xyz")

    config = load_config_from_file("config_orb.json")
    config.optimization.multiplicity = 3
    results = gpuma.optimize_batch_multi_xyz_file(
        input_file=input_file,
        output_file=output_file,
        config=config,
    )

    print("  ORB-v3 batch optimization from multi-XYZ successful!")
    print(f"  Successfully optimized: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Structure {i + 1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_batch_from_xyz_directory_orb():
    """Example 7: Batch optimize structures from a directory of XYZ files (ORB-v3)."""
    print("\n=== Example 7: Batch optimization from XYZ directory (ORB-v3) ===")

    input_dir = "example_input_xyzs/multi_xyz_dir"
    output_file = os.path.join(OUTPUT_DIR, "python_batch_from_directory_orb.xyz")

    config = load_config_from_file("config_orb.json")
    results = gpuma.optimize_batch_xyz_directory(
        input_directory=input_dir,
        output_file=output_file,
        config=config,
    )

    print("  ORB-v3 batch optimization from directory successful!")
    print(f"  Successfully optimized: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Structure {i + 1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


def example_ensemble_from_smiles_orb_sequential():
    """Example 8: Conformer ensemble optimization (ORB-v3, sequential mode).

    Falls back to ASE/BFGS per-structure optimization.  Useful when no GPU
    is available or for small ensembles.
    """
    print("\n=== Example 8: Ensemble optimization from SMILES (ORB-v3, sequential) ===")

    smiles = "CCC(CC)CCOOC(CC)CCOC"
    config = load_config_from_file("config_orb.json")
    config.conformer_generation.max_num_conformers = 10
    config.optimization.batch_optimization_mode = "sequential"
    print(f"Generating conformers for {smiles} and optimizing with ORB-v3 (sequential)...")

    output_file = os.path.join(OUTPUT_DIR, "python_ensemble_orb_sequential.xyz")

    results = gpuma.optimize_ensemble_smiles(
        smiles=smiles,
        output_file=output_file,
        config=config,
    )

    print("  ORB-v3 sequential ensemble optimization successful!")
    print(f"  Generated conformers: {len(results)}")
    for i, s in enumerate(results):
        print(f"  Conformer {i + 1}: {s.energy:.6f} eV")
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    print("GPUMA - Ensemble and Batch Optimization Examples")
    print("=" * 70)

    example_ensemble_from_smiles()
    example_batch_from_multi_xyz()
    example_batch_from_xyz_directory()
    example_ensemble_from_smiles_orb()
    example_batch_from_multi_xyz_orb()
    example_batch_from_xyz_directory_orb()
    example_ensemble_from_smiles_orb_sequential()

    print("\n" + "=" * 70)
    print("Examples completed! Check the generated XYZ files.")
