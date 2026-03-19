#!/usr/bin/env python3
"""Example: Large-scale batch optimization with GPUMA.

Demonstrates batch optimization of a large multi-XYZ file (~4000 structures)
using both the Fairchem UMA and ORB-v3 backends.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gpuma
from gpuma.config import load_config_from_file

INPUT_FILE = "example_input_xyzs/many_structures.xyz"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "example_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def example_large_batch_fairchem():
    """Example 1: Large batch optimization with Fairchem UMA."""
    print("=== Example 1: Large batch optimization (Fairchem UMA) ===")

    config = load_config_from_file("config.json")
    output_file = os.path.join(OUTPUT_DIR, "python_large_batch_fairchem.xyz")

    structures = gpuma.read_multi_xyz(INPUT_FILE)
    print(f"  Loaded {len(structures)} structures from {INPUT_FILE}")

    results = gpuma.optimize_structure_batch(structures, config)

    gpuma.save_multi_xyz(results, output_file)
    print(f"  Optimization complete: {len(results)}/{len(structures)} successful")
    if results:
        energies = [s.energy for s in results if s.energy is not None]
        print(f"  Energy range: {min(energies):.4f} to {max(energies):.4f} eV")
    print(f"  Output saved to: {output_file}")


def example_large_batch_orb():
    """Example 2: Large batch optimization with ORB-v3."""
    print("\n=== Example 2: Large batch optimization (ORB-v3) ===")

    config = load_config_from_file("config_orb.json")
    output_file = os.path.join(OUTPUT_DIR, "python_large_batch_orb.xyz")

    structures = gpuma.read_multi_xyz(INPUT_FILE)
    print(f"  Loaded {len(structures)} structures from {INPUT_FILE}")

    results = gpuma.optimize_structure_batch(structures, config)

    gpuma.save_multi_xyz(results, output_file)
    print(f"  Optimization complete: {len(results)}/{len(structures)} successful")
    if results:
        energies = [s.energy for s in results if s.energy is not None]
        print(f"  Energy range: {min(energies):.4f} to {max(energies):.4f} eV")
    print(f"  Output saved to: {output_file}")


if __name__ == "__main__":
    print("GPUMA - Large-Scale Batch Optimization Examples")
    print("=" * 70)

    example_large_batch_fairchem()
    # example_large_batch_orb()

    print("\n" + "=" * 70)
    print("Examples completed! Check the generated XYZ files.")
