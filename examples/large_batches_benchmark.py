#!/usr/bin/env python3
"""Benchmark: Large-scale batch optimization across models.

Compares Fairchem UMA (small v1.1, small v1.2, medium) and ORB-v3
(direct and conservative) on a ~4000-structure molecular dataset.
Results are saved as a CSV file.
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gpuma
from gpuma.config import load_config_from_file

INPUT_FILE = "example_input_xyzs/many_structures.xyz"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "example_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    {
        "name": "Fairchem UMA-s-1p1",
        "config_file": "config.json",
        "overrides": {"model": {"model_name": "uma-s-1p1"}},
        "output": "benchmark_fairchem_uma_s_1p1.xyz",
    },
    {
        "name": "Fairchem UMA-s-1p2",
        "config_file": "config.json",
        "overrides": {"model": {"model_name": "uma-s-1p2"}},
        "output": "benchmark_fairchem_uma_s_1p2.xyz",
    },
    {
        "name": "Fairchem UMA-m-1p1",
        "config_file": "config.json",
        "overrides": {"model": {"model_name": "uma-m-1p1"}},
        "output": "benchmark_fairchem_uma_m_1p1.xyz",
    },
    {
        "name": "ORB-v3 direct omol",
        "config_file": "config_orb.json",
        "overrides": {"model": {"model_name": "orb_v3_direct_omol"}},
        "output": "benchmark_orb_v3_omol_direct.xyz",
    },
    {
        "name": "ORB-v3 conservative omol",
        "config_file": "config_orb.json",
        "overrides": {"model": {"model_name": "orb_v3_conservative_omol"}},
        "output": "benchmark_orb_v3_omol_conservative.xyz",
    },
]

CSV_FIELDS = [
    "model",
    "structures_input",
    "structures_output",
    "success_rate_pct",
    "total_time_sec",
    "avg_time_per_structure_sec",
    "throughput_structures_per_sec",
    "total_atoms",
    "atoms_min",
    "atoms_max",
    "atoms_avg",
    "atom_throughput_per_sec",
    "energy_min_eV",
    "energy_max_eV",
    "energy_mean_eV",
    "energy_spread_eV",
]


def run_benchmark(model_spec: dict, structures: list) -> dict:
    """Run a single benchmark and return stats."""
    name = model_spec["name"]
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    config = load_config_from_file(model_spec["config_file"])
    for section, values in model_spec["overrides"].items():
        for key, val in values.items():
            setattr(getattr(config, section), key, val)

    output_file = os.path.join(OUTPUT_DIR, model_spec["output"])

    t0 = time.perf_counter()
    results = gpuma.optimize_structure_batch(list(structures), config)
    elapsed = time.perf_counter() - t0

    gpuma.save_multi_xyz(results, output_file)

    n_input = len(structures)
    n_output = len(results)
    energies = [s.energy for s in results if s.energy is not None]
    atom_counts = [s.n_atoms for s in results]
    total_atoms = sum(atom_counts) if atom_counts else 0

    row = {
        "model": name,
        "structures_input": n_input,
        "structures_output": n_output,
        "success_rate_pct": round(100 * n_output / n_input, 1) if n_input else 0,
        "total_time_sec": round(elapsed, 2),
        "avg_time_per_structure_sec": round(elapsed / n_output, 3) if n_output else None,
        "throughput_structures_per_sec": round(n_output / elapsed, 1) if elapsed > 0 else None,
        "total_atoms": total_atoms,
        "atoms_min": min(atom_counts) if atom_counts else None,
        "atoms_max": max(atom_counts) if atom_counts else None,
        "atoms_avg": round(total_atoms / len(atom_counts), 1) if atom_counts else None,
        "atom_throughput_per_sec": round(total_atoms / elapsed) if elapsed > 0 else None,
        "energy_min_eV": round(min(energies), 4) if energies else None,
        "energy_max_eV": round(max(energies), 4) if energies else None,
        "energy_mean_eV": round(sum(energies) / len(energies), 4) if energies else None,
        "energy_spread_eV": round(max(energies) - min(energies), 4) if energies else None,
    }

    print(f"  Structures: {n_output}/{n_input} successful")
    print(f"  Time:       {elapsed:.1f} sec ({n_output / elapsed:.1f} struct/sec)")
    if energies:
        print(f"  Energy:     {min(energies):.4f} to {max(energies):.4f} eV")
    print(f"  Output:     {output_file}")

    return row


def print_summary(rows: list[dict]) -> None:
    """Print a comparison table to stdout."""
    print(f"\n{'=' * 70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Model':<30s} {'Success':>8s} {'Time (s)':>10s} {'Struct/s':>10s}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 10} {'-' * 10}")
    for r in rows:
        rate = r["throughput_structures_per_sec"] or 0
        print(
            f"  {r['model']:<30s} "
            f"{r['structures_output']:>4d}/{r['structures_input']:<4d}"
            f"{r['total_time_sec']:>10.1f}"
            f"{rate:>10.1f}"
        )
    print(f"{'=' * 70}")


def save_csv(rows: list[dict], path: str) -> None:
    """Write benchmark results to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to: {path}")


if __name__ == "__main__":
    print("GPUMA - Large Batch Benchmark")
    print(f"Input: {INPUT_FILE}")

    structures = gpuma.read_multi_xyz(INPUT_FILE)
    print(f"Loaded {len(structures)} structures")

    benchmark_results = []
    for spec in MODELS:
        try:
            row = run_benchmark(spec, structures)
            benchmark_results.append(row)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            benchmark_results.append({f: None for f in CSV_FIELDS})
            benchmark_results[-1]["model"] = spec["name"]
            benchmark_results[-1]["structures_input"] = len(structures)
            benchmark_results[-1]["structures_output"] = 0

    print_summary(benchmark_results)
    save_csv(benchmark_results, os.path.join(OUTPUT_DIR, "benchmark_results.csv"))
