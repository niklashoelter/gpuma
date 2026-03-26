#!/usr/bin/env python3
"""Benchmark: Large-scale batch optimization across models.

Compares Fairchem UMA (small v1.1, small v1.2, medium) and ORB-v3
(direct and conservative) on a ~4000-structure molecucllar dataset.
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

FORCE_CRITERIA = [5e-1, 1e-1, 5e-2, 1e-2]
BASE_MODELS = [
    {
        "name": "Fairchem UMA-s-1p1",
        "config_file": "config.json",
        "overrides": {"model": {"model_name": "uma-s-1p1"}},
        "output_prefix": "benchmark_fairchem_uma_s_1p1",
    },
    {
        "name": "Fairchem UMA-s-1p2",
        "config_file": "config.json",
        "overrides": {"model": {"model_name": "uma-s-1p2"}},
        "output_prefix": "benchmark_fairchem_uma_s_1p2",
    },
    {
        "name": "Fairchem UMA-m-1p1",
        "config_file": "config.json",
        "overrides": {"model": {"model_name": "uma-m-1p1"}},
        "output_prefix": "benchmark_fairchem_uma_m_1p1",
    },
    {
        "name": "ORB-v3 direct omol",
        "config_file": "config_orb.json",
        "overrides": {"model": {"model_name": "orb_v3_direct_omol"}},
        "output_prefix": "benchmark_orb_v3_omol_direct",
    },
    {
        "name": "ORB-v3 conservative omol",
        "config_file": "config_orb.json",
        "overrides": {"model": {"model_name": "orb_v3_conservative_omol"}},
        "output_prefix": "benchmark_orb_v3_omol_conservative",
    },
]

# Expand each model with every force convergence criterion
MODELS = []
for _base in BASE_MODELS:
    for _fc in FORCE_CRITERIA:
        _label = f"{_fc:.0e}".replace("+", "").replace("-0", "-")
        MODELS.append(
            {
                "name": f"{_base['name']} (fconv={_fc})",
                "config_file": _base["config_file"],
                "overrides": {
                    **_base["overrides"],
                    "optimization": {"force_convergence_criterion": _fc},
                },
                "output": f"{_base['output_prefix']}_fconv{_label}.xyz",
            }
        )

CSV_FIELDS = [
    "model",
    "optimizer",
    "force_convergence_criterion",
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

    force_crit = config.optimization.force_convergence_criterion
    batch_optimizer = str(config.optimization.batch_optimizer)
    row = {
        "model": name,
        "optimizer": batch_optimizer,
        "force_convergence_criterion": force_crit,
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

    print(f"  Optimizer:  {batch_optimizer}")
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
    print(f"  {'Model':<35s} {'Optim':>10s} {'fconv':>8s} {'Success':>8s} {'Time (s)':>10s} {'Struct/s':>10s}")
    print(f"  {'-' * 35} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10}")
    for r in rows:
        rate = r["throughput_structures_per_sec"] or 0
        fc = r.get("force_convergence_criterion")
        fc_str = f"{fc:.0e}" if fc is not None else "N/A"
        optim = r.get("optimizer") or "N/A"
        print(
            f"  {r['model']:<35s} "
            f"{optim:>10s}"
            f" {fc_str:>8s}"
            f" {r['structures_output']:>4d}/{r['structures_input']:<4d}"
            f" {r['total_time_sec']:>10.1f}"
            f" {rate:>10.1f}"
        )
    print(f"{'=' * 90}")


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

    csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    benchmark_results = []
    for spec in MODELS:
        try:
            row = run_benchmark(spec, structures)
            benchmark_results.append(row)
        except BaseException as exc:
            print(f"  FAILED: {exc}")
            failed_row = {f: None for f in CSV_FIELDS}
            failed_row["model"] = spec["name"]
            failed_row["optimizer"] = spec["overrides"].get(
                "optimization", {}
            ).get("batch_optimizer", "fire")
            failed_row["force_convergence_criterion"] = spec["overrides"].get(
                "optimization", {}
            ).get("force_convergence_criterion")
            failed_row["structures_input"] = len(structures)
            failed_row["structures_output"] = 0
            benchmark_results.append(failed_row)
        save_csv(benchmark_results, csv_path)

    print_summary(benchmark_results)
