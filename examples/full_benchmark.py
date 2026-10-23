#!/usr/bin/env python3
"""Scientific benchmark: vary models × optimizers × force-convergence on a
multi-structure xyz file. All technical autobatcher knobs are fixed at the
values determined by screening campaigns 1-3.
"""
import argparse
import csv
import itertools
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gpuma
from gpuma.config import load_config_from_file
from gpuma.utils.decorators import capture_timings

_HERE = os.path.dirname(__file__)
DEFAULT_INPUT  = os.path.join(_HERE, "example_input_xyzs", "many_structures_4k.xyz")
DEFAULT_OUTPUT = os.path.join(_HERE, "example_output", "full_benchmark")
DEFAULT_DEVICE = "cuda:0"


# ---- Axes ------------------------------------------------------------------

@dataclass(frozen=True)
class ModelChoice:
    label: str
    config_file: str
    model_name: str


MODELS = [
    ModelChoice("uma-s-1p1",           "config.json",     "uma-s-1p1"),
    ModelChoice("uma-s-1p2",           "config.json",     "uma-s-1p2"),
    ModelChoice("uma-s-1p2p1",         "config.json",     "uma-s-1p2p1"),
    ModelChoice("uma-s-1p2p1-d4rev",   "config.json",     "uma-s-1p2p1-ft-d4rev"),
    ModelChoice("uma-m-1p1",           "config.json",     "uma-m-1p1"),
    ModelChoice("orb-v3-direct",       "config_orb.json", "orb_v3_direct_omol"),
    ModelChoice("orb-v3-conservative", "config_orb.json", "orb_v3_conservative_omol"),
]

OPTIMIZERS = ["fire", "lbfgs", "bfgs"]
FCONV      = [5e-1, 1e-1, 5e-2]

# ---- Per-run config building -----------------------------------------------

def build_config(model, optimizer, fconv, device):
    cfg = load_config_from_file(os.path.join(_HERE, model.config_file))
    cfg.model.model_name    = model.model_name
    cfg.model.d3_correction = False
    cfg.optimization.batch_optimization_mode     = "batch"
    cfg.optimization.batch_optimizer             = optimizer
    cfg.optimization.force_convergence_criterion = fconv
    cfg.technical.memory_scaling_factor = 1.75
    cfg.technical.max_memory_padding    = 0.95
    cfg.technical.steps_between_swaps   = 1
    cfg.technical.device = device
    return cfg


# ---- CSV -------------------------------------------------------------------

CSV_FIELDS = [
    "run_id", "model", "optimizer", "force_convergence_criterion",
    "structures_input", "structures_output", "success_rate_pct",
    "time_model_loading", "time_memory_estimation", "time_overhead",
    "time_optimization", "time_total",
    "avg_time_per_structure_sec", "throughput_structures_per_sec",
    "total_atoms", "atoms_min", "atoms_max", "atoms_avg",
    "atom_throughput_per_sec",
    "energy_min_eV", "energy_max_eV", "energy_mean_eV", "energy_spread_eV",
    "error",
]


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def make_row(run_id, model, optimizer, fconv, results, timings, n_input, error):
    n_out = len(results)
    energies = [s.energy for s in results if s.energy is not None]
    atoms = [s.n_atoms for s in results]
    total_atoms = sum(atoms)
    t = timings.as_dict()  # zeros if the run errored before any phase

    return {
        "run_id": run_id,
        "model": model.label,
        "optimizer": optimizer,
        "force_convergence_criterion": fconv,
        "structures_input": n_input,
        "structures_output": n_out,
        "success_rate_pct": round(100 * n_out / n_input, 1) if n_input else 0,
        "time_model_loading":     round(t["model_loading"], 1),
        "time_memory_estimation": round(t["memory_estimation"], 1),
        "time_overhead":          round(t["overhead"], 1),
        "time_optimization":      round(t["optimization"], 1),
        "time_total":             round(t["total"], 1),
        "avg_time_per_structure_sec":     round(t["total"] / n_out, 3) if n_out else None,
        "throughput_structures_per_sec":  round(n_out / t["total"], 1) if t["total"] > 0 else None,
        "total_atoms": total_atoms,
        "atoms_min": min(atoms) if atoms else None,
        "atoms_max": max(atoms) if atoms else None,
        "atoms_avg": round(total_atoms / len(atoms), 1) if atoms else None,
        "atom_throughput_per_sec": round(total_atoms / t["total"]) if t["total"] > 0 else None,
        "energy_min_eV":    round(min(energies), 4) if energies else None,
        "energy_max_eV":    round(max(energies), 4) if energies else None,
        "energy_mean_eV":   round(sum(energies) / len(energies), 4) if energies else None,
        "energy_spread_eV": round(max(energies) - min(energies), 4) if energies else None,
        "error": error,
    }


# ---- Main loop -------------------------------------------------------------

def main(input_file=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT, device=DEFAULT_DEVICE):
    struct_dir = os.path.join(output_dir, "structures")
    csv_path   = os.path.join(output_dir, "results.csv")
    os.makedirs(struct_dir, exist_ok=True)

    structures = gpuma.read_multi_xyz(input_file)
    n_input = len(structures)
    grid = list(itertools.product(MODELS, OPTIMIZERS, FCONV))
    total = len(grid)

    print(f"GPUMA full benchmark   in: {input_file}   out: {output_dir}   device: {device}")
    print(f"{n_input} structures × {total} runs\n")

    rows = []
    t_overall = time.perf_counter()
    for i, (model, optimizer, fconv) in enumerate(grid, 1):
        elapsed = time.perf_counter() - t_overall
        eta_h = (elapsed / max(i - 1, 1)) * (total - i + 1) / 3600

        print(f"\n[{i:>4}/{total}] {model.label} | opt={optimizer} | fconv={fconv:.0e}")
        print(f"  elapsed {elapsed/60:.1f} min   ETA {eta_h:.1f} h")

        cfg = build_config(model, optimizer, fconv, device)

        results, error = [], None
        with capture_timings() as timings:
            try:
                results = gpuma.optimize_structure_batch(structures, cfg)
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"  FAILED: {error}")

        if not error:
            gpuma.save_multi_xyz(results, os.path.join(struct_dir, f"run_{i:04d}.xyz"))
            print(f"  ok: {len(results)}/{n_input}  "
                  f"total={timings.total:.1f}s  "
                  f"(load={timings.model_loading:.2f}, "
                  f"mem={timings.memory_estimation:.2f}, "
                  f"opt={timings.optimization:.2f})")

        rows.append(make_row(i, model, optimizer, fconv, results, timings, n_input, error))
        write_csv(rows, csv_path)

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      default=DEFAULT_INPUT,  help="Path to input xyz")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory")
    p.add_argument("--device",     default=DEFAULT_DEVICE, help="Torch device, e.g. cuda:0")
    args = p.parse_args()
    main(input_file=args.input, output_dir=args.output_dir, device=args.device)
