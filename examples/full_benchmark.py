#!/usr/bin/env python3
import csv
import itertools
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gpuma
from gpuma import capture_timings
from gpuma.config import load_config_from_file


# ---- Paths -----------------------------------------------------------------

INPUT_FILE = "example_input_xyzs/many_structures.xyz"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "example_output", "full_benchmark")
STRUCT_DIR = os.path.join(OUTPUT_DIR, "structures")
CSV_PATH = os.path.join(OUTPUT_DIR, "results.csv")
os.makedirs(STRUCT_DIR, exist_ok=True)


# ---- Axes ------------------------------------------------------------------

@dataclass(frozen=True)
class ModelChoice:
    label: str
    config_file: str
    model_name: str


MODELS = [
    # ModelChoice("uma-s-1p1",           "config.json",     "uma-s-1p1"),
    ModelChoice("uma-s-1p2",           "config.json",     "uma-s-1p2"),
    ModelChoice("uma-m-1p1",           "config.json",     "uma-m-1p1"),
    ModelChoice("orb-v3-direct",       "config_orb.json", "orb_v3_direct_omol"),
    ModelChoice("orb-v3-conservative", "config_orb.json", "orb_v3_conservative_omol"),
]
#D3_VARIANTS  = [("off", False), ("on", True)]            # PBE/BJ when on
D3_VARIANTS  = [("on", True)]            # PBE/BJ when on
#OPTIMIZERS   = ["fire", "gradient_descent", "lbfgs", "bfgs"]
OPTIMIZERS   = ["fire"]
#FCONV        = [5e-1, 1e-1, 5e-2]
FCONV        = [1e-1]
SWAPS        = [1, 3, 5, 10]
MSCALE       = [1.25, 1.50, 1.75]


# ---- Per-run config building -----------------------------------------------

def build_config(model, d3_on, optimizer, fconv, swaps, mscale):
    cfg = load_config_from_file(model.config_file)
    cfg.model.model_name = model.model_name
    cfg.model.d3_correction = d3_on
    cfg.model.d3_functional = "PBE"
    cfg.model.d3_damping    = "BJ"
    cfg.optimization.batch_optimization_mode      = "batch"
    cfg.optimization.batch_optimizer              = optimizer
    cfg.optimization.force_convergence_criterion  = fconv
    cfg.optimization.steps_between_swaps          = swaps
    cfg.technical.max_memory_padding   = 0.95
    cfg.technical.memory_scaling_factor = mscale
    cfg.technical.max_atoms_to_try     = 100_000
    cfg.technical.devide = "cuda:3"
    return cfg


# ---- CSV -------------------------------------------------------------------

CSV_FIELDS = [
    "run_id", "model", "d3", "optimizer",
    "force_convergence_criterion", "steps_between_swaps", "memory_scaling_factor",
    "structures_input", "structures_output", "success_rate_pct",
    "time_model_loading", "time_memory_estimation", "time_overhead",
    "time_optimization", "time_total",
    "avg_time_per_structure_sec", "throughput_structures_per_sec",
    "total_atoms", "atoms_min", "atoms_max", "atoms_avg",
    "atom_throughput_per_sec",
    "energy_min_eV", "energy_max_eV", "energy_mean_eV", "energy_spread_eV",
    "error",
]


def write_csv(rows):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def make_row(run_id, model, d3_label, optimizer, fconv, swaps, mscale,
             results, timings, n_input, error):
    n_out = len(results)
    energies = [s.energy for s in results if s.energy is not None]
    atoms = [s.n_atoms for s in results]
    total_atoms = sum(atoms)
    t = timings.as_dict()  # zeros if the run errored before any phase

    return {
        "run_id": run_id,
        "model": model.label,
        "d3": d3_label,
        "optimizer": optimizer,
        "force_convergence_criterion": fconv,
        "steps_between_swaps": swaps,
        "memory_scaling_factor": mscale,
        "structures_input": n_input,
        "structures_output": n_out,
        "success_rate_pct": round(100 * n_out / n_input, 1) if n_input else 0,
        "time_model_loading":     round(t["model_loading"], 3),
        "time_memory_estimation": round(t["memory_estimation"], 3),
        "time_overhead":          round(t["overhead"], 3),
        "time_optimization":      round(t["optimization"], 3),
        "time_total":             round(t["total"], 3),
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

def main():
    structures = gpuma.read_multi_xyz(INPUT_FILE)
    n_input = len(structures)
    grid = list(itertools.product(MODELS, D3_VARIANTS, OPTIMIZERS, FCONV, SWAPS, MSCALE))
    total = len(grid)

    print(f"GPUMA full benchmark   in: {INPUT_FILE}   out: {OUTPUT_DIR}")
    print(f"{n_input} structures × {total} runs\n")

    rows = []
    t_overall = time.perf_counter()
    for i, (model, (d3_label, d3_on), optimizer, fconv, swaps, mscale) in enumerate(grid, 1):
        elapsed = time.perf_counter() - t_overall
        eta_h = (elapsed / max(i - 1, 1)) * (total - i + 1) / 3600

        print(f"\n[{i:>4}/{total}] {model.label} | d3={d3_label} | opt={optimizer} | "
              f"fconv={fconv:.0e} | swaps={swaps} | mscale={mscale}")
        print(f"  elapsed {elapsed/60:.1f} min   ETA {eta_h:.1f} h")

        cfg = build_config(model, d3_on, optimizer, fconv, swaps, mscale)

        results, error = [], None
        with capture_timings() as timings:
            try:
                results = gpuma.optimize_structure_batch(structures, cfg)
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"  FAILED: {error}")

        if not error:
            gpuma.save_multi_xyz(results, os.path.join(STRUCT_DIR, f"run_{i:04d}.xyz"))
            print(f"  ok: {len(results)}/{n_input}  "
                  f"total={timings.total:.1f}s  "
                  f"(load={timings.model_loading:.2f}, "
                  f"mem={timings.memory_estimation:.2f}, "
                  f"opt={timings.optimization:.2f})")

        rows.append(make_row(i, model, d3_label, optimizer, fconv, swaps, mscale,
                             results, timings, n_input, error))
        write_csv(rows)

    print(f"\nCSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
