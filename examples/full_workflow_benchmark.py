#!/usr/bin/env python3
"""End-to-end workflow benchmark for GPUMA.

Exercises the *entire* pipeline:

    SMILES  --(nvMolKit GPU / morfeus CPU conformer generation)-->  conformers
            --(torch-sim batch optimization)-->  optimized structures

across three dataset sizes (small / medium / large SMILES sets) and all model
backends that make sense for isolated molecules:

  - uma-s-1p2p1-ft-d4rev    (Fairchem UMA, omol, default model)
  - orb_v3_direct_omol      (ORB-v3)
  - orb_v3_conservative_omol(ORB-v3)
  - 7net-0                  (SevenNet)   <- one SevenNet model

Force convergence is fixed at fconv = 5e-2 for every run (per request).

Conformer generation is done ONCE per dataset size (it is model-agnostic
geometry) and the resulting conformer set is then optimized by each model, so
the nvMolKit/morfeus embedding path is exercised at all three scales and every
backend is exercised on identical inputs.

Backend for conformer generation and optimization follows ``technical.device``:
run this with ``CUDA_VISIBLE_DEVICES`` pointing at a free GPU and ``--device
cuda:0``.
"""
import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gpuma
from gpuma.config import load_config_from_file
from gpuma.utils.decorators import capture_timings
from gpuma.conformer_generation import generate_ensembles

_HERE = os.path.dirname(__file__)
SMI_DIR = os.path.join(_HERE, "example_input_smiles")
DEFAULT_OUTPUT = os.path.join(_HERE, "example_output", "full_workflow_benchmark")
DEFAULT_DEVICE = "cuda:0"
FCONV = 5e-2


# ---- Axes ------------------------------------------------------------------

@dataclass(frozen=True)
class ModelChoice:
    label: str
    config_file: str
    model_type: str
    model_name: str


MODELS = [
    ModelChoice("uma-s-1p2",            "config.json",     "fairchem", "uma-s-1p2"),
    ModelChoice("uma-s-1p2p1",          "config.json",     "fairchem", "uma-s-1p2p1"),
    ModelChoice("uma-s-1p2p1-d4rev",    "config.json",     "fairchem", "uma-s-1p2p1-ft-d4rev"),
    ModelChoice("uma-m-1p1",            "config.json",     "fairchem", "uma-m-1p1"),
    ModelChoice("orb-v3-direct",        "config_orb.json", "orb",      "orb_v3_direct_omol"),
    ModelChoice("orb-v3-conservative",  "config_orb.json", "orb",      "orb_v3_conservative_omol"),
    ModelChoice("7net-0",               "config.json",     "sevennet", "7net-0"),
]

# Default set: the default UMA model plus ORB and SevenNet; pass --models for the others.
DEFAULT_MODEL_LABELS = ["uma-s-1p2p1-d4rev", "orb-v3-direct", "orb-v3-conservative", "7net-0"]

# (label, smiles file, conformers per molecule) -> aims for ~50 / ~1000 / ~4000 structures
DATASETS = [
    ("small",  "small.smi",  10),
    ("medium", "medium.smi", 50),
    ("large",  "large.smi",  100),
]


def read_smiles(path):
    smiles = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smiles.append(line.split()[0])
    return smiles


# ---- Config building -------------------------------------------------------

def build_config(model, device):
    cfg = load_config_from_file(os.path.join(_HERE, model.config_file))
    cfg.model.model_type = model.model_type
    cfg.model.model_name = model.model_name
    cfg.model.d3_correction = False  # keep the axis clean/comparable across backends
    cfg.optimization.batch_optimization_mode = "batch"
    cfg.optimization.batch_optimizer = "fire"
    cfg.optimization.force_convergence_criterion = FCONV
    cfg.optimization.charge = 0
    cfg.optimization.multiplicity = 1
    cfg.technical.memory_scaling_factor = 1.75
    cfg.technical.max_memory_padding = 0.95
    cfg.technical.steps_between_swaps = 1
    cfg.technical.device = device
    return cfg


def gen_config(device):
    cfg = load_config_from_file(os.path.join(_HERE, "config.json"))
    cfg.optimization.charge = 0
    cfg.optimization.multiplicity = 1
    cfg.technical.device = device
    return cfg


# ---- CSV -------------------------------------------------------------------

CSV_FIELDS = [
    "run_id", "dataset", "model", "force_convergence_criterion",
    "n_smiles", "structures_input", "structures_output", "success_rate_pct",
    "gen_time_sec", "opt_time_total_sec",
    "throughput_structures_per_sec", "total_atoms",
    "energy_min_eV", "energy_max_eV", "energy_mean_eV",
    "error",
]


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def make_row(run_id, dataset, model, n_smiles, conformers, results, gen_time, opt_time, error):
    n_in = len(conformers)
    n_out = len(results)
    energies = [s.energy for s in results if s.energy is not None]
    atoms = [s.n_atoms for s in results]
    return {
        "run_id": run_id,
        "dataset": dataset,
        "model": model.label,
        "force_convergence_criterion": FCONV,
        "n_smiles": n_smiles,
        "structures_input": n_in,
        "structures_output": n_out,
        "success_rate_pct": round(100 * n_out / n_in, 1) if n_in else 0,
        "gen_time_sec": round(gen_time, 1),
        "opt_time_total_sec": round(opt_time, 1),
        "throughput_structures_per_sec": round(n_out / opt_time, 1) if opt_time > 0 else None,
        "total_atoms": sum(atoms),
        "energy_min_eV": round(min(energies), 4) if energies else None,
        "energy_max_eV": round(max(energies), 4) if energies else None,
        "energy_mean_eV": round(sum(energies) / len(energies), 4) if energies else None,
        "error": error,
    }


# ---- Main ------------------------------------------------------------------

def main(output_dir=DEFAULT_OUTPUT, device=DEFAULT_DEVICE, models=None, datasets=None):
    struct_dir = os.path.join(output_dir, "structures")
    csv_path = os.path.join(output_dir, "results.csv")
    os.makedirs(struct_dir, exist_ok=True)

    labels = models or DEFAULT_MODEL_LABELS
    selected = [m for m in MODELS if m.label in labels]
    if not selected:
        raise SystemExit(f"No models matched {labels}; choices: {[m.label for m in MODELS]}")
    active_datasets = [d for d in DATASETS if not datasets or d[0] in datasets]

    print(f"GPUMA full-workflow benchmark   out: {output_dir}   device: {device}")
    print(f"models={[m.label for m in selected]}  datasets={[d[0] for d in active_datasets]}  "
          f"fconv={FCONV:.0e}\n")

    rows = []
    run_id = 0
    for ds_label, smi_file, n_confs in active_datasets:
        smiles = read_smiles(os.path.join(SMI_DIR, smi_file))
        print(f"\n########## dataset={ds_label}  ({len(smiles)} SMILES, {n_confs} conf/mol) ##########")

        # --- conformer generation (nvMolKit GPU / morfeus CPU), once per size
        t0 = time.perf_counter()
        try:
            ensembles = generate_ensembles(
                smiles, max_num_confs=n_confs, config=gen_config(device),
                n_confs=n_confs, seed=42,
            )
        except BaseException as exc:  # noqa: BLE001
            gen_time = time.perf_counter() - t0
            print(f"  CONFORMER GENERATION FAILED: {type(exc).__name__}: {exc}")
            for model in selected:
                run_id += 1
                rows.append(make_row(run_id, ds_label, model, len(smiles), [], [], gen_time,
                                     0.0, f"gen: {type(exc).__name__}: {exc}"))
            write_csv(rows, csv_path)
            continue
        gen_time = time.perf_counter() - t0
        conformers = [c for ens in ensembles if ens for c in ens]
        n_failed = sum(1 for ens in ensembles if not ens)
        print(f"  generated {len(conformers)} conformers from {len(smiles)} SMILES "
              f"({n_failed} molecules failed) in {gen_time:.1f}s")
        gpuma.save_multi_xyz(conformers, os.path.join(struct_dir, f"{ds_label}_conformers.xyz"))

        # --- optimize with each model
        for model in selected:
            run_id += 1
            print(f"\n[{run_id}] {ds_label} | {model.label} | fconv={FCONV:.0e} "
                  f"| {len(conformers)} structures")
            cfg = build_config(model, device)
            results, error, opt_time = [], None, 0.0
            with capture_timings() as timings:
                try:
                    results = gpuma.optimize_structure_batch(conformers, cfg)
                except BaseException as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"  FAILED: {error}")
            opt_time = timings.total
            if not error:
                gpuma.save_multi_xyz(
                    results, os.path.join(struct_dir, f"{ds_label}_{model.label}.xyz")
                )
                print(f"  ok: {len(results)}/{len(conformers)}  opt_total={opt_time:.1f}s")
            rows.append(make_row(run_id, ds_label, model, len(smiles), conformers,
                                 results, gen_time, opt_time, error))
            write_csv(rows, csv_path)

    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--models", default=None,
                   help="Comma-separated model labels (default: all except uma-m-1p1). "
                        f"Choices: {[m.label for m in MODELS]}")
    p.add_argument("--datasets", default=None,
                   help="Comma-separated dataset labels to run (default: all). "
                        "Choices: small, medium, large")
    args = p.parse_args()
    models = [s.strip() for s in args.models.split(",")] if args.models else None
    datasets = [s.strip() for s in args.datasets.split(",")] if args.datasets else None
    main(output_dir=args.output_dir, device=args.device, models=models, datasets=datasets)
