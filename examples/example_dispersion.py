#!/usr/bin/env python3
"""Example: DFT-D3(BJ) Dispersion Correction with GPUMA.

Demonstrates how to enable DFT-D3 dispersion correction on top of either
backend supported by GPUMA, in both single-structure and batch
optimization modes:

- **Fairchem UMA**: layered with torch-sim's ``D3DispersionModel`` via
  ``SumModel`` for batch and via a thin ASE wrapper for single-structure
  optimization (added in torch-sim 0.6.0).
- **ORB-v3**: uses orb-models' native ``D3SumModel``.

Both backends share the same ``nvalchemiops`` D3 GPU kernel underneath
and accept the same configuration knobs:

```json
"model": {
  "d3_correction": true,
  "d3_functional": "PBE",
  "d3_damping": "BJ"
}
```

For each backend this script runs:
1. A *single-structure* optimization with D3 off vs. on (same SMILES).
2. A *batch* optimization of an ensemble of conformers with D3 off vs. on.

The reported energies show the D3 contribution. This only demonstrates the
mechanics: UMA and ORB OMol are trained on data that already includes
dispersion, so gpuma logs a double-counting warning here, and the default
``uma-s-1p2p1-ft-d4rev`` rejects D3. Use D3 for models trained without it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gpuma
from gpuma.config import load_config_from_file

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "example_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# A short alkane chain — D3 contributes a non-trivial attractive correction
# for systems where intramolecular dispersion matters.
SINGLE_SMILES = "CCCCCCCC"
# A small ensemble for the batch demo.
BATCH_SMILES = "CCCCCCCC"
BATCH_NUM_CONFORMERS = 5


def _build_config(
    backend: str, *, d3: bool, batch: bool = False
):
    """Build a Config for ``backend`` with D3 toggled and the requested mode."""
    config_file = "config.json" if backend == "fairchem" else "config_orb.json"
    cfg = load_config_from_file(config_file)
    if backend == "fairchem":
        cfg.model.model_name = "uma-s-1p2p1"
    cfg.model.d3_correction = d3
    if d3:
        cfg.model.d3_functional = "PBE"
        cfg.model.d3_damping = "BJ"
    cfg.optimization.batch_optimization_mode = "batch" if batch else "sequential"
    return cfg


def _run_single(backend: str, d3: bool) -> float:
    """Run a single-structure optimization and return the final energy."""
    cfg = _build_config(backend, d3=d3, batch=False)
    out_name = f"dispersion_{backend}_{'d3' if d3 else 'no_d3'}_single.xyz"
    structure = gpuma.optimize_single_smiles(
        smiles=SINGLE_SMILES,
        output_file=os.path.join(OUTPUT_DIR, out_name),
        config=cfg,
    )
    return float(structure.energy)


def _run_batch(backend: str, d3: bool) -> list[float]:
    """Run a batch optimization on a small conformer ensemble and return energies."""
    cfg = _build_config(backend, d3=d3, batch=True)
    cfg.conformer_generation.max_num_conformers = BATCH_NUM_CONFORMERS
    out_name = f"dispersion_{backend}_{'d3' if d3 else 'no_d3'}_batch.xyz"
    results = gpuma.optimize_ensemble_smiles(
        smiles=BATCH_SMILES,
        output_file=os.path.join(OUTPUT_DIR, out_name),
        config=cfg,
    )
    return [float(s.energy) for s in results if s.energy is not None]


def example_for_backend(backend: str, label: str) -> None:
    """Run single + batch comparisons (D3 off vs on) for one backend."""
    print(f"=== {label}: D3 off vs on ===")
    print(f"Substrate (single + batch): {SINGLE_SMILES}")

    # --- Single optimization
    e_off = _run_single(backend, d3=False)
    e_on = _run_single(backend, d3=True)
    print("  Single optimization:")
    print(f"    no D3            E = {e_off:.6f} eV")
    print(f"    + D3(PBE/BJ)     E = {e_on:.6f} eV")
    print(f"    Δ(D3 - no D3)    = {e_on - e_off:+.6f} eV")

    # --- Batch optimization
    es_off = _run_batch(backend, d3=False)
    es_on = _run_batch(backend, d3=True)
    print(
        f"  Batch optimization ({len(es_off)} / {len(es_on)} conformers converged):"
    )
    if es_off and es_on:
        print(
            f"    no D3            E_min = {min(es_off):.6f} eV   "
            f"E_mean = {sum(es_off)/len(es_off):.6f} eV"
        )
        print(
            f"    + D3(PBE/BJ)     E_min = {min(es_on):.6f} eV   "
            f"E_mean = {sum(es_on)/len(es_on):.6f} eV"
        )
        print(
            f"    Δ(mean energy)   = "
            f"{(sum(es_on)/len(es_on)) - (sum(es_off)/len(es_off)):+.6f} eV"
        )
    print()


if __name__ == "__main__":
    print("GPUMA - D3 Dispersion Correction Example")
    print("=" * 70)
    example_for_backend("fairchem", "Fairchem UMA")
    example_for_backend("orb", "ORB-v3")
    print("=" * 70)
    print("Done. Optimized geometries saved to examples/example_output/.")
