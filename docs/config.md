# Configuration

The configuration is organized into four top-level sections:

| Section | Purpose |
|---|---|
| `optimization` | Batch mode, convergence criteria, charge & multiplicity |
| `model` | Model backend, name, SevenNet modal, checkpoints, tokens, D3 correction |
| `conformer_generation` | Conformer count and random seed |
| `technical` | Device selection, memory & autobatcher knobs, logging |

Unknown fields are preserved. **Always use a config file for CLI and API calls.**

---

## Full Example (JSON, Fairchem UMA)

```json
{
  "optimization": {
    "batch_optimization_mode": "batch",
    "batch_optimizer": "fire",

    "charge": 0,
    "multiplicity": 1,

    "force_convergence_criterion": 5e-2,
    "energy_convergence_criterion": null
  },

  "model": {
    "model_type": "fairchem",
    "model_name": "uma-s-1p2p1-ft-d4rev",
    "model_modal": null,

    "model_path": null,
    "model_cache_dir": null,

    "huggingface_token": null,
    "huggingface_token_file": "/home/hf_secret",

    "d3_correction": false,
    "d3_functional": "PBE",
    "d3_damping": "BJ"
  },

  "conformer_generation": {
    "max_num_conformers": 20,
    "conformer_seed": 42
  },

  "technical": {
    "device": "cuda",
    "max_memory_padding": 0.95,
    "memory_scaling_factor": 1.75,

    "max_atoms_to_try": 100000,
    "memory_scales_with": "auto",
    "steps_between_swaps": 1,

    "logging_level": "INFO"
  }
}
```

## Minimal Example (JSON, ORB-v3)

```json
{
  "model": {
    "model_type": "orb",
    "model_name": "orb_v3_direct_omol"
  },
  "technical": {
    "device": "cuda"
  }
}
```

## Minimal Example (JSON, SevenNet)

For multi-modal checkpoints such as `7net-omni` or `7net-mf-ompa`, a
`model_modal` fidelity must be supplied. Single-modal checkpoints (e.g.
`7net-0`) ignore it.

```json
{
  "model": {
    "model_type": "sevennet",
    "model_name": "7net-omni",
    "model_modal": "omol25_high"
  },
  "technical": {
    "device": "cuda"
  }
}
```

## D3 Dispersion Correction

D3 is off by default and adds GPU memory per batch; enable it only for models
trained without dispersion. `uma-s-1p2p1-ft-d4rev` already includes dispersion
and rejects D3. The other UMA models and the ORB `*_omol` models are trained on
ωB97M-V, which also includes it, so gpuma logs a warning when D3 is enabled for
them.

DFT-D3(BJ) dispersion correction is supported for the ORB-v3,
Fairchem/UMA, and SevenNet backends.  Internally ORB uses orb-models' native
``D3SumModel`` and Fairchem uses torch-sim's ``D3DispersionModel``
(via ``SumModel`` in batch mode); both share the same
``nvalchemiops`` GPU kernel. SevenNet uses its own native D3 implementation
(``SevenNetD3Model`` for the batch path, ``SevenNetD3Calculator`` for the
single-structure path).

```json
{
  "model": {
    "model_type": "sevennet",
    "model_name": "7net-0",

    "d3_correction": true,
    "d3_functional": "PBE",
    "d3_damping": "BJ"
  },
  "technical": {
    "device": "cuda"
  }
}
```

## Full Example (YAML)

```yaml
optimization:
  batch_optimization_mode: batch
  batch_optimizer: fire

  charge: 0
  multiplicity: 1

  force_convergence_criterion: 5.0e-2
  energy_convergence_criterion: null

model:
  model_type: fairchem
  model_name: uma-s-1p2p1-ft-d4rev
  model_modal: null

  model_path: null
  model_cache_dir: null

  huggingface_token: null
  huggingface_token_file: /home/hf_secret

  d3_correction: false
  d3_functional: PBE
  d3_damping: BJ

conformer_generation:
  max_num_conformers: 20
  conformer_seed: 42

technical:
  device: cuda
  max_memory_padding: 0.95
  memory_scaling_factor: 1.75

  max_atoms_to_try: 100000
  memory_scales_with: auto
  steps_between_swaps: 1

  logging_level: INFO
```

---

## Parameter Reference

### `optimization`

| Parameter | Default | Description |
|---|---|---|
| `batch_optimization_mode` | `"batch"` | `"sequential"` (ASE per structure) or `"batch"` (torch-sim GPU-accelerated) |
| `batch_optimizer` | `"fire"` | Optimizer for both single and batch modes: `"fire"`, `"gradient_descent"`, `"lbfgs"`, or `"bfgs"`. In single-structure (ASE) mode, maps to ASE FIRE/BFGS/LBFGS (`"gradient_descent"` falls back to FIRE). Defaults to `"fire"` if unset; an unknown value is an error |
| `charge` | `0` | Total charge of the system (inferred from SMILES, not overridden) |
| `multiplicity` | `1` | Spin multiplicity of the system |
| `force_convergence_criterion` | `5e-2` | Force convergence threshold (eV/A). Used for both single and batch modes |
| `energy_convergence_criterion` | `null` | Energy convergence threshold. Batch mode only; force takes precedence if both set |

### `model`

| Parameter | Default | Description |
|---|---|---|
| `model_type` | `"fairchem"` | Backend: `"fairchem"` / `"uma"`, `"orb"` / `"orb-v3"`, or `"sevennet"` / `"7net"` |
| `model_name` | `"uma-s-1p2p1-ft-d4rev"` | Model identifier (see [Available Models](#available-models)) |
| `model_modal` | `null` | SevenNet multi-modal fidelity selector. Valid values depend on the checkpoint (`7net-omni`: e.g. `"omol25_high"`, `"omol25_low"`, `"spice"`, `"qcml"`; `7net-mf-ompa`: `"mpa"`, `"omat24"`). Required for multi-modal checkpoints; ignored by single-modal checkpoints and by other backends |
| `model_path` | `null` | Local checkpoint path (Fairchem and SevenNet; overrides `model_name`) |
| `model_cache_dir` | `null` | Directory for cached model downloads |
| `huggingface_token` | `null` | HuggingFace token for model access |
| `huggingface_token_file` | `null` | File path to read the HF token from |
| `d3_correction` | `false` | Enable D3 dispersion correction. Supported for the ORB, Fairchem/UMA, and SevenNet backends; rejected for `uma-s-1p2p1-ft-d4rev` |
| `d3_functional` | `"PBE"` | DFT functional for D3 correction (ORB/Fairchem use the orb-models BJ-damping table; SevenNet uses its own native D3) |
| `d3_damping` | `"BJ"` | Damping scheme for D3 correction (`"BJ"` or `"BJM"`) |

### `conformer_generation`

| Parameter | Default | Description |
|---|---|---|
| `max_num_conformers` | `20` | Maximum number of conformers to generate from SMILES |
| `conformer_seed` | `42` | Random seed for conformer generation |

### `technical`

| Parameter | Default | Description |
|---|---|---|
| `device` | `"cuda"` | `"cpu"`, `"cuda"`, or `"cuda:N"` (e.g. `"cuda:0"`). A GPU index that doesn't exist, or CUDA without a usable GPU, is an error |
| `max_memory_padding` | `0.95` | Fraction of GPU memory the autobatcher is allowed to fill during calibration. Lower = more headroom, smaller batches |
| `memory_scaling_factor` | `1.75` | Factor by which the autobatcher grows the probe size during calibration. Larger = faster calibration but coarser final batch size; smaller = slower but tighter. A job runs as one batch without probing only if its forward-pass peak times this factor fits in `max_memory_padding` of free memory. Must be > 1 |
| `max_atoms_to_try` | `100000` | Upper bound on the autobatcher's calibration probe size (atoms) |
| `memory_scales_with` | `"auto"` | Per-structure memory metric for batch sizing: `"n_atoms"`, `"n_edges"` or `"auto"`. `"auto"` uses `n_atoms` for UMA, whose large fixed memory per structure makes edge counts overcharge small molecules, and `n_edges` for ORB and SevenNet |
| `steps_between_swaps` | `1` | Optimization steps between batch swaps in the in-flight autobatcher. `1` is fastest on this codebase's screenings (uma-s/uma-m/orb); higher values are monotonically slower |
| `logging_level` | `"INFO"` | Logging verbosity: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |

> You can also control the GPU with the `CUDA_VISIBLE_DEVICES` environment variable.

Before calibration, torch-sim measures every system's neighbor-list edge count so it can bin-pack batches. torch-sim runs that measurement over the whole submitted batch in one pass, which holds roughly ten kilobytes per atom and overflows a 44 GB card at a few million atoms. GPUMA routes the measurement through chunks sized from the device's free memory, so batch size is bounded by the optimization itself, not by this bookkeeping. The per-system numbers are identical either way, and there is no knob to set.

---

## Available Models

### Fairchem UMA (`model_type: "fairchem"`)

| Model Name | Description |
|---|---|
| `uma-s-1p2p1-ft-d4rev` | UMA small 1.2.1 fine-tuned on ωB97M-D4rev (default); includes dispersion, so D3 is rejected |
| `uma-s-1p2p1` | UMA small, version 1.2.1 |
| `uma-s-1p2` | UMA small, version 1.2 |
| `uma-s-1p1` | UMA small, version 1.1 |
| `uma-m-1p1` | UMA medium, version 1.1 |

> Fairchem UMA models require a HuggingFace token for download.

### ORB-v3 (`model_type: "orb"`)

Use the **underscored** form as `model_name`.

| Model Name | Description |
|---|---|
| `orb_v3_direct_omol` | ORB-v3 direct, omol (recommended) |
| `orb_v3_conservative_omol` | ORB-v3 conservative, omol |
| `orb_v3_direct_20_omat` | ORB-v3 direct, 20-layer omat |
| `orb_v3_direct_inf_omat` | ORB-v3 direct, inf omat |
| `orb_v3_conservative_20_omat` | ORB-v3 conservative, 20-layer omat |
| `orb_v3_conservative_inf_omat` | ORB-v3 conservative, inf omat |
| `orb_v3_direct_20_mpa` | ORB-v3 direct, 20-layer mpa |
| `orb_v3_direct_inf_mpa` | ORB-v3 direct, inf mpa |
| `orb_v3_conservative_20_mpa` | ORB-v3 conservative, 20-layer mpa |
| `orb_v3_conservative_inf_mpa` | ORB-v3 conservative, inf mpa |

### SevenNet (`model_type: "sevennet"`)

Multi-modal checkpoints (`7net-mf-ompa`, `7net-omni`, `7net-omni-i8`,
`7net-omni-i12`) also require a `model_modal` fidelity whose valid values
depend on the checkpoint (e.g. `omol25_high`, `omol25_low`, `spice`, `qcml`
for `7net-omni`; `mpa`, `omat24` for `7net-mf-ompa`). A local checkpoint may
instead be supplied via `model_path`.

| Model Name | Description |
|---|---|
| `7net-0` | SevenNet-0 (single-modal) |
| `7net-0_22may2024` | SevenNet-0, 22 May 2024 checkpoint |
| `7net-l3i5` | SevenNet l3i5 |
| `7net-mf-0` | SevenNet multi-fidelity 0 |
| `7net-mf-ompa` | SevenNet multi-fidelity OMPA (multi-modal) |
| `7net-omat` | SevenNet OMat |
| `7net-omni` | SevenNet Omni (multi-modal) |
| `7net-omni-i8` | SevenNet Omni i8 (multi-modal) |
| `7net-omni-i12` | SevenNet Omni i12 (multi-modal) |

> D3 dispersion correction can be enabled for any ORB, Fairchem/UMA, or SevenNet model with `d3_correction: true`, except `uma-s-1p2p1-ft-d4rev`. gpuma warns when it is enabled for UMA or ORB `*_omol` models.

These model names are also available programmatically as
`gpuma.AVAILABLE_FAIRCHEM_MODELS`, `gpuma.AVAILABLE_ORB_MODELS`, and
`gpuma.AVAILABLE_SEVENNET_MODELS`.

---

See the `examples/` folder for:
- Single optimization (`example_single_optimization.py`)
- Ensemble / batch optimization (`example_ensemble_optimization.py`)
- Dispersion (D3) correction with Fairchem and ORB (`example_dispersion.py`)
- Full multi-axis benchmark across models / optimizers / convergence (`full_benchmark.py`)

The example configs are sanitized (no tokens in plain text).
