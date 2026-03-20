# Configuration

The configuration is organized into four top-level sections:

| Section | Purpose |
|---|---|
| `optimization` | Batch mode, convergence criteria, charge & multiplicity |
| `model` | Model backend, name, checkpoints, tokens, D3 correction |
| `conformer_generation` | Conformer count and random seed |
| `technical` | Device selection, memory padding, logging |

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
    "energy_convergence_criterion": null,
    "steps_between_swaps": 3
  },

  "model": {
    "model_type": "fairchem",
    "model_name": "uma-s-1p2",

    "model_path": null,
    "model_cache_dir": null,

    "huggingface_token": null,
    "huggingface_token_file": "/home/hf_secret"
  },

  "conformer_generation": {
    "max_num_conformers": 20,
    "conformer_seed": 42
  },

  "technical": {
    "device": "cuda",
    "max_memory_padding": 0.95,
    "memory_scaling_factor": 1.6,

    "max_atoms_to_try": 100000,

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

## ORB-v3 with D3 Dispersion Correction

```json
{
  "model": {
    "model_type": "orb",
    "model_name": "orb_v3_direct_omol",

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
  steps_between_swaps: 3

model:
  model_type: fairchem
  model_name: uma-s-1p2

  model_path: null
  model_cache_dir: null

  huggingface_token: null
  huggingface_token_file: /home/hf_secret

conformer_generation:
  max_num_conformers: 20
  conformer_seed: 42

technical:
  device: cuda
  max_memory_padding: 0.95
  memory_scaling_factor: 1.6

  max_atoms_to_try: 100000

  logging_level: INFO
```

---

## Parameter Reference

### `optimization`

| Parameter | Default | Description |
|---|---|---|
| `batch_optimization_mode` | `"batch"` | `"sequential"` (ASE/BFGS per structure) or `"batch"` (torch-sim GPU-accelerated) |
| `batch_optimizer` | `"fire"` | Optimizer for batch mode: `"fire"` or `"gradient_descent"` |
| `charge` | `0` | Total charge of the system (inferred from SMILES, not overridden) |
| `multiplicity` | `1` | Spin multiplicity of the system |
| `force_convergence_criterion` | `5e-2` | Force convergence threshold (eV/A). Used for both single and batch modes |
| `energy_convergence_criterion` | `null` | Energy convergence threshold. Batch mode only; force takes precedence if both set |
| `steps_between_swaps` | `3` | Steps between batch swaps in the autobatcher. Lower = more frequent swaps |

### `model`

| Parameter | Default | Description |
|---|---|---|
| `model_type` | `"fairchem"` | Backend: `"fairchem"` / `"uma"` or `"orb"` / `"orb-v3"` |
| `model_name` | `"uma-s-1p2"` | Model identifier (see [Available Models](#available-models)) |
| `model_path` | `null` | Local checkpoint path (Fairchem only; overrides `model_name`) |
| `model_cache_dir` | `null` | Directory for cached model downloads |
| `huggingface_token` | `null` | HuggingFace token for model access |
| `huggingface_token_file` | `null` | File path to read the HF token from |
| `d3_correction` | `false` | Enable D3 dispersion correction (ORB models only) |
| `d3_functional` | `"PBE"` | DFT functional for D3 correction |
| `d3_damping` | `"BJ"` | Damping scheme for D3 correction |

### `conformer_generation`

| Parameter | Default | Description |
|---|---|---|
| `max_num_conformers` | `20` | Maximum number of conformers to generate from SMILES |
| `conformer_seed` | `42` | Random seed for conformer generation |

### `technical`

| Parameter | Default | Description |
|---|---|---|
| `device` | `"cuda"` | `"cpu"`, `"cuda"`, or `"cuda:N"` (e.g. `"cuda:0"`). Falls back to `cuda:0` if the requested index doesn't exist |
| `max_memory_padding` | `0.95` | Fraction of GPU memory to use for batch optimization. Lower = more headroom |
| `memory_scaling_factor` | `1.6` | Factor to multiply batch size by during autobatcher calibration. Larger = faster calibration, smaller = more accurate limit. Must be > 1 |
| `max_atoms_to_try` | `100000` | Maximum atoms for autobatcher calibration probe |
| `logging_level` | `"INFO"` | Logging verbosity: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |

> You can also control the GPU with the `CUDA_VISIBLE_DEVICES` environment variable.

---

## Available Models

### Fairchem UMA (`model_type: "fairchem"`)

| Model Name | Description |
|---|---|
| `uma-s-1p2` | UMA small, version 1.2 (default) |
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

> D3 dispersion correction can be enabled for any ORB model with `d3_correction: true`.

These model names are also available programmatically as
`gpuma.AVAILABLE_FAIRCHEM_MODELS` and `gpuma.AVAILABLE_ORB_MODELS`.

---

See the `examples/` folder for:
- Single optimization (`example_single_optimization.py`)
- Ensemble / batch optimization (`example_ensemble_optimization.py`)
- Large-scale batch optimization (`large_batches_benchmark.py`)

The example configs are sanitized (no tokens in plain text).
