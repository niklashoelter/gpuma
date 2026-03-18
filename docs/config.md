# Configuration

All supported parameters live under the `optimization` key. Only these parameters are effective in the code; unknown fields are preserved.

**Always use a config file for CLI and API calls.**

Example (JSON, Fairchem UMA):

```json
{
  "optimization": {
    "batch_optimization_mode": "batch",
    "batch_optimizer": "fire",
    "max_memory_padding": 0.95,
    "steps_between_swaps": 5,
    "max_num_conformers": 20,
    "conformer_seed": 42,

    "charge": 0,
    "multiplicity": 1,

    "force_convergence_criterion": 5e-2,
    "energy_convergence_criterion": null,

    "model_type": "fairchem",
    "model_name": "uma-s-1p1",
    "model_path": null,
    "model_cache_dir": null,
    "device": "cuda",
    "huggingface_token": null,
    "huggingface_token_file": "/home/hf_secret",

    "logging_level": "INFO"
  }
}
```

Example (JSON, ORB-v3):

```json
{
  "optimization": {
    "model_type": "orb",
    "model_name": "orb_v3_direct_omol",
    "device": "cuda",
    "d3_correction": false
  }
}
```

Example (JSON, ORB-v3 with D3 dispersion correction):

```json
{
  "optimization": {
    "model_type": "orb",
    "model_name": "orb_v3_direct_omol",
    "device": "cuda",
    "d3_correction": true,
    "d3_functional": "PBE",
    "d3_damping": "BJ"
  }
}
```

Example (YAML):

```yaml
optimization:
  batch_optimization_mode: batch
  batch_optimizer: fire
  max_memory_padding: 0.95
  steps_between_swaps: 5
  max_num_conformers: 20
  conformer_seed: 42

  charge: 0
  multiplicity: 1

  force_convergence_criterion: 5.0e-2
  energy_convergence_criterion: null

  model_type: fairchem
  model_name: uma-s-1p1
  model_path: null
  model_cache_dir: null
  device: cuda
  huggingface_token: null
  huggingface_token_file: /home/hf_secret

  logging_level: INFO
```
### Supported Parameters

- `batch_optimization_mode`: controls the ensemble mode
  - `sequential`: ASE/BFGS per conformer with a shared calculator
  - `batch`: torch-sim batch optimization (accelerated for larger ensembles)
- `batch_optimizer`: optimizer for batch mode; `fire` (default) or `gradient_descent`
- `max_memory_padding`: fraction of estimated GPU memory to use for batch optimization
  (default: `0.95`). Lower values leave more headroom and reduce the risk of OOM errors
  at the cost of smaller batch sizes. Useful to tune when running large models on shared
  GPUs or when the FIRE optimizer requires significant additional memory.
- `steps_between_swaps`: number of optimization steps between batch swaps in the
  autobatcher (default: `5`). Lower values swap batches more frequently, which can
  improve GPU utilization for workloads with heterogeneous structure sizes.
- `max_num_conformers`: max number of conformers to generate from SMILES (if applicable)
- `conformer_seed`: random seed for conformer generation (if applicable)
- `charge`: total charge of the system (for SMILES this is inferred from the
  input and not overridden by this setting)
- `multiplicity`: spin multiplicity of the system
- `force_convergence_criterion`: force convergence threshold (default: 5e-2).
  Used for both single and batch optimizations.
- `energy_convergence_criterion`: energy convergence threshold (default: None).
  If provided, it is used for batch optimization (unless force is also set).
  Not supported for single structure optimization.
- `model_type`: model backend to use; one of `fairchem` (or `uma`) for Fairchem UMA
  models, or `orb` (or `orb-v3`) for ORB-v3 models (default: `fairchem`).
  ORB-v3 uses the `orb-models` package (included in core dependencies).
- `model_name`: model identifier (see [Available Models](#available-models) below).
  - For **Fairchem**: e.g. `uma-s-1p1` (default).
  - For **ORB-v3**: e.g. `orb_v3_direct_omol` (recommended).
- `model_path`: local path to a Fairchem UMA model checkpoint (overrides `model_name`
  if set; not used for ORB models)
- `model_cache_dir`: directory to cache downloaded models (default: `~/.cache/fairchem`)
- `device`: compute device string; `cpu`, `cuda` (use default/next available GPU),
  or `cuda:N` to select a specific GPU (e.g. `cuda:0`, `cuda:1`).
  If the requested GPU index does not exist, GPUMA falls back to `cuda:0` with
  a warning. Alternatively, you can use the `CUDA_VISIBLE_DEVICES` environment
  variable to control GPU visibility before running the CLI or Python code.

- `huggingface_token`: optional HF token for model access (if required)
- `huggingface_token_file`: optional file path to read the HF token from
- `d3_correction`: enable D3 dispersion correction for ORB models (default: `false`).
  Wraps the model with `D3SumModel` from `orb-models`. Has no effect on Fairchem models.
- `d3_functional`: DFT functional for D3 correction (default: `"PBE"`). Only used
  when `d3_correction` is `true`.
- `d3_damping`: damping scheme for D3 correction (default: `"BJ"`). Only used
  when `d3_correction` is `true`.
- `logging_level`: logging verbosity; e.g., `DEBUG`, `INFO`, `WARNING`

### Available Models

#### Fairchem UMA models (`model_type: "fairchem"`)

| Model Name | Description |
|---|---|
| `uma-s-1p2` | UMA small, version 1.2 |
| `uma-s-1p1` | UMA small, version 1.1 (default) |
| `uma-m-1p1` | UMA medium, version 1.1 |

> **Note:** Fairchem UMA models require a HuggingFace token for download.

#### ORB-v3 models (`model_type: "orb"`)

Use the **underscored** form as `model_name` in configuration.

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

> **Note:** D3 dispersion correction can be enabled for any ORB model by setting
> `d3_correction: true`.

These model names are also available programmatically as
`gpuma.AVAILABLE_FAIRCHEM_MODELS` and `gpuma.AVAILABLE_ORB_MODELS`.

See the `examples/` folder for:
- Simple single optimization (`example_single_optimization.py`)
- Ensemble optimization from SMILES, multi-XYZ, directories (`example_ensemble_optimization.py`)
- CLI examples (`example_cli_usage.py`)

The example configs are sanitized (no tokens in plain text).
