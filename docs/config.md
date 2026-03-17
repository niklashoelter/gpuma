# Configuration

All supported parameters live under the `optimization` key. Only these parameters are effective in the code; unknown fields are preserved.

**Always use a config file for CLI and API calls.**

Example (JSON, Fairchem UMA):

```json
{
  "optimization": {
    "batch_optimization_mode": "batch",
    "batch_optimizer": "fire",
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
- `model_name`: model identifier.
  - For **Fairchem**: e.g. `uma-s-1p1` (default). See
    [fairchem-core](https://github.com/FAIR-Chem/fairchem) for available models.
  - For **ORB-v3**: a pretrained model function name from
    `orb_models.forcefield.pretrained`, e.g. `orb_v3_direct_omol` (recommended),
    `orb_v3_conservative_inf_omat`, `orb_v3_direct_inf_omat`, etc.
- `model_path`: local path to a Fairchem UMA model checkpoint (overrides `model_name`
  if set; not used for ORB models)
- `model_cache_dir`: directory to cache downloaded models (default: `~/.cache/fairchem`)
- `device`: compute device string; one of `cpu` or `cuda`.
  Fairchem only distinguishes between CPU and CUDA; selection of specific
  GPUs should be handled via the `CUDA_VISIBLE_DEVICES` environment variable
  before running the CLI or Python code.

- `huggingface_token`: optional HF token for model access (if required)
- `huggingface_token_file`: optional file path to read the HF token from
- `d3_correction`: enable D3 dispersion correction for ORB models (default: `false`).
  Wraps the model with `D3SumModel` from `orb-models`. Has no effect on Fairchem models.
- `d3_functional`: DFT functional for D3 correction (default: `"PBE"`). Only used
  when `d3_correction` is `true`.
- `d3_damping`: damping scheme for D3 correction (default: `"BJ"`). Only used
  when `d3_correction` is `true`.
- `logging_level`: logging verbosity; e.g., `DEBUG`, `INFO`, `WARNING`

See the `examples/` folder for:
- Simple single optimization (`example_single_optimization.py`)
- Ensemble optimization from SMILES, multi-XYZ, directories (`example_ensemble_optimization.py`)
- CLI examples (`example_cli_usage.py`)

The example configs are sanitized (no tokens in plain text).
