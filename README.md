# GPUMA

A minimalist toolkit for molecular geometry optimization using fairchem UMA models and torch-sim. Includes a simple Python API and a CLI.

## Installation

Using pip:

```bash
# install this package first
git clone https://github.com/niklashoelter/gpuma.git
cd gpuma

uv pip install -e . # or plain pip: pip install -e .
```

**Dependencies that need to be manually installed**:
- pyyaml (optional if you use YAML configs)
- pytest (for running tests)

Note: Some ML/simulation dependencies (torch, rdkit, fairchem) provide OS- and GPU-specific wheels. Please consult project-specific installation notes as needed.

## CLI Usage (Config-Driven)

The CLI is provided via the command `gpuma`. For best results, create a
config file (JSON or YAML) and reference it in all CLI calls.

**Example config file (JSON):**

```json
{
  "optimization": {
    "batch_optimization_mode": "batch",
    "batch_optimizer": "fire",
    "max_num_conformers": 20,
    "conformer_seed": 42,

    "model_name": "uma-m-1p1",
    "model_path": null,
    "model_cache_dir": null,
    "device": "cuda",
    "huggingface_token": null,
    "huggingface_token_file": "/home/hf_secret",

    "logging_level": "INFO"
  }
}
```

**Example config file (YAML):**

```yaml
optimization:
  batch_optimization_mode: batch
  batch_optimizer: fire
  max_num_conformers: 20
  conformer_seed: 42

  model_name: uma-m-1p1
  model_path: null
  model_cache_dir: null
  device: cuda
  huggingface_token: null
  huggingface_token_file: /home/hf_secret

  logging_level: INFO
```

**Recommended CLI usage:**

```bash
# Optimize a single structure from SMILES using a config file
gpuma optimize --smiles "CCO" --output ethanol_opt.xyz --config examples/config.json

# Optimize a single structure from an XYZ file
gpuma optimize --xyz test.xyz --output test_opt.xyz --config examples/config.json

# Create and optimize a conformer ensemble from SMILES
gpuma ensemble --smiles "c1ccccc1" --conformers 10 --output benzene_ensemble.xyz --config examples/config.json

# Batch optimization from a multi-XYZ file
gpuma batch --multi-xyz examples/read_multiple_xyz_file/conf0_confsearch_ensemble.xyz \
  --output optimized_ensemble.xyz --config examples/config.json

# Batch optimization from a directory of XYZ files
gpuma batch --xyz-dir examples/read_multiple_xyz_dir/ --output optimized_dir.xyz --config examples/config.json

# Convert SMILES to XYZ (no optimization)
gpuma convert --smiles "CCO" --output ethanol.xyz --config examples/config.json

# Generate conformers from SMILES (no optimization)
gpuma generate --smiles "c1ccccc1" --conformers 5 --output benzene_conformers.xyz --config examples/config.json

# Create or validate configuration files
gpuma config --create examples/config.json
gpuma config --validate examples/config.json

# Verbose vs. quiet (set in config file)
gpuma optimize --smiles "CCO" --output ethanol.xyz --config examples/config.json
gpuma ensemble --smiles "CCO" --conformers 3 --output ethanol.xyz --config examples/config.json
```

**Note:**
- If `--config` is not specified, `config.json` in the current directory is loaded by default.
- Direct CLI flags are supported, but using a config file is preferred for all workflows.

## Python API (short)

```python
import gpuma
from gpuma import Config, Structure

# Convenience (top-level, via the public api module): optimize a single
# molecule from SMILES and optionally save
cfg = Config()  # or gpuma.load_config_from_file("config.json")
optimized: Structure = gpuma.optimize_single_smiles(
    "CCO", output_file="ethanol_opt.xyz", config=cfg
)
print(optimized.energy)

# Convenience: optimize a single molecule from an XYZ file
optimized2: Structure = gpuma.optimize_single_xyz_file(
    "test.xyz", output_file="test_opt.xyz", config=cfg
)

# Convenience: optimize a conformer ensemble generated from SMILES
optimized_confs = gpuma.optimize_smiles_ensemble(
    "c1ccccc1", num_conformers=5, output_file="benzene_ensemble.xyz", config=cfg
)

# Lower-level building blocks are available via the package root as well
s: Structure = gpuma.smiles_to_xyz("CCO")
opt_s: Structure = gpuma.optimize_single_structure(s, cfg)
gpuma.save_xyz_file(opt_s, "ethanol_opt.xyz")

# You can also import directly from the dedicated API module if you prefer
from gpuma.api import optimize_single_smiles
opt3 = optimize_single_smiles("CCO", config=cfg)
```

## Configuration

All supported parameters live under the `optimization` key. Only these parameters are effective in the code; unknown fields are preserved.

**Always use a config file for CLI and API calls.**

Example (JSON):

```json
{
  "optimization": {
    "batch_optimization_mode": "batch",
    "batch_optimizer": "fire",
    "max_num_conformers": 20,
    "conformer_seed": 42,

    "model_name": "uma-m-1p1",
    "model_path": null,
    "model_cache_dir": null,
    "device": "cuda",
    "huggingface_token": null,
    "huggingface_token_file": "/home/hf_secret",

    "logging_level": "INFO"
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

  model_name: uma-m-1p1
  model_path: null
  model_cache_dir: null
  device: cuda
  huggingface_token: null
  huggingface_token_file: /home/hf_secret

  logging_level: INFO
```

- `batch_optimization_mode`: controls the ensemble mode
  - `sequential`: ASE/BFGS per conformer with a shared calculator
  - `batch`: torch-sim batch optimization (accelerated for larger ensembles)
- `batch_optimizer`: optimizer for batch mode; `fire` (default) or `gradient_descent`
- `device`: CPU/GPU (CUDA). Batch mode requires a GPU to run truly batched; on CPU it falls back to sequential.
- Hugging Face token: either set directly or reference a file.
- `logging_level`: set logging verbosity (`info`, `warning`, `error`).

## Examples

See the `examples/` folder for:
- Simple single optimization (`example_single_optimization.py`)
- Ensemble optimization from SMILES, multi-XYZ, directories (`example_ensemble_optimization.py`)
- CLI examples (`example_cli_usage.py`)

The example configs are sanitized (no tokens in plain text).

## Development

```bash
# Lint/Typecheck (optional, if tools are available)
python -m compileall -q src

# Run examples
python examples/example_single_optimization.py
python examples/example_ensemble_optimization.py
```

## Troubleshooting
- Missing libraries: install optional dependencies like `pyyaml` if you use YAML configs.
- CUDA/GPU: ensure a compatible PyTorch is installed and set `device: "cuda"` in your config.
- Fairchem/UMA: ensure network access for model downloads and optionally set `huggingface_token` (e.g., via a token file).

## License
MIT License (see LICENSE)
