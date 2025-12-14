# GPUMA

GPUMA is a minimalist Python toolkit for facile and rapid high-throughput molecular geometry optimization 
based on the [UMA/OMol25 machine-learning interatomic potential](https://arxiv.org/abs/2505.08762).  

GPUMA is especially designed for batch optimizations of many structures (conformer ensembles, datasets) on GPU,
ensuring efficient parallelization and maximum GPU utilization by leveraging the [torch-sim library](https://arxiv.org/abs/2508.06628).
It wraps Fairchem UMA models and torch-sim functionality to provide both a simple command-line 
interface (CLI) and a small but expressive Python API for single- and multi-structure optimizations.

If conformer sampling is desired, GPUMA can generate conformer ensembles on the fly from SMILES strings 
using the [morfeus library](https://digital-chemistry-laboratory.github.io/morfeus/). Alternative input formats
are described in the CLI section below.

Feedback and improvements are always welcome!

## Installation

### Option 1: Install from PyPI (recommended)

```bash
pip install gpuma
```

This installs `gpuma` together with its core dependencies. Make sure you are using
Python 3.12 or newer.

### Option 2: Install from source

```bash
# clone the repository
git clone https://github.com/niklashoelter/gpuma.git
cd gpuma

# install using (uv) pip
uv pip install .
# or, without uv:
pip install .
```

## CLI Usage

The CLI is provided via the command `gpuma`. For best results, create a
config file (JSON or YAML) and reference it in all CLI calls.

**Important:** For the optimatzion of very large ensembles or high-throughput workflows, using
the batch optimization mode is recommended (set in the config file, see below).
In this case, make sure to use a multi-XYZ input file or a directory of XYZ files
and only start one GPUMA process to leverage maximum efficient GPU parallelization
and avoid runtime overhead for model initialization and memory estimation.

**Recommended CLI usage:**

```bash
# Optimize a single structure from SMILES using a config file
gpuma optimize --smiles "CCO" --output ethanol_opt.xyz --config examples/config.json

# Optimize a single structure from an XYZ file
gpuma optimize --xyz test.xyz --output test_opt.xyz --config examples/config.json

# Create and optimize a conformer ensemble from SMILES
gpuma ensemble --smiles "c1c(CCOCC)cccc1" --conformers 10 --output benzene_ensemble.xyz --config examples/config.json

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
cfg = gpuma.load_config_from_file("config.json")
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
- `max_num_conformers`: max number of conformers to generate from SMILES (if applicable)
- `conformer_seed`: random seed for conformer generation (if applicable)
- `model_name`: Fairchem UMA model name (e.g., `uma-m-1p1`)
- `model_path`: local path to a Fairchem UMA model (overrides `model_name` if set)
- `model_cache_dir`: directory to cache downloaded models (default: `~/.cache/fairchem`)
- `device`: compute device, e.g., `cuda` or `cpu`
- `huggingface_token`: optional HF token for model access (if required)
- `huggingface_token_file`: optional file path to read the HF token from
- `logging_level`: logging verbosity; e.g., `DEBUG`, `INFO`, `WARNING

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
- Fairchem/UMA: ensure network access for model downloads and optionally set or provide 
`huggingface_token` (e.g., via a token file) to access the UMA model family.

## License
MIT License (see LICENSE)
