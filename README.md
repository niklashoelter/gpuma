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
gpuma optimize --smiles "C=C" --output examples/example_output/ethylene_opt.xyz --config examples/config.json

# Optimize a triplet state from SMILES (multiplicity = 3)
# Charge is inferred from the SMILES; multiplicity is set via CLI
gpuma optimize --smiles "C=C" --multiplicity 3 --output examples/example_output/ethylene_triplet_opt.xyz --config examples/config.json

# Optimize a single structure from an XYZ file
gpuma optimize --xyz examples/example_input_xyzs/single_xyz_file.xyz --output examples/example_output/single_from_xyz_cli.xyz --config examples/config.json

# Create and optimize a conformer ensemble from SMILES
gpuma ensemble --smiles "c1c(CCOCC)cccc1" --conformers 10 --output examples/example_output/benzene_ensemble.xyz --config examples/config.json

# Batch optimization from a multi-XYZ file
gpuma batch --multi-xyz examples/example_input_xyzs/multi_xyz_file.xyz \
  --output examples/example_output/optimized_ensemble.xyz --config examples/config.json

# Batch optimization from a directory of XYZ files
gpuma batch --xyz-dir examples/multi_xyz_dir/ --output examples/example_output/optimized_dir.xyz --config examples/config.json

# Batch optimization from a directory of XYZ files with modified charge/spin
gpuma batch --xyz-dir examples/multi_xyz_dir/ --output examples/example_output/optimized_dir_ch1_mult2.xyz --charge 1 --multiplicity 2 --config examples/config.json

# Convert SMILES to XYZ (no optimization)
gpuma convert --smiles "CCO" --output examples/example_output/ethanol.xyz --config examples/config.json

# Generate conformers from SMILES (no optimization)
gpuma generate --smiles "c1ccccc1" --conformers 5 --output examples/example_output/benzene_conformers.xyz --config examples/config.json

# Create or validate configuration files
gpuma config --create examples/config.json
gpuma config --validate examples/config.json

# Verbose vs. quiet (set in config file)
gpuma optimize --smiles "CCO" --output examples/example_output/ethanol_verbose.xyz --config examples/config.json
gpuma ensemble --smiles "CCO" --conformers 3 --output examples/example_output/ethanol_ensemble_verbose.xyz --config examples/config.json
```

**Note:**
- If `--config` is not specified, `config.json` in the current directory is loaded by default.
- Direct CLI flags are supported, but using a config file is preferred for all workflows.
- Unless explicitly overridden, the electronic state defaults are always
  `charge = 0` and `multiplicity = 1`.

- For **SMILES inputs**, the total charge is automatically inferred from the
  SMILES via RDKit/MORFEUS. The multiplicity can be controlled globally via
  the config (`optimization.multiplicity`) or overridden per CLI call with
  `--multiplicity` where supported. Internally this is passed to the models
  as the `spin` channel.
- For **XYZ inputs** (single and batch), both charge and multiplicity can be
  set via CLI flags (`--charge`, `--multiplicity`) or via the config
  (`optimization.charge`, `optimization.multiplicity`). CLI flags override the
  config values. These values are passed down to the models as
  `Atoms.info = {"charge": charge, "spin": multiplicity}` and are written to
  the XYZ comments as `Charge: ... | Multiplicity: ...`.

You can control the compute device globally in the config or from the CLI with `--device` (which overrides the config).
Accepted values are `cpu` for CPU-only execution and `cuda` to enable GPU acceleration.

**GPU selection:** Fairchem currently does not support selecting a specific GPU
via the `device` argument. To target particular GPUs you should set the
`CUDA_VISIBLE_DEVICES` environment variable before calling `gpuma`, e.g.:

```bash
# use only GPU 1
CUDA_VISIBLE_DEVICES=1 gpuma optimize --smiles "C=C" --output examples/example_output/ethylene_opt_gpu1.xyz --config examples/config.json

# use GPUs 1,2,3
CUDA_VISIBLE_DEVICES=1,2,3 gpuma batch --xyz-dir examples/multi_xyz_dir/ --output examples/example_output/optimized_dir_gpu123.xyz --config examples/config.json
```

## Python API (short)

```python
import gpuma
from gpuma import Config, Structure

# Convenience (top-level, via the public api module): optimize a single
# molecule from SMILES and optionally save
cfg = gpuma.load_config_from_file("config.json")
optimized: Structure = gpuma.optimize_single_smiles(
    "CCO", output_file="examples/example_output/ethanol_opt.xyz", config=cfg
)
print(optimized.energy)

# Convenience: optimize a single molecule from an XYZ file
optimized2: Structure = gpuma.optimize_single_xyz_file(
    "test.xyz", output_file="examples/example_output/test_opt.xyz", config=cfg
)

# Convenience: optimize a conformer ensemble generated from SMILES
optimized_confs = gpuma.optimize_smiles_ensemble(
    "c1ccccc1", num_conformers=5, output_file="examples/example_output/benzene_ensemble.xyz", config=cfg
)

# Lower-level building blocks are available via the package root as well
s: Structure = gpuma.smiles_to_xyz("CCO")
opt_s: Structure = gpuma.optimize_single_structure(s, cfg)
gpuma.save_xyz_file(opt_s, "examples/example_output/ethanol_opt.xyz")
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

    "charge": 0,
    "multiplicity": 1,

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

  charge: 0
  multiplicity: 1

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
- `charge`: total charge of the system (for SMILES this is inferred from the
  input and not overridden by this setting)
- `multiplicity`: spin multiplicity of the system
- `model_name`: Fairchem UMA model name (e.g., `uma-m-1p1`)
- `model_path`: local path to a Fairchem UMA model (overrides `model_name` if set)
- `model_cache_dir`: directory to cache downloaded models (default: `~/.cache/fairchem`)
- `device`: compute device string; one of `cpu` or `cuda`.
  Fairchem only distinguishes between CPU and CUDA; selection of specific
  GPUs should be handled via the `CUDA_VISIBLE_DEVICES` environment variable
  before running the CLI or Python code.

- `huggingface_token`: optional HF token for model access (if required)
- `huggingface_token_file`: optional file path to read the HF token from
- `logging_level`: logging verbosity; e.g., `DEBUG`, `INFO`, `WARNING`

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
