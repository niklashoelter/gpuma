# GPUMA

<div align="center">
  <img src="logo.png"/>
</div>

---

GPUMA is a minimalist Python toolkit for facile and rapid high-throughput molecular geometry optimization
based on machine-learning interatomic potentials (MLIPs).

Two model backends are supported out of the box:

- **Fairchem UMA** ([UMA/OMol25](https://arxiv.org/abs/2505.08762)) — the default backend.
- **ORB-v3** ([orbital-materials/orb-models](https://github.com/orbital-materials/orb-models)) — included in core dependencies, with optional D3 dispersion correction.

GPUMA is especially designed for batch optimizations of many structures (conformer ensembles, datasets) on GPU,
ensuring efficient parallelization and maximum GPU utilization by leveraging the [torch-sim library](https://arxiv.org/abs/2508.06628).
It wraps model backends and torch-sim functionality to provide both a simple command-line
interface (CLI) and a small but expressive Python API for single- and multi-structure optimizations.

If conformer sampling is desired, GPUMA can generate conformer ensembles on the fly from SMILES strings
using the [morfeus library](https://digital-chemistry-laboratory.github.io/morfeus/). Alternative input formats
are described in the CLI section below.

Feedback and improvements are always welcome!

## Installation

> **Required for UMA models:**</br>
> To access the UMA models on Hugging Face, **you must provide a token** either via the `HUGGINGFACE_TOKEN` environment variable or via the config (direct token string or path to a file containing the token).

### Option 1: Install from PyPI (recommended)

This installs `gpuma` together with all dependencies (including both the
Fairchem UMA and ORB-v3 backends).
At the moment, installation and tests have only been
validated under Python 3.12; using other Python versions is currently
considered experimental.

> **GPU support:** By default, `pip install` may pull a CPU-only build of
> PyTorch (especially on Windows). To enable GPU acceleration, install
> PyTorch with CUDA **before** installing GPUMA. Visit
> [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to
> get the install command for your platform and CUDA version, e.g.:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu124
> ```

- **Using a `uv` virtual environment**
  ```powershell
  # create and activate a fresh environment
  uv venv .venv

  # activate the environment

  # install PyTorch with CUDA support (pick your CUDA version at https://pytorch.org)
  uv pip install torch --index-url https://download.pytorch.org/whl/cu124

  # install gpuma from PyPI inside the environment
  uv pip install gpuma
  ```

- **Using a `conda` environment**
  ```powershell
  # create and activate a fresh environment with Python 3.12
  conda create -n gpuma-py312 python=3.12
  conda activate gpuma-py312

  # install PyTorch with CUDA support (pick your CUDA version at https://pytorch.org)
  pip install torch --index-url https://download.pytorch.org/whl/cu124

  # install gpuma from PyPI inside the environment
  pip install gpuma
  ```

### ORB-v3 support

ORB-v3 models are included in the standard installation. To use them,
set `"model_type": "orb"` and `"model_name": "orb_v3_direct_omol"` in the
`model` section of your configuration file (see [Configuration](config.md)
and `examples/config_orb.json`).


### Option 2: Install from source

```bash
# clone the repository
git clone https://github.com/niklashoelter/gpuma.git
cd gpuma

# install PyTorch with CUDA support (pick your CUDA version at https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# install using (uv) pip
uv pip install .
# or, without uv:
pip install .
```
## Documentation

Full documentation is available at [https://niklashoelter.github.io/gpuma/](https://niklashoelter.github.io/gpuma/).

Please refer to the documentation for detailed configuration options and advanced usage. Using a configuration file is highly recommended for reproducibility and ease of use.

Also check the examples folder in the repository for sample config files and usage examples.

## CLI Usage

The CLI is provided via the command `gpuma`. For best results, create a
config file (JSON or YAML) and reference it in all CLI calls.

Refer to the documentation for details on configuration options and CLI usage.

## Python API

A minimalistic and high-level Python API is provided for easy integration into custom scripts and workflows.

Please refer to the documentation for detailed usage examples and API reference.

## Known limitations

When a run is started from SMILES, an RDKit force field (via the morfeus library) is used to generate an initial structure. Spin is not taken into account during this step, so the initial estimated geometries can be incorrect. When the MLIP models are applied subsequently, the structure can sometimes be optimized to a maximum rather than a minimum because the model is not provided with Hessian matrices. This behavior only affects runs originating from SMILES; it does not occur with better starting geometries (e.g., when starting from XYZ files).

## Troubleshooting
- Missing libraries: install optional dependencies like `pyyaml` if you use YAML configs.
- Fairchem/UMA: ensure network access for model downloads and optionally set or provide
`huggingface_token` (e.g., via a token file) to access the UMA model family.
- ORB-v3: set `"model_type": "orb"` in the `model` section of your config.
  Enable D3 dispersion correction with `"d3_correction": true`
  (see [Configuration](config.md)).

## License
MIT License (see LICENSE)
