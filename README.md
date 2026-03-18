# GPUMA

<div align="center">
  <img src="docs/logo_bg.png" alt="GPUMA Logo"/>
</div>

---

GPUMA is a minimalist Python toolkit for facile and rapid high-throughput molecular geometry optimization
using machine-learning interatomic potentials (MLIPs).

Two model backends are supported out of the box:

- **Fairchem UMA** ([UMA/OMol25](https://arxiv.org/abs/2505.08762)) — the default backend.
- **ORB-v3** ([orbital-materials/orb-models](https://github.com/orbital-materials/orb-models)) — with optional D3 dispersion correction.

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

For local browsing of the Markdown sources, see in particular:
- [docs/index.md](docs/index.md) – overview and getting started
- [docs/install.md](docs/install.md) – installation details
- [docs/cli.md](docs/cli.md) – CLI options and input formats
- [docs/config.md](docs/config.md) – configuration file schema and examples
- [docs/reference.md](docs/reference.md) – API and configuration reference

Using a configuration file is highly recommended for reproducibility and ease of use.

Also check the [examples/](examples) folder in the repository for sample config files and usage examples:
- [examples/config.json](examples/config.json) – Fairchem UMA configuration
- [examples/config_orb.json](examples/config_orb.json) – ORB-v3 configuration (with D3 options)
- [examples/example_single_optimization.py](examples/example_single_optimization.py) – single-structure optimization from Python
- [examples/example_ensemble_optimization.py](examples/example_ensemble_optimization.py) – ensemble/multi-structure optimization from Python

## CLI Usage

The CLI is provided via the command `gpuma`. For best results, create a
config file (JSON or YAML) and reference it in all CLI calls (see [examples/config.json](examples/config.json) for a minimal example).

### Examples: Batch optimization of multiple XYZ structures

Optimize all XYZ files in a directory (each file containing a single structure):

```bash
gpuma batch --xyz-dir examples/example_input_xyzs/multi_xyz_dir/ --output output.xyz --config examples/config.json
```

Optimize multiple structures contained in a single multi-XYZ file:

```bash
gpuma batch --multi-xyz examples/example_input_xyzs/multi_xyz_file.xyz --output output.xyz --config examples/config.json
```

Refer to the [CLI documentation](docs/cli.md) for details on configuration options, supported input formats (SMILES, XYZ, directories, multi-XYZ files), and additional CLI examples.

## Python API

A minimalistic and high-level Python API is provided for easy integration into custom scripts and workflows.

For example usage, see:
- [examples/example_single_optimization.py](examples/example_single_optimization.py)
- [examples/example_ensemble_optimization.py](examples/example_ensemble_optimization.py)

Please refer to the documentation and examples for detailed usage examples and API reference.

## Known limitations

When a run is started from SMILES, an RDKit force field (via the morfeus library) is used to generate an initial structure. Spin is not taken into account during this step, so the initial estimated geometries can be incorrect. When the MLIP models are applied subsequently, the structure can sometimes be optimized to a maximum rather than a minimum because the model is not provided with Hessian matrices. This behavior only affects runs originating from SMILES; it does not occur with better starting geometries (e.g., when starting from XYZ files).

## Troubleshooting
- Fairchem/UMA: ensure network access for model downloads and optionally set or provide
`huggingface_token` (e.g., via a token file) to access the UMA model family.
- ORB-v3: models are downloaded automatically on first use. To enable D3
  dispersion correction, set `"d3_correction": true` in the config (see
  [docs/config.md](docs/config.md)).

## License
MIT License (see LICENSE)
