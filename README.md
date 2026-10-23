# GPUMA

<div align="center">
  <img src="docs/logo_bg.png" alt="GPUMA Logo"/>
</div>

<p align="center">
  <a href="https://github.com/niklashoelter/gpuma/actions/workflows/python-app.yml"><img src="https://img.shields.io/github/actions/workflow/status/niklashoelter/gpuma/python-app.yml?branch=master&label=tests" alt="CI status"/></a>
  <a href="https://pypi.org/project/gpuma/"><img src="https://img.shields.io/pypi/v/gpuma" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/gpuma/"><img src="https://img.shields.io/pypi/pyversions/gpuma" alt="Python versions"/></a>
  <a href="https://niklashoelter.github.io/gpuma/"><img src="https://img.shields.io/badge/docs-online-brightgreen" alt="Documentation"/></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"/></a>
</p>

---

GPUMA is a minimalist Python toolkit for facile and rapid high-throughput molecular geometry optimization
using machine-learning interatomic potentials (MLIPs).

Three model backends are supported out of the box:

- **Fairchem UMA** ([UMA/OMol25](https://arxiv.org/abs/2505.08762)) — the default backend.
- **ORB-v3** ([orbital-materials/orb-models](https://github.com/orbital-materials/orb-models)).
- **SevenNet** ([SevenNet](https://github.com/MDIL-SNU/SevenNet)), including multi-modal checkpoints such as `7net-omni`.

All three backends support optional DFT-D3(BJ) dispersion correction. It is off by
default and rejected for the default `uma-s-1p2p1-ft-d4rev`, which already includes dispersion.

GPUMA is especially designed for batch optimizations of many structures (conformer ensembles, datasets) on GPU,
ensuring efficient parallelization and maximum GPU utilization by leveraging the [torch-sim library](https://arxiv.org/abs/2508.06628).
It wraps model backends and torch-sim functionality to provide both a simple command-line
interface (CLI) and a small but expressive Python API for single- and multi-structure optimizations.

If conformer sampling is desired, GPUMA can generate conformer ensembles on the fly from SMILES strings.
The SMILES → 3D embedding runs on the GPU via [nvMolKit](https://nvidia-bionemo.github.io/nvMolKit/)
when a CUDA device is configured, and on the CPU via the
[morfeus library](https://digital-chemistry-laboratory.github.io/morfeus/) otherwise; the backend is
chosen from `technical.device` like everywhere else in GPUMA. Alternative input formats
are described in the CLI section below.

Feedback and improvements are always welcome!

## Installation

### Option 1: Install from PyPI (recommended)

This installs `gpuma` together with all dependencies (including the
Fairchem UMA, ORB-v3, and SevenNet backends, as well as the nvMolKit GPU
conformer-embedding backend). Every dependency is pinned to the exact tested
version, so install gpuma into a fresh environment.
At the moment, installation and tests have only been
validated under Python 3.12; using other Python versions is currently
considered experimental.

> **GPU support:** By default, `pip install` may pull a CPU-only build of
> PyTorch (especially on Windows). To enable GPU acceleration, install
> PyTorch with CUDA **before** installing GPUMA. Visit
> [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to
> get the install command for your platform and CUDA version. Keep
> `torch==2.13.0`, which gpuma pins, and only change the `cuXXX` index, e.g.:
> ```bash
> pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
> ```
> The `cu126` build runs on drivers from 525 on; the default PyPI build needs driver 580 or newer.

- **Using a `uv` virtual environment**
  ```powershell
  # create and activate a fresh environment (Python 3.12)
  uv venv .venv --python 3.12

  # activate the environment

  # install PyTorch 2.13.0 with CUDA support (keep the version, pick the cuXXX index for your driver)
  uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126

  # install gpuma from PyPI inside the environment
  uv pip install gpuma
  ```

- **Using a `conda` environment**
  ```powershell
  # create and activate a fresh environment with Python 3.12
  conda create -n gpuma-py312 python=3.12
  conda activate gpuma-py312

  # install PyTorch 2.13.0 with CUDA support (keep the version, pick the cuXXX index for your driver)
  pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126

  # install gpuma from PyPI inside the environment
  pip install gpuma
  ```

> **GPU conformer embedding:** SMILES → 3D conversion can run on the GPU via
> [nvMolKit](https://nvidia-bionemo.github.io/nvMolKit/), which is a core
> dependency and installs automatically with GPUMA (it pins `rdkit==2026.3.1`
> to match the RDKit build it is linked against). The backend is selected by
> `technical.device` in the config, as everywhere else in GPUMA: a CUDA device
> uses nvMolKit, otherwise the CPU (morfeus) backend is used and results stay
> valid. The GPU path needs an NVIDIA GPU (compute capability 7.0+); on
> CPU-only machines the package still installs and the CPU backend is used.


### Option 2: Install from source

```bash
# clone the repository
git clone https://github.com/niklashoelter/gpuma.git
cd gpuma

# install PyTorch 2.13.0 with CUDA support (keep the version, pick the cuXXX index for your driver)
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126

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
- [docs/agents.md](docs/agents.md) – using GPUMA with AI agents
- [docs/config.md](docs/config.md) – configuration file schema and examples
- [docs/reference.md](docs/reference.md) – API and configuration reference

Using a configuration file is highly recommended for reproducibility and ease of use.

Also check the [examples/](examples) folder in the repository for sample config files and usage examples:
- [examples/config.json](examples/config.json) – Fairchem UMA configuration
- [examples/config_orb.json](examples/config_orb.json) – ORB-v3 configuration
- [examples/config_sevennet.json](examples/config_sevennet.json) – SevenNet configuration (multi-modal `7net-omni`)
- [examples/example_single_optimization.py](examples/example_single_optimization.py) – single-structure optimization from Python
- [examples/example_ensemble_optimization.py](examples/example_ensemble_optimization.py) – ensemble/multi-structure optimization from Python
- [examples/example_dispersion.py](examples/example_dispersion.py) – DFT-D3(BJ) correction with Fairchem and ORB (single + batch)

> **Required for UMA models:**</br>
> To access the UMA models on Hugging Face, **you must provide a token** either via the `HUGGINGFACE_TOKEN` environment variable or via the config (direct token string or path to a file containing the token).

## Using with AI agents

GPUMA ships a skill for AI coding agents (Claude Code, Codex, Gemini CLI, Copilot, Cursor), so
chemists can run it in plain language without writing Python. Install it with
`gpuma skills install` and check the machine with `gpuma doctor`; see
[docs/agents.md](docs/agents.md).

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

When a run is started from SMILES, an RDKit/MMFF force field (via nvMolKit on GPU or the morfeus library on CPU) is used to generate an initial structure. Spin is not taken into account during this step, so the initial estimated geometries can be incorrect. When the MLIP models are applied subsequently, the structure can sometimes be optimized to a maximum rather than a minimum because the model is not provided with Hessian matrices. This behavior only affects runs originating from SMILES; it does not occur with better starting geometries (e.g., when starting from XYZ files).

## Troubleshooting
- Fairchem/UMA: ensure network access for model downloads and optionally set or provide
`huggingface_token` (e.g., via a token file) to access the UMA model family.
- ORB-v3: models are downloaded automatically on first use.
- SevenNet: models are downloaded automatically on first use; multi-modal
  checkpoints (e.g. `7net-omni`, `7net-mf-ompa`) also require a
  `model_modal` fidelity (see [docs/config.md](docs/config.md)).
- D3 dispersion correction: off by default. Set `"d3_correction": true` to
  enable DFT-D3(BJ) for models trained without dispersion. It is rejected for
  `uma-s-1p2p1-ft-d4rev` and logs a warning for the other UMA models and ORB
  `*_omol` models (see [docs/config.md](docs/config.md)).

## License
MIT License (see LICENSE)
