# Installation

## Option 1: Install from PyPI (recommended)

```bash
pip install gpuma
```

This installs `gpuma` together with its core dependencies. At the moment, installation and tests have only been
validated under Python 3.12; using other Python versions is currently
considered experimental.

### GPU support

By default, `pip install` may pull a **CPU-only** build of PyTorch
(especially on Windows). To enable GPU acceleration, install PyTorch with
CUDA **before** installing GPUMA.

Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
to get the install command matching your platform and CUDA version. Keep
`torch==2.13.0`, which gpuma pins, and only change the `cuXXX` index, e.g.:

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
```

The `cu126` build runs on NVIDIA drivers from 525 on. The default PyPI build of
torch 2.13 targets CUDA 13 and needs driver 580 or newer.

Then install GPUMA. pip keeps the installed torch 2.13.0; any other torch
version is replaced by the default PyPI build:

```bash
pip install gpuma
```

This pulls in all model backends (Fairchem UMA, ORB-v3, and SevenNet) as
core dependencies — there is no separate optional extra to install.

### Pinned dependencies

Every dependency in `pyproject.toml` is pinned to the exact version gpuma is
tested with (`==`, no ranges), including torch, fairchem-core, torch-sim, the
`dev` extra and the build backend. Installing gpuma into an environment that
already holds other versions of these packages replaces them, so use a fresh
environment.

### GPU conformer embedding

SMILES → 3D conversion can run on the GPU via
[nvMolKit](https://nvidia-bionemo.github.io/nvMolKit/), which is a core
dependency and installs automatically from PyPI (it pins `rdkit==2026.3.1`
to match the RDKit build it is linked against, so no extra package index is
needed). The backend is selected by `technical.device`: a CUDA device uses
nvMolKit, otherwise the CPU (morfeus) backend is used. The GPU path needs an
NVIDIA GPU (compute capability 7.0+); on CPU-only machines the package still
installs and the CPU backend is used.

### Environment setup examples

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

> ⚠️ **Required for UMA models:**</br>
> To access the UMA models on Hugging Face, **you must provide a token** either via the `HUGGINGFACE_TOKEN` environment variable or via the config (direct token string or path to a file containing the token).

## Option 2: Install from source

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

# for development: tests, linting and docs tools, all pinned
uv pip install -e ".[dev]"
```
