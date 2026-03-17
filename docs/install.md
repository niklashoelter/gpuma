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
to get the install command matching your platform and CUDA version, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then install GPUMA — pip will see that a CUDA-enabled PyTorch is already
present and will not replace it:

```bash
pip install gpuma
```

### Environment setup examples

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

### ORB-v3 model support

ORB-v3 models are included in the standard `gpuma` installation. To use them,
set `"model_type": "orb"` and `"model_name": "orb_v3_direct_omol"` in your
configuration file (see [Configuration](config.md) and `examples/config_orb.json`).

D3 dispersion correction can be enabled by setting `"d3_correction": true` in the config.

> ⚠️ **Required for UMA models:**</br>
> To access the UMA models on Hugging Face, **you must provide a token** either via the `HUGGINGFACE_TOKEN` environment variable or via the config (direct token string or path to a file containing the token).

## Option 2: Install from source

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
