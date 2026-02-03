# GPUMA

<div align="center">
  <img src="docs/logo.png" alt="PAYN Logo"/>
</div>

---

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

## Documentation

Full documentation is available at [https://GloriusGroup.github.io/gpuma/](https://GloriusGroup.github.io/gpuma/).

Please refer to the documentation for detailed configuration options and advanced usage. Using a configuration file is highly recommended for reproducibility and ease of use.

## CLI Usage

The CLI is provided via the command `gpuma`. For best results, create a
config file (JSON or YAML) and reference it in all CLI calls. 
Refer to the 
documentation for details on configuration options and CLI usage.

## Python API

A minimalistic and high-level Python API is provided for easy integration into custom scripts and workflows.
Please refer to the documentation for detailed usage examples and API reference.

## Known limitations

When a run is started from SMILES, an RDKit force field (via the morfeus library) is used to generate an initial structure. Spin is not taken into account during this step, so the initial estimated geometries can be incorrect. When the UMA/Omol25 models are applied subsequently, the structure can sometimes be optimized to a maximum rather than a minimum because the model is not provided with Hessian matrices. This behavior only affects runs originating from SMILES; it does not occur with better starting geometries (e.g., when starting from XYZ files).

## Troubleshooting
- Missing libraries: install optional dependencies like `pyyaml` if you use YAML configs.
- Fairchem/UMA: ensure network access for model downloads and optionally set or provide 
`huggingface_token` (e.g., via a token file) to access the UMA model family.

## License
MIT License (see LICENSE)
