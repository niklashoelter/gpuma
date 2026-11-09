# CLI Usage

The CLI is provided via the command `gpuma`. For best results, create a config file (JSON or YAML) and reference it in all CLI calls.

**Important:** For the optimization of very large ensembles or high-throughput workflows, using the batch optimization mode is recommended (set in the config file, see below). In this case, make sure to use a multi-XYZ input file or a directory of XYZ files and only start one GPUMA process to leverage maximum efficient GPU parallelization and avoid runtime overhead for model initialization and memory estimation.

**Recommended CLI usage:**

```bash
# Optimize a single structure from SMILES using a config file
gpuma optimize --smiles "C=C" --output examples/example_output/cli_smiles_ethylene_opt.xyz --config examples/config.json

# Optimize a triplet state from SMILES (multiplicity = 3)
# Charge is inferred from the SMILES; multiplicity is set via CLI
gpuma optimize --smiles "C=C" --multiplicity 3 --output examples/example_output/cli_smiles_ethylene_triplet_opt.xyz --config examples/config.json

# Optimize a single structure from an XYZ file
gpuma optimize --xyz examples/example_input_xyzs/single_xyz_file.xyz --output examples/example_output/cli_single_xyz_optimization.xyz --config examples/config.json

# Create and optimize a conformer ensemble from SMILES
gpuma ensemble --smiles "c1c(CCOCC)cccc1" --conformers 10 --output examples/example_output/cli_ensemble_optimization.xyz --config examples/config.json

# Batch optimization from a multi-XYZ file
gpuma batch --multi-xyz examples/example_input_xyzs/multi_xyz_file.xyz \
  --output examples/example_output/cli_multi_xyz_file_optimization.xyz --config examples/config.json

# Batch optimization from a directory of XYZ files
gpuma batch --xyz-dir examples/example_input_xyzs/multi_xyz_dir/ --output examples/example_output/cli_multi_xyz_dir_optimization.xyz  --config examples/config.json

# Batch optimization from a directory of XYZ files with modified charge/spin
gpuma batch --xyz-dir examples/example_input_xyzs/multi_xyz_dir/ --output examples/example_output/cli_multi_xyz_dir_optimization_modified_charge_spin.xyz --charge 1 --multiplicity 2 --config examples/config.json

# Convert SMILES to XYZ (no optimization)
gpuma convert --smiles "CCO" --output examples/example_output/cli_no_optimization.xyz --config examples/config.json

# Generate conformers from SMILES (no optimization)
gpuma generate --smiles "c1ccccc1" --conformers 5 --output examples/example_output/cli_conformer_generation.xyz --config examples/config.json

# Create or validate configuration files
gpuma config --create my_config.json
gpuma config --validate examples/config.json

# Verbose vs. quiet (set in config file)
gpuma optimize --smiles "CCO" --output examples/example_output/ethanol_verbose.xyz --config examples/config.json
gpuma ensemble --smiles "CCO" --conformers 3 --output examples/example_output/ethanol_ensemble_verbose.xyz --config examples/config.json
```

### `gpuma smiles` shorthand

The `gpuma smiles` command is a shorthand for single-structure optimization directly from a SMILES string (equivalent to `gpuma optimize --smiles`):

```bash
gpuma smiles --smiles "CCO" --output examples/example_output/ethanol_opt.xyz --config examples/config.json
```

### Using ORB-v3 models

To use ORB-v3 instead of Fairchem UMA, set `"model_type": "orb"` and an ORB `model_name` in your config file (see `examples/config_orb.json`). `--model-type orb` only switches the backend; the config must still name an ORB model, otherwise the run stops with exit code 3:

```bash
# Single optimization with ORB-v3
gpuma --model-type orb optimize --smiles "CCO" --output output.xyz --config examples/config_orb.json

# Ensemble with ORB-v3
gpuma ensemble --smiles "CCO" --conformers 10 --output ensemble.xyz --config examples/config_orb.json

# Batch optimization from a multi-XYZ file with ORB-v3
gpuma batch --multi-xyz examples/example_input_xyzs/multi_xyz_file.xyz --output batch_orb.xyz --config examples/config_orb.json

# Batch optimization from a directory of XYZ files with ORB-v3
gpuma batch --xyz-dir examples/example_input_xyzs/multi_xyz_dir/ --output batch_dir_orb.xyz --config examples/config_orb.json
```

### Using SevenNet models

To use SevenNet, set `"model_type": "sevennet"` (alias `"7net"`) in your config file (see `examples/config_sevennet.json`). Multi-modal checkpoints such as `7net-omni` also need a `"model_modal"` fidelity (e.g. `"omol25_high"`) in the `model` section. The commands are identical to the other backends, just pointed at a SevenNet config:

```bash
# Single optimization with SevenNet
gpuma optimize --smiles "CCO" --output output.xyz --config examples/config_sevennet.json

# Batch optimization from a directory of XYZ files with SevenNet
gpuma batch --xyz-dir examples/example_input_xyzs/multi_xyz_dir/ --output batch_dir_7net.xyz --config examples/config_sevennet.json
```

You can also select SevenNet with the `--model-type` flag (`--model-type sevennet` or `--model-type 7net`), just like the other backends.

### Using AIMNet2 models

Set `"model_type": "aimnet"` (alias `"aimnet2"`) and `"model_name"` to `"aimnet2"`, `"aimnet2-nse"` (open-shell) or `"aimnet2-2025"` (see `examples/config_aimnet.json`), or pass `--model-type aimnet`. Inputs outside the model's range (elements, open-shell on a closed-shell model, H–H contacts below 0.8 Å) are rejected with exit code 3 before the model loads:

```bash
# Batch optimization from a multi-XYZ file with AIMNet2
gpuma batch --multi-xyz examples/example_input_xyzs/multi_xyz_file.xyz --output batch_aimnet.xyz --config examples/config_aimnet.json

# Open-shell structure with AIMNet2-NSE (set "model_name": "aimnet2-nse" in the config)
gpuma optimize --xyz examples/example_input_xyzs/butene_triplet.xyz --multiplicity 3 --output triplet.xyz --config my_nse_config.json
```

**Note:**

- If `--config` is not specified, `config.json` in the current directory is loaded by default. If that file does not exist either, GPUMA runs on its built-in defaults and logs a warning naming the default model. A `--config` path that does not exist is an error.

- Global flags (`--device`, `--model-type`, `--verbose`, `--quiet`, `--json`, `--dry-run`) work before or after the subcommand; if given in both places, the one after the subcommand wins.

- Direct CLI flags are supported, but using a config file is preferred for all workflows.

- Unless explicitly overridden, the electronic state defaults are always `charge = 0` and `multiplicity = 1`.

- For **SMILES inputs**, the total charge is automatically inferred from the SMILES via RDKit/MORFEUS. The multiplicity can be controlled globally via the config (`optimization.multiplicity`) or overridden per CLI call with `--multiplicity` where supported. Internally this is passed to the models as the `spin` channel.

- For **XYZ inputs** (single and batch), both charge and multiplicity can be set via CLI flags (`--charge`, `--multiplicity`) or via the config (`optimization.charge`, `optimization.multiplicity`). CLI flags override the config values; without the flags, the config values are used. These values are passed down to the models as `Atoms.info = {"charge": charge, "spin": multiplicity}` and are written to the XYZ comments as `Charge: ... | Multiplicity: ...`.

The config file is organized into four sections: `optimization` (batch settings, convergence, charge/multiplicity), `model` (backend, name, SevenNet modal, tokens, D3), `conformer_generation` (conformer count, seed), and `technical` (device, memory padding, logging). See [Configuration](https://NiklasHoelter.github.io/gpuma/config/index.md) for details.

You can control the compute device globally in the config or from the CLI with `--device` (which overrides the config). Accepted values are `cpu`, `cuda` (default GPU), or `cuda:N` to select a specific GPU (e.g. `cuda:0`, `cuda:1`). A GPU index that does not exist, or a CUDA device on a machine without usable CUDA, is an error (exit code 3); optimization never silently switches to another GPU or to the CPU. Conformer generation alone may still fall back to the CPU, with a warning.

```bash
# use GPU 1 explicitly
gpuma --device cuda:1 optimize --smiles "C=C" --output output.xyz --config examples/config.json

# or set it in the config file: "device": "cuda:3"
```

Alternatively, you can use the `CUDA_VISIBLE_DEVICES` environment variable to control GPU visibility:

```bash
CUDA_VISIBLE_DEVICES=1 gpuma optimize --smiles "C=C" --output output.xyz --config examples/config.json
```

## Machine-readable output and exit codes

Add `--json` to any command to get exactly one JSON object on stdout; all log output goes to stderr. This is meant for scripts and AI agents:

```bash
gpuma batch --multi-xyz in.xyz --output out.xyz --config config.json --json > summary.json
```

The summary contains:

| Key                                              | Content                                                                                                                                   |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `status`                                         | `ok`, `partial`, `error` or `dry_run`                                                                                                     |
| `exit_code`                                      | The process exit code (see below)                                                                                                         |
| `command`, `output_file`                         | What ran and where results were written                                                                                                   |
| `device`, `model`, `config`                      | The resolved device, the model (`type`, `name`, and `path` when a checkpoint was loaded), and the full config (Hugging Face token masked) |
| `n_input`, `n_returned`                          | Number of input structures and of returned structures                                                                                     |
| `n_generated`                                    | `ensemble` only: conformers generated (pruning can leave fewer than requested)                                                            |
| `structures`                                     | Per returned structure: `index`, `n_atoms`, `charge`, `multiplicity`, `energy_ev`, `comment`, `converged`                                 |
| `diagnostics`                                    | The engine diagnostic report (embedding and optimization events, convergence where known)                                                 |
| `error`                                          | `{"type", "message"}` when the command failed, else `null`                                                                                |
| `wall_time_s`, `gpuma_version`, `schema_version` | Run metadata                                                                                                                              |
| `checks`, `recommended_device`                   | `doctor` only: each check (`name`, `status`, `detail`, `hint`, `data`) and the GPU to use                                                 |
| `schema`                                         | `config --schema` only: the JSON Schema of the config                                                                                     |
| `installed`                                      | `skills install` only: the paths the skill was written to                                                                                 |

`config --show --json` puts the resolved config only in `config`. A usage error (exit code 2) also prints a summary, with `error.type` `UsageError` and `device`, `model` and `config` set to `null`.

| Exit code | Meaning                                                                                         |
| --------- | ----------------------------------------------------------------------------------------------- |
| `0`       | Success                                                                                         |
| `1`       | Runtime failure (for example an optimization crashed)                                           |
| `2`       | Command-line usage error                                                                        |
| `3`       | Invalid input, config value, model name or device, as detected by GPUMA's own checks            |
| `4`       | Partial result: some structures failed to optimize or stopped at `max_steps` without converging |

## Checking before running

`--dry-run` (for `optimize`, `smiles`, `ensemble` and `batch`) validates the inputs, the config and the device without loading a model, and reports the number of input structures:

```bash
gpuma batch --multi-xyz in.xyz --output out.xyz --config config.json --dry-run
```

`gpuma config --show --config config.json` prints the configuration that will actually be used, after `--config` and global overrides such as `--device`. `gpuma config --schema` prints the JSON Schema of the config file, with every key's type, default and allowed values.

## Environment checks

`gpuma doctor` checks the environment without loading a model: CUDA, the free memory of each GPU, the configured device, the Hugging Face token (for UMA models), the model cache and the installed agent skill. It prints the GPU with the most free memory as the recommended device and exits with 1 if any check fails.

```bash
gpuma doctor --config config.json
```

## Agent skill

`gpuma skills install` installs the GPUMA skill for AI coding agents; see [Using GPUMA with AI agents](https://NiklasHoelter.github.io/gpuma/agents/index.md).

| Flag           | Description                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------------- |
| `--target DIR` | Skills folder to install into (repeatable; default: `~/.claude/skills` and `~/.agents/skills`) |
| `--force`      | Replace a `gpuma` skill folder that gpuma did not install                                      |
