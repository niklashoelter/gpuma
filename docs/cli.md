# CLI Usage

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
