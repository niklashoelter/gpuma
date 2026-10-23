---
name: gpuma
description: Optimize molecular geometries and conformer ensembles on the GPU with gpuma, using machine-learned interatomic potentials (Fairchem UMA, ORB-v3, SevenNet). Use when the user wants to relax or optimize structures from SMILES or XYZ files, generate and optimize conformers, find the lowest-energy conformer, batch-optimize many molecules, choose a model or optimizer, write a gpuma config, or find out why a gpuma run failed.
---

# Running gpuma

gpuma relaxes many molecules at once on one GPU with machine-learned
interatomic potentials. Drive it through the `gpuma` command line only; the
user should never need to write Python. You run directly on a GPU node, with
no job scheduler.

## Workflow

1. **Check the machine.** Run `gpuma doctor --json`. Resolve every check with
   status `fail` before anything else. Use `recommended_device`, the GPU with
   the most free memory, as `technical.device`.
2. **Write a config** called `config.json` in the working directory and pass
   it to every command with `-c config.json`. A good starting point for organic
   molecules:

   ```json
   {
     "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
     "optimization": {"batch_optimizer": "lbfgs"},
     "technical": {"device": "cuda:0"}
   }
   ```

   `gpuma config --schema` lists every key with its type, default and allowed
   values. `gpuma config --show -c config.json` prints what will actually run.
   Read [references/config.md](references/config.md) before choosing another
   model or changing convergence or memory settings.
3. **Pick the command.**

   | Goal | Command |
   | --- | --- |
   | One molecule | `gpuma optimize --smiles S -o out.xyz` or `gpuma optimize --xyz in.xyz -o out.xyz` |
   | Conformers of one molecule, optimized | `gpuma ensemble --smiles S --conformers N -o out.xyz` |
   | Many structures | `gpuma batch --multi-xyz in.xyz -o out.xyz` or `gpuma batch --xyz-dir dir -o out.xyz` |
   | 3D structure only, no optimization | `gpuma convert --smiles S -o out.xyz` |
   | Conformers only, no optimization | `gpuma generate --smiles S --conformers N -o out.xyz` |

4. **Validate first.** Add `--dry-run --json` to the command. It checks the
   input files or SMILES, the config and the device without loading a model,
   and reports `n_input`.
5. **Run with `--json`.** stdout then carries exactly one JSON summary and the
   logs go to stderr. Your shell commands time out after a few minutes, so
   start anything that may take longer in the background and poll:

   ```bash
   nohup gpuma batch --multi-xyz in.xyz -o out.xyz -c config.json --json > summary.json 2> run.log &
   echo $!
   ```

   Note the process ID that `echo $!` prints and check progress with
   `tail -n 5 run.log`. The run has finished once `kill -0 <pid>` fails. Then
   read `summary.json`; if it is empty, the process died before writing it
   (killed, out of memory, or a command-line error), and `run.log` says why.
   Never start a second gpuma run on the same GPU while one is running.
6. **Report back** from `summary.json`: `status`, `n_input` and `n_returned`,
   `structures[].energy_ev`, and anything unusual in `diagnostics`. Energies are
   in eV (1 eV = 96.485 kJ/mol = 23.061 kcal/mol). For conformers, report
   energies relative to the lowest one; absolute MLIP energies mean little on
   their own.

## Exit codes

| Code | Meaning | What to do |
| --- | --- | --- |
| 0 | Success | Report the results |
| 1 | Runtime failure | Read `error` in the summary and `run.log` |
| 2 | Wrong command-line usage | Read stderr and fix the command |
| 3 | Invalid input, config or device | `error.message` names the problem; fix it, rerun with `--dry-run` |
| 4 | Some structures failed | `n_returned` < `n_input`; `diagnostics` shows which |

For any code other than 0, read
[references/troubleshooting.md](references/troubleshooting.md).

## Judgement calls

- **Model.** Use ORB (`orb_v3_direct_omol`) for organic molecules unless the
  user asks otherwise: no token needed and it loads fast. UMA needs a Hugging
  Face token and its first load takes minutes. For conformer searches on
  organic molecules the model choice changes the results little.
- **Optimizer.** Set `optimization.batch_optimizer` to `lbfgs`. It is faster
  than the default `fire` and at least as accurate. Avoid `bfgs`, which is
  slower than `fire`.
- **Dispersion.** Leave `model.d3_correction` off for UMA and the ORB OMol
  models. Their training data (ωB97M-D4rev or ωB97M-V) already includes dispersion.
- **Charge and spin.** A SMILES string carries its own charge. For XYZ input,
  set `optimization.charge` and `optimization.multiplicity` or pass `--charge`
  and `--multiplicity`. A batch applies one charge and multiplicity to every
  structure, so group inputs accordingly and ask the user when unsure.
- **One process, many structures.** Every process loads the model again. Put
  all structures of one charge state into one multi-XYZ file or directory and
  run `gpuma batch` once instead of looping over `gpuma optimize`. For many
  SMILES, create the conformers with `gpuma generate` (fast, no model), combine
  the files, then optimize them in one `gpuma batch`.
- **Starting geometries.** A start from SMILES comes from a force field and can
  occasionally converge to a saddle point. When the result matters, start from
  a good XYZ structure or optimize an ensemble and keep the lowest energy.
- **Reproducibility.** On the GPU, conformer generation is not reproducible even
  with a fixed `conformer_generation.conformer_seed`. Treat two GPU runs as
  independent samples; to compare settings, run replicates.

## Reference files

- [references/config.md](references/config.md): read when choosing a model,
  optimizer, convergence criteria or memory settings.
- [references/gpu-node.md](references/gpu-node.md): read when picking a GPU,
  running out of GPU memory, setting up a Hugging Face token or model
  downloads, or planning a long run.
- [references/troubleshooting.md](references/troubleshooting.md): read when a
  command exits with a code other than 0 or the diagnostics look wrong.
