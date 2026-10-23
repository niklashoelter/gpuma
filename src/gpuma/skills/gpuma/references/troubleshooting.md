# Troubleshooting

Start from the exit code and the `error` field of the `--json` summary, then
read the end of the log (stderr).

## Symptoms

| Symptom | Cause | Fix |
| --- | --- | --- |
| Exit 3, `Unknown batch_optimizer` or another unknown value | Misspelled config value | Use a value listed by `gpuma config --schema` |
| Exit 3, `CUDA is not available` | Not on a GPU node, a broken GPU driver, or a torch build for a newer CUDA than the driver supports | Run `gpuma doctor`; work on the GPU node; if it says the driver is too old, tell the user and show its reinstall hint |
| Exit 3, `Requested GPU ... but only N GPU(s) are visible` | Wrong device index | Use `recommended_device` from `gpuma doctor --json` |
| Exit 3, `Invalid SMILES` | SMILES does not parse | Check the SMILES with the user |
| Exit 3, file not found | Wrong input path | Check the path; paths are relative to the working directory |
| Exit 2, `config file not found` | Wrong `-c` path | Fix the path |
| Warning `No --config given` | No config file, so the defaults (UMA) are used | Always pass `-c config.json` |
| Hugging Face authentication or "gated repo" error | UMA without a token, or license not accepted | See gpu-node.md, or use ORB |
| Exit 1 with an `IndexError` inside `warp`, or `gpuma doctor` reports `warp` FAIL | The installed warp-lang build needs a newer NVIDIA driver | Tell the user; the environment needs a warp-lang version built for the installed driver |
| CUDA out of memory, "Failed to allocate" | GPU too full or shared | Lower `technical.max_memory_padding` to `0.8`; pick a freer GPU |
| `model_modal` error with SevenNet | Multi-modal checkpoint | Set `model.model_modal` |
| Energies of ions or radicals look wrong | Charge or multiplicity not set for XYZ input | Set `--charge` and `--multiplicity` |
| Different conformers on a rerun | GPU conformer generation is not reproducible | Expected; not an error |
| Exit 4 | Some structures failed to optimize | See "Finding failed structures" below |

## Reading the diagnostics

`diagnostics.events` in the summary is a list of observations. Each has a
`stage` and a `reason`; fields that were not observed are `null`.

| `stage` / `reason` | Meaning |
| --- | --- |
| `embed` / `embedding_returned` | Conformers generated for input `input_index`: `generated` of `requested` |
| `embed` / `embedding_shortfall` | Fewer conformers than requested; normal for rigid molecules |
| `embed` / `rmsd_pruned` | Conformer dropped as a duplicate of another |
| `embed` / `return_cap` | Conformer dropped because more than requested survived |
| `mlip_optimize` / `criterion_result` | Convergence of input `input_index`: `converged` is `true`, `false`, or `null` (unknown, not a failure) |
| `mlip_optimize` / `optimization_returned` | Input `input_index` produced output `output_index` |
| any / `operation_exception` | That step raised `exception_type`; the log has the message |

`diagnostics.complete: false` means the report itself lost observations. It
says nothing about whether the run succeeded.

## Finding failed structures

In a partial batch (exit 4), the failed inputs are the `input_index` values with
no `optimization_returned` event, counting from 0 in input order. Structures
with `converged: false` were returned but did not reach the force criterion.
Report both to the user, with their positions in the input file. Note that
`structures[].index` in the summary counts the returned structures, not the
inputs.
