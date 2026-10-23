# AGENTS.md

Guidance for AI coding agents working on the GPUMA repository. (Agents that
*use* GPUMA get the skill in `src/gpuma/skills/gpuma/` instead.)

## Layout

| Path | Content |
| --- | --- |
| `src/gpuma/api.py`, `src/gpuma/__init__.py` | Public Python API. Don't change signatures without the maintainer's agreement. |
| `src/gpuma/cli.py` | argparse CLI; handlers return a `CommandResult` |
| `src/gpuma/cli_report.py` | Exit codes and the `--json` summary |
| `src/gpuma/config.py`, `config_schema.py` | Config defaults, validation, JSON Schema |
| `src/gpuma/optimizer.py`, `autobatching.py` | ASE and torch-sim optimization, batching |
| `src/gpuma/models/` | Model backends (Fairchem UMA, ORB-v3, SevenNet) and device handling |
| `src/gpuma/conformer_generation/` | SMILES embedding (nvMolKit on GPU, morfeus on CPU), RMSD pruning |
| `src/gpuma/utils/reporting.py` | Opt-in diagnostic events used by `--json` |
| `src/gpuma/doctor.py`, `agent_skills.py` | `gpuma doctor` and `gpuma skills install` |
| `src/gpuma/skills/gpuma/` | Agent skill shipped to end users |
| `docs/` | mkdocs site |

## Environment and tests

- uv-managed, Python 3.12 only. Run tests with `.venv/bin/python -m pytest`.
  Avoid `uv run` while someone else's job may hold the project environment.
- Tests use real models (ORB by default) and real packages. Only the CLI tests
  fake the optimization API. GPU tests skip without CUDA; Fairchem tests need
  `HF_TOKEN`.
- `tests/test_skill_content.py` fails when the skill names a subcommand, flag or
  config key that doesn't exist. After renaming any of these, update the skill.
- Lint the files you change with `ruff check` and `ruff format` (line length
  100). Some untouched files have older lint findings; leave those alone.
- Docs: `mkdocs build --strict` must pass.
- Pin every dependency exactly (`==`): runtime, the `dev` extra, the build
  backend and the CI tools in `.github/workflows`. Bump them together after
  testing; never use ranges.

## Conventions

- Keep docstrings and comments short; match the surrounding file's idiom.
- Fail loudly: an invalid value is an error with the allowed values in the
  message, never a silent fallback.
- The `--json` summary is a contract for agents. Only add keys; bump
  `SUMMARY_SCHEMA_VERSION` in `cli_report.py` for anything else.
- No backward-compatibility shims unless asked for.
- Don't add AI co-author trailers or "generated with" footers to commits or PRs.
