# Using GPUMA with AI agents

GPUMA ships an [Agent Skill](https://agentskills.io) that teaches AI coding agents (Claude Code, OpenAI Codex, Gemini CLI, GitHub Copilot, Cursor) how to run it. With the skill installed you can ask for results in plain language, for example *"optimize the conformers of these 40 molecules and give me the lowest-energy one of each"*. The agent writes the config, checks the GPU, runs GPUMA and reports back. You don't need to know Python.

## One-time setup on the GPU node

1. **Install a coding agent**, for example Claude Code:

   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   ```

1. **Install GPUMA** as a stand-alone tool with [uv](https://docs.astral.sh/uv/), which also fetches the right Python version:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh   # only if uv is missing
   uv tool install gpuma --python 3.12 \
       --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
   ```

   The extra index gives torch's CUDA 12.6 build, which runs on NVIDIA drivers from 525 on. Without it uv installs PyPI's default torch build, which needs driver 580 or newer.

1. **Install the skill** for your agents:

   ```bash
   gpuma skills install
   ```

   This copies the skill into `~/.claude/skills/gpuma` (Claude Code) and `~/.agents/skills/gpuma` (Codex, Gemini CLI, Copilot, Cursor). Use `--target DIR` to install somewhere else, for example a project's `.claude/skills` folder.

1. **Check the machine:**

   ```bash
   gpuma doctor
   ```

   It reports the visible GPUs and their free memory, whether the configured device works, whether a Hugging Face token is available (only needed for UMA models) and whether the skill is installed. It exits with 1 if anything needs fixing.

## Working with the agent

Start the agent in a folder that contains your input files (SMILES lists, XYZ files) and describe what you want:

- *"Optimize ibuprofen and tell me its energy."*
- *"Generate 50 conformers of this SMILES, optimize them, and list the five lowest relative energies in kcal/mol."*
- *"Optimize every structure in `anions.xyz`; they all have charge -1."*
- *"The last run failed, find out why and fix it."*

The skill tells the agent to check the inputs with `--dry-run` before a run, start long runs in the background, and read the machine-readable `--json` summary instead of guessing from the logs.

## Updating

Reinstall the skill after each upgrade, so the agent's instructions match the installed version:

```bash
uv tool install --upgrade gpuma --python 3.12 \
    --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
gpuma skills install
```

`gpuma doctor` warns when the installed skill is from a different GPUMA version.

## For agent and tool developers

The CLI is designed to be driven by programs as well as people:

- `--json` prints exactly one JSON object to stdout (all logs go to stderr), including per-structure energies and the engine diagnostics.
- Exit codes: `0` success, `1` runtime failure, `2` usage error, `3` invalid input, config or device, `4` some structures in a batch failed.
- `--dry-run` validates inputs, config and device without loading a model.
- `gpuma config --schema` prints the config JSON Schema; `gpuma config --show` prints the resolved config.

See [Command Line Interface](https://NiklasHoelter.github.io/gpuma/cli/index.md) for details. The documentation is also available as [llms.txt](https://niklashoelter.github.io/gpuma/llms.txt) and [llms-full.txt](https://niklashoelter.github.io/gpuma/llms-full.txt) for tools that read documentation in that format.
