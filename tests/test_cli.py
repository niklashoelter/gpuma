"""Tests for the CLI — real end-to-end with ORB models."""

import json

import pytest

from gpuma.cli import main
from gpuma.config import Config, save_config_to_file

from conftest import (
    DEVICE,
    MULTI_XYZ,
    MULTI_XYZ_DIR,
    SINGLE_XYZ,
    requires_gpu,
    requires_hf_token,
)


def _write_orb_config(tmp_path, mode="sequential", **extra_optimization):
    """Write an ORB config JSON file and return its path."""
    cfg = {
        "optimization": {
            "batch_optimization_mode": mode,
            "force_convergence_criterion": 0.5,
            **extra_optimization,
        },
        "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
        "technical": {"device": DEVICE, "max_atoms_to_try": 1000},
    }
    path = tmp_path / "config.json"
    with open(path, "w") as f:
        json.dump(cfg, f)
    return str(path)


# ---------------------------------------------------------------------------
# Parser / help / no-args
# ---------------------------------------------------------------------------


class TestCliBasics:
    """CLI invocation without commands or with --help."""

    def test_no_args_returns_1(self):
        """No subcommand prints help and returns exit code 1."""
        assert main([]) == 1

    def test_help(self):
        """--help prints usage and exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# optimize (single structure)
# ---------------------------------------------------------------------------


class TestCliOptimize:
    """CLI single-structure optimization from SMILES or XYZ."""

    def test_optimize_smiles(self, tmp_path):
        """optimize --smiles produces an output XYZ file."""
        cfg = _write_orb_config(tmp_path)
        out = tmp_path / "out.xyz"
        assert main(["optimize", "--smiles", "C", "-o", str(out), "-c", cfg]) == 0
        assert out.exists()

    def test_optimize_smiles_output_has_energy(self, tmp_path):
        """Output XYZ contains optimized energy in the comment line."""
        cfg = _write_orb_config(tmp_path)
        out = tmp_path / "out.xyz"
        main(["optimize", "--smiles", "C", "-o", str(out), "-c", cfg])
        content = out.read_text()
        assert "Energy:" in content
        lines = [line for line in content.strip().split("\n") if line.strip()]
        assert int(lines[0].strip()) == 5

    def test_optimize_xyz(self, tmp_path):
        """optimize --xyz reads and optimizes an XYZ file."""
        cfg = _write_orb_config(tmp_path)
        out = tmp_path / "out.xyz"
        assert main(["optimize", "--xyz", str(SINGLE_XYZ), "-o", str(out), "-c", cfg]) == 0
        assert out.exists()

    def test_smiles_alias(self, tmp_path):
        """'smiles' subcommand is an alias for 'optimize --smiles'."""
        cfg = _write_orb_config(tmp_path)
        out = tmp_path / "out.xyz"
        assert main(["smiles", "--smiles", "C", "-o", str(out), "-c", cfg]) == 0
        assert out.exists()

    @requires_hf_token
    def test_optimize_smiles_fairchem(self, tmp_path):
        """Fairchem backend works through the CLI."""
        cfg_data = {
            "optimization": {
                "batch_optimization_mode": "sequential",
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "fairchem", "model_name": "uma-s-1p1"},
            "technical": {"device": DEVICE},
        }
        cfg_path = tmp_path / "config.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg_data, f)
        out = tmp_path / "out.xyz"
        assert main(["optimize", "--smiles", "C", "-o", str(out), "-c", str(cfg_path)]) == 0


# ---------------------------------------------------------------------------
# ensemble
# ---------------------------------------------------------------------------


class TestCliEnsemble:
    """CLI conformer ensemble optimization."""

    def test_ensemble(self, tmp_path):
        """ensemble generates and optimizes conformers."""
        cfg = _write_orb_config(tmp_path)
        out = tmp_path / "ensemble.xyz"
        assert main([
            "ensemble", "--smiles", "C", "--conformers", "2",
            "-o", str(out), "-c", cfg,
        ]) == 0
        assert out.exists()


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


class TestCliBatch:
    """CLI batch optimization from files."""

    @requires_gpu
    def test_batch_multi_xyz(self, tmp_path):
        """batch --multi-xyz runs GPU batch optimization."""
        cfg = _write_orb_config(tmp_path, mode="batch")
        out = tmp_path / "batch.xyz"
        assert main([
            "batch", "--multi-xyz", str(MULTI_XYZ), "-o", str(out), "-c", cfg,
        ]) == 0
        assert out.exists()

    def test_batch_sequential(self, tmp_path):
        """batch in sequential mode works without GPU."""
        cfg = _write_orb_config(tmp_path, mode="sequential")
        out = tmp_path / "batch_seq.xyz"
        assert main([
            "batch", "--multi-xyz", str(MULTI_XYZ), "-o", str(out), "-c", cfg,
        ]) == 0
        assert out.exists()

    @requires_gpu
    def test_batch_xyz_dir(self, tmp_path):
        """batch --xyz-dir reads and optimizes a directory of XYZ files."""
        cfg = _write_orb_config(tmp_path, mode="batch")
        out = tmp_path / "dir.xyz"
        assert main([
            "batch", "--xyz-dir", str(MULTI_XYZ_DIR), "-o", str(out), "-c", cfg,
        ]) == 0
        assert out.exists()


# ---------------------------------------------------------------------------
# convert / generate (no optimization)
# ---------------------------------------------------------------------------


class TestCliUtilities:
    """CLI utility commands that don't run optimization."""

    def test_convert(self, tmp_path):
        """convert produces an XYZ file from SMILES without optimization."""
        out = tmp_path / "converted.xyz"
        assert main(["convert", "--smiles", "CCO", "-o", str(out)]) == 0
        assert out.exists()
        assert "O" in out.read_text()

    def test_generate(self, tmp_path):
        """generate produces a multi-XYZ conformer ensemble file."""
        out = tmp_path / "conformers.xyz"
        assert main(["generate", "--smiles", "CCCC", "--conformers", "3", "-o", str(out)]) == 0
        assert out.exists()


# ---------------------------------------------------------------------------
# config management
# ---------------------------------------------------------------------------


class TestCliConfig:
    """CLI config create and validate commands."""

    def test_config_create(self, tmp_path):
        """config --create writes a valid default config file."""
        cfg_file = tmp_path / "new_config.json"
        assert main(["config", "--create", str(cfg_file)]) == 0
        assert cfg_file.exists()
        with open(cfg_file) as f:
            data = json.load(f)
        assert data["model"]["model_name"] == "uma-s-1p2"

    def test_config_validate(self, tmp_path):
        """config --validate accepts a valid config file."""
        cfg_file = tmp_path / "test_config.json"
        save_config_to_file(Config(), str(cfg_file))
        assert main(["config", "--validate", str(cfg_file)]) == 0


# ---------------------------------------------------------------------------
# Global flags
# ---------------------------------------------------------------------------


class TestCliGlobalFlags:
    """CLI global flags: --verbose, --quiet, --device, --model-type."""

    def test_verbose(self, tmp_path):
        """--verbose flag enables DEBUG logging."""
        out = tmp_path / "out.xyz"
        assert main(["-v", "convert", "--smiles", "C", "-o", str(out)]) == 0

    def test_quiet(self, tmp_path):
        """--quiet flag suppresses output."""
        out = tmp_path / "out.xyz"
        assert main(["-q", "convert", "--smiles", "C", "-o", str(out)]) == 0

    def test_device_override(self, tmp_path):
        """--device overrides the config device setting."""
        cfg = _write_orb_config(tmp_path)
        out = tmp_path / "out.xyz"
        assert main([
            "--device", "cpu", "optimize", "--smiles", "C", "-o", str(out), "-c", cfg,
        ]) == 0

    def test_model_type_override(self, tmp_path):
        """--model-type overrides the config model_type setting."""
        cfg = _write_orb_config(tmp_path)
        out = tmp_path / "out.xyz"
        assert main([
            "--model-type", "orb", "optimize", "--smiles", "C", "-o", str(out), "-c", cfg,
        ]) == 0
