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
        assert data["model"]["model_name"] == "uma-s-1p2p1-ft-d4rev"

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


# ---------------------------------------------------------------------------
# Conformer count resolution
# ---------------------------------------------------------------------------


def test_conformers_zero_is_rejected():
    """--conformers 0 used to fall through to the config value.

    ``args.conformers or config...`` treats 0 as "not given", so asking for
    none quietly produced however many the config named.
    """
    import argparse

    from gpuma.cli import _resolve_conformer_count

    args = argparse.Namespace(conformers=0)
    with pytest.raises(ValueError, match="--conformers must be positive"):
        _resolve_conformer_count(args, Config())


def test_conformers_negative_is_rejected():
    """Symmetric to the zero case."""
    import argparse

    from gpuma.cli import _resolve_conformer_count

    with pytest.raises(ValueError, match="--conformers must be positive"):
        _resolve_conformer_count(argparse.Namespace(conformers=-3), Config())


def test_conformers_omitted_falls_back_to_config():
    """Omitting the flag still takes the configured count."""
    import argparse

    from gpuma.cli import _resolve_conformer_count

    config = Config({"conformer_generation": {"max_num_conformers": 7}})
    assert _resolve_conformer_count(argparse.Namespace(conformers=None), config) == 7


def test_conformers_explicit_value_wins():
    """An explicit count overrides the config."""
    import argparse

    from gpuma.cli import _resolve_conformer_count

    config = Config({"conformer_generation": {"max_num_conformers": 7}})
    assert _resolve_conformer_count(argparse.Namespace(conformers=3), config) == 3


def test_generate_passes_configured_multiplicity(tmp_path, monkeypatch):
    """'generate' ignored config multiplicity while 'ensemble' honoured it."""
    import argparse

    import gpuma.cli as cli
    from gpuma.structure import Structure

    seen = {}

    def fake_ensemble(smiles, num_confs, multiplicity=None, seed=None, config=None):
        seen["multiplicity"] = multiplicity
        return [
            Structure(symbols=["He"], coordinates=[(0.0, 0.0, 0.0)],
                      charge=0, multiplicity=multiplicity or 1)
        ]

    monkeypatch.setattr(cli, "smiles_to_ensemble", fake_ensemble)

    config = Config({"optimization": {"multiplicity": 3}})
    args = argparse.Namespace(
        smiles="[He]", conformers=1, output=str(tmp_path / "out.xyz")
    )
    cli.cmd_generate(args, config)

    assert seen["multiplicity"] == 3


# ---------------------------------------------------------------------------
# Config handling: --config on every command, missing files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", [["convert"], ["generate", "--conformers", "2"]])
def test_convert_and_generate_accept_config(command):
    from gpuma.cli import setup_parser

    args = setup_parser().parse_args([*command, "--smiles", "C", "-o", "x.xyz", "-c", "c.json"])

    assert args.config == "c.json"


def test_missing_explicit_config_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        main(["optimize", "--smiles", "C", "-o", str(tmp_path / "x.xyz"), "-c", "nope.json"])
    assert exc_info.value.code == 2


def test_validate_rejects_a_missing_file(tmp_path):
    assert main(["config", "--validate", str(tmp_path / "nope.json")]) == 3


def test_no_config_anywhere_reports_defaults(tmp_path, monkeypatch):
    import argparse

    from gpuma.cli import _resolve_config_path, setup_parser

    monkeypatch.chdir(tmp_path)
    path, using_defaults = _resolve_config_path(argparse.Namespace(config=None), setup_parser())
    assert (path, using_defaults) == ("config.json", True)

    (tmp_path / "config.json").write_text("{}")
    assert _resolve_config_path(argparse.Namespace(config=None), setup_parser())[1] is False


# ---------------------------------------------------------------------------
# Global flags in any position; charge/multiplicity from config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--device", "cpu", "batch", "--multi-xyz", "a.xyz", "-o", "o.xyz"],
        ["batch", "--multi-xyz", "a.xyz", "-o", "o.xyz", "--device", "cpu"],
    ],
)
def test_global_flags_before_or_after_subcommand(argv):
    from gpuma.cli import setup_parser

    assert setup_parser().parse_args(argv).device == "cpu"


def test_subcommand_position_wins():
    from gpuma.cli import setup_parser

    args = setup_parser().parse_args(
        ["--device", "cpu", "batch", "--multi-xyz", "a", "-o", "o", "--device", "cuda:1"]
    )
    assert args.device == "cuda:1"


def test_flag_before_subcommand_survives_parent_default():
    from gpuma.cli import setup_parser

    args = setup_parser().parse_args(["-v", "--json", "convert", "--smiles", "C", "-o", "o"])
    assert args.verbose is True and args.json is True


def _record_batch_config(monkeypatch):
    import gpuma.cli as cli
    from gpuma.structure import Structure

    seen = {}

    def fake_batch(input_file, output_file=None, config=None):
        seen["charge"] = config.optimization.charge
        seen["multiplicity"] = config.optimization.multiplicity
        return [Structure(symbols=["He"], coordinates=[(0.0, 0.0, 0.0)], charge=0, multiplicity=1)]

    monkeypatch.setattr(cli, "optimize_batch_multi_xyz_file", fake_batch)
    return seen


def test_batch_uses_config_charge_when_flag_omitted(tmp_path, monkeypatch):
    seen = _record_batch_config(monkeypatch)
    cfg = _write_orb_config(tmp_path, charge=-1, multiplicity=2)
    main(["batch", "--multi-xyz", str(MULTI_XYZ), "-o", str(tmp_path / "o.xyz"), "-c", cfg])
    assert (seen["charge"], seen["multiplicity"]) == (-1, 2)


def test_batch_charge_flag_overrides_config(tmp_path, monkeypatch):
    seen = _record_batch_config(monkeypatch)
    cfg = _write_orb_config(tmp_path, charge=-1, multiplicity=2)
    main(["batch", "--multi-xyz", str(MULTI_XYZ), "-o", str(tmp_path / "o.xyz"), "-c", cfg,
          "--charge", "1"])
    assert (seen["charge"], seen["multiplicity"]) == (1, 2)
