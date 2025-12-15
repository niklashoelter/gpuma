import argparse

import pytest

import gpuma.cli as cli
from gpuma.config import Config


def test_setup_parser_returns_parser():
    parser = cli.setup_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_optimize_parser_has_charge_and_multiplicity():
    parser = cli.setup_parser()
    args = parser.parse_args(
        [
            "optimize",
            "--xyz",
            "file.xyz",
            "--output",
            "out.xyz",
            "--charge",
            "-1",
            "--multiplicity",
            "2",
        ]
    )
    assert args.charge == -1
    assert args.multiplicity == 2


def test_batch_parser_has_charge_and_multiplicity():
    parser = cli.setup_parser()
    args = parser.parse_args(
        [
            "batch",
            "--multi-xyz",
            "m.xyz",
            "--output",
            "out.xyz",
            "--charge",
            "1",
            "--multiplicity",
            "1",
        ]
    )
    assert args.charge == 1
    assert args.multiplicity == 1


def test_help_exits():
    parser = cli.setup_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_cmd_optimize_xyz_uses_cli_charge_and_multiplicity(monkeypatch, tmp_path):
    # Arrange: fake args and config
    parser = cli.setup_parser()
    out = tmp_path / "out.xyz"
    args = parser.parse_args(
        [
            "optimize",
            "--xyz",
            "in.xyz",
            "--output",
            str(out),
            "--charge",
            "-1",
            "--multiplicity",
            "2",
        ]
    )
    config = Config()
    config.optimization.charge = 0
    config.optimization.multiplicity = 1

    captured = {}

    def fake_opt_single_xyz_file(input_file, output_file, config, charge, multiplicity):  # noqa: D401
        captured["input_file"] = input_file
        captured["output_file"] = output_file
        captured["charge"] = charge
        captured["multiplicity"] = multiplicity
        class Dummy:
            energy = 0.0
        return Dummy()

    monkeypatch.setattr(cli, "optimize_single_xyz_file", fake_opt_single_xyz_file)

    # Act
    cli.cmd_optimize(args, config)

    # Assert: CLI flags override config
    assert captured["charge"] == -1
    assert captured["multiplicity"] == 2


def test_cmd_batch_uses_cli_or_config_for_charge_and_multiplicity(monkeypatch, tmp_path):
    parser = cli.setup_parser()
    out = tmp_path / "out.xyz"
    # use xyz-dir branch
    args = parser.parse_args(
        [
            "batch",
            "--xyz-dir",
            str(tmp_path),
            "--output",
            str(out),
            "--charge",
            "3",
            "--multiplicity",
            "4",
        ]
    )
    config = Config()
    config.optimization.charge = 0
    config.optimization.multiplicity = 1

    called = {}

    def fake_read_xyz_directory(directory_path, charge, multiplicity):
        called["dir"] = directory_path
        called["charge"] = charge
        called["multiplicity"] = multiplicity
        return []

    def fake_optimize_batch(structures, config):  # pragma: no cover - not reached
        return structures

    monkeypatch.setattr(cli, "read_xyz_directory", fake_read_xyz_directory)
    monkeypatch.setattr(cli, "save_multi_xyz", lambda *a, **k: None)
    monkeypatch.setattr(cli, "Structure", object)

    # Avoid heavy optimization: patch the local import target used in cmd_batch
    from gpuma import optimizer as optimizer_mod
    monkeypatch.setattr(optimizer_mod, "optimize_structure_batch", fake_optimize_batch)

    # Act: since read_xyz_directory returns [], cmd_batch will exit early
    with pytest.raises(SystemExit):
        cli.cmd_batch(args, config)

    # Assert: CLI values were forwarded
    assert called["charge"] == 3
    assert called["multiplicity"] == 4

