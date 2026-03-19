import json
from unittest.mock import patch

import pytest

from gpuma.cli import main


def _write_sequential_config(tmp_path):
    """Write a config that forces sequential mode to avoid torch-sim device issues."""
    cfg_file = tmp_path / "test_config.json"
    cfg_file.write_text(json.dumps({
        "optimization": {"batch_optimization_mode": "sequential"}
    }))
    return str(cfg_file)


def test_cli_optimize_smiles(tmp_path):
    output = tmp_path / "out.xyz"
    args = ["optimize", "--smiles", "C", "-o", str(output)]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()
        assert "Energy:" in output.read_text()


def test_cli_optimize_smiles_orb(tmp_path):
    """CLI optimize with --model-type orb."""
    output = tmp_path / "out.xyz"
    args = ["--model-type", "orb", "optimize", "--smiles", "C", "-o", str(output)]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()


def test_cli_optimize_xyz(tmp_path, sample_xyz_content):
    inp = tmp_path / "in.xyz"
    inp.write_text(sample_xyz_content)
    output = tmp_path / "out.xyz"
    args = ["optimize", "--xyz", str(inp), "-o", str(output), "--charge", "1"]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()
        content = output.read_text()
        assert "Charge: 1" in content


def test_cli_smiles_alias(tmp_path):
    output = tmp_path / "out.xyz"
    args = ["smiles", "--smiles", "C", "-o", str(output)]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()


def test_cli_ensemble(tmp_path):
    output = tmp_path / "out.xyz"
    cfg = _write_sequential_config(tmp_path)
    args = ["ensemble", "--smiles", "C", "-o", str(output), "--conformers", "2", "-c", cfg]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()


def test_cli_batch_multi_xyz(tmp_path, sample_multi_xyz_content):
    inp = tmp_path / "multi.xyz"
    inp.write_text(sample_multi_xyz_content)
    output = tmp_path / "out.xyz"
    cfg = _write_sequential_config(tmp_path)
    args = ["batch", "--multi-xyz", str(inp), "-o", str(output), "-c", cfg]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()


def test_cli_convert(tmp_path):
    output = tmp_path / "out.xyz"
    args = ["convert", "--smiles", "C", "-o", str(output)]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()
        assert "Energy:" not in output.read_text()


def test_cli_generate(tmp_path):
    output = tmp_path / "out.xyz"
    args = ["generate", "--smiles", "C", "-o", str(output), "--conformers", "2"]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()


def test_cli_config_create(tmp_path):
    output = tmp_path / "config.json"
    args = ["config", "--create", str(output)]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()
        # Verify new default model name
        data = json.loads(output.read_text())
        assert data["model"]["model_name"] == "uma-s-1p2"


def test_cli_config_validate(tmp_path):
    output = tmp_path / "config.json"
    args_create = ["config", "--create", str(output)]
    with patch("sys.argv", ["gpuma"] + args_create):
        main()

    args_val = ["config", "--validate", str(output)]
    with patch("sys.argv", ["gpuma"] + args_val):
        ret = main()
        assert ret == 0


def test_cli_no_args(capsys):
    with patch("sys.argv", ["gpuma"]):
        ret = main()
        assert ret == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out or "usage:" in captured.err


def test_cli_verbose(tmp_path, caplog):
    output = tmp_path / "out.xyz"
    args = ["-v", "optimize", "--smiles", "C", "-o", str(output)]

    import logging
    with caplog.at_level(logging.DEBUG):
        with patch("sys.argv", ["gpuma"] + args):
            main()


def test_cli_help(capsys):
    with patch("sys.argv", ["gpuma", "--help"]):
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "GPUMA" in captured.out or "usage:" in captured.out


def test_cli_device_override(tmp_path):
    """Verify --device flag overrides config."""
    output = tmp_path / "out.xyz"
    args = ["--device", "cpu", "optimize", "--smiles", "C", "-o", str(output)]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0


def test_cli_model_type_override(tmp_path):
    """Verify --model-type flag overrides config."""
    output = tmp_path / "out.xyz"
    args = ["--model-type", "orb", "optimize", "--smiles", "C", "-o", str(output)]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
