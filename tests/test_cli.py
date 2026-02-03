from unittest.mock import patch

from gpuma.cli import main


def test_cli_optimize_smiles(tmp_path):
    output = tmp_path / "out.xyz"
    args = ["optimize", "--smiles", "C", "-o", str(output)]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()
        assert "Energy:" in output.read_text()

def test_cli_optimize_xyz(tmp_path, sample_xyz_content):
    inp = tmp_path / "in.xyz"
    inp.write_text(sample_xyz_content)
    output = tmp_path / "out.xyz"
    args = ["optimize", "--xyz", str(inp), "-o", str(output), "--charge", "1"]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()
        # Verify charge was passed (our mock optimize returns energy -50 but respects charge
        # in structure if IO reads it)
        # But optimize_single_xyz_file reads it.
        # Check structure in file
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
    args = ["ensemble", "--smiles", "C", "-o", str(output), "--conformers", "2"]

    with patch("sys.argv", ["gpuma"] + args):
        ret = main()
        assert ret == 0
        assert output.exists()

def test_cli_batch_multi_xyz(tmp_path, sample_multi_xyz_content):
    inp = tmp_path / "multi.xyz"
    inp.write_text(sample_multi_xyz_content)
    output = tmp_path / "out.xyz"
    args = ["batch", "--multi-xyz", str(inp), "-o", str(output)]

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
        # Convert does NO optimization, so no energy in comment
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

def test_cli_config_validate(tmp_path):
    # First create one
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

    # Check for debug logs if any (depends on what's logged)
    # The config logic sets logging level to DEBUG
    pass
