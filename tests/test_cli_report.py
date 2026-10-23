"""Tests for CLI exit codes and the --json run summary (CPU-only, API faked)."""

import json

import pytest

import gpuma.cli as cli
from gpuma.cli import main
from gpuma.structure import Structure
from gpuma.utils.io_handler import read_multi_xyz

from conftest import MULTI_XYZ

N_MULTI = len(read_multi_xyz(str(MULTI_XYZ)))


def _write_config(tmp_path, **model):
    cfg = {
        "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol", **model},
        "technical": {"device": "cpu"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def _he(energy=-1.0):
    return Structure(
        symbols=["He"], coordinates=[(0.0, 0.0, 0.0)], charge=0, multiplicity=1, energy=energy
    )


def _fake_batch(monkeypatch, n_returned, effect=None):
    def fake(input_file, output_file=None, config=None):
        if effect:
            effect()
        return [_he(-float(i)) for i in range(n_returned)]

    monkeypatch.setattr(cli, "optimize_batch_multi_xyz_file", fake)


def _batch_argv(tmp_path, *extra, input_file=MULTI_XYZ):
    return [
        "batch", "--multi-xyz", str(input_file), "-o", str(tmp_path / "o.xyz"),
        "-c", _write_config(tmp_path), *extra,
    ]


def test_json_summary_on_success(tmp_path, monkeypatch, capsys):
    _fake_batch(monkeypatch, N_MULTI)
    assert main(_batch_argv(tmp_path, "--json")) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["status"] == "ok" and s["exit_code"] == 0
    assert s["n_input"] == N_MULTI and s["n_returned"] == N_MULTI
    assert s["structures"][0].keys() >= {"index", "energy_ev", "n_atoms"}
    assert s["diagnostics"]["schema_version"] == 1
    assert s["model"] == {"type": "orb", "name": "orb_v3_direct_omol"}
    assert s["device"] == "cpu"


def test_partial_batch_exit_code(tmp_path, monkeypatch, capsys):
    _fake_batch(monkeypatch, N_MULTI - 1)
    assert main(_batch_argv(tmp_path, "--json")) == 4
    assert json.loads(capsys.readouterr().out)["status"] == "partial"


def test_invalid_input_exit_code(tmp_path, monkeypatch, capsys):
    _fake_batch(monkeypatch, 1)
    argv = _batch_argv(tmp_path, "--json", input_file=tmp_path / "missing.xyz")
    assert main(argv) == 3
    s = json.loads(capsys.readouterr().out)
    assert s["status"] == "error" and s["error"]["type"] == "FileNotFoundError"


def test_invalid_config_file_exit_code(tmp_path, capsys):
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"optimization": {"batch_optimizer": "lbgfs"}}))
    argv = ["batch", "--multi-xyz", str(MULTI_XYZ), "-o", "o.xyz", "-c", str(cfg), "--json"]
    assert main(argv) == 3
    s = json.loads(capsys.readouterr().out)
    assert "lbgfs" in s["error"]["message"] and s["config"] is None


def test_runtime_failure_exit_code(tmp_path, monkeypatch, capsys):
    def boom():
        raise RuntimeError("kaboom")

    _fake_batch(monkeypatch, 1, effect=boom)
    assert main(_batch_argv(tmp_path, "--json")) == 1
    s = json.loads(capsys.readouterr().out)
    assert s["error"] == {"type": "RuntimeError", "message": "kaboom"}


def test_json_stdout_is_single_object_despite_prints(tmp_path, monkeypatch, capsys):
    _fake_batch(monkeypatch, N_MULTI, effect=lambda: print("noise from a backend"))
    main(_batch_argv(tmp_path, "--json"))
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "ok"
    assert "noise from a backend" in captured.err


def test_summary_redacts_token(tmp_path, monkeypatch, capsys):
    _fake_batch(monkeypatch, N_MULTI)
    cfg = _write_config(tmp_path, huggingface_token="hf_secret")
    argv = ["batch", "--multi-xyz", str(MULTI_XYZ), "-o", str(tmp_path / "o.xyz"), "-c", cfg]
    main([*argv, "--json"])
    out = capsys.readouterr().out
    assert "hf_secret" not in out
    assert json.loads(out)["config"]["model"]["huggingface_token"] == "***"


def test_without_json_stdout_is_empty(tmp_path, monkeypatch, capsys):
    _fake_batch(monkeypatch, N_MULTI)
    assert main(_batch_argv(tmp_path)) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("exc", "code"),
    [(ValueError("x"), 3), (FileNotFoundError("x"), 3), (RuntimeError("x"), 1)],
)
def test_exit_code_for(exc, code):
    from gpuma.cli_report import exit_code_for

    assert exit_code_for(exc) == code


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def _forbid_api(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not call the optimization API")

    for name in (
        "optimize_single_smiles", "optimize_single_xyz_file", "optimize_ensemble_smiles",
        "optimize_batch_multi_xyz_file", "optimize_batch_xyz_directory",
    ):
        monkeypatch.setattr(cli, name, forbidden)


def test_dry_run_does_not_call_api(tmp_path, monkeypatch, capsys):
    _forbid_api(monkeypatch)
    assert main(_batch_argv(tmp_path, "--dry-run", "--json")) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["status"] == "dry_run" and s["n_input"] == N_MULTI


def test_dry_run_ensemble_reports_conformer_count(tmp_path, monkeypatch, capsys):
    _forbid_api(monkeypatch)
    argv = ["ensemble", "--smiles", "CCO", "--conformers", "3", "-o", "o.xyz",
            "-c", _write_config(tmp_path), "--dry-run", "--json"]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["n_input"] == 3


def test_dry_run_bad_smiles_exit_3(tmp_path, monkeypatch, capsys):
    _forbid_api(monkeypatch)
    argv = ["optimize", "--smiles", "C1CC", "-o", "o.xyz", "-c", _write_config(tmp_path),
            "--dry-run", "--json"]
    assert main(argv) == 3
    assert "SMILES" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_dry_run_bad_device_exit_3(tmp_path, monkeypatch, capsys):
    import torch

    _forbid_api(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    argv = _batch_argv(tmp_path, "--dry-run", "--json", "--device", "cuda")
    assert main(argv) == 3
    assert "CUDA is not available" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_dry_run_on_convert_is_usage_error():
    with pytest.raises(SystemExit) as exc_info:
        main(["convert", "--smiles", "C", "-o", "o.xyz", "--dry-run"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_stdout_to_stderr_flushes_dunder_and_libc():
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    script = textwrap.dedent(
        """
        import ctypes, sys
        from gpuma.cli_report import stdout_to_stderr
        with stdout_to_stderr():
            sys.__stdout__.write("NOISE_DUNDER\\n")
            ctypes.CDLL(None).printf(b"NOISE_LIBC\\n")
        print('{"ok": true}')
        """
    )
    src = str(Path(__file__).resolve().parent.parent / "src")
    env = {**os.environ, "PYTHONPATH": src, "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=300
    )
    assert out.stdout == '{"ok": true}\n'
    assert "NOISE_DUNDER" in out.stderr and "NOISE_LIBC" in out.stderr


def test_optimize_unusable_device_exit_3(tmp_path, monkeypatch, capsys):
    import torch

    def wrapped_failure(**kwargs):
        raise RuntimeError("Optimization failed: CUDA is not available")

    monkeypatch.setattr(cli, "optimize_single_smiles", wrapped_failure)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    argv = ["optimize", "--smiles", "CCO", "-o", "o.xyz", "-c", _write_config(tmp_path),
            "--device", "cuda", "--json"]
    assert main(argv) == 3
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "ValueError"


def test_dry_run_unknown_model_name_exit_3(tmp_path, monkeypatch, capsys):
    _forbid_api(monkeypatch)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"technical": {"device": "cpu"}}))
    argv = ["optimize", "--smiles", "CCO", "-o", "o.xyz", "-c", str(cfg), "--model-type", "orb",
            "--dry-run", "--json"]
    assert main(argv) == 3
    message = json.loads(capsys.readouterr().out)["error"]["message"]
    assert "uma-s-1p2p1-ft-d4rev" in message and "orb_v3_direct_omol" in message


def test_library_value_error_is_runtime_failure(tmp_path, monkeypatch, capsys):
    _fake_batch(monkeypatch, 1, effect=lambda: int("not a number"))
    assert main(_batch_argv(tmp_path, "--json")) == 1
    assert json.loads(capsys.readouterr().out)["error"]["type"] == "ValueError"


def test_gpuma_value_error_is_invalid_input():
    from gpuma.cli_report import exit_code_for
    from gpuma.models.device import _parse_device_string

    with pytest.raises(ValueError) as exc_info:
        _parse_device_string("tpu")
    assert exit_code_for(exc_info.value) == 3
