"""Tests for ``gpuma doctor`` (CPU-only; CUDA, GPUs and tokens faked)."""

import json
from pathlib import Path

import pytest
import torch

import gpuma.doctor as doctor
from gpuma.cli import main
from gpuma.config import Config
from gpuma.doctor import Check, recommended_device, run_checks

GPUS = [
    {"index": 0, "name": "A", "free_gib": 2.0, "total_gib": 24.0},
    {"index": 1, "name": "B", "free_gib": 20.0, "total_gib": 24.0},
]


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)


@pytest.fixture
def two_gpus(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(doctor, "_gpu_memory", lambda: GPUS)
    _fake_warp(monkeypatch, 2)


@pytest.fixture
def no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.setattr(doctor, "_hub_token", lambda: None)


def _by_name(checks):
    return {c.name: c for c in checks}


def _orb(device):
    return Config({"model": {"model_type": "orb"}, "technical": {"device": device}})


def test_no_cuda_is_fail(no_cuda):
    checks = _by_name(run_checks(_orb("cuda")))
    assert checks["cuda"].status == "fail" and checks["device"].status == "fail"


def test_no_cuda_with_cpu_device_is_warn(no_cuda):
    assert _by_name(run_checks(_orb("cpu")))["cuda"].status == "warn"


def test_old_driver_names_torch_build(no_cuda, monkeypatch):
    def init():
        raise RuntimeError("The NVIDIA driver on your system is too old (found version 12040).")

    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "init", init)
    check = _by_name(run_checks(_orb("cuda")))["cuda"]
    assert "driver is too old" in check.detail and "cu126" in check.hint


def test_old_driver_with_cuda12_build_suggests_driver_update(no_cuda, monkeypatch):
    def init():
        raise RuntimeError("The NVIDIA driver on your system is too old (found version 11080).")

    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(torch.cuda, "init", init)
    check = _by_name(run_checks(_orb("cuda")))["cuda"]
    assert "driver is too old" in check.detail and "cu126" not in check.hint


def test_uma_without_token_is_fail(two_gpus, no_token):
    config = Config({"model": {"model_type": "fairchem"}, "technical": {"device": "cuda:1"}})
    check = _by_name(run_checks(config))["hf_token"]
    assert check.status == "fail" and "orb_v3_direct_omol" in check.hint


def test_uma_with_env_token_is_ok(two_gpus, no_token, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    config = Config({"model": {"model_type": "fairchem"}, "technical": {"device": "cuda:1"}})
    assert _by_name(run_checks(config))["hf_token"].status == "ok"


def test_orb_skips_token_check(two_gpus, no_token):
    assert "hf_token" not in _by_name(run_checks(_orb("cuda:1")))


def test_busy_configured_gpu_warns(two_gpus):
    assert _by_name(run_checks(_orb("cuda:0")))["gpus"].status == "warn"
    assert _by_name(run_checks(_orb("cuda:1")))["gpus"].status == "ok"


def test_recommended_device_picks_most_free(two_gpus):
    assert recommended_device(run_checks(_orb("cuda:0"))) == "cuda:1"


def test_recommended_device_none_without_gpus(no_cuda):
    assert recommended_device(run_checks(_orb("cpu"))) is None


def test_stale_skill_warns(no_cuda, monkeypatch):
    monkeypatch.setattr(doctor, "installed_skill_versions", lambda: {Path("x"): "0.0.1"})
    check = _by_name(run_checks(_orb("cpu")))["skill"]
    assert check.status == "warn" and "gpuma skills install" in check.hint


def test_missing_skill_warns(no_cuda, monkeypatch):
    monkeypatch.setattr(doctor, "installed_skill_versions", lambda: {Path("x"): None})
    assert _by_name(run_checks(_orb("cpu")))["skill"].status == "warn"


def test_doctor_exit_code_reflects_failures(no_cuda, tmp_path, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"model": {"model_type": "orb"}, "technical": {"device": "cuda"}}))
    assert main(["doctor", "-c", str(cfg), "--json"]) == 1
    s = json.loads(capsys.readouterr().out)
    assert {c["name"] for c in s["checks"]} >= {"versions", "cuda", "device", "skill"}
    assert s["recommended_device"] is None and "cuda" in s["error"]["message"]


def test_doctor_ok_exit_0(two_gpus, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(doctor, "installed_skill_versions", lambda: {})
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"model": {"model_type": "orb"}, "technical": {"device": "cuda:1"}}))
    assert main(["doctor", "-c", str(cfg), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["recommended_device"] == "cuda:1"


def test_check_is_frozen():
    with pytest.raises(AttributeError):
        Check("a", "ok", "d").status = "fail"


def _fake_warp(monkeypatch, n_cuda):
    import sys
    import types

    fake = types.SimpleNamespace(__version__="9.9", get_cuda_device_count=lambda: n_cuda)
    monkeypatch.setitem(sys.modules, "warp", types.ModuleType("warp"))
    monkeypatch.setattr(sys.modules["warp"], "__version__", fake.__version__, raising=False)
    monkeypatch.setattr(
        sys.modules["warp"], "get_cuda_device_count", fake.get_cuda_device_count, raising=False
    )


def test_warp_without_gpu_is_fail(two_gpus, monkeypatch):
    _fake_warp(monkeypatch, 0)
    check = _by_name(run_checks(_orb("cuda:1")))["warp"]
    assert check.status == "fail" and "driver" in check.hint


def test_warp_with_gpu_is_ok(two_gpus, monkeypatch):
    _fake_warp(monkeypatch, 2)
    assert _by_name(run_checks(_orb("cuda:1")))["warp"].status == "ok"


def test_warp_check_skipped_without_cuda(no_cuda):
    assert "warp" not in _by_name(run_checks(_orb("cpu")))
