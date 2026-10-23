"""Environment checks behind ``gpuma doctor``; never loads a model."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from .agent_skills import installed_skill_versions
from .cli_report import gpuma_version
from .config import Config, resolve_model_type
from .models.device import _parse_device_string

_GIB = 1024**3
_BUSY_FRACTION = 0.5


@dataclass(frozen=True)
class Check:
    """Result of one environment check."""

    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str
    hint: str | None = None
    data: dict[str, Any] | None = None


def _dist_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _visible_physical_indices() -> list[int] | None:
    """Map CUDA indices to nvidia-smi indices; ``None`` if CUDA_VISIBLE_DEVICES is not numeric."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return list(range(torch.cuda.device_count()))
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    return [int(e) for e in entries] if all(e.isdigit() for e in entries) else None


def _gpu_memory() -> list[dict[str, Any]]:
    """Return name and free/total memory per visible GPU, indexed as ``cuda:N``."""
    physical = _visible_physical_indices()
    if physical is not None and shutil.which("nvidia-smi"):
        # nvidia-smi reads memory without creating a CUDA context on every GPU.
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode == 0:
            rows = {}
            for line in out.stdout.strip().splitlines():
                idx, name, free, total = (p.strip() for p in line.split(","))
                rows[int(idx)] = (name, float(free) / 1024, float(total) / 1024)
            if all(p in rows for p in physical):
                return [
                    {
                        "index": i,
                        "name": rows[p][0],
                        "free_gib": round(rows[p][1], 2),
                        "total_gib": round(rows[p][2], 2),
                    }
                    for i, p in enumerate(physical)
                ]
    gpus = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        gpus.append(
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "free_gib": round(free / _GIB, 2),
                "total_gib": round(total / _GIB, 2),
            }
        )
    return gpus


def _hub_token() -> str | None:
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def _check_versions() -> Check:
    versions = {
        "gpuma": gpuma_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch-sim-atomistic": _dist_version("torch-sim-atomistic"),
    }
    detail = ", ".join(f"{k} {v}" for k, v in versions.items())
    return Check("versions", "ok", detail, data=versions)


def _check_cuda(wants_cuda: bool) -> Check:
    if torch.cuda.is_available():
        return Check("cuda", "ok", f"{torch.cuda.device_count()} GPU(s) visible")
    detail = "CUDA is not available"
    hint = "Run on a GPU node and check that nvidia-smi lists the GPUs."
    if torch.version.cuda is None:
        detail += " (this torch build has no CUDA support)"
    elif _driver_too_old():
        detail += f" (this torch build targets CUDA {torch.version.cuda}; the driver is too old)"
        hint = "Update the NVIDIA driver, or install a torch build for an older CUDA."
        if int(torch.version.cuda.split(".")[0]) >= 13:
            hint = (
                "Reinstall torch for CUDA 12.6: "
                f"pip install torch=={torch.__version__.split('+')[0]} "
                "--index-url https://download.pytorch.org/whl/cu126"
            )
    return Check("cuda", "fail" if wants_cuda else "warn", detail, hint=hint)


def _driver_too_old() -> bool:
    try:
        torch.cuda.init()
    except (RuntimeError, AssertionError) as exc:
        return "too old" in str(exc)
    return False


def _check_gpus(device: str | None) -> Check | None:
    if not torch.cuda.is_available():
        return None
    gpus = _gpu_memory()
    detail = "; ".join(
        f"cuda:{g['index']} {g['name']} {g['free_gib']:.1f}/{g['total_gib']:.1f} GiB free"
        for g in gpus
    )
    configured = next((g for g in gpus if device and f"cuda:{g['index']}" == device), None)
    if device == "cuda" and gpus:
        configured = gpus[0]
    if configured and configured["free_gib"] < _BUSY_FRACTION * configured["total_gib"]:
        return Check(
            "gpus",
            "warn",
            detail,
            hint=f"cuda:{configured['index']} is busy; use recommended_device instead.",
            data={"gpus": gpus},
        )
    return Check("gpus", "ok", detail, data={"gpus": gpus})


def _check_warp() -> Check | None:
    """Warp runs the neighbor lists of batch optimization; it needs its own CUDA support."""
    if not torch.cuda.is_available():
        return None
    try:
        import warp
    except ImportError:
        return None
    n_cuda = warp.get_cuda_device_count()
    if n_cuda:
        return Check("warp", "ok", f"warp {warp.__version__} sees {n_cuda} GPU(s)")
    return Check(
        "warp",
        "fail",
        f"warp {warp.__version__} sees no GPU although torch does",
        hint=(
            "This warp-lang build needs a newer NVIDIA driver than the installed one; "
            "install a warp-lang version built for this driver's CUDA version."
        ),
    )


def _check_device(config: Config) -> tuple[Check, str | None]:
    raw = str(config.technical.device)
    try:
        resolved = _parse_device_string(raw)
    except ValueError as exc:
        return Check("device", "fail", str(exc), hint="Set technical.device or --device."), None
    return Check("device", "ok", f"configured device {raw!r} resolves to {resolved}"), resolved


def _check_hf_token(config: Config) -> Check | None:
    if resolve_model_type(config) != "fairchem":
        return None
    sources = {
        "config": config.model.get_huggingface_token(),
        "HF_TOKEN": os.environ.get("HF_TOKEN"),
        "HUGGINGFACE_TOKEN": os.environ.get("HUGGINGFACE_TOKEN"),
        "huggingface-cli login": _hub_token(),
    }
    found = [name for name, token in sources.items() if token]
    if found:
        return Check("hf_token", "ok", f"Hugging Face token found ({found[0]})")
    return Check(
        "hf_token",
        "fail",
        "UMA models need a Hugging Face token; none found",
        hint=(
            "Set model.huggingface_token_file, or use model_type orb with model_name "
            "orb_v3_direct_omol, which needs no token."
        ),
    )


def _check_model_cache(config: Config) -> Check:
    cache = config.model.model_cache_dir or os.environ.get("HF_HOME") or "~/.cache/huggingface"
    path = Path(cache).expanduser()
    existing = next(p for p in (path, *path.parents) if p.exists())
    if os.access(existing, os.W_OK):
        return Check("model_cache", "ok", f"model cache {path} is writable")
    return Check(
        "model_cache",
        "warn",
        f"model cache {path} is not writable",
        hint="Set model.model_cache_dir to a writable directory with enough space.",
    )


def _check_skill() -> Check:
    versions = installed_skill_versions()
    installed = {str(p): v for p, v in versions.items() if v is not None}
    current = gpuma_version()
    if not installed:
        return Check("skill", "warn", "agent skill not installed", hint="Run gpuma skills install.")
    stale = [p for p, v in installed.items() if v != current]
    if stale:
        return Check(
            "skill",
            "warn",
            f"agent skill in {', '.join(stale)} is from another gpuma version",
            hint="Run gpuma skills install to update it.",
        )
    return Check("skill", "ok", f"agent skill installed in {', '.join(installed)}")


def run_checks(config: Config) -> list[Check]:
    """Run every environment check for ``config`` and return the results."""
    wants_cuda = str(config.technical.device).strip().lower().startswith("cuda")
    device_check, resolved = _check_device(config)
    checks = [
        _check_versions(),
        _check_cuda(wants_cuda),
        _check_gpus(resolved),
        _check_warp(),
        device_check,
        _check_hf_token(config),
        _check_model_cache(config),
        _check_skill(),
    ]
    return [c for c in checks if c is not None]


def recommended_device(checks: list[Check]) -> str | None:
    """Return the ``cuda:N`` with the most free memory, or ``None`` without GPUs."""
    gpus = next((c.data["gpus"] for c in checks if c.name == "gpus" and c.data), [])
    if not gpus:
        return None
    return f"cuda:{max(gpus, key=lambda g: g['free_gib'])['index']}"
