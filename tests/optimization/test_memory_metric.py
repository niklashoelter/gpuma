"""Tests for choosing the autobatcher memory metric per backend."""

import pytest

from gpuma.config import Config
from gpuma.optimizer import _memory_scales_with, _warn_batch_optimizer_memory


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("fairchem", "n_atoms"),
        ("uma", "n_atoms"),
        ("orb", "n_edges"),
        ("sevennet", "n_edges"),
        ("aimnet", "n_atoms"),
    ],
)
def test_auto_picks_metric_by_backend(model_type, expected):
    cfg = Config({"model": {"model_type": model_type}})
    assert _memory_scales_with(cfg) == expected


@pytest.mark.parametrize("explicit", ["n_atoms", "n_edges"])
def test_explicit_metric_overrides_backend(explicit):
    cfg = Config({
        "model": {"model_type": "fairchem"},
        "technical": {"memory_scales_with": explicit},
    })
    assert _memory_scales_with(cfg) == explicit


def test_value_set_after_construction_is_checked():
    cfg = Config({"model": {"model_type": "orb"}})
    cfg.technical.memory_scales_with = "N_ATOMS"
    assert _memory_scales_with(cfg) == "n_atoms"
    cfg.technical.memory_scales_with = "n_atoms_x_density"
    with pytest.raises(ValueError, match="memory_scales_with"):
        _memory_scales_with(cfg)


@pytest.mark.parametrize(
    ("model_type", "optimizer", "warned"),
    [
        ("orb", "bfgs", "BFGS"),
        ("fairchem", "bfgs", "BFGS"),
        ("aimnet", "bfgs", "BFGS"),
        ("aimnet", "lbfgs", None),
        ("aimnet", "fire", None),
        ("orb", "lbfgs", None),
    ],
)
def test_batch_optimizer_memory_warning(caplog, model_type, optimizer, warned):
    """The memory estimate misses BFGS's Hessian; other optimizers fit."""
    cfg = Config(
        {"model": {"model_type": model_type}, "optimization": {"batch_optimizer": optimizer}}
    )
    with caplog.at_level("WARNING", logger="gpuma.optimizer"):
        _warn_batch_optimizer_memory(cfg)
    messages = [r.getMessage() for r in caplog.records]
    if warned is None:
        assert not messages
    else:
        assert len(messages) == 1 and messages[0].startswith(warned)
        assert "fire" in messages[0]


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="batch mode needs a GPU")
def test_batch_mode_warns_once_per_job(caplog, monkeypatch):
    from gpuma import optimizer as opt
    from gpuma.structure import Structure

    monkeypatch.setattr(opt, "_optimize_batch_or_each", lambda structures, config: [None])
    cfg = Config(
        {
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "optimization": {"batch_optimizer": "bfgs"},
            "technical": {"device": "cuda:0"},
        }
    )
    he = Structure(["He"], [(0.0, 0.0, 0.0)], 0, 1)
    with caplog.at_level("WARNING", logger="gpuma.optimizer"):
        opt.optimize_structure_batch([he], cfg)
    assert sum(r.getMessage().startswith("BFGS") for r in caplog.records) == 1
