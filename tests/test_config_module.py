import logging

import pytest
import torch

from gpuma.config import (
    DEFAULT_CONFIG,
    Config,
    default_device,
    get_huggingface_token,
    load_config_from_file,
    save_config_to_file,
    validate_config,
)


def test_config_defaults_and_attribute_access():
    cfg = Config()
    # default merge
    assert (
        cfg.to_dict()["optimization"]["model_name"] == DEFAULT_CONFIG["optimization"]["model_name"]
    )
    # attribute access and set
    assert isinstance(cfg.optimization.to_dict(), dict)
    cfg.optimization.device = "cpu"
    assert cfg.optimization.device == "cpu"


def test_default_device_matches_torch():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert default_device == expected


def test_load_and_save_config_json(tmp_path):
    # Save custom config
    cfg = Config()
    cfg.optimization.logging_level = "ERROR"
    out = tmp_path / "config.json"
    save_config_to_file(cfg, str(out))
    assert out.exists()

    # Load and merge
    loaded = load_config_from_file(str(out))
    assert isinstance(loaded, Config)
    assert loaded.optimization.logging_level == "ERROR"
    # unknown keys preserved
    data = loaded.to_dict()
    data["custom"] = {"a": 1}
    loaded2 = Config.from_dict(data)
    assert loaded2.to_dict()["custom"]["a"] == 1


def test_get_huggingface_token_from_inline_and_file(tmp_path):
    token_file = tmp_path / "token.txt"
    token_file.write_text("secrettoken\n", encoding="utf-8")

    cfg = Config()
    cfg.optimization.huggingface_token = "inline"
    assert cfg.optimization.get_huggingface_token() == "inline"

    cfg2 = Config()
    cfg2.optimization.huggingface_token = None
    cfg2.optimization.huggingface_token_file = str(token_file)
    assert cfg2.optimization.get_huggingface_token() == "secrettoken"

    # dict helper
    d = {"optimization": {"huggingface_token_file": str(token_file)}}
    assert get_huggingface_token(d) == "secrettoken"


def test_get_huggingface_token_missing_file_logs_warning(tmp_path, caplog):
    cfg = Config()
    cfg.optimization.huggingface_token = None
    cfg.optimization.huggingface_token_file = str(tmp_path / "missing.txt")
    with caplog.at_level(logging.WARNING):
        token = cfg.optimization.get_huggingface_token()
    assert token is None
    assert any("Could not read HuggingFace token" in rec.message for rec in caplog.records)


def test_validate_config_device_and_charge_multiplicity():
    cfg = Config()
    # defaults should be valid
    validate_config(cfg)

    # valid cuda device strings
    cfg.optimization.device = "cuda"
    validate_config(cfg)
    cfg.optimization.device = "cuda:0"
    validate_config(cfg)

    # invalid device
    cfg.optimization.device = "gpu42"
    with pytest.raises(ValueError):
        validate_config(cfg)

    # invalid charge/multiplicity
    cfg.optimization.device = default_device
    cfg.optimization.charge = "x"
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg.optimization.charge = 0
    cfg.optimization.multiplicity = 0
    with pytest.raises(ValueError):
        validate_config(cfg)
