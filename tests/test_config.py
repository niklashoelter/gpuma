"""Tests for the Config system — validation, load/save, and section access."""

import json

import pytest
import yaml

from gpuma.config import (
    VALID_BATCH_OPTIMIZERS,
    Config,
    load_config_from_file,
    resolve_model_type,
    save_config_to_file,
)


def test_config_initialization():
    """Default Config has correct initial values."""
    cfg = Config()
    assert cfg.optimization.charge == 0
    assert cfg.model.model_name == "uma-s-1p2"


def test_config_override():
    """User overrides are merged; unspecified defaults persist."""
    data = {"optimization": {"charge": 1, "new_key": "value"}}
    cfg = Config(data)
    assert cfg.optimization.charge == 1
    assert cfg.optimization.new_key == "value"
    assert cfg.optimization.multiplicity == 1


def test_section_access():
    """Attribute access reads and writes through to the underlying dict."""
    cfg = Config()
    opt = cfg.optimization
    assert opt.charge == 0

    opt.charge = 2
    assert cfg.optimization.charge == 2

    with pytest.raises(AttributeError):
        _ = opt.non_existent


def test_load_save_json(tmp_path):
    """Config round-trips through JSON correctly."""
    config_file = tmp_path / "config.json"
    cfg = Config({"optimization": {"charge": -1}})
    save_config_to_file(cfg, str(config_file))

    loaded_cfg = load_config_from_file(str(config_file))
    assert loaded_cfg.optimization.charge == -1

    with open(config_file) as f:
        data = json.load(f)
    assert data["optimization"]["charge"] == -1


def test_load_save_yaml(tmp_path):
    """Config round-trips through YAML correctly."""
    config_file = tmp_path / "config.yaml"
    cfg = Config({"optimization": {"charge": -2}})
    save_config_to_file(cfg, str(config_file))

    loaded_cfg = load_config_from_file(str(config_file))
    assert loaded_cfg.optimization.charge == -2

    with open(config_file) as f:
        data = yaml.safe_load(f)
    assert data["optimization"]["charge"] == -2


def test_load_non_existent():
    """Loading a missing file returns defaults without error."""
    cfg = load_config_from_file("non_existent_file.json")
    assert cfg.optimization.charge == 0


def test_validate_config():
    """Invalid charge, multiplicity, and device raise ValueError."""
    with pytest.raises(ValueError, match="Invalid charge"):
        Config({"optimization": {"charge": "invalid"}})

    with pytest.raises(ValueError, match="Multiplicity must be a positive integer"):
        Config({"optimization": {"multiplicity": 0}})

    with pytest.raises(ValueError, match="Device must be"):
        Config({"technical": {"device": "invalid_device"}})


def test_validate_config_convergence():
    """Convergence criteria must be positive floats."""
    with pytest.raises(ValueError, match="force_convergence_criterion must be a positive float"):
        Config({"optimization": {"force_convergence_criterion": -0.01}})

    with pytest.raises(ValueError, match="force_convergence_criterion must be a positive float"):
        Config({"optimization": {"force_convergence_criterion": "invalid"}})

    with pytest.raises(ValueError, match="energy_convergence_criterion must be a positive float"):
        Config({"optimization": {"energy_convergence_criterion": 0.0}})


def test_validate_config_device_empty():
    """Empty or whitespace device strings raise ValueError."""
    with pytest.raises(ValueError, match="Device string in config cannot be empty"):
        Config({"technical": {"device": ""}})

    with pytest.raises(ValueError, match="Device string in config cannot be empty"):
        Config({"technical": {"device": "   "}})


def test_huggingface_token(tmp_path, monkeypatch):
    """HF token can come from config, file, or both (config takes priority)."""
    cfg = Config({"model": {"huggingface_token": "token_in_config"}})
    assert cfg.model.get_huggingface_token() == "token_in_config"

    token_file = tmp_path / "token.txt"
    token_file.write_text("token_in_file")
    cfg = Config({"model": {"huggingface_token_file": str(token_file)}})
    assert cfg.model.get_huggingface_token() == "token_in_file"

    cfg = Config({
        "model": {
            "huggingface_token": "priority_token",
            "huggingface_token_file": str(token_file),
        }
    })
    assert cfg.model.get_huggingface_token() == "priority_token"


def test_section_to_dict():
    """Section.to_dict() returns a deep copy that doesn't affect the config."""
    cfg = Config()
    d = cfg.optimization.to_dict()
    assert isinstance(d, dict)
    assert d["charge"] == 0

    d["charge"] = 100
    assert cfg.optimization.charge == 0


def test_config_to_dict():
    """Config.to_dict() returns the full nested dict."""
    cfg = Config()
    d = cfg.to_dict()
    assert d["optimization"]["charge"] == 0


# --- model_type resolution ---


def test_config_default_model_type():
    """Default model_type is 'fairchem'."""
    cfg = Config()
    assert cfg.model.model_type == "fairchem"


def test_resolve_model_type_aliases():
    """'uma' resolves to 'fairchem', 'orb-v3' resolves to 'orb'."""
    assert resolve_model_type(Config({"model": {"model_type": "fairchem"}})) == "fairchem"
    assert resolve_model_type(Config({"model": {"model_type": "uma"}})) == "fairchem"
    assert resolve_model_type(Config({"model": {"model_type": "orb"}})) == "orb"
    assert resolve_model_type(Config({"model": {"model_type": "orb-v3"}})) == "orb"


def test_resolve_model_type_case_insensitive():
    """Model type resolution is case-insensitive."""
    assert resolve_model_type(Config({"model": {"model_type": "FAIRCHEM"}})) == "fairchem"
    assert resolve_model_type(Config({"model": {"model_type": "ORB"}})) == "orb"


def test_resolve_model_type_from_dict():
    """resolve_model_type works with plain dicts as well as Config objects."""
    assert resolve_model_type({"model": {"model_type": "orb-v3"}}) == "orb"
    assert resolve_model_type({"model": {"model_type": "uma"}}) == "fairchem"
    assert resolve_model_type({"model": {}}) == "fairchem"


def test_resolve_model_type_invalid():
    """Unknown model_type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model_type"):
        resolve_model_type(Config({"model": {"model_type": "invalid"}}))


def test_validate_config_invalid_model_type():
    """Invalid model_type in Config constructor raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model_type"):
        Config({"model": {"model_type": "nonexistent"}})


# --- conformer_generation section ---


def test_conformer_generation_defaults():
    """Conformer generation has sensible defaults."""
    cfg = Config()
    assert cfg.conformer_generation.max_num_conformers == 20
    assert cfg.conformer_generation.conformer_seed == 42


def test_conformer_generation_override():
    """Conformer generation parameters can be overridden."""
    cfg = Config({"conformer_generation": {"max_num_conformers": 50, "conformer_seed": 123}})
    assert cfg.conformer_generation.max_num_conformers == 50
    assert cfg.conformer_generation.conformer_seed == 123


def test_conformer_generation_to_dict():
    """Conformer generation section serializes to dict."""
    cfg = Config()
    d = cfg.to_dict()
    assert d["conformer_generation"]["max_num_conformers"] == 20
    assert d["conformer_generation"]["conformer_seed"] == 42


# --- technical section ---


def test_technical_defaults():
    """Technical section has correct default values."""
    cfg = Config()
    assert cfg.technical.max_memory_padding == 0.95
    assert cfg.technical.memory_scaling_factor == 1.6
    assert cfg.technical.logging_level == "INFO"
    assert cfg.technical.device in ("cuda", "cpu")


def test_technical_override():
    """Technical parameters can be overridden."""
    cfg = Config({"technical": {"max_memory_padding": 0.5, "logging_level": "DEBUG"}})
    assert cfg.technical.max_memory_padding == 0.5
    assert cfg.technical.logging_level == "DEBUG"


def test_technical_to_dict():
    """Technical section serializes to dict."""
    cfg = Config()
    d = cfg.to_dict()
    assert d["technical"]["max_memory_padding"] == 0.95
    assert d["technical"]["memory_scaling_factor"] == 1.6
    assert d["technical"]["logging_level"] == "INFO"


# --- batch_optimizer validation ---


def test_valid_batch_optimizers_constant():
    """VALID_BATCH_OPTIMIZERS contains all 4 supported optimizers."""
    assert VALID_BATCH_OPTIMIZERS == {"fire", "gradient_descent", "lbfgs", "bfgs"}


@pytest.mark.parametrize("optimizer", ["fire", "gradient_descent", "lbfgs", "bfgs"])
def test_batch_optimizer_valid_values(optimizer):
    """Each valid optimizer name is accepted and preserved."""
    cfg = Config({"optimization": {"batch_optimizer": optimizer}})
    assert cfg.optimization.batch_optimizer == optimizer


def test_batch_optimizer_default_is_fire():
    """Default batch_optimizer is 'fire'."""
    cfg = Config()
    assert cfg.optimization.batch_optimizer == "fire"


def test_batch_optimizer_invalid_defaults_to_fire():
    """Invalid optimizer name silently defaults to 'fire'."""
    cfg = Config({"optimization": {"batch_optimizer": "invalid_optimizer"}})
    assert cfg.optimization.batch_optimizer == "fire"


def test_batch_optimizer_none_defaults_to_fire():
    """None optimizer defaults to 'fire'."""
    cfg = Config({"optimization": {"batch_optimizer": None}})
    assert cfg.optimization.batch_optimizer == "fire"


def test_batch_optimizer_case_normalized():
    """Optimizer names are normalized to lowercase."""
    cfg = Config({"optimization": {"batch_optimizer": "LBFGS"}})
    assert cfg.optimization.batch_optimizer == "lbfgs"

    cfg = Config({"optimization": {"batch_optimizer": "Fire"}})
    assert cfg.optimization.batch_optimizer == "fire"
