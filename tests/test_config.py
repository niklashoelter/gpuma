import json

import pytest
import yaml

from gpuma.config import (
    Config,
    load_config_from_file,
    resolve_model_type,
    save_config_to_file,
)


def test_config_initialization():
    cfg = Config()
    # Check default values
    assert cfg.optimization.charge == 0
    assert cfg.model.model_name == "uma-s-1p2"

def test_config_override():
    data = {"optimization": {"charge": 1, "new_key": "value"}}
    cfg = Config(data)
    assert cfg.optimization.charge == 1
    assert cfg.optimization.new_key == "value"
    # Defaults should persist for other keys
    assert cfg.optimization.multiplicity == 1

def test_section_access():
    cfg = Config()
    opt = cfg.optimization
    assert opt.charge == 0

    # Test setting attribute
    opt.charge = 2
    assert cfg.optimization.charge == 2

    # Test unknown attribute
    with pytest.raises(AttributeError):
        _ = opt.non_existent

def test_load_save_json(tmp_path):
    config_file = tmp_path / "config.json"
    cfg = Config({"optimization": {"charge": -1}})
    save_config_to_file(cfg, str(config_file))

    loaded_cfg = load_config_from_file(str(config_file))
    assert loaded_cfg.optimization.charge == -1

    # Verify file content
    with open(config_file) as f:
        data = json.load(f)
    assert data["optimization"]["charge"] == -1

def test_load_save_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    cfg = Config({"optimization": {"charge": -2}})
    save_config_to_file(cfg, str(config_file))

    loaded_cfg = load_config_from_file(str(config_file))
    assert loaded_cfg.optimization.charge == -2

    # Verify file content
    with open(config_file) as f:
        data = yaml.safe_load(f)
    assert data["optimization"]["charge"] == -2

def test_load_non_existent():
    # Should load defaults
    cfg = load_config_from_file("non_existent_file.json")
    assert cfg.optimization.charge == 0

def test_validate_config():
    # Invalid charge
    with pytest.raises(ValueError, match="Invalid charge"):
        Config({"optimization": {"charge": "invalid"}})

    # Invalid multiplicity
    with pytest.raises(ValueError, match="Multiplicity must be a positive integer"):
        Config({"optimization": {"multiplicity": 0}})

    # Invalid device
    with pytest.raises(ValueError, match="Device must be"):
        Config({"technical": {"device": "invalid_device"}})

def test_validate_config_convergence():
    # Test invalid convergence criteria
    with pytest.raises(ValueError, match="force_convergence_criterion must be a positive float"):
        Config({"optimization": {"force_convergence_criterion": -0.01}})

    with pytest.raises(ValueError, match="force_convergence_criterion must be a positive float"):
        Config({"optimization": {"force_convergence_criterion": "invalid"}})

    with pytest.raises(ValueError, match="energy_convergence_criterion must be a positive float"):
        Config({"optimization": {"energy_convergence_criterion": 0.0}})

def test_validate_config_device_empty():
    with pytest.raises(ValueError, match="Device string in config cannot be empty"):
        Config({"technical": {"device": ""}})

    with pytest.raises(ValueError, match="Device string in config cannot be empty"):
        Config({"technical": {"device": "   "}})

def test_huggingface_token(tmp_path, monkeypatch):
    # Case 1: Token in config
    cfg = Config({"model": {"huggingface_token": "token_in_config"}})
    assert cfg.model.get_huggingface_token() == "token_in_config"

    # Case 2: Token in file
    token_file = tmp_path / "token.txt"
    token_file.write_text("token_in_file")
    cfg = Config({"model": {"huggingface_token_file": str(token_file)}})
    assert cfg.model.get_huggingface_token() == "token_in_file"

    # Case 3: Priority (config > file)
    cfg = Config({
        "model": {
            "huggingface_token": "priority_token",
            "huggingface_token_file": str(token_file)
        }
    })
    assert cfg.model.get_huggingface_token() == "priority_token"

def test_section_to_dict():
    cfg = Config()
    d = cfg.optimization.to_dict()
    assert isinstance(d, dict)
    assert d["charge"] == 0

    # Modifying dict shouldn't affect config
    d["charge"] = 100
    assert cfg.optimization.charge == 0

def test_config_to_dict():
    cfg = Config()
    d = cfg.to_dict()
    assert d["optimization"]["charge"] == 0


def test_config_default_model_type():
    cfg = Config()
    assert cfg.model.model_type == "fairchem"


def test_resolve_model_type_aliases():
    assert resolve_model_type(Config({"model": {"model_type": "fairchem"}})) == "fairchem"
    assert resolve_model_type(Config({"model": {"model_type": "uma"}})) == "fairchem"
    assert resolve_model_type(Config({"model": {"model_type": "orb"}})) == "orb"
    assert resolve_model_type(Config({"model": {"model_type": "orb-v3"}})) == "orb"


def test_resolve_model_type_case_insensitive():
    assert resolve_model_type(Config({"model": {"model_type": "FAIRCHEM"}})) == "fairchem"
    assert resolve_model_type(Config({"model": {"model_type": "ORB"}})) == "orb"


def test_resolve_model_type_from_dict():
    assert resolve_model_type({"model": {"model_type": "orb-v3"}}) == "orb"
    assert resolve_model_type({"model": {"model_type": "uma"}}) == "fairchem"
    # Missing model_type defaults to fairchem
    assert resolve_model_type({"model": {}}) == "fairchem"


def test_resolve_model_type_invalid():
    with pytest.raises(ValueError, match="Unknown model_type"):
        resolve_model_type(Config({"model": {"model_type": "invalid"}}))


def test_validate_config_invalid_model_type():
    with pytest.raises(ValueError, match="Unknown model_type"):
        Config({"model": {"model_type": "nonexistent"}})


# --- conformer_generation section ---


def test_conformer_generation_defaults():
    cfg = Config()
    assert cfg.conformer_generation.max_num_conformers == 20
    assert cfg.conformer_generation.conformer_seed == 42


def test_conformer_generation_override():
    cfg = Config({"conformer_generation": {"max_num_conformers": 50, "conformer_seed": 123}})
    assert cfg.conformer_generation.max_num_conformers == 50
    assert cfg.conformer_generation.conformer_seed == 123


def test_conformer_generation_to_dict():
    cfg = Config()
    d = cfg.to_dict()
    assert d["conformer_generation"]["max_num_conformers"] == 20
    assert d["conformer_generation"]["conformer_seed"] == 42


# --- technical section ---


def test_technical_defaults():
    cfg = Config()
    assert cfg.technical.max_memory_padding == 0.95
    assert cfg.technical.memory_scaling_factor == 1.6
    assert cfg.technical.logging_level == "INFO"
    # device is dynamic (cuda or cpu), just check it's set
    assert cfg.technical.device in ("cuda", "cpu")


def test_technical_override():
    cfg = Config({"technical": {"max_memory_padding": 0.5, "logging_level": "DEBUG"}})
    assert cfg.technical.max_memory_padding == 0.5
    assert cfg.technical.logging_level == "DEBUG"


def test_technical_to_dict():
    cfg = Config()
    d = cfg.to_dict()
    assert d["technical"]["max_memory_padding"] == 0.95
    assert d["technical"]["memory_scaling_factor"] == 1.6
    assert d["technical"]["logging_level"] == "INFO"
