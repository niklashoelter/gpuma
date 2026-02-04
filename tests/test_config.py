import json

import pytest
import yaml

from gpuma.config import Config, get_huggingface_token, load_config_from_file, save_config_to_file


def test_config_initialization():
    cfg = Config()
    # Check default values
    assert cfg.optimization.charge == 0
    assert cfg.optimization.model_name == "uma-s-1p1"

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
        Config({"optimization": {"device": "invalid_device"}})

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
        Config({"optimization": {"device": ""}})

    with pytest.raises(ValueError, match="Device string in config cannot be empty"):
        Config({"optimization": {"device": "   "}})

def test_huggingface_token(tmp_path, monkeypatch):
    # Case 1: Token in config
    cfg = Config({"optimization": {"huggingface_token": "token_in_config"}})
    assert cfg.optimization.get_huggingface_token() == "token_in_config"
    assert get_huggingface_token(cfg) == "token_in_config"

    # Case 2: Token in file
    token_file = tmp_path / "token.txt"
    token_file.write_text("token_in_file")
    cfg = Config({"optimization": {"huggingface_token_file": str(token_file)}})
    assert cfg.optimization.get_huggingface_token() == "token_in_file"

    # Case 3: Priority (config > file)
    cfg = Config({
        "optimization": {
            "huggingface_token": "priority_token",
            "huggingface_token_file": str(token_file)
        }
    })
    assert cfg.optimization.get_huggingface_token() == "priority_token"

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
