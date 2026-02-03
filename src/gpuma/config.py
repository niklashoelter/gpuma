"""Dict-based configuration for GPUMA.

This module provides a simple JSON/YAML-backed configuration represented as a
nested Python dict while still exposing a Config class API. Unknown keys are
preserved, and only a small set of known keys have defaults derived from
``examples/config.json`` so users can add new keys without changing the code.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

default_device = "cuda" if torch.cuda.is_available() else "cpu"

# Default configuration aligned with examples/config.json
# This is the single source of truth for required/known configuration keys.
DEFAULT_CONFIG: dict[str, Any] = {
    "optimization": {
        "batch_optimization_mode": "batch",
        "batch_optimizer": "fire",
        "max_num_conformers": 20,
        "conformer_seed": 42,
        # Default electronic structure settings
        # Charge and multiplicity can be overridden via CLI for XYZ inputs
        # and via config or CLI for SMILES multiplicities.
        "charge": 0,
        "multiplicity": 1,
        # Convergence criteria
        "force_convergence_criterion": 5e-2,
        "energy_convergence_criterion": None,
        "model_name": "uma-s-1p1",
        "model_path": None,
        # Optional local model cache directory; can be overridden by user config
        "model_cache_dir": None,
        "device": default_device,
        "huggingface_token": None,
        "huggingface_token_file": None,
        # Logging level control: one of "ERROR", "WARNING", "INFO", "DEBUG"
        "logging_level": "INFO",
    }
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dicts, with values from override taking precedence.

    Leaves inputs unmodified and returns a new merged dict.
    """
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class _Section:
    """Lightweight wrapper to provide attribute access to a nested dict section."""

    def __init__(self, root: dict[str, Any], path: list[str]):
        """Initialize the section wrapper.

        Parameters
        ----------
        root:
            The root dictionary of the configuration.
        path:
            List of keys to traverse to reach this section.

        """
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_path", path)

    def _node(self) -> dict[str, Any]:
        """Resolve the path to the current node in the dictionary."""
        node = self._root
        for key in self._path:
            node = node.setdefault(key, {})
        return node

    def __getattr__(self, name: str):
        """Get a value or a subsection by attribute name."""
        node = self._node()
        if name in node:
            val = node[name]
            if isinstance(val, dict):
                return _Section(self._root, self._path + [name])
            return val
        raise AttributeError(
            f"{name} not found in section {'.'.join(self._path) if self._path else 'root'}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a value in the configuration by attribute name."""
        node = self._node()
        node[name] = value

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the section as a dictionary."""
        return copy.deepcopy(self._node())

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key, similar to dict.get."""
        return self._node().get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set a default value for a key, similar to dict.setdefault."""
        return self._node().setdefault(key, default)

    def get_huggingface_token(self) -> str | None:
        """Return the HuggingFace token from this section if available.

        The token is retrieved either directly from the configuration or from a
        file path specified in ``huggingface_token_file``.
        """
        opt = self._node()
        token = opt.get("huggingface_token")
        if token:
            return str(token)
        token_file = opt.get("huggingface_token_file")
        if not token_file:
            return None
        try:
            with open(str(token_file), encoding="utf-8") as fh:
                content = fh.read().strip()
            return content or None
        except OSError as e:
            logger.warning("Could not read HuggingFace token from %s: %s", token_file, e)
            return None


class Config:
    """Dict-backed configuration with attribute access for sections.

    Example:
    -------
    >>> cfg = load_config_from_file()
    >>> print(cfg.optimization.logging_level)
    >>> cfg.optimization.device = "cuda"
    >>> save_config_to_file(cfg, "config.json")

    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize configuration with optional overrides.

        Parameters
        ----------
        data:
            Optional dictionary of configuration overrides.

        """
        merged = _deep_merge(DEFAULT_CONFIG, data or {})
        self._data: dict[str, Any] = merged
        # basic validation to catch obvious mistakes early
        validate_config(self)

    @property
    def optimization(self) -> _Section:
        """Return the optimization section of the configuration."""
        return _Section(self._data, ["optimization"])

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the underlying configuration dictionary."""
        return copy.deepcopy(self._data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create a :class:`Config` instance from a plain dictionary."""
        return cls(data)


@functools.lru_cache(maxsize=1)
def _read_config_file(filepath: str) -> dict[str, Any]:
    """Read and parse configuration file with caching.

    Returns an empty dict if the file does not exist.
    """
    if not os.path.exists(filepath):
        return {}

    with open(filepath, encoding="utf-8") as f:
        if filepath.endswith(".json"):
            return json.load(f)
        elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
            try:
                from yaml import safe_load as _yaml_safe_load  # type: ignore

                return _yaml_safe_load(f)
            except ImportError as e:
                raise ImportError("PyYAML is required to load YAML config files") from e
        else:
            raise ValueError("Config file must be JSON or YAML format")


def load_config_from_file(filepath: str = "config.json") -> Config:
    """Load configuration from a JSON/YAML file and deep-merge with defaults.

    This function caches the raw dictionary loaded from the file to avoid
    repeated I/O and parsing. The returned Config object is always a new
    instance, safe to modify.

    Args:
        filepath: Path to the config file. If it doesn't exist, defaults are used.

    Returns:
        A :class:`Config` instance with data merged with :data:`DEFAULT_CONFIG`.
        Unknown keys are preserved.

    """
    user_cfg = _read_config_file(filepath)

    if user_cfg is not None and not isinstance(user_cfg, dict):
        raise ValueError("Configuration file must contain a JSON/YAML object at the root")

    cfg = Config.from_dict(copy.deepcopy(user_cfg))
    validate_config(cfg)
    return cfg


def save_config_to_file(config: Any, filepath: str) -> None:
    """Save configuration to JSON/YAML file.

    Accepts either a :class:`Config` instance or a plain dictionary.
    """
    cfg_dict = config.to_dict() if isinstance(config, Config) else dict(config)

    with open(filepath, "w", encoding="utf-8") as f:
        if filepath.endswith(".json"):
            json.dump(cfg_dict, f, indent=2)
        elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
            try:
                from yaml import safe_dump as _yaml_safe_dump  # type: ignore

                _yaml_safe_dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
            except ImportError as e:
                raise ImportError("PyYAML is required to save YAML config files") from e
        else:
            raise ValueError("Config file must be JSON or YAML format")

    # Invalidate cache since the file on disk has changed
    _read_config_file.cache_clear()


def get_huggingface_token(config: Config | dict[str, Any]) -> str | None:
    """Return the HuggingFace token from config or a file, if available.

    Works with either :class:`Config` or a plain dict. Checks
    ``optimization.huggingface_token`` first, then
    ``optimization.huggingface_token_file``.
    """
    if isinstance(config, Config):
        return config.optimization.get_huggingface_token()

    opt = (config or {}).get("optimization", {})
    token = opt.get("huggingface_token")
    if token:
        return str(token)

    token_file = opt.get("huggingface_token_file")
    if not token_file:
        return None

    try:
        with open(str(token_file), encoding="utf-8") as fh:
            content = fh.read().strip()
        return content or None
    except OSError as e:
        logger.warning("Could not read HuggingFace token from %s: %s", token_file, e)
        return None


def validate_config(config: Config) -> None:
    """Validate core optimization settings in a Config instance.

    Checks basic types and value ranges for commonly used options.
    Raises ValueError if an invalid value is found.
    """
    opt = config.optimization

    # Charge and multiplicity must be integers; multiplicity > 0
    charge = getattr(opt, "charge", 0)
    multiplicity = getattr(opt, "multiplicity", 1)
    try:
        int(charge)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid charge value in config: {charge!r}") from exc
    try:
        mult_int = int(multiplicity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid multiplicity value in config: {multiplicity!r}") from exc
    if mult_int <= 0:
        raise ValueError("Multiplicity must be a positive integer")

    # Convergence criteria
    force_crit = getattr(opt, "force_convergence_criterion", None)
    energy_crit = getattr(opt, "energy_convergence_criterion", None)

    if force_crit is not None:
        try:
            val = float(force_crit)
            if val <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"force_convergence_criterion must be a positive float, got {force_crit!r}"
            ) from exc

    if energy_crit is not None:
        try:
            val = float(energy_crit)
            if val <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"energy_convergence_criterion must be a positive float, got {energy_crit!r}"
            ) from exc

    # Device must be cpu, cuda or cuda:N
    dev = str(getattr(opt, "device", default_device) or "").strip().lower()
    if not dev:
        raise ValueError("Device string in config cannot be empty")
    if dev != "cpu" and not dev.startswith("cuda"):
        raise ValueError(f"Device must be 'cpu', 'cuda' or 'cuda:N' (e.g. 'cuda:0'), got {dev!r}")

    # If CUDA is requested but not available, we don't fail here; the
    # runtime will transparently fall back to CPU via model helpers.


# Default configuration instance for convenience
default_config = Config()
