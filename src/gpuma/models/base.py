"""Shared constants and helpers for the GPUMA model backends.

This module holds the available-model-name registries and the small
config-validation helpers reused by every backend loader.  It sits at the
bottom of the ``gpuma.models`` import graph: the per-backend modules import
from here, never the other way around.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Available model names
# ---------------------------------------------------------------------------

#: Fairchem UMA model names accepted by ``fairchem.core.pretrained_mlip``.
AVAILABLE_FAIRCHEM_MODELS: tuple[str, ...] = (
    "uma-s-1p2",
    "uma-s-1p2p1",
    "uma-s-1p2p1-ft-d4rev",
    "uma-s-1p1",
    "uma-m-1p1",
)

#: ORB-v3 model names accepted by ``orb_models.forcefield.pretrained``.
#: Use the **underscored** form (e.g. ``orb_v3_direct_omol``) as
#: ``model_name`` in the configuration.
AVAILABLE_ORB_MODELS: tuple[str, ...] = (
    # ORB-v3 — omol
    "orb_v3_conservative_omol",
    "orb_v3_direct_omol",
    # ORB-v3 — omat
    "orb_v3_conservative_20_omat",
    "orb_v3_conservative_inf_omat",
    "orb_v3_direct_20_omat",
    "orb_v3_direct_inf_omat",
    # ORB-v3 — mpa
    "orb_v3_conservative_20_mpa",
    "orb_v3_conservative_inf_mpa",
    "orb_v3_direct_20_mpa",
    "orb_v3_direct_inf_mpa",
)

#: SevenNet pretrained model names accepted by ``sevenn`` (``sevenn.util``
#: ``pretrained_name_to_path``).  Multi-modal checkpoints (``7net-mf-ompa``,
#: ``7net-omni``) additionally require ``config.model.model_modal`` to pick a
#: fidelity.  A local checkpoint may instead be supplied via
#: ``config.model.model_path``.
AVAILABLE_SEVENNET_MODELS: tuple[str, ...] = (
    "7net-0",
    "7net-0_22may2024",
    "7net-l3i5",
    "7net-mf-0",
    "7net-mf-ompa",
    "7net-omat",
    "7net-omni",
    "7net-omni-i8",
    "7net-omni-i12",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_hf_token_to_env(config: Config) -> None:
    """Set the ``HF_TOKEN`` environment variable from config if available."""
    hf_token = config.model.get_huggingface_token()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token


def _verify_model_name_and_cache_dir(config: Config) -> tuple[str, Path | None]:
    """Return ``(model_name, cache_dir)`` after validating the config values.

    Raises :class:`ValueError` if ``model_name`` is empty or missing.
    """
    opt = config.model
    model_name = opt.model_name
    if not model_name:
        raise ValueError("Model name must be specified in the configuration")
    model_cache_dir = Path(opt.model_cache_dir) if opt.model_cache_dir else None
    if model_cache_dir and not model_cache_dir.exists():
        try:
            os.makedirs(model_cache_dir, exist_ok=True)
        except OSError as e:
            logger.warning("Could not create model cache directory at %s: %s", model_cache_dir, e)
            model_cache_dir = None
    if model_cache_dir is not None and not model_cache_dir.exists():
        model_cache_dir = None
    return model_name, model_cache_dir


def _verify_model_path(config: Config) -> Path | None:
    """Return the model checkpoint path if it exists, else ``None``."""
    opt = config.model
    if opt.model_path:
        p = Path(opt.model_path)
        return p if p.exists() else None
    return None
