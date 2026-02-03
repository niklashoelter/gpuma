"""Logging utilities for GPUMA."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO, logger_name: str | None = None) -> None:
    """Configure the root logger or a named logger for GPUMA.

    Parameters
    ----------
    level:
        Logging level from :mod:`logging` (defaults to ``logging.INFO``).
    logger_name:
        Optional name of the logger to configure. If omitted, the root logger
        is configured.

    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
