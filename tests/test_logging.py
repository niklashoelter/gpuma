"""Tests for logging utilities."""

import logging

from gpuma.config import Config
from gpuma.logging_utils import configure_logging, log_optimization_summary
from gpuma.structure import Structure


def test_configure_logging_root():
    """Root logger is configured with the specified level and a StreamHandler."""
    logging.getLogger().handlers = []
    configure_logging(level=logging.DEBUG)

    logger = logging.getLogger()
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_configure_logging_named():
    """Named loggers can be configured independently."""
    logger_name = "test_logger"
    logging.getLogger(logger_name).handlers = []

    configure_logging(level=logging.WARNING, logger_name=logger_name)

    logger = logging.getLogger(logger_name)
    assert logger.level == logging.WARNING
    assert len(logger.handlers) >= 1


def test_configure_logging_idempotent():
    """Calling configure_logging twice does not duplicate handlers."""
    logging.getLogger().handlers = []
    configure_logging(level=logging.INFO)
    handlers_count = len(logging.getLogger().handlers)
    configure_logging(level=logging.INFO)
    assert len(logging.getLogger().handlers) == handlers_count


def test_log_optimization_summary(caplog):
    """Summary includes model, optimizer, structure counts, and energy stats."""
    config = Config({
        "optimization": {"batch_optimizer": "lbfgs"},
        "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
    })
    inputs = [
        Structure(symbols=["C"], coordinates=[(0, 0, 0)], charge=0, multiplicity=1),
    ]
    results = [
        Structure(
            symbols=["C"],
            coordinates=[(0, 0, 0)],
            charge=0,
            multiplicity=1,
            energy=-10.5,
        ),
    ]

    with caplog.at_level(logging.INFO, logger="gpuma.logging_utils"):
        log_optimization_summary(inputs, results, total_time=1.5, mode="batch", config=config)

    assert "GPUMA Optimization Summary" in caplog.text
    assert "orb / orb_v3_direct_omol" in caplog.text
    assert "Optimizer:           lbfgs" in caplog.text
    assert "Structures input:    1" in caplog.text
    assert "Energy min:" in caplog.text


def test_log_optimization_summary_no_results(caplog):
    """Summary handles zero results without errors."""
    config = Config({
        "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
    })

    with caplog.at_level(logging.INFO, logger="gpuma.logging_utils"):
        log_optimization_summary([], [], total_time=0.1, mode="sequential", config=config)

    assert "Structures input:    0" in caplog.text
    assert "Structures output:   0" in caplog.text
