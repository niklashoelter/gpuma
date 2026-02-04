import logging
from unittest.mock import patch

from gpuma.logging_utils import configure_logging


def test_configure_logging_root():
    # Reset root logger handlers
    logging.getLogger().handlers = []

    configure_logging(level=logging.DEBUG)

    logger = logging.getLogger()
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

def test_configure_logging_named():
    logger_name = "test_logger"
    # Ensure fresh start
    logging.getLogger(logger_name).handlers = []

    configure_logging(level=logging.WARNING, logger_name=logger_name)

    logger = logging.getLogger(logger_name)
    assert logger.level == logging.WARNING
    assert len(logger.handlers) >= 1

def test_configure_logging_idempotent():
    # Test that calling it twice doesn't add multiple handlers if not needed
    # (The implementation checks `if not logger.handlers`)
    logging.getLogger().handlers = []

    configure_logging(level=logging.INFO)
    handlers_count = len(logging.getLogger().handlers)

    configure_logging(level=logging.INFO)
    assert len(logging.getLogger().handlers) == handlers_count
