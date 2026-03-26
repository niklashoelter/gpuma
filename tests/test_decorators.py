"""Tests for timing decorators."""

import logging
import time

from gpuma.decorators import time_it, timed_block


def test_time_it(caplog):
    """@time_it logs the function name and execution time."""
    @time_it
    def dummy():
        time.sleep(0.01)
        return 42

    with caplog.at_level(logging.INFO, logger="gpuma.decorators"):
        result = dummy()

    assert result == 42
    assert "dummy" in caplog.text
    assert "took" in caplog.text


def test_time_it_wraps():
    """@time_it preserves the wrapped function's name and docstring."""
    @time_it
    def my_function():
        """My docstring."""
        pass

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring."


def test_timed_block_logs_and_stores_elapsed(caplog):
    """timed_block logs the block name and stores elapsed time."""
    with caplog.at_level(logging.INFO, logger="gpuma.decorators"):
        with timed_block("test operation") as tb:
            time.sleep(0.01)

    assert tb.elapsed >= 0.01
    assert "test operation" in caplog.text


def test_timed_block_custom_level(caplog):
    """timed_block respects a custom logging level."""
    with caplog.at_level(logging.DEBUG, logger="gpuma.decorators"):
        with timed_block("debug op", level=logging.DEBUG) as tb:
            pass

    assert tb.elapsed >= 0
    assert "debug op" in caplog.text
