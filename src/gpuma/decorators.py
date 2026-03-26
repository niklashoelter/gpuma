"""Decorators and timing utilities used across the GPUMA package."""

import logging
from functools import wraps
from time import perf_counter

logger = logging.getLogger(__name__)


def time_it(func):
    """Measure the execution time of a function and log the result.

    Parameters
    ----------
    func:
        Callable to be wrapped.

    Returns
    -------
    callable
        Wrapped function that logs its runtime at :mod:`logging.INFO` level.

    """

    @wraps(func)
    def wrap(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start_time
        logger.info("Function: %r took: %.2f sec", func.__name__, elapsed)
        return result

    return wrap


class timed_block:
    """Context manager that measures and logs a named code block.

    The elapsed time (in seconds) is available via the :attr:`elapsed`
    attribute after the block exits.

    Example
    -------
    >>> with timed_block("model loading") as tb:
    ...     model = load_model()
    >>> print(tb.elapsed)
    """

    def __init__(self, name: str, *, level: int = logging.INFO):
        self.name = name
        self.elapsed: float = 0.0
        self._level = level

    def __enter__(self):
        """Start the timer and return ``self`` for attribute access."""
        self._start = perf_counter()
        return self

    def __exit__(self, *exc_info):
        """Stop the timer, store elapsed time, and log the result."""
        self.elapsed = perf_counter() - self._start
        logger.log(self._level, "%s took %.2f sec", self.name, self.elapsed)
