"""Decorators used across the GPUMA package."""

import logging
from functools import wraps
from time import time

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
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        logger.info("Function: %r took: %.2f sec to complete", func.__name__, end_time - start_time)
        return result

    return wrap
