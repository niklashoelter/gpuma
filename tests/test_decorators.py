import logging
from gpuma.decorators import time_it

def test_time_it(caplog):
    @time_it
    def sample_func(x):
        return x + 1

    with caplog.at_level(logging.INFO):
        res = sample_func(1)
        assert res == 2

    assert "Function: 'sample_func' took:" in caplog.text

def test_time_it_wraps():
    @time_it
    def sample_func(x):
        """Docstring."""
        return x

    assert sample_func.__name__ == "sample_func"
    assert sample_func.__doc__ == "Docstring."
