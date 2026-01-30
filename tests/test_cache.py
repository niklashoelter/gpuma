from unittest.mock import MagicMock, patch

from gpuma.config import Config
from gpuma.optimizer import _get_cached_calculator


def test_bounded_cache_behavior():
    """Verify that the cache is bounded and evicts old entries."""
    # Access the cached function implementation
    from gpuma.optimizer import _load_calculator_impl

    # Clear cache to start fresh
    _load_calculator_impl.cache_clear()

    # Mock load_model_fairchem
    with patch("gpuma.optimizer.load_model_fairchem") as mock_load:
        mock_load.return_value = MagicMock()

        # 1. Load first config
        config1 = Config({"optimization": {"model_name": "model_1"}})
        _get_cached_calculator(config1)

        info = _load_calculator_impl.cache_info()
        assert info.currsize == 1
        assert info.misses == 1
        assert info.hits == 0

        # 2. Load same config again -> should hit cache
        _get_cached_calculator(config1)
        info = _load_calculator_impl.cache_info()
        assert info.currsize == 1
        assert info.misses == 1
        assert info.hits == 1

        # 3. Load different config -> should evict old one (maxsize=1) and miss
        config2 = Config({"optimization": {"model_name": "model_2"}})
        _get_cached_calculator(config2)
        info = _load_calculator_impl.cache_info()
        assert info.currsize == 1  # Still 1 because maxsize is 1
        assert info.misses == 2
        assert info.hits == 1

        # 4. Load first config again -> should miss (was evicted)
        _get_cached_calculator(config1)
        info = _load_calculator_impl.cache_info()
        assert info.currsize == 1
        assert info.misses == 3
        assert info.hits == 1


if __name__ == "__main__":
    test_bounded_cache_behavior()
