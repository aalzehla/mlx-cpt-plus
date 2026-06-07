"""Test configuration."""

import pytest


@pytest.fixture
def sample_sequences():
    """Sample sequences for testing."""
    return [
        [1, 2, 3, 4, 5],
        [1, 2, 4, 5, 6],
        [2, 3, 4, 7, 8],
        [1, 3, 5, 7, 9],
    ]


@pytest.fixture
def config():
    """Default test configuration."""
    from mlx_cpt_plus.config import Config
    return Config()