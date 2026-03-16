"""Shared test fixtures for DS2."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def random_eigenvalues(rng):
    """100 random eigenvalues in [-1, 1]."""
    return rng.uniform(-1.0, 1.0, size=100)
