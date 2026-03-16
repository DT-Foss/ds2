"""Tests for ds2.correction — cascade-correction duality."""

import numpy as np
import pytest

from ds2_toolkit.correction import (
    capacity,
    cascade_eigenvalue,
    cascade_with_correction,
    correct,
    correct_f3,
    correction_polynomial,
    duality_fixed_points,
)


class TestCorrection:
    def test_f3_closed_form(self):
        """f_3(lambda) = (3*lambda - lambda^3) / 2."""
        for lam in np.linspace(-1, 1, 100):
            expected = (3 * lam - lam**3) / 2
            assert correct(lam, k=3) == pytest.approx(expected, abs=1e-14)

    def test_f3_matches_closed(self):
        for lam in np.linspace(-1, 1, 50):
            assert correct(lam, k=3) == pytest.approx(correct_f3(lam), abs=1e-15)

    def test_fixed_points(self):
        """Fixed points at {-1, 0, +1}."""
        for fp in duality_fixed_points():
            assert correct(fp, k=3) == pytest.approx(fp, abs=1e-15)
            assert correct(fp, k=5) == pytest.approx(fp, abs=1e-15)

    def test_repels_from_zero(self):
        """Correction pushes eigenvalues AWAY from 0."""
        for lam in [0.1, 0.3, 0.5, 0.7]:
            corrected = correct(lam, k=3)
            assert abs(corrected) > abs(lam)

    def test_even_k_rejected(self):
        with pytest.raises(ValueError):
            correct(0.5, k=4)

    def test_k_less_than_3(self):
        with pytest.raises(ValueError):
            correct(0.5, k=1)


class TestCascade:
    def test_cascade_contracts(self):
        """Cascade pushes eigenvalues TOWARD 0."""
        for lam in [0.3, 0.5, 0.7, 0.9]:
            assert abs(cascade_eigenvalue(lam, 3)) < abs(lam)

    def test_cascade_fixed_points(self):
        for fp in [0.0, 1.0, -1.0]:
            assert cascade_eigenvalue(fp, 5) == pytest.approx(fp, abs=1e-15)


class TestDuality:
    def test_shared_fixed_points(self):
        fps = duality_fixed_points()
        assert fps == [-1.0, 0.0, 1.0]

    def test_cascade_correction_improvement(self):
        """Correction every 3 stages maintains eigenvalue above pure cascade."""
        pure = cascade_eigenvalue(0.9, 20)
        with_corr = cascade_with_correction(0.9, 20, correction_interval=3, correction_k=3)
        assert abs(with_corr[-1]) > abs(pure)

    def test_paper_values(self):
        """26.5x capacity improvement from paper."""
        pure_lam = 0.9**20  # 0.1216
        trace = cascade_with_correction(0.9, 20, correction_interval=3, correction_k=3)
        corr_lam = trace[-1]
        c_pure = capacity(pure_lam)
        c_corr = capacity(corr_lam)
        improvement = c_corr / c_pure
        assert improvement > 20.0  # At least 20x (paper says 26.5x)


class TestCapacity:
    def test_perfect_channel(self):
        assert capacity(1.0) == pytest.approx(1.0)
        assert capacity(-1.0) == pytest.approx(1.0)

    def test_useless_channel(self):
        assert capacity(0.0) == pytest.approx(0.0)

    def test_bsc_half(self):
        # lambda = 0.5 → eps = 0.25 → C = 1 - H(0.25) ≈ 0.189
        assert capacity(0.5) == pytest.approx(0.1887, abs=0.01)
