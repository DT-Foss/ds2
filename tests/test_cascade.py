"""Tests for ds2.cascade — cascade-by-multiplication."""

import numpy as np
import pytest

from ds2.cascade import cascade, link_budget


class TestCascade:
    def test_single_device(self):
        r = cascade([0.7])
        assert r.eigenvalue == pytest.approx(0.7)
        assert len(r.individual) == 1

    def test_two_devices(self):
        r = cascade([0.9, 0.8])
        assert r.eigenvalue == pytest.approx(0.72)

    def test_many_devices(self):
        evs = [0.9] * 20
        r = cascade(evs)
        assert r.eigenvalue == pytest.approx(0.9**20, rel=1e-10)

    def test_sign_flip(self):
        r = cascade([0.5, -0.5])
        assert r.eigenvalue == pytest.approx(-0.25)

    def test_cumulative(self):
        r = cascade([0.9, 0.8, 0.7])
        assert r.cumulative[0] == pytest.approx(0.9)
        assert r.cumulative[1] == pytest.approx(0.72)
        assert r.cumulative[2] == pytest.approx(0.504)

    def test_empty(self):
        with pytest.raises(ValueError):
            cascade([])

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            cascade([0.5, 1.5])

    def test_il_additive(self):
        """IL in dB should be additive under cascade (positive eigenvalues)."""
        from ds2.core import insertion_loss
        evs = [0.9, 0.8, 0.7]
        r = cascade(evs)
        # log|lambda| is additive: log|prod| = sum(log|lambda_i|)
        log_sum = sum(np.log(abs(lam)) for lam in evs)
        log_prod = np.log(abs(r.eigenvalue))
        assert log_prod == pytest.approx(log_sum, rel=1e-10)


class TestLinkBudget:
    def test_basic(self):
        devices = [("coupler", -0.002), ("mismatch", -0.589)]
        budget = link_budget(devices)
        assert len(budget) == 2
        assert budget[0]["device"] == "coupler"
        assert budget[1]["cumulative_eigenvalue"] == pytest.approx(-0.002 * -0.589)
