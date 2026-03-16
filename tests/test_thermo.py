"""Tests for ds2.thermo — entropy, Landauer cost, efficiency."""

import numpy as np
import pytest

from ds2_toolkit.thermo import entropy, entropy_change, landauer_cost, landauer_efficiency


class TestEntropy:
    def test_uniform(self):
        assert entropy(0.5) == pytest.approx(1.0)

    def test_deterministic(self):
        assert entropy(0.0) == 0.0
        assert entropy(1.0) == 0.0

    def test_symmetric(self):
        assert entropy(0.3) == pytest.approx(entropy(0.7))


class TestEntropyChange:
    def test_positive(self):
        """DS2 Second Law: entropy never decreases."""
        for lam in [0.1, 0.5, 0.9, -0.3, -0.7]:
            for delta in [0.1, 0.5, 0.9]:
                dh = entropy_change(lam, delta)
                assert dh >= -1e-15, f"Entropy decreased for lam={lam}, delta={delta}"

    def test_identity_no_change(self):
        """lambda = 1 or -1 preserves distribution."""
        assert entropy_change(1.0, 0.5) == pytest.approx(0.0, abs=1e-15)
        assert entropy_change(-1.0, 0.5) == pytest.approx(0.0, abs=1e-15)


class TestLandauerCost:
    def test_one_bit(self):
        """1 bit at 300K costs k_B * T * ln(2)."""
        cost = landauer_cost(1.0, temperature=300.0)
        assert cost == pytest.approx(2.8534e-21, rel=0.01)

    def test_zero_bits(self):
        assert landauer_cost(0.0) == 0.0


class TestLandauerEfficiency:
    def test_increases_with_k(self):
        """More steps → higher efficiency (quasistatic limit)."""
        etas = []
        for k in [1, 10, 100, 1000]:
            result = landauer_efficiency(0.9, 0.1, k)
            etas.append(result["eta"])
        for i in range(len(etas) - 1):
            assert etas[i] < etas[i + 1]

    def test_approaches_one(self):
        result = landauer_efficiency(0.9, 0.1, 1000)
        assert result["eta"] > 0.99

    def test_specific_values(self):
        """Match paper values."""
        r1 = landauer_efficiency(0.9, 0.1, 1)
        assert r1["eta"] == pytest.approx(0.416, abs=0.02)
        r100 = landauer_efficiency(0.9, 0.1, 100)
        assert r100["eta"] == pytest.approx(0.982, abs=0.01)

    def test_invalid(self):
        with pytest.raises(ValueError):
            landauer_efficiency(0.5, 0.9, 10)  # delta_f > delta_0
