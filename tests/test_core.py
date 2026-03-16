"""Tests for ds2.core — matrix construction, eigenvalue extraction, IL."""

import numpy as np
import pytest

from ds2.core import (
    eigenvalue,
    from_insertion_loss,
    insertion_loss,
    is_doubly_stochastic,
    matrix,
    product,
)


class TestMatrix:
    def test_identity(self):
        P = matrix(1.0)
        np.testing.assert_allclose(P, np.eye(2))

    def test_uniform(self):
        P = matrix(0.0)
        np.testing.assert_allclose(P, 0.5 * np.ones((2, 2)))

    def test_negative_one(self):
        P = matrix(-1.0)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(P, expected)

    def test_ds_property(self, random_eigenvalues):
        for lam in random_eigenvalues:
            P = matrix(lam)
            assert is_doubly_stochastic(P)

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            matrix(1.5)
        with pytest.raises(ValueError):
            matrix(-1.1)


class TestEigenvalue:
    def test_roundtrip(self, random_eigenvalues):
        for lam in random_eigenvalues:
            P = matrix(lam)
            recovered = eigenvalue(P)
            np.testing.assert_allclose(recovered, lam, atol=1e-15)

    def test_not_2x2(self):
        with pytest.raises(ValueError):
            eigenvalue(np.eye(3))

    def test_not_ds(self):
        with pytest.raises(ValueError):
            eigenvalue(np.array([[0.5, 0.3], [0.5, 0.7]]))


class TestInsertionLoss:
    def test_full_transmission(self):
        # lambda = -1 → T = 1 → IL = 0 dB
        assert insertion_loss(-1.0) == pytest.approx(0.0)

    def test_3db_coupler(self):
        # lambda = 0 → T = 0.5 → IL = 3.01 dB
        assert insertion_loss(0.0) == pytest.approx(3.0103, rel=1e-3)

    def test_roundtrip(self, random_eigenvalues):
        for lam in random_eigenvalues:
            if abs(lam) < 0.999:
                il = insertion_loss(lam)
                recovered = from_insertion_loss(il)
                np.testing.assert_allclose(recovered, lam, atol=1e-10)


class TestProduct:
    def test_monoid_product(self, random_eigenvalues):
        """P(a) * P(b) = P(ab) — the fundamental theorem."""
        for i in range(0, len(random_eigenvalues) - 1, 2):
            a, b = random_eigenvalues[i], random_eigenvalues[i + 1]
            Pa = matrix(a)
            Pb = matrix(b)
            Pab = matrix(product(a, b))
            np.testing.assert_allclose(Pa @ Pb, Pab, atol=1e-15)

    def test_commutativity(self, random_eigenvalues):
        for i in range(0, len(random_eigenvalues) - 1, 2):
            a, b = random_eigenvalues[i], random_eigenvalues[i + 1]
            np.testing.assert_allclose(
                matrix(a) @ matrix(b), matrix(b) @ matrix(a), atol=1e-15
            )

    def test_associativity(self, random_eigenvalues):
        for i in range(0, len(random_eigenvalues) - 2, 3):
            a, b, c = random_eigenvalues[i], random_eigenvalues[i + 1], random_eigenvalues[i + 2]
            lhs = matrix(a) @ (matrix(b) @ matrix(c))
            rhs = (matrix(a) @ matrix(b)) @ matrix(c)
            np.testing.assert_allclose(lhs, rhs, atol=1e-14)

    def test_identity_element(self, random_eigenvalues):
        for lam in random_eigenvalues[:10]:
            P = matrix(lam)
            np.testing.assert_allclose(matrix(1.0) @ P, P, atol=1e-15)

    def test_absorbing_element(self, random_eigenvalues):
        J = matrix(0.0)
        for lam in random_eigenvalues[:10]:
            np.testing.assert_allclose(J @ matrix(lam), J, atol=1e-15)
