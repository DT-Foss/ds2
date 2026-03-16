"""Tests for ds2.phase — interferometric phase recovery."""

import numpy as np
import pytest

from ds2.phase import (
    generate_test_unitary,
    phase_measurements_needed,
    recover_phase_2port,
)


class TestPhaseRecovery:
    def test_recovery_random(self):
        """Test phase recovery on 100 random unitaries."""
        rng = np.random.default_rng(123)
        n_success = 0
        n_trials = 100

        for _ in range(n_trials):
            U = generate_test_unitary(rng)
            power = np.abs(U) ** 2
            interf = np.abs(U + 1.0) ** 2
            S_rec = recover_phase_2port(power, interf)

            # Check power matrix matches
            power_rec = np.abs(S_rec) ** 2
            if np.allclose(power_rec, power, atol=1e-6):
                n_success += 1

        assert n_success >= 90, f"Only {n_success}/100 recoveries succeeded"

    def test_identity_matrix(self):
        """S = I should recover to something with |S|^2 = I."""
        S = np.eye(2, dtype=complex)
        power = np.abs(S) ** 2
        interf = np.abs(S + 1.0) ** 2
        S_rec = recover_phase_2port(power, interf)
        np.testing.assert_allclose(np.abs(S_rec) ** 2, power, atol=1e-10)


class TestMeasurementCount:
    def test_2port(self):
        r = phase_measurements_needed(2)
        assert r["power"] == 4
        assert r["interferometric"] == 3
        assert r["total"] == 7

    def test_scaling(self):
        """Total measurements = n^2 + 2n - 1."""
        for n in range(2, 10):
            r = phase_measurements_needed(n)
            assert r["total"] == n * n + 2 * n - 1


class TestHaarRandom:
    def test_unitary(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            U = generate_test_unitary(rng)
            np.testing.assert_allclose(
                U @ U.conj().T, np.eye(2), atol=1e-12
            )
