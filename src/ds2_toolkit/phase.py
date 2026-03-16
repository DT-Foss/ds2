"""Minimum-measurement phase recovery for 2-port S-matrices.

The power matrix |S|^2 discards 2n-1 phase parameters. For n=2,
three phases (theta, phi, psi) are lost. Interferometric measurements
of |S_ij + 1|^2 recover cos(arg(S_ij)), and sign disambiguation
gives complete reconstruction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def recover_phase_2port(
    power_matrix: NDArray[np.float64],
    interferometric: NDArray[np.float64],
) -> NDArray[np.complexfloating]:
    """Recover a 2x2 unitary S-matrix from power and interferometric measurements.

    Parameters
    ----------
    power_matrix : ndarray, shape (2, 2)
        |S_ij|^2 values.
    interferometric : ndarray, shape (2, 2)
        |S_ij + 1|^2 values for each entry.

    Returns
    -------
    S : ndarray, shape (2, 2), complex
        Reconstructed unitary S-matrix (up to global phase).

    Notes
    -----
    Uses |S_ij + 1|^2 = 1 + |S_ij|^2 + 2*|S_ij|*cos(arg(S_ij))
    to extract cos(arg(S_ij)).

    The reconstruction is unique up to global phase.
    """
    P = np.asarray(power_matrix, dtype=np.float64)
    I = np.asarray(interferometric, dtype=np.float64)

    if P.shape != (2, 2) or I.shape != (2, 2):
        raise ValueError("Both matrices must be 2x2")

    magnitudes = np.sqrt(np.clip(P, 0.0, None))
    phases = np.zeros((2, 2), dtype=np.float64)

    for i in range(2):
        for j in range(2):
            mag = magnitudes[i, j]
            if mag > 1e-12:
                cos_arg = (I[i, j] - 1.0 - P[i, j]) / (2.0 * mag)
                cos_arg = np.clip(cos_arg, -1.0, 1.0)
                phases[i, j] = np.arccos(cos_arg)

    # Construct S with phases
    S = magnitudes * np.exp(1j * phases)

    # Fix global phase: set S[0,0] phase to recovered value
    # Check unitarity and adjust signs if needed
    S = _disambiguate_signs(S, P, I)

    return S


def _disambiguate_signs(
    S: NDArray[np.complexfloating],
    P: NDArray[np.float64],
    I: NDArray[np.float64],
) -> NDArray[np.complexfloating]:
    """Try sign combinations and pick the one closest to unitary."""
    best_S = S.copy()
    best_err = _unitarity_error(S)

    # Try flipping signs on phases (2^3 = 8 combinations for off-diagonal)
    magnitudes = np.abs(S)
    base_phases = np.angle(S)

    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                trial_phases = base_phases.copy()
                trial_phases[0, 1] *= s1
                trial_phases[1, 0] *= s2
                trial_phases[1, 1] *= s3
                trial_S = magnitudes * np.exp(1j * trial_phases)
                err = _unitarity_error(trial_S)
                if err < best_err:
                    best_err = err
                    best_S = trial_S.copy()

    return best_S


def _unitarity_error(S: NDArray[np.complexfloating]) -> float:
    """Frobenius norm of S*S^H - I."""
    return float(np.linalg.norm(S @ S.conj().T - np.eye(2)))


def phase_measurements_needed(n: int) -> dict[str, int]:
    """Minimum measurements for complete n-port S-matrix recovery.

    Parameters
    ----------
    n : int
        Port count.

    Returns
    -------
    dict
        'power': n^2 (from |S|^2), 'interferometric': 2n-1,
        'total': n^2 + 2n - 1.
    """
    return {
        "power": n * n,
        "interferometric": 2 * n - 1,
        "total": n * n + 2 * n - 1,
    }


def generate_test_unitary(rng: np.random.Generator | None = None) -> NDArray[np.complexfloating]:
    """Generate a Haar-random 2x2 unitary matrix for testing.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    U : ndarray, shape (2, 2), complex
        A Haar-random unitary matrix.
    """
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    Q, R = np.linalg.qr(Z)
    # Make Haar-distributed
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q * ph
