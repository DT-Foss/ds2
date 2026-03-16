"""Core DS2 operations: matrix construction, eigenvalue extraction, insertion loss.

The fundamental result: every 2x2 doubly stochastic matrix has the form

    P(lambda) = 1/2 * [[1+lambda, 1-lambda],
                        [1-lambda, 1+lambda]]

for lambda in [-1, 1], and P(a) * P(b) = P(a*b).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def matrix(lam: float) -> NDArray[np.float64]:
    """Construct the 2x2 doubly stochastic matrix P(lambda).

    Parameters
    ----------
    lam : float
        Eigenvalue in [-1, 1].

    Returns
    -------
    P : ndarray, shape (2, 2)
        The doubly stochastic matrix.

    Raises
    ------
    ValueError
        If lambda is outside [-1, 1].

    Examples
    --------
    >>> matrix(1.0)
    array([[1., 0.],
           [0., 1.]])
    >>> matrix(0.0)
    array([[0.5, 0.5],
           [0.5, 0.5]])
    """
    if not -1.0 <= lam <= 1.0:
        raise ValueError(f"lambda must be in [-1, 1], got {lam}")
    return 0.5 * np.array(
        [[1.0 + lam, 1.0 - lam], [1.0 - lam, 1.0 + lam]], dtype=np.float64
    )


def eigenvalue(P: NDArray[np.float64]) -> float:
    """Extract the non-trivial eigenvalue lambda from a 2x2 DS matrix.

    Parameters
    ----------
    P : ndarray, shape (2, 2)
        A 2x2 doubly stochastic matrix.

    Returns
    -------
    lam : float
        The eigenvalue lambda = P[0,0] - P[0,1].

    Raises
    ------
    ValueError
        If P is not 2x2 or not doubly stochastic (within tolerance).
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape != (2, 2):
        raise ValueError(f"Expected 2x2 matrix, got shape {P.shape}")
    tol = 1e-10
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)
    if not (np.allclose(row_sums, 1.0, atol=tol) and np.allclose(col_sums, 1.0, atol=tol)):
        raise ValueError(
            f"Matrix is not doubly stochastic: row sums={row_sums}, col sums={col_sums}"
        )
    if np.any(P < -tol):
        raise ValueError("Matrix has negative entries")
    return float(P[0, 0] - P[0, 1])


def insertion_loss(lam: float) -> float:
    """Compute insertion loss in dB from eigenvalue lambda.

    IL = -10 * log10((1 - lambda) / 2)

    Parameters
    ----------
    lam : float
        Eigenvalue in [-1, 1]. For lam = -1 (full transmission), IL = 0 dB.

    Returns
    -------
    il_db : float
        Insertion loss in decibels. Returns inf for lam = 1.
    """
    if not -1.0 <= lam <= 1.0:
        raise ValueError(f"lambda must be in [-1, 1], got {lam}")
    t = (1.0 - lam) / 2.0
    if t <= 0.0:
        return float("inf")
    return -10.0 * np.log10(t)


def from_insertion_loss(il_db: float) -> float:
    """Recover eigenvalue lambda from insertion loss in dB.

    lambda = 1 - 2 * 10^(-IL/10)

    Parameters
    ----------
    il_db : float
        Insertion loss in decibels (>= 0).

    Returns
    -------
    lam : float
        Eigenvalue in [-1, 1].
    """
    if il_db < 0.0:
        raise ValueError(f"Insertion loss must be >= 0, got {il_db}")
    t = 10.0 ** (-il_db / 10.0)
    return 1.0 - 2.0 * t


def product(lam1: float, lam2: float) -> float:
    """DS2 monoid product: P(a) * P(b) = P(a*b).

    Parameters
    ----------
    lam1, lam2 : float
        Eigenvalues in [-1, 1].

    Returns
    -------
    float
        Product eigenvalue lam1 * lam2.
    """
    return lam1 * lam2


def is_doubly_stochastic(P: NDArray[np.float64], tol: float = 1e-10) -> bool:
    """Check if a matrix is doubly stochastic.

    Parameters
    ----------
    P : ndarray
        Matrix to check.
    tol : float
        Tolerance for row/column sum checks.

    Returns
    -------
    bool
        True if P is doubly stochastic within tolerance.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    if np.any(P < -tol):
        return False
    row_ok = np.allclose(P.sum(axis=1), 1.0, atol=tol)
    col_ok = np.allclose(P.sum(axis=0), 1.0, atol=tol)
    return bool(row_ok and col_ok)
