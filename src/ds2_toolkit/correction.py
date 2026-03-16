"""Analog error correction via cascade-correction duality.

Cascade: lambda -> lambda^k  (attracts to 0)
Correction: lambda -> f_k(lambda)  (repels from 0)

Both share fixed points {-1, 0, +1}. For cascade, 0 is stable;
for correction, 0 is unstable and +/-1 are stable.

f_3(lambda) = (3*lambda - lambda^3) / 2
"""

from __future__ import annotations

from math import comb

import numpy as np


def correct(lam: float, k: int = 3) -> float:
    """Apply k-fold repetition code correction to eigenvalue lambda.

    Parameters
    ----------
    lam : float
        Eigenvalue in [-1, 1].
    k : int
        Repetition code length (must be odd, >= 3).

    Returns
    -------
    float
        Corrected eigenvalue f_k(lambda).

    Examples
    --------
    >>> abs(correct(0.6, k=3) - (3*0.6 - 0.6**3)/2) < 1e-15
    True
    """
    if k < 3 or k % 2 == 0:
        raise ValueError(f"k must be odd and >= 3, got {k}")
    if not -1.0 <= lam <= 1.0:
        raise ValueError(f"lambda must be in [-1, 1], got {lam}")
    return correction_polynomial(lam, k)


def correct_f3(lam: float) -> float:
    """Closed-form k=3 correction: f_3(lambda) = (3*lambda - lambda^3) / 2.

    Parameters
    ----------
    lam : float
        Eigenvalue in [-1, 1].

    Returns
    -------
    float
        Corrected eigenvalue.
    """
    return (3.0 * lam - lam**3) / 2.0


def correction_polynomial(lam: float, k: int) -> float:
    """General k-fold correction polynomial f_k(lambda).

    f_k(lambda) = 1 - 2 * sum_{i=ceil(k/2)}^{k} C(k,i) * eps^i * (1-eps)^(k-i)

    where eps = (1 - lambda) / 2 is the BSC error rate.

    Parameters
    ----------
    lam : float
        Eigenvalue in [-1, 1].
    k : int
        Repetition code length (odd).

    Returns
    -------
    float
        f_k(lambda).
    """
    eps = (1.0 - lam) / 2.0
    threshold = (k + 1) // 2
    # Probability of majority error
    p_error = sum(
        comb(k, i) * eps**i * (1.0 - eps) ** (k - i) for i in range(threshold, k + 1)
    )
    return 1.0 - 2.0 * p_error


def cascade_eigenvalue(lam: float, k: int) -> float:
    """k-fold cascade: lambda -> lambda^k.

    Parameters
    ----------
    lam : float
        Eigenvalue in [-1, 1].
    k : int
        Number of cascade stages.

    Returns
    -------
    float
        lambda^k.
    """
    return lam**k


def duality_fixed_points() -> list[float]:
    """Return the shared fixed points of cascade and correction maps.

    Both lambda -> lambda^k and lambda -> f_k(lambda) share
    fixed points at {-1, 0, +1}.

    Returns
    -------
    list of float
        [-1.0, 0.0, 1.0]
    """
    return [-1.0, 0.0, 1.0]


def cascade_with_correction(
    lam_per_stage: float,
    n_stages: int,
    correction_interval: int = 3,
    correction_k: int = 3,
) -> list[float]:
    """Simulate cascade with periodic correction.

    Parameters
    ----------
    lam_per_stage : float
        Eigenvalue per cascade stage.
    n_stages : int
        Total number of stages.
    correction_interval : int
        Apply correction every this many stages.
    correction_k : int
        Repetition code length for correction.

    Returns
    -------
    list of float
        Effective eigenvalue after each stage.
    """
    trace = []
    lam = 1.0
    for i in range(1, n_stages + 1):
        lam *= lam_per_stage
        if i % correction_interval == 0:
            lam = correct(lam, k=correction_k)
        trace.append(lam)
    return trace


def capacity(lam: float) -> float:
    """Binary symmetric channel capacity C = 1 - H(eps).

    Parameters
    ----------
    lam : float
        Eigenvalue in [-1, 1].

    Returns
    -------
    float
        Channel capacity in bits.
    """
    eps = (1.0 - abs(lam)) / 2.0
    if eps <= 0.0:
        return 1.0
    if eps >= 0.5:
        return 0.0
    return 1.0 + eps * np.log2(eps) + (1.0 - eps) * np.log2(1.0 - eps)
