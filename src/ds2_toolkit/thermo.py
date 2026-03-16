"""Thermodynamic analysis: entropy, Landauer cost, quasistatic efficiency.

The discrete Second Law: H(P(lambda) * x) >= H(x) for any probability
vector x and |lambda| < 1. Each cascade step erases information with
minimum energy cost k_B * T * ln(2) per bit (Landauer's principle).
"""

from __future__ import annotations

import numpy as np


# Boltzmann constant times ln(2) at T=300K, in Joules
K_B_T_LN2_300K = 2.8534e-21  # J (approx 0.018 eV)


def entropy(p: float) -> float:
    """Binary entropy H(p) in bits.

    Parameters
    ----------
    p : float
        Probability in [0, 1].

    Returns
    -------
    float
        H(p) = -p*log2(p) - (1-p)*log2(1-p).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def _bias_after(lam: float, delta: float) -> float:
    """Bias after applying P(lambda) to a distribution with bias delta."""
    return abs(lam) * delta


def entropy_change(lam: float, delta: float) -> float:
    """Entropy increase from applying P(lambda) to a distribution with bias delta.

    Parameters
    ----------
    lam : float
        Eigenvalue in [-1, 1].
    delta : float
        Input bias |2p - 1| in [0, 1].

    Returns
    -------
    float
        Delta_H = H(output) - H(input) >= 0.
    """
    p_in = 0.5 + delta / 2.0
    p_out = 0.5 + _bias_after(lam, delta) / 2.0
    return entropy(p_out) - entropy(p_in)


def landauer_cost(delta_h: float, temperature: float = 300.0) -> float:
    """Minimum energy cost (Landauer bound) for erasing delta_h bits.

    Parameters
    ----------
    delta_h : float
        Entropy increase (bits erased).
    temperature : float
        Temperature in Kelvin (default 300K).

    Returns
    -------
    float
        Energy in Joules: delta_h * k_B * T * ln(2).
    """
    k_b = 1.380649e-23  # J/K (exact, SI 2019)
    return delta_h * k_b * temperature * np.log(2.0)


def _kl_divergence(p: float, q: float) -> float:
    """KL divergence D_KL(Ber(p) || Ber(q)) for Bernoulli distributions."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    if q <= 0.0 or q >= 1.0:
        return float("inf")
    return p * np.log2(p / q) + (1.0 - p) * np.log2((1.0 - p) / (1.0 - q))


def landauer_efficiency(
    delta_0: float,
    delta_f: float,
    k: int,
) -> dict[str, float]:
    """Compute Landauer efficiency for k-step quasistatic erasure.

    Uses equal eigenvalues lambda_i = (delta_f/delta_0)^(1/k) per step.
    Efficiency eta = Delta_H / (Delta_H + D(k)), where D(k) is the
    total KL-divergence dissipation (path-dependent).

    Parameters
    ----------
    delta_0 : float
        Initial bias in (0, 1].
    delta_f : float
        Final bias in [0, delta_0).
    k : int
        Number of equal steps (>= 1).

    Returns
    -------
    dict
        Keys: 'eta' (efficiency), 'delta_h' (entropy change in bits),
        'dissipation' (total KL divergence), 'lambda_per_step'.
    """
    if not (0.0 < delta_f < delta_0 <= 1.0):
        raise ValueError(f"Need 0 < delta_f < delta_0 <= 1, got {delta_f}, {delta_0}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    lam_step = (delta_f / delta_0) ** (1.0 / k)
    p_in = 0.5 + delta_0 / 2.0
    p_out = 0.5 + delta_f / 2.0
    delta_h = entropy(p_out) - entropy(p_in)

    # Compute path-dependent KL dissipation
    total_kl = 0.0
    delta = delta_0
    for _ in range(k):
        p_before = 0.5 + delta / 2.0
        delta = delta * lam_step
        p_after = 0.5 + delta / 2.0
        total_kl += _kl_divergence(p_after, p_before)

    eta = delta_h / (delta_h + total_kl) if (delta_h + total_kl) > 0.0 else 1.0

    return {
        "eta": eta,
        "delta_h": delta_h,
        "dissipation": total_kl,
        "lambda_per_step": lam_step,
    }
