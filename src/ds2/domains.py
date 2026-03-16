"""DS2 universality: thirteen domains, one algebra.

Every physical process that mixes two states with a fixed ratio has a
transition matrix P(lambda). This module provides domain-specific
constructors that map physical parameters to the DS2 eigenvalue.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Domain 1: Microwave scattering
# --------------------------------------------------------------------------- #

def scattering(s11_sq: float, s21_sq: float | None = None) -> float:
    """Eigenvalue from scattering parameters.

    lambda = |S11|^2 - |S21|^2. For lossless devices, |S21|^2 = 1 - |S11|^2.

    Parameters
    ----------
    s11_sq : float
        |S11|^2 (reflection coefficient squared).
    s21_sq : float, optional
        |S21|^2 (transmission coefficient squared). If None, assumes lossless.
    """
    if s21_sq is None:
        s21_sq = 1.0 - s11_sq
    return s11_sq - s21_sq


# --------------------------------------------------------------------------- #
# Domain 2: Quantum gates
# --------------------------------------------------------------------------- #

def quantum_gate(alpha_sq: float, beta_sq: float | None = None) -> float:
    """Eigenvalue from single-qubit gate probabilities.

    lambda = |alpha|^2 - |beta|^2 where U|0> = alpha|0> + beta|1>.
    """
    if beta_sq is None:
        beta_sq = 1.0 - alpha_sq
    return alpha_sq - beta_sq


# --------------------------------------------------------------------------- #
# Domain 3: Markov chains (2-state)
# --------------------------------------------------------------------------- #

def markov(p01: float, p10: float) -> float:
    """Eigenvalue of a 2-state Markov chain.

    The non-trivial eigenvalue is lambda = 1 - p01 - p10.

    Parameters
    ----------
    p01 : float
        Transition probability 0 -> 1.
    p10 : float
        Transition probability 1 -> 0.
    """
    return 1.0 - p01 - p10


# --------------------------------------------------------------------------- #
# Domain 4: Softmax / neural networks
# --------------------------------------------------------------------------- #

def softmax(delta_z: float, temperature: float = 1.0) -> float:
    """Eigenvalue from softmax with logit difference delta_z.

    lambda = tanh(delta_z / (2*T)).
    """
    return float(np.tanh(delta_z / (2.0 * temperature)))


# --------------------------------------------------------------------------- #
# Domain 5: Optical beam splitters
# --------------------------------------------------------------------------- #

def beam_splitter(reflectance: float) -> float:
    """Eigenvalue from beam splitter reflectance R.

    lambda = R - T = R - (1 - R) = 2R - 1.
    """
    return 2.0 * reflectance - 1.0


# --------------------------------------------------------------------------- #
# Domain 6: Population genetics
# --------------------------------------------------------------------------- #

def selection(fitness_coefficient: float) -> float:
    """Eigenvalue from selection coefficient s.

    Fitness of allele A = 1, allele B = 1-s.
    lambda = 1 - s.
    """
    return 1.0 - fitness_coefficient


# --------------------------------------------------------------------------- #
# Domain 7: Resistor voltage dividers
# --------------------------------------------------------------------------- #

def voltage_divider(r1: float, r2: float) -> float:
    """Eigenvalue from resistor values.

    lambda = (R2 - R1) / (R1 + R2).
    """
    if r1 + r2 <= 0:
        raise ValueError("R1 + R2 must be positive")
    return (r2 - r1) / (r1 + r2)


# --------------------------------------------------------------------------- #
# Domain 8: Polarization (Malus's law)
# --------------------------------------------------------------------------- #

def polarization(theta: float) -> float:
    """Eigenvalue from polarizer angle theta (radians).

    lambda = cos(2*theta).
    """
    return float(np.cos(2.0 * theta))


# --------------------------------------------------------------------------- #
# Domain 9: Digital filters (1st-order IIR)
# --------------------------------------------------------------------------- #

def iir_filter(alpha: float) -> float:
    """Eigenvalue from IIR filter coefficient alpha.

    y_k = alpha * x_k + (1-alpha) * y_{k-1}.
    lambda = 1 - 2*alpha.
    """
    return 1.0 - 2.0 * alpha


# --------------------------------------------------------------------------- #
# Domain 10: Binary symmetric channel
# --------------------------------------------------------------------------- #

def bsc(epsilon: float) -> float:
    """Eigenvalue from BSC crossover probability epsilon.

    lambda = 1 - 2*epsilon.
    """
    return 1.0 - 2.0 * epsilon


# --------------------------------------------------------------------------- #
# Domain 11: Synaptic transmission
# --------------------------------------------------------------------------- #

def synapse(p_release: float) -> float:
    """Eigenvalue from vesicle release probability.

    lambda = 2*p_release - 1.
    """
    return 2.0 * p_release - 1.0


# --------------------------------------------------------------------------- #
# Domain 12: Gene regulation cascades
# --------------------------------------------------------------------------- #

def gene_regulation(sensitivity: float) -> float:
    """Eigenvalue from gene regulation sensitivity.

    lambda = sensitivity (input-output transfer coefficient).
    """
    if not -1.0 <= sensitivity <= 1.0:
        raise ValueError(f"sensitivity must be in [-1,1], got {sensitivity}")
    return sensitivity


# --------------------------------------------------------------------------- #
# Domain 13: Epidemic transmission
# --------------------------------------------------------------------------- #

def epidemic(beta: float) -> float:
    """Eigenvalue from per-contact infection probability beta.

    lambda = 2*beta - 1. Serial contacts cascade as lambda^k.
    """
    return 2.0 * beta - 1.0


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

DOMAIN_TABLE: list[dict[str, str]] = [
    {"domain": "Microwave scattering", "lambda": "|S11|^2 - |S21|^2", "function": "scattering"},
    {"domain": "Quantum gates", "lambda": "Gate power eigenvalue", "function": "quantum_gate"},
    {"domain": "Markov chains (2-state)", "lambda": "1 - p01 - p10", "function": "markov"},
    {"domain": "Softmax / neural networks", "lambda": "tanh(dz/2T)", "function": "softmax"},
    {"domain": "Optical beam splitters", "lambda": "R - T", "function": "beam_splitter"},
    {"domain": "Population genetics", "lambda": "1 - s", "function": "selection"},
    {"domain": "Resistor voltage dividers", "lambda": "(R2-R1)/(R1+R2)", "function": "voltage_divider"},
    {"domain": "Polarization (Malus's law)", "lambda": "cos(2*theta)", "function": "polarization"},
    {"domain": "Digital filters (1st-order IIR)", "lambda": "1 - 2*alpha", "function": "iir_filter"},
    {"domain": "Binary symmetric channel", "lambda": "1 - 2*epsilon", "function": "bsc"},
    {"domain": "Synaptic transmission", "lambda": "2*p_release - 1", "function": "synapse"},
    {"domain": "Gene regulation cascades", "lambda": "sensitivity", "function": "gene_regulation"},
    {"domain": "Epidemic transmission", "lambda": "2*beta - 1", "function": "epidemic"},
]
