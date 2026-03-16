"""Cascade-by-multiplication: reduce k-device chain to scalar arithmetic.

The DS2 monoid product P(a) * P(b) = P(ab) means cascading k lossless
two-port devices with eigenvalues lambda_1, ..., lambda_k gives:

    lambda_total = prod(lambda_i)
    IL_total = sum(IL_i)       [in dB]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ds2_toolkit.core import insertion_loss, matrix


@dataclass(frozen=True)
class CascadeResult:
    """Result of cascading multiple DS2 devices.

    Attributes
    ----------
    eigenvalue : float
        Product eigenvalue of the cascade.
    insertion_loss_db : float
        Total insertion loss in dB.
    matrix : np.ndarray
        The 2x2 doubly stochastic matrix P(eigenvalue).
    individual : tuple[float, ...]
        Individual eigenvalues in cascade order.
    cumulative : tuple[float, ...]
        Cumulative eigenvalue after each stage.
    """

    eigenvalue: float
    insertion_loss_db: float
    matrix: np.ndarray = field(repr=False)
    individual: tuple[float, ...] = field(repr=False)
    cumulative: tuple[float, ...] = field(repr=False)

    def __repr__(self) -> str:
        return (
            f"CascadeResult(eigenvalue={self.eigenvalue:.6f}, "
            f"insertion_loss_db={self.insertion_loss_db:.4f}, "
            f"stages={len(self.individual)})"
        )


def cascade(eigenvalues: Sequence[float]) -> CascadeResult:
    """Cascade k devices by multiplying their eigenvalues.

    Parameters
    ----------
    eigenvalues : sequence of float
        Eigenvalues lambda_i in [-1, 1] for each device.

    Returns
    -------
    CascadeResult
        Contains product eigenvalue, total IL, cumulative trace.

    Raises
    ------
    ValueError
        If any eigenvalue is outside [-1, 1] or sequence is empty.

    Examples
    --------
    >>> r = cascade([0.9, 0.8, 0.7])
    >>> abs(r.eigenvalue - 0.9 * 0.8 * 0.7) < 1e-15
    True
    """
    evs = list(eigenvalues)
    if not evs:
        raise ValueError("Need at least one eigenvalue")
    for i, lam in enumerate(evs):
        if not -1.0 <= lam <= 1.0:
            raise ValueError(f"eigenvalue[{i}] = {lam} outside [-1, 1]")

    cum = []
    prod = 1.0
    for lam in evs:
        prod *= lam
        cum.append(prod)

    lam_total = cum[-1]
    il_db = insertion_loss(lam_total)
    P = matrix(lam_total)

    return CascadeResult(
        eigenvalue=lam_total,
        insertion_loss_db=il_db,
        matrix=P,
        individual=tuple(evs),
        cumulative=tuple(cum),
    )


def link_budget(
    devices: Sequence[tuple[str, float]],
) -> list[dict[str, float | str]]:
    """Compute a link budget table for named devices.

    Parameters
    ----------
    devices : sequence of (name, eigenvalue) tuples
        Each device with its name and eigenvalue.

    Returns
    -------
    list of dict
        Each dict has keys: 'device', 'eigenvalue', 'cumulative_eigenvalue',
        'cumulative_il_db'.

    Examples
    --------
    >>> budget = link_budget([("coupler", -0.002), ("mismatch", -0.589)])
    >>> budget[1]['cumulative_il_db']  # doctest: +SKIP
    3.01...
    """
    rows = []
    prod = 1.0
    for name, lam in devices:
        if not -1.0 <= lam <= 1.0:
            raise ValueError(f"eigenvalue for '{name}' = {lam} outside [-1, 1]")
        prod *= lam
        rows.append(
            {
                "device": name,
                "eigenvalue": lam,
                "cumulative_eigenvalue": prod,
                "cumulative_il_db": insertion_loss(prod),
            }
        )
    return rows
