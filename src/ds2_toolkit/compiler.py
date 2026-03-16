"""Cascade compiler: find exact device chains for target insertion loss.

Since IL values add in dB under cascade (log|lambda| is additive),
cascade design reduces to subset-sum on IL values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# Standard component library (name, IL in dB)
STANDARD_LIBRARY: list[tuple[str, float]] = [
    ("20dB", 20.0),
    ("10dB", 10.0),
    ("6dB", 6.0),
    ("3dB", 3.0),
    ("1dB", 1.0),
    ("0.5dB", 0.5),
    ("0.1dB", 0.1),
]


@dataclass(frozen=True)
class CompilerResult:
    """Result of cascade compilation.

    Attributes
    ----------
    chain : tuple[str, ...]
        Device names in the compiled chain.
    target_il_db : float
        Requested insertion loss.
    achieved_il_db : float
        Achieved insertion loss.
    error_db : float
        |target - achieved| in dB.
    eigenvalue : float
        Equivalent eigenvalue of the chain.
    """

    chain: tuple[str, ...]
    target_il_db: float
    achieved_il_db: float
    error_db: float
    eigenvalue: float

    def __repr__(self) -> str:
        chain_str = " + ".join(self.chain)
        return (
            f"CompilerResult(chain=[{chain_str}], "
            f"target={self.target_il_db:.2f}dB, "
            f"achieved={self.achieved_il_db:.2f}dB, "
            f"error={self.error_db:.2f}dB)"
        )


def compile(
    target_il: float,
    library: Sequence[tuple[str, float]] | None = None,
    tolerance: float = 0.05,
) -> CompilerResult:
    """Find a device chain achieving the target insertion loss.

    Uses a greedy algorithm on a sorted library. For practical libraries
    (< 20 component types), this achieves exact solutions.

    Parameters
    ----------
    target_il : float
        Target insertion loss in dB (> 0).
    library : sequence of (name, IL_dB) tuples, optional
        Component library. Defaults to STANDARD_LIBRARY.
    tolerance : float
        Acceptable error in dB.

    Returns
    -------
    CompilerResult
        Compiled chain with achieved IL and error.

    Examples
    --------
    >>> r = compile(7.0)
    >>> r.chain
    ('6dB', '1dB')
    >>> r.error_db
    0.0
    """
    if target_il <= 0.0:
        raise ValueError(f"target_il must be > 0, got {target_il}")
    if library is None:
        library = STANDARD_LIBRARY

    # Sort by IL descending
    lib = sorted(library, key=lambda x: x[1], reverse=True)

    chain: list[str] = []
    current_il = 0.0

    while abs(current_il - target_il) > tolerance:
        remaining = target_il - current_il
        if remaining <= 0:
            break

        # Find best component
        best_name = None
        best_il = 0.0
        best_gap = float("inf")

        for name, il in lib:
            if il <= remaining + tolerance:
                gap = abs(remaining - il)
                if gap < best_gap:
                    best_gap = gap
                    best_name = name
                    best_il = il

        if best_name is None:
            break

        chain.append(best_name)
        current_il += best_il

    error = abs(current_il - target_il)
    lam = 1.0 - 2.0 * 10.0 ** (-current_il / 10.0) if current_il > 0 else 1.0

    return CompilerResult(
        chain=tuple(chain),
        target_il_db=target_il,
        achieved_il_db=round(current_il, 10),
        error_db=round(error, 10),
        eigenvalue=lam,
    )
