"""DS2: Algebraic toolkit for 2x2 doubly stochastic matrices.

The DS2 monoid: every 2x2 doubly stochastic matrix is P(lambda) for a unique
lambda in [-1, 1], and P(a) * P(b) = P(ab). This single-scalar parameterization
replaces four complex scattering parameters for power-level two-port analysis.

Five core tools:
    1. Cascade-by-multiplication (cascade)
    2. Thermodynamic analysis (thermo)
    3. Analog error correction (correction)
    4. Phase recovery (phase)
    5. Cascade compiler (compiler)

Universal across 13 domains (domains).

Example::

    >>> import ds2
    >>> P = ds2.matrix(0.7)
    >>> ds2.cascade([0.9, 0.8, -0.5, 0.95])
    CascadeResult(eigenvalue=-0.342, insertion_loss_db=...)
    >>> ds2.correct(0.3, k=3)
    0.4365...

References:
    D. T. Foss, "Algebraic Structure of Doubly Stochastic Power Matrices
    in Electromagnetic Scattering," Proc. ICEAA/APWC, Toyama, 2026.

    D. T. Foss, "The DS2 Toolkit: Algebraic Methods for Two-Port Network
    Analysis Across Thirteen Domains," Proc. ICEAA/APWC, Toyama, 2026.
"""

__version__ = "0.1.0"

from ds2.core import matrix, eigenvalue, insertion_loss, from_insertion_loss
from ds2.cascade import cascade, CascadeResult
from ds2.thermo import entropy, entropy_change, landauer_cost, landauer_efficiency
from ds2.correction import correct, correction_polynomial, duality_fixed_points
from ds2.phase import recover_phase_2port
from ds2.compiler import compile, CompilerResult, STANDARD_LIBRARY
from ds2 import domains

__all__ = [
    "__version__",
    # core
    "matrix",
    "eigenvalue",
    "insertion_loss",
    "from_insertion_loss",
    # cascade
    "cascade",
    "CascadeResult",
    # thermo
    "entropy",
    "entropy_change",
    "landauer_cost",
    "landauer_efficiency",
    # correction
    "correct",
    "correction_polynomial",
    "duality_fixed_points",
    # phase
    "recover_phase_2port",
    # compiler
    "compile",
    "CompilerResult",
    "STANDARD_LIBRARY",
    # domains
    "domains",
]
