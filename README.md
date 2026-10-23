# DS2 — Algebraic Toolkit for 2x2 Doubly Stochastic Matrices

[![PyPI version](https://img.shields.io/pypi/v/ds2-toolkit)](https://pypi.org/project/ds2-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/ds2-toolkit)](https://pypi.org/project/ds2-toolkit/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()

**One scalar replaces four complex scattering parameters.**

Every 2x2 doubly stochastic matrix is uniquely determined by a single eigenvalue
`lambda in [-1, 1]`, and `P(a) * P(b) = P(a*b)`. This monoid structure reduces
cascade analysis to scalar multiplication — no matrix algebra needed.

## Installation

```bash
pip install ds2-toolkit
```

## Quick Start

```python
import ds2_toolkit

# Build a P(lambda) matrix
P = ds2_toolkit.matrix(0.7)
# array([[0.85, 0.15],
#        [0.15, 0.85]])

# Cascade 4 devices: just multiply eigenvalues
result = ds2_toolkit.cascade([0.9, 0.8, -0.5, 0.95])
print(result.eigenvalue)          # -0.342
print(result.insertion_loss_db)   # 3.33 dB

# Analog error correction (k=3 repetition code)
ds2_toolkit.correct(0.3, k=3)  # 0.4365 — repels from 0

# Compile a target IL from standard components
r = ds2_toolkit.compile(target_il=7.0)
print(r.chain)   # ('6dB', '1dB')
print(r.error_db) # 0.0

# Landauer efficiency (quasistatic erasure)
ds2_toolkit.landauer_efficiency(delta_0=0.9, delta_f=0.1, k=100)
# {'eta': 0.982, 'delta_h': 0.469, 'dissipation': 0.009, ...}

# 13-domain universality
ds2_toolkit.domains.bsc(epsilon=0.1)          # 0.8
ds2_toolkit.domains.beam_splitter(reflectance=0.3)  # -0.4
ds2_toolkit.domains.scattering(s11_sq=0.3)    # -0.4
```

## Five Tools

| Tool | What it does | Key result |
|------|-------------|------------|
| **Cascade** | `lambda_total = prod(lambda_i)` | Error < 10^-16 |
| **Thermodynamics** | Entropy budget + Landauer cost | eta -> 1 as k -> inf |
| **Correction** | `f_3(lambda) = (3*lambda - lambda^3)/2` | 26.5x capacity gain |
| **Phase recovery** | Interferometric S-matrix reconstruction | 100% on 1000 unitaries |
| **Compiler** | Target IL -> device chain | 0.00 dB error |

## Thirteen Domains

The DS2 monoid appears in: microwave scattering, quantum gates, Markov chains,
softmax/neural networks, optical beam splitters, population genetics, resistor
dividers, polarization (Malus's law), IIR filters, binary symmetric channels,
synaptic transmission, gene regulation, and epidemic transmission.

```python
from ds2_toolkit.domains import DOMAIN_TABLE
for d in DOMAIN_TABLE:
    print(f"{d['domain']:30s} lambda = {d['lambda']}")
```

## References

- D. T. Foss, "Algebraic Structure of Doubly Stochastic Power Matrices in
  Electromagnetic Scattering," Proc. ICEAA/APWC, Toyama, Japan, Sep. 2026.
- D. T. Foss, "The DS2 Toolkit: Algebraic Methods for Two-Port Network Analysis
  Across Thirteen Domains," Proc. ICEAA/APWC, Toyama, Japan, Sep. 2026.

## License

PolyForm Noncommercial License 1.0.0. See [LICENSE](LICENSE). Commercial licenses: Wirkung Inc. (d.foss@ieee.org).
