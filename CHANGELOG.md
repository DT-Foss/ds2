# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-16

### Added
- Core DS2 operations: `matrix()`, `eigenvalue()`, `insertion_loss()`, `from_insertion_loss()`
- Cascade-by-multiplication: `cascade()`, `link_budget()`
- Thermodynamic analysis: `entropy()`, `entropy_change()`, `landauer_cost()`, `landauer_efficiency()`
- Analog error correction: `correct()`, `correct_f3()`, `cascade_with_correction()`, `capacity()`
- Phase recovery: `recover_phase_2port()`, `generate_test_unitary()`
- Cascade compiler: `compile()` with `STANDARD_LIBRARY`
- 13-domain universality mappings in `ds2.domains`
- Full test suite with pytest
