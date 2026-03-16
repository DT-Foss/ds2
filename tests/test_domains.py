"""Tests for ds2.domains — 13-domain universality."""

import numpy as np
import pytest

from ds2_toolkit import domains


class TestDomainMappings:
    def test_scattering_lossless(self):
        # |S11|^2 = 0.3, |S21|^2 = 0.7 (lossless)
        lam = domains.scattering(0.3)
        assert lam == pytest.approx(-0.4)

    def test_scattering_explicit(self):
        lam = domains.scattering(0.3, 0.7)
        assert lam == pytest.approx(-0.4)

    def test_quantum_gate(self):
        lam = domains.quantum_gate(0.8)
        assert lam == pytest.approx(0.6)

    def test_markov(self):
        lam = domains.markov(0.3, 0.3)
        assert lam == pytest.approx(0.4)

    def test_softmax_zero(self):
        assert domains.softmax(0.0) == pytest.approx(0.0)

    def test_softmax_high_temp(self):
        # High temperature → lambda near 0
        assert abs(domains.softmax(1.0, temperature=100.0)) < 0.01

    def test_beam_splitter(self):
        # 50/50 splitter: R = 0.5
        assert domains.beam_splitter(0.5) == pytest.approx(0.0)
        # Full reflection: R = 1.0
        assert domains.beam_splitter(1.0) == pytest.approx(1.0)

    def test_selection(self):
        assert domains.selection(0.0) == pytest.approx(1.0)  # No selection
        assert domains.selection(1.0) == pytest.approx(0.0)  # Lethal

    def test_voltage_divider(self):
        assert domains.voltage_divider(50.0, 50.0) == pytest.approx(0.0)
        assert domains.voltage_divider(0.0, 100.0) == pytest.approx(1.0)

    def test_polarization(self):
        # theta = 0 → cos(0) = 1
        assert domains.polarization(0.0) == pytest.approx(1.0)
        # theta = pi/4 → cos(pi/2) = 0
        assert domains.polarization(np.pi / 4) == pytest.approx(0.0, abs=1e-15)

    def test_iir_filter(self):
        assert domains.iir_filter(0.0) == pytest.approx(1.0)
        assert domains.iir_filter(0.5) == pytest.approx(0.0)

    def test_bsc(self):
        assert domains.bsc(0.0) == pytest.approx(1.0)
        assert domains.bsc(0.5) == pytest.approx(0.0)

    def test_synapse(self):
        assert domains.synapse(1.0) == pytest.approx(1.0)
        assert domains.synapse(0.5) == pytest.approx(0.0)

    def test_gene_regulation(self):
        assert domains.gene_regulation(0.5) == pytest.approx(0.5)
        with pytest.raises(ValueError):
            domains.gene_regulation(1.5)

    def test_epidemic(self):
        assert domains.epidemic(1.0) == pytest.approx(1.0)
        assert domains.epidemic(0.5) == pytest.approx(0.0)


class TestDomainTable:
    def test_thirteen_domains(self):
        assert len(domains.DOMAIN_TABLE) == 13

    def test_all_functions_exist(self):
        for entry in domains.DOMAIN_TABLE:
            func_name = entry["function"]
            assert hasattr(domains, func_name), f"Missing: {func_name}"


class TestMonoidUniversality:
    """The cascade law lambda_total = prod(lambda_i) holds in every domain."""

    def test_cascade_scattering(self):
        from ds2_toolkit.core import matrix
        l1 = domains.scattering(0.3)
        l2 = domains.scattering(0.6)
        P1 = matrix(l1)
        P2 = matrix(l2)
        np.testing.assert_allclose(P1 @ P2, matrix(l1 * l2), atol=1e-15)

    def test_cascade_bsc(self):
        from ds2_toolkit.core import matrix
        l1 = domains.bsc(0.1)
        l2 = domains.bsc(0.2)
        np.testing.assert_allclose(
            matrix(l1) @ matrix(l2), matrix(l1 * l2), atol=1e-15
        )
