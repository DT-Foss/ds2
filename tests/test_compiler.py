"""Tests for ds2.compiler — cascade compilation."""

import pytest

from ds2_toolkit.compiler import STANDARD_LIBRARY, CompilerResult, compile


class TestCompiler:
    def test_exact_match(self):
        """Standard library targets should be exact."""
        for target in [1, 3, 5, 7, 10, 13, 15, 20, 25, 30]:
            r = compile(float(target))
            assert r.error_db == pytest.approx(0.0, abs=0.1)

    def test_chain_content(self):
        r = compile(7.0)
        assert r.chain == ("6dB", "1dB")

    def test_single_component(self):
        r = compile(3.0)
        assert r.chain == ("3dB",)

    def test_large_target(self):
        r = compile(30.0)
        assert r.error_db < 0.1

    def test_small_target(self):
        r = compile(0.5)
        assert r.chain == ("0.5dB",)

    def test_repr(self):
        r = compile(7.0)
        assert "7.00dB" in repr(r)

    def test_invalid_target(self):
        with pytest.raises(ValueError):
            compile(-1.0)
        with pytest.raises(ValueError):
            compile(0.0)

    def test_custom_library(self):
        lib = [("big", 10.0), ("small", 1.0)]
        r = compile(12.0, library=lib)
        assert r.achieved_il_db == pytest.approx(12.0)
