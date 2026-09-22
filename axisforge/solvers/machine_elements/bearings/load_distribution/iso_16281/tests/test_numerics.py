# test_numerics.py
"""Testes de numerics.py -- run_root, o driver hybr-com-fallback-lm
partilhado por todos os solvers ISO/TS 16281."""
import numpy as np
import pytest

from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281 import numerics as num_module
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.numerics import run_root


class _FakeSol:
    def __init__(self, x, fun, success, nfev=None):
        self.x = np.asarray(x, dtype=float)
        self.fun = np.asarray(fun, dtype=float)
        self.success = success
        if nfev is not None:
            self.nfev = nfev


class TestRunRootHappyPath:
    def test_solves_simple_linear_system(self):
        x, nfev, res, ok = run_root(lambda x: np.array([2.0 * x[0] - 4.0]), [0.0], tol=1e-9)
        assert ok
        assert x[0] == pytest.approx(2.0, abs=1e-6)
        assert res < 1e-6
        assert nfev > 0

    def test_result_is_at_least_1d(self):
        x, *_ = run_root(lambda x: np.array([x[0] - 1.0]), [0.0], tol=1e-9)
        assert x.ndim == 1


class TestRunRootFallback:
    def test_falls_back_to_lm_when_hybr_fails(self, monkeypatch):
        calls = []

        def fake_root(fun, x0, method, tol):
            calls.append(method)
            if method == "hybr":
                return _FakeSol([0.0], [10.0], success=False, nfev=5)
            return _FakeSol([2.0], [0.0], success=True, nfev=3)

        monkeypatch.setattr(num_module, "root", fake_root)
        x, nfev, res, ok = run_root(lambda x: x, [0.0], tol=1e-9)

        assert calls == ["hybr", "lm"]
        assert ok is True
        assert x[0] == pytest.approx(2.0)
        assert res == pytest.approx(0.0)
        assert nfev == 3

    def test_keeps_hybr_result_when_its_residual_is_smaller_despite_failure_flag(self, monkeypatch):
        """hybr reporta success=False mas com resíduo MENOR que o lm --
        run_root deve manter hybr mesmo assim (ver docstring do próprio
        módulo: 'not unusual for it to land closer than a nominally
        successful lm run')."""
        def fake_root(fun, x0, method, tol):
            if method == "hybr":
                return _FakeSol([1.0], [0.001], success=False, nfev=4)
            return _FakeSol([5.0], [1.0], success=True, nfev=2)

        monkeypatch.setattr(num_module, "root", fake_root)
        x, nfev, res, ok = run_root(lambda x: x, [0.0], tol=1e-9)

        assert x[0] == pytest.approx(1.0)
        assert ok is False  # mantém o próprio success flag da tentativa aceite
        assert res == pytest.approx(0.001)
        assert nfev == 4

    def test_does_not_call_lm_when_hybr_succeeds(self, monkeypatch):
        calls = []

        def fake_root(fun, x0, method, tol):
            calls.append(method)
            return _FakeSol([1.0], [0.0], success=True, nfev=1)

        monkeypatch.setattr(num_module, "root", fake_root)
        run_root(lambda x: x, [0.0], tol=1e-9)
        assert calls == ["hybr"]

    def test_nfev_defaults_to_zero_when_missing(self, monkeypatch):
        monkeypatch.setattr(
            num_module, "root",
            lambda fun, x0, method, tol: _FakeSol([1.0], [0.0], success=True),
        )
        x, nfev, res, ok = run_root(lambda x: x, [0.0], tol=1e-9)
        assert nfev == 0
