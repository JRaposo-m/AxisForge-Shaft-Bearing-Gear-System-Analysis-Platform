# test_validation.py
"""Testes de validation.py -- check_bearing_ready e
warn_if_floating_loaded, os dois checks contact-agnosticos corridos
antes/à volta de um solve ISO/TS 16281."""
import warnings
from types import SimpleNamespace

import pytest

from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.validation import (
    FA_FLOATING_EPS, check_bearing_ready, warn_if_floating_loaded,
)


class TestCheckBearingReady:
    def test_passes_when_all_attrs_present_and_not_none(self):
        bearing = SimpleNamespace(A=1.0, alpha_0=0.2)
        check_bearing_ready(bearing, "b1", ("A", "alpha_0"))  # não deve levantar

    def test_passes_with_empty_required_attrs(self):
        bearing = SimpleNamespace()
        check_bearing_ready(bearing, "b1", ())  # não deve levantar

    def test_raises_when_attr_missing(self):
        bearing = SimpleNamespace(A=1.0)
        with pytest.raises(RuntimeError, match=r"missing \['alpha_0'\]"):
            check_bearing_ready(bearing, "b1", ("A", "alpha_0"))

    def test_raises_when_attr_present_but_none(self):
        """Um atributo presente mas None conta como 'setup não correu',
        igual a estar ausente."""
        bearing = SimpleNamespace(A=1.0, alpha_0=None)
        with pytest.raises(RuntimeError, match="alpha_0"):
            check_bearing_ready(bearing, "b1", ("A", "alpha_0"))

    def test_lists_every_missing_attribute_at_once(self):
        bearing = SimpleNamespace()
        with pytest.raises(RuntimeError) as excinfo:
            check_bearing_ready(bearing, "b1", ("A", "alpha_0", "cp"))
        message = str(excinfo.value)
        assert "A" in message and "alpha_0" in message and "cp" in message

    def test_error_message_includes_label(self):
        bearing = SimpleNamespace()
        with pytest.raises(RuntimeError, match="b1"):
            check_bearing_ready(bearing, "b1", ("A",))


class TestWarnIfFloatingLoaded:
    def test_warns_when_floating_and_loaded(self):
        bearing = SimpleNamespace(arrangement="floating")
        with pytest.warns(UserWarning, match="floating but Fa"):
            warn_if_floating_loaded(bearing, "b1", Fa=10.0)

    def test_no_warning_when_floating_and_unloaded(self):
        bearing = SimpleNamespace(arrangement="floating")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_floating_loaded(bearing, "b1", Fa=0.0)

    def test_no_warning_when_not_floating(self):
        bearing = SimpleNamespace(arrangement="locating")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_floating_loaded(bearing, "b1", Fa=1000.0)

    def test_no_warning_when_arrangement_attribute_absent(self):
        bearing = SimpleNamespace()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_floating_loaded(bearing, "b1", Fa=1000.0)

    def test_eps_boundary_is_exclusive(self):
        """Fa exatamente igual a eps não avisa -- a condição é
        abs(Fa) > eps, não >=."""
        bearing = SimpleNamespace(arrangement="floating")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_floating_loaded(bearing, "b1", Fa=FA_FLOATING_EPS)

    def test_custom_eps_is_honoured(self):
        bearing = SimpleNamespace(arrangement="floating")
        with pytest.warns(UserWarning):
            warn_if_floating_loaded(bearing, "b1", Fa=0.5, eps=0.1)

    def test_negative_fa_also_warns(self):
        bearing = SimpleNamespace(arrangement="floating")
        with pytest.warns(UserWarning):
            warn_if_floating_loaded(bearing, "b1", Fa=-10.0)
