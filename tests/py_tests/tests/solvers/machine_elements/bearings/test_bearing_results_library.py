"""
tests/solvers/machine_elements/bearings/test_bearing_results_library.py

Unit tests for ISO_16281/library.py -- the contact-agnostic half of the
package: three shared utilities and the cross-type results registry.

No ISO math here. This file is about guards, bookkeeping and the numerical
root helper every per-type solver leans on.

ASCII only.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    BearingResultBundle,
    BearingResultsLibrary,
    check_bearing_ready,
    run_root,
    warn_if_floating_loaded,
)


# ===========================================================================
# check_bearing_ready
# ===========================================================================

class TestCheckBearingReady:

    REQUIRED = ("A", "alpha_0", "phi_j", "Ri", "cp", "Dpw", "Z")

    def test_a_fully_assembled_bearing_passes(self, brg_A):
        check_bearing_ready(brg_A, "A", self.REQUIRED)   # must not raise

    def test_a_missing_attribute_is_named_in_the_error(self):
        class Half:
            A = 0.3
            alpha_0 = 0.17
            phi_j = np.zeros(8)
            Ri = 17.0
            cp = None          # <- not computed
            Dpw = 33.5
            Z = 8

        with pytest.raises(RuntimeError, match="cp"):
            check_bearing_ready(Half(), "half", self.REQUIRED)

    def test_the_label_is_named_in_the_error(self):
        class Empty:
            pass

        with pytest.raises(RuntimeError, match="brg_XYZ"):
            check_bearing_ready(Empty(), "brg_XYZ", self.REQUIRED)

    def test_an_empty_requirement_list_always_passes(self):
        class Empty:
            pass

        check_bearing_ready(Empty(), "empty", ())


# ===========================================================================
# warn_if_floating_loaded
# ===========================================================================

class TestWarnIfFloatingLoaded:

    def test_a_loaded_floating_bearing_warns(self, brg_B):
        """
        A floating bearing has no shoulder to react against, so a non-zero
        Fa on one means the arrangement and the load case disagree.
        """
        assert brg_B.arrangement == "floating"
        with pytest.warns(UserWarning):
            warn_if_floating_loaded(brg_B, "B", Fa=500.0)

    def test_an_unloaded_floating_bearing_is_silent(self, brg_B):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_floating_loaded(brg_B, "B", Fa=0.0)

    def test_a_locating_bearing_never_warns(self, brg_A):
        assert brg_A.arrangement == "locating"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_floating_loaded(brg_A, "A", Fa=5000.0)

    def test_numerical_noise_is_below_the_floor(self, brg_B):
        """The 1e-6 N floor exists so FEM round-off does not trip the warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_floating_loaded(brg_B, "B", Fa=1.0e-12)

    def test_the_label_appears_in_the_warning(self, brg_B):
        with pytest.warns(UserWarning, match="B"):
            warn_if_floating_loaded(brg_B, "B", Fa=500.0)


# ===========================================================================
# run_root
# ===========================================================================

class TestRunRoot:

    def test_solves_a_linear_system(self):
        def f(u):
            return np.array([u[0] - 3.0, u[1] + 2.0])

        x, nfev, residual, ok = run_root(f, [0.0, 0.0], 1e-10)
        assert ok is True
        assert x[0] == pytest.approx(3.0)
        assert x[1] == pytest.approx(-2.0)
        assert residual < 1e-8

    def test_solves_a_nonlinear_system(self):
        """Intersection of a circle and a line: (1, 1)."""
        def f(u):
            return np.array([u[0] ** 2 + u[1] ** 2 - 2.0, u[0] - u[1]])

        x, _, residual, ok = run_root(f, [0.5, 0.4], 1e-10)
        assert ok is True
        assert abs(x[0]) == pytest.approx(1.0, rel=1e-6)
        assert residual < 1e-8

    def test_reports_the_function_evaluation_count(self):
        def f(u):
            return np.array([u[0] - 1.0])

        _, nfev, _, _ = run_root(f, [0.0], 1e-10)
        assert isinstance(nfev, (int, np.integer))
        assert nfev > 0

    def test_returns_the_residual_norm_not_the_vector(self):
        def f(u):
            return np.array([u[0] - 1.0, u[1] - 1.0])

        _, _, residual, _ = run_root(f, [0.0, 0.0], 1e-10)
        assert np.isscalar(residual) or np.ndim(residual) == 0

    def test_an_unsolvable_system_reports_failure_rather_than_raising(self):
        """
        x^2 + 1 = 0 has no real root. The helper must come back with
        ok=False and a finite residual -- the ball solver relies on being
        able to inspect a failed solve, not on catching an exception.
        """
        def f(u):
            return np.array([u[0] ** 2 + 1.0])

        x, _, residual, ok = run_root(f, [1.0], 1e-10)
        assert ok is False
        assert residual > 0.0
        assert math.isfinite(float(np.asarray(x).ravel()[0]))

    def test_an_already_converged_start_stays_put(self):
        def f(u):
            return np.array([u[0] - 5.0])

        x, _, _, ok = run_root(f, [5.0], 1e-10)
        assert ok is True
        assert x[0] == pytest.approx(5.0)


# ===========================================================================
# BearingResultBundle
# ===========================================================================

class TestBearingResultBundle:

    def test_everything_but_the_label_starts_empty(self):
        bundle = BearingResultBundle(label="A")
        assert bundle.label == "A"
        assert bundle.bearing_type is None
        assert bundle.load_distribution is None
        assert bundle.stiffness is None
        assert bundle.capacity is None
        assert bundle.dynamic_equivalent_load is None
        assert bundle.extra == {}

    def test_extra_is_per_instance_not_shared(self):
        """A mutable default shared across bundles would be a nasty bug."""
        a = BearingResultBundle(label="A")
        b = BearingResultBundle(label="B")
        a.extra["k"] = 1
        assert b.extra == {}


# ===========================================================================
# BearingResultsLibrary
# ===========================================================================

class TestBearingResultsLibrary:

    def test_starts_empty(self):
        assert list(BearingResultsLibrary().labels()) == []

    def test_unknown_label_raises(self):
        with pytest.raises(KeyError):
            BearingResultsLibrary().get("nope")

    def test_capacity_round_trip(self):
        lib = BearingResultsLibrary()
        lib.set_capacity("A", (4050.0, 9195.0))
        assert lib.get("A").capacity == (4050.0, 9195.0)

    def test_setting_a_result_creates_the_label(self):
        lib = BearingResultsLibrary()
        lib.set_stiffness("A", object())
        assert "A" in list(lib.labels())

    def test_the_setters_are_independent(self):
        """Writing one slot must not disturb the others."""
        lib = BearingResultsLibrary()
        stiffness = object()
        lib.set_capacity("A", (1.0, 2.0))
        lib.set_stiffness("A", stiffness)
        lib.set_dynamic_equivalent_load("A", ["derel"])

        bundle = lib.get("A")
        assert bundle.capacity == (1.0, 2.0)
        assert bundle.stiffness is stiffness
        assert bundle.dynamic_equivalent_load == ["derel"]

    def test_setters_overwrite_rather_than_accumulate(self):
        lib = BearingResultsLibrary()
        lib.set_capacity("A", (1.0, 2.0))
        lib.set_capacity("A", (3.0, 4.0))
        assert lib.get("A").capacity == (3.0, 4.0)

    def test_extra_is_an_open_slot_keyed_by_name(self):
        lib = BearingResultsLibrary()
        lib.set_extra("A", "lubrication", {"lambda": 2.1})
        lib.set_extra("A", "fatigue", "pending")
        assert lib.get("A").extra["lubrication"] == {"lambda": 2.1}
        assert lib.get("A").extra["fatigue"] == "pending"

    def test_labels_are_kept_apart(self):
        lib = BearingResultsLibrary()
        lib.set_capacity("A", (1.0, 2.0))
        lib.set_capacity("B", (3.0, 4.0))
        assert lib.get("A").capacity == (1.0, 2.0)
        assert lib.get("B").capacity == (3.0, 4.0)
        assert set(lib.labels()) == {"A", "B"}

    def test_a_whole_load_distribution_library_can_be_handed_over(self):
        """
        The documented primary path: one call per BearingType group, taking
        the per-type local library wholesale rather than label by label.
        """
        class LocalLib:
            def __init__(self, entries):
                self._e = entries

            def labels(self):
                return list(self._e)

            def get(self, label):
                return self._e[label]

        class FakeResult:
            def __init__(self, rows):
                self.rows = rows

        local = LocalLib({"A": FakeResult(["row_A"]), "B": FakeResult(["row_B"])})

        lib = BearingResultsLibrary()
        lib.add_load_distribution_library(BearingType.DEEP_GROOVE_BALL, local)

        assert set(lib.labels()) == {"A", "B"}
        assert lib.get("A").load_distribution == ["row_A"]
        assert lib.get("A").bearing_type is BearingType.DEEP_GROOVE_BALL

    def test_the_sub_library_can_be_retrieved_by_type(self):
        class LocalLib:
            def labels(self):
                return []

            def get(self, label):
                raise KeyError(label)

        local = LocalLib()
        lib = BearingResultsLibrary()
        lib.add_load_distribution_library(BearingType.CYLINDRICAL_ROLLER, local)
        assert lib.load_distribution_library(BearingType.CYLINDRICAL_ROLLER) is local
