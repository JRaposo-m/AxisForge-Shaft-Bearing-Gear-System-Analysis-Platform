"""What ``build_load_cases`` does with the gear-mesh loads of a resolved ``ShaftSystem`` (finding X7
of the gears review, question Q11), on the pinion shaft of baseline case 7a: F_r 674.505 N at 210 deg
and F_tw 1659.831 N at 300 deg at x = 100 mm, axial load +449.236 N and the bending moment of the axial
force, 16442.08 N mm (Q05, Q06, José 2026-10-15).

Real objects (no fake): the shaft is built and resolved by
``axisforge.core.machine_elements.gears.tests.baseline_cases``. ``build_load_cases`` is the only
source of the load vector of ``RigidSupportFEMSolver.solve``, so a force that is not in its output
is not in the solve.
"""
import pytest

from axisforge.core.machine_elements.gears.tests import baseline_cases as bc
from axisforge.solvers.machine_elements.shaft.fem_solvers import build_load_cases


def _pinion_shaft_load_cases(distribute_loads, labels):
    """Load cases of the pinion shaft of case 7a; ``labels`` is the ``distribute_gear_labels`` of the solver."""
    sP = bc.resolved_case7("pinion", 1, distribute_loads=distribute_loads).sP
    # same two lines as RigidSupportFEMSolver.__init__ (rigid_support.py:176-177)
    return build_load_cases(sP, distribute_all=labels == {"*"}, distribute_labels=labels or set())


def _has_lateral_force(cases):
    return any(c["radial_xz"] or c["radial_xy"] or c.get("distributed_xz") or c.get("distributed_xy") for c in cases)


def test_point_loads_assembled_by_default():
    cases = _pinion_shaft_load_cases(False, None)
    assert [c["label"] for c in cases] == ["g_pinion:Fr", "g_pinion:Ft", "g_pinion:Fa", "g_pinion:M"]


@pytest.mark.parametrize("labels", [{"g_pinion"}, {"*"}])
def test_point_loads_dropped_when_label_flagged_and_link_not_distributed(labels):
    """Pins TODAY's behaviour (Q11 / X7): with ``distribute_gear_labels`` set and the link resolved
    with ``distribute_loads=False``, Fr and Ft are skipped (vector_external_forces.py:86-91) and no
    distributed counterpart exists, so the shaft is solved with no lateral force at all; only the
    axial load and the bending moment remain."""
    cases = _pinion_shaft_load_cases(False, labels)
    assert [c["label"] for c in cases] == ["g_pinion:Fa", "g_pinion:M"]
    assert not _has_lateral_force(cases)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Q11: X7, FEM drops gear-mesh point loads under distribute_gear_labels")
@pytest.mark.parametrize("labels", [{"g_pinion"}, {"*"}])
def test_alternative_flagged_mesh_force_still_loads_the_shaft(labels):
    """The alternative: a flagged gear force reaches the solve in some form (point or distributed)."""
    assert _has_lateral_force(_pinion_shaft_load_cases(False, labels))


def test_distributed_loads_assembled_when_link_distributed():
    """The supported path: ``distribute_loads=True`` on the link stores ``DistributedRadialLoad``
    objects over the face width, which are assembled whatever the labels."""
    cases = _pinion_shaft_load_cases(True, {"g_pinion"})
    assert [c["label"] for c in cases] == ["g_pinion:Fr", "g_pinion:Ft", "g_pinion:Fa", "g_pinion:M"]
    assert all(c["distributed_xz"] and not c["radial_xz"] for c in cases[:2])
