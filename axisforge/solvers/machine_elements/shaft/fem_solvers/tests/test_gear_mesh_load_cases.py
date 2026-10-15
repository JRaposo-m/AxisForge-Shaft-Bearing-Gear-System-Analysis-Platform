"""What ``build_load_cases`` and ``RigidSupportFEMSolver`` do with the gear-mesh loads of a resolved
``ShaftSystem`` (finding X7 of the gears review, question Q11), on the pinion shaft of baseline case 7a: F_r 674.505 N at 210 deg
and F_tw 1659.831 N at 300 deg at x = 100 mm, axial load +449.236 N and the bending moment of the axial
force, 16442.08 N mm (Q05, Q06, José 2026-10-15).

Real objects (no fake): the shaft is built and resolved by
``axisforge.core.machine_elements.gears.tests.baseline_cases``. ``build_load_cases`` is the only
source of the load vector of ``RigidSupportFEMSolver.solve``, so a force that is not in its output
is not in the solve. Since Q11 (José, 2026-10-15) the solver drops the point loads of a flagged gear
label only when its distributed version is on the shaft.
"""
import numpy as np
import pytest

from axisforge.core.machine_elements.gears.tests import baseline_cases as bc
from axisforge.core.machine_elements.gears.tests.baseline_cases import s355  # noqa: F401 (pytest fixture)
from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.solvers.machine_elements.shaft.fem_solvers import RigidSupportFEMSolver, build_load_cases


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


def _reactions(distribute_loads, labels):
    """Radial bearing loads [N] of the pinion shaft of case 7a solved by the FEM (Euler-Bernoulli); the callers
    use the fixture ``s355`` (shaft material)."""
    sP = bc.resolved_case7("pinion", 1, distribute_loads=distribute_loads).sP
    solver = RigidSupportFEMSolver(BeamModelSettings("euler_bernoulli", None, None), distribute_gear_labels=labels)
    return np.asarray(solver.solve(sP).R)


def test_build_load_cases_drops_the_point_loads_of_a_flagged_label():
    """``build_load_cases`` itself leaves out the point loads of the labels it is given; the solver passes only
    the labels whose distributed load is present (Q11)."""
    cases = _pinion_shaft_load_cases(False, {"g_pinion"})
    assert [c["label"] for c in cases] == ["g_pinion:Fa", "g_pinion:M"] and not _has_lateral_force(cases)


@pytest.mark.parametrize("labels", [{"g_pinion"}, {"*"}])
def test_flagged_label_without_distributed_load_keeps_its_point_loads(labels, s355):
    """Q11 (José, 2026-10-15): with ``distribute_gear_labels`` set and the link resolved with
    ``distribute_loads=False`` (no distributed load), the solver uses the point loads: the reactions are those of
    ``distribute_gear_labels=None``, 865.581 / 934.053 N (with the D-0010 loads and the bending moment of F_a,
    block 3; 895.82 / 895.82 N before them)."""
    reference = _reactions(False, None)
    assert reference == pytest.approx([865.581, 934.053], abs=5e-4)
    assert _reactions(False, labels) == pytest.approx(reference, rel=1e-12)


def test_flagged_label_with_neither_point_nor_distributed_load_warns(s355):
    """Q11: a flagged label that has no gear-mesh load on the shaft gives a ``UserWarning``; the solve goes on."""
    with pytest.warns(UserWarning, match="'g_absent' has no gear-mesh load"):
        R = _reactions(False, {"g_absent"})
    assert R == pytest.approx(_reactions(False, None), rel=1e-12)


def test_distributed_loads_assembled_when_link_distributed():
    """The supported path: ``distribute_loads=True`` on the link stores ``DistributedRadialLoad``
    objects over the face width, which are assembled whatever the labels."""
    cases = _pinion_shaft_load_cases(True, {"g_pinion"})
    assert [c["label"] for c in cases] == ["g_pinion:Fr", "g_pinion:Ft", "g_pinion:Fa", "g_pinion:M"]
    assert all(c["distributed_xz"] and not c["radial_xz"] for c in cases[:2])
