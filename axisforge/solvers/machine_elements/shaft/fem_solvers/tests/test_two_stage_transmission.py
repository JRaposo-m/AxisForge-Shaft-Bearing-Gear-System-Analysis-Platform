"""A two-stage cylindrical transmission written as a power path, resolved, and its three loaded
shafts solved by the shaft FEM (the "done when" of step 5 of the gears refactor).

Real objects (no fake). Stage 1 is the helical pair of baseline case 2 (H501, z 20/30, mn 3.5 mm,
beta 15 deg) at x = 100 mm, line of centres at 30 deg; stage 2 is the spur pair of case 1 (C14,
z 16/24, mn 4.5 mm) at x = 50 mm, line of centres at 90 deg. 9224.5 W enter shaft 1 at 1450 rpm
(60.75 N m); shaft 3 is closed by a sink at its right end. Shafts and bearings are those of
baseline case 7 (``baseline_cases._shaft_system``: stepped shaft of 200 mm, deep groove ball
bearing at x = 10 mm, locating, cylindrical roller bearing at x = 190 mm).

What the test proves: the path builds and resolves, and for each shaft, taken alone, the reactions
of the FEM balance the loads stored on it TODAY (forces and moments).

What it does not prove: that those loads are right. The mesh loads of the three shafts together sum
to (0, 3319.673, 2874.912) N instead of zero: 2 F_t per stage, because both stored F_t of a stage
point the same way (Q05). Shaft 2 carries two meshes, so the MAGNITUDES of its bearing loads are open
too, not only their directions: radial bearing loads 1825.8 / 793.6 N today, against 1947.0 / 990.1 N
with the loads of the D-0010 crossing including the moment F_a r_w, and 2061.2 / 944.1 N without that
moment (statics on two supports; vault script ``questions_values_step5.py``). No moment F_a r_w is
stored (Q06). The SIGN of the torque diagram is not asserted, only its largest magnitude (Q05).

Units: forces [N], torques [N m], positions [mm], speeds [rpm].
"""
import numpy as np
import pytest

from axisforge.core.loads import AxialLoad, RadialLoad, TorqueLoad
from axisforge.core.machine_elements.gears import GearPair
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc
from axisforge.core.materials.base import IsotropicElastic, Material, StrengthProperties, available_materials, register
from axisforge.core.mechanical_system import GearElement, PairStage, Transmission
from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.solvers.machine_elements.shaft.fem_solvers import RigidSupportFEMSolver

X1, X2 = 100.0, 50.0        # [mm] axial stations of stage 1 and of stage 2
X_SINK = 200.0              # [mm] output end of shaft 3


@pytest.fixture(scope="module")
def two_stage():
    """``(transmission, resolved)`` of the two-stage system described in the module docstring."""
    if "S355" not in available_materials():
        # the default material of a Shaft; AxisForge registers none. Values of the shaft validation
        # cases of AxisForge-Design-Studies (EN 10025-2 S355: E 210 GPa, nu 0.3)
        register(Material(material_id="S355", density=7850.0, elastic=IsotropicElastic(E=210_000.0, poisson_ratio=0.3),
                          strength=StrengthProperties(Sut=590.0, Sy=355.0), description="EN 10025-2 S355"))
    s1, s2, s3 = (bc._shaft_system(name) for name in ("shaft_1", "shaft_2", "shaft_3"))
    p1, w1 = bc.pinion_wheel("H501", position=X1)
    p2, w2 = bc.pinion_wheel("C14", position=X2)
    elements = {name: GearElement(g, role=role, label=name) for name, g, role in (
        ("g_p1", p1, "driver"), ("g_w1", w1, "driven"), ("g_p2", p2, "driver"), ("g_w2", w2, "driven"))}
    s1.add_gear(elements["g_p1"])
    s2.add_gear(elements["g_w1"])
    s2.add_gear(elements["g_p2"])       # shaft 2 carries the wheel of stage 1 and the pinion of stage 2
    s3.add_gear(elements["g_w2"])
    path = [
        PairStage(pair=GearPair.assemble(bc.FAMILY, gear1=p1, gear2=w1, label="stage1"), shaft_a=s1,
                  gear_a=elements["g_p1"], shaft_b=s2, gear_b=elements["g_w1"], phi_deg=30.0, label="stage1"),
        PairStage(pair=GearPair.assemble(bc.FAMILY, gear1=p2, gear2=w2, label="stage2"), shaft_a=s2,
                  gear_a=elements["g_p2"], shaft_b=s3, gear_b=elements["g_w2"], phi_deg=90.0, label="stage2"),
    ]
    transmission = Transmission([s1, s2, s3], path, source=s1, sinks=((s3, X_SINK),), label="two_stage")
    return transmission, transmission.resolve(bc.T_NM * bc.OMEGA, bc.N_RPM, 1)


def test_two_stages_resolve_along_the_written_path(two_stage):
    """Torque, speed and sense of rotation down the path: 60.75 N m at 1450 rpm; x 30/20 and
    x 24/16 in torque, the inverse in speed; the sense alternates. The power is conserved."""
    transmission, resolved = two_stage
    assert transmission.warnings == () and transmission.validate() == []
    assert resolved.torque_Nm == pytest.approx((60.75, 91.125, 136.6875), rel=1e-12)
    assert resolved.speed_rpm == pytest.approx((1450.0, 966.6666666666666, 644.4444444444445), rel=1e-12)
    assert resolved.rotation == (1, -1, 1)
    assert [len(s.loads) for s in resolved.shafts] == [5, 7, 4]     # shaft 2: two meshes (4 + 3 loads: stage 2 is spur)
    assert len(resolved.meshes) == 2 and resolved.meshes[1].F_a_N == 0.0
    # axis positions [mm]: a_w1 along 30 deg, then a_w2 along 90 deg
    a1, a2 = (stage.pair.al for stage in transmission.path)
    assert resolved.shafts[1].shaft_position == pytest.approx((a1 * 3 ** 0.5 / 2, a1 / 2), rel=1e-12)
    assert resolved.shafts[2].shaft_position == pytest.approx((a1 * 3 ** 0.5 / 2, a1 / 2 + a2), rel=1e-12)
    closure = {name: ok for name, _, ok in resolved.checks}
    assert closure["train: power leaving the end shafts - power entering [W]"]
    assert all(closure[f"shaft '{s.name}': sum of the torque loads [N m]"] for s in resolved.shafts)
    assert [s.loads for s in transmission.shafts] == [[], [], []]   # the user's shafts are untouched


@pytest.mark.parametrize("index", [0, 1, 2])
def test_each_loaded_shaft_solved_by_the_fem_is_in_equilibrium(two_stage, index):
    """Reactions of the two bearings against the applied loads of the shaft, in the global frame
    (X along the shaft; a load at theta has F_y = F cos(theta) in the XY plane and F_z = F sin(theta)
    in the XZ plane): sum of forces zero in X, Y and Z; sum of moments about the left end zero in both
    bending planes; torque loads summing to zero, and the torque diagram of the solver within the
    torque that enters the shaft. Tolerance: 1e-9 of the largest load (linear solve of a small system)."""
    shaft = two_stage[1].shafts[index]
    result = RigidSupportFEMSolver(BeamModelSettings("euler_bernoulli", None, None)).solve(shaft)

    radial = [ld for ld in shaft.loads if isinstance(ld, RadialLoad)]
    Fy, Fz = sum(ld.Fy for ld in radial), sum(ld.Fz for ld in radial)
    Fx = sum(ld.magnitude for ld in shaft.loads if isinstance(ld, AxialLoad))
    scale = max(ld.magnitude for ld in radial)                       # [N] largest tooth force on this shaft
    x_b = np.asarray(result.bearing_positions)
    assert list(x_b) == [10.0, 190.0]
    assert np.sum(result.R_xy) + Fy == pytest.approx(0.0, abs=1e-9 * scale)
    assert np.sum(result.R_xz) + Fz == pytest.approx(0.0, abs=1e-9 * scale)
    assert np.sum(result.R_axial) + Fx == pytest.approx(0.0, abs=1e-9 * scale)
    # moments about the left end [N mm]: lever arm x of every transverse force
    arm = 200.0 * scale
    assert np.dot(x_b, result.R_xy) + sum(ld.position * ld.Fy for ld in radial) == pytest.approx(0.0, abs=1e-9 * arm)
    assert np.dot(x_b, result.R_xz) + sum(ld.position * ld.Fz for ld in radial) == pytest.approx(0.0, abs=1e-9 * arm)
    # the axial thrust of the helical stage goes to the locating (ball) bearing alone
    assert result.R_axial[1] == 0.0 and (Fx == 0.0) == (index == 2)
    # torsion [N m]
    T_in = abs(two_stage[1].torque_Nm[index])
    assert sum(ld.magnitude for ld in shaft.loads if isinstance(ld, TorqueLoad)) == pytest.approx(0.0, abs=1e-9 * T_in)
    assert np.max(np.abs(result.T)) == pytest.approx(T_in, rel=1e-9)
    assert result.R[0] > 0.0 and result.R[1] > 0.0 and result.v_max > 0.0
