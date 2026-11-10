"""A two-stage cylindrical transmission written as a power path, resolved, and its three loaded
shafts solved by the shaft FEM (the "done when" of step 5 of the gears refactor).

Real objects (no fake). Stage 1 is the helical pair of baseline case 2 (H501, z 20/30, mn 3.5 mm,
beta 15 deg, right-hand pinion, left-hand wheel) at x = 100 mm, line of centres at 30 deg; stage 2 is
the spur pair of case 1 (C14, z 16/24, mn 4.5 mm) at x = 50 mm, line of centres at 90 deg. 9224.5 W
enter shaft 1 at 1450 rpm (60.75 N m); shaft 3 is closed by a sink at its right end. Shafts and
bearings are those of baseline case 7 (``baseline_cases._shaft_system``: stepped shaft of 200 mm, deep
groove ball bearing at x = 10 mm, locating, cylindrical roller bearing at x = 190 mm).

What the tests prove (Q05, Q06, José 2026-10-15): the path builds and resolves with the D-0010 loads
(every closure check passes); for each shaft, the six equilibrium equations hold between the stored
load list (including the bending moment of the axial force, ``ExternalMoment``, mapped as the FEM
consumes it) and the bearing reactions of the FEM; those reactions equal a hand solution of the
statically determinate two-support shaft built from the ``MeshForces`` alone (3D force at the pitch
point, independent of the FEM and of the ``ExternalMoment`` mapping): radial bearing loads 865.581 /
934.053, 1946.975 / 990.089 and 2095.100 / 598.600 N (vault statics, ``questions.md`` rows Q05 added in
step 5 and Q06, script ``questions_values_step5.py``; physics review of block 3). Shaft 2, with two
meshes, had 1825.8 / 793.6 N with the loads before the rulings.

Units: forces [N], torques [N m], bending moments [N mm], positions [mm], speeds [rpm].
"""
import numpy as np
import pytest

from axisforge.core.loads import AxialLoad, ExternalMoment, RadialLoad, TorqueLoad
from axisforge.core.machine_elements.gears import GearPair
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc
from axisforge.core.machine_elements.gears.tests.baseline_cases import s355  # noqa: F401 (pytest fixture)
from axisforge.core.mechanical_system import GearElement, PairStage, Transmission
from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.solvers.machine_elements.shaft.fem_solvers import RigidSupportFEMSolver

X1, X2 = 100.0, 50.0        # [mm] axial stations of stage 1 and of stage 2
X_SINK = 200.0              # [mm] output end of shaft 3


@pytest.fixture(scope="module")
def two_stage(s355):
    """``(transmission, resolved)`` of the two-stage system described in the module docstring (shaft material
    S355: fixture ``s355`` of baseline_cases)."""
    s1, s2, s3 = (bc._shaft_system(name) for name in ("shaft_1", "shaft_2", "shaft_3"))
    p1, w1 = bc.pinion_wheel("H501", position=X1)
    p2, w2 = bc.pinion_wheel("C14", position=X2)
    elements = {name: GearElement(g, label=name) for name, g in (("g_p1", p1), ("g_w1", w1), ("g_p2", p2), ("g_w2", w2))}
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


def _solve(shaft):
    return RigidSupportFEMSolver(BeamModelSettings("euler_bernoulli", None, None)).solve(shaft)


def _applied(shaft):
    """Resultant of the applied loads of one shaft about the origin of its axis (x = 0) [N], [N mm].

    A force (F_x, F_y, F_z) at the axial station x gives the moment r x F with r = (x, 0, 0):
    (0, -x F_z, x F_y). An ``ExternalMoment`` of magnitude M at theta is, as the FEM consumes it, the
    vector M (cos(theta + 90), sin(theta + 90)) in (Y, Z) (``core/loads.py``, Warnings). A
    ``TorqueLoad`` [N m] is a moment about X. Returns (F, M) as numpy arrays (X, Y, Z).
    """
    F, M = np.zeros(3), np.zeros(3)
    for ld in shaft.loads:
        if isinstance(ld, RadialLoad):
            f = np.array([0.0, ld.Fy, ld.Fz])
        elif isinstance(ld, AxialLoad):
            f = np.array([ld.magnitude, 0.0, 0.0])
        elif isinstance(ld, TorqueLoad):
            M[0] += 1000.0 * ld.magnitude                                  # [N m] -> [N mm]
            continue
        elif isinstance(ld, ExternalMoment):
            t = np.radians(ld.theta_deg + 90.0)
            M[1:] += ld.magnitude * np.array([np.cos(t), np.sin(t)])
            continue
        else:
            raise AssertionError(f"unexpected load {ld!r}")
        F += f
        M += np.cross([ld.position, 0.0, 0.0], f)
    return F, M


def _hand_reactions(two_stage, index, x_b):
    """Bearing reactions (R_y, R_z) [N] of shaft ``index`` on two simple supports at x_b = (x_A, x_B) [mm],
    from the ``MeshForces`` of the stages alone (not from the stored loads, so independent of the FEM AND of the
    mapping of ``ExternalMoment``): each mesh force on a gear of this shaft is a 3D force F = F_c e_c + F_t e_t +
    F_x e_x at the pitch point C = O + rho e_c (O: the axis at the gear station; e_c from the axis of gear 1 to
    that of gear 2, at phi or phi + 180 deg; e_t = e_x x e_c), with C placed from the axis positions
    ``shaft_position`` of both shafts; its moment r_C x F about the origin of the shaft axis enters sum M = 0,
    the force sum F = 0 (statically determinate: the bearings take no moment)."""
    transmission, resolved = two_stage
    shaft = resolved.shafts[index]
    F, M = np.zeros(3), np.zeros(3)
    for stage, mesh in zip(transmission.path, resolved.meshes):
        a_is_gear2 = stage.gear_a.gear is stage.pair.gear2
        theta = np.radians(stage.phi_deg + (180.0 if a_is_gear2 else 0.0))
        e_c, e_t = np.array([0.0, np.cos(theta), np.sin(theta)]), np.array([0.0, -np.sin(theta), np.cos(theta)])
        i_a, i_b = (transmission.shafts.index(s) for s in stage.shafts)
        C_global = []
        for i, element, side in ((i_a, stage.gear_a, 2 if a_is_gear2 else 1), (i_b, stage.gear_b, 1 if a_is_gear2 else 2)):
            rho = mesh.rho_mm[side - 1]
            y, z = resolved.shafts[i].shaft_position
            C_global.append(np.array([element.position, y, z]) + rho * e_c)
            if i == index:
                F_c, F_t, F_x = mesh.on_1 if side == 1 else mesh.on_2
                f = F_c * e_c + F_t * e_t + F_x * np.array([1.0, 0.0, 0.0])
                r_C = C_global[-1] - np.array([0.0, y, z])          # from the origin of this shaft axis
                F, M = F + f, M + np.cross(r_C, f)
        assert C_global[0] == pytest.approx(C_global[1], abs=1e-9)  # one pitch point, seen from both shafts
    A = np.array([[1.0, 1.0], [x_b[0], x_b[1]]])
    R_y = np.linalg.solve(A, [-F[1], -M[2]])          # sum F_y = 0; sum M_z = x R_y + M_z = 0
    R_z = np.linalg.solve(A, [-F[2], M[1]])           # sum F_z = 0; sum M_y = -x R_z + M_y = 0
    return R_y, R_z


def test_two_stages_resolve_along_the_written_path(two_stage):
    """Torque, speed and sense of rotation down the path: 60.75 N m at 1450 rpm; the torque delivered to each
    output shaft is -T_a z_b / z_a (signed, T omega > 0 on every shaft), the speed the inverse ratio; the sense
    alternates. Every closure check passes (power conserved, action = reaction, separating radial forces)."""
    transmission, resolved = two_stage
    assert transmission.validate() == []
    assert resolved.torque_Nm == pytest.approx((60.75, -91.125, 136.6875), rel=1e-12)
    assert resolved.speed_rpm == pytest.approx((1450.0, 966.6666666666666, 644.4444444444445), rel=1e-12)
    assert resolved.rotation == (1, -1, 1)
    assert all(T * rot > 0.0 for T, rot in zip(resolved.torque_Nm, resolved.rotation))
    # shaft 1: input + helical mesh (Fr, Ft, Fa, T, M); shaft 2: helical (5) + spur (3); shaft 3: spur (3) + sink
    assert [len(s.loads) for s in resolved.shafts] == [6, 8, 4]
    assert len(resolved.meshes) == 2 and resolved.meshes[1].F_a_N == 0.0
    # axis positions [mm]: a_w1 along 30 deg, then a_w2 along 90 deg
    a1, a2 = (stage.pair.al for stage in transmission.path)
    assert resolved.shafts[1].shaft_position == pytest.approx((a1 * 3 ** 0.5 / 2, a1 / 2), rel=1e-12)
    assert resolved.shafts[2].shaft_position == pytest.approx((a1 * 3 ** 0.5 / 2, a1 / 2 + a2), rel=1e-12)
    assert all(ok for _, _, ok in resolved.checks), [name for name, _, ok in resolved.checks if not ok]
    assert [s.loads for s in transmission.shafts] == [[], [], []]   # the user's shafts are untouched


@pytest.mark.parametrize("index", [0, 1, 2])
def test_each_loaded_shaft_solved_by_the_fem_is_in_equilibrium(two_stage, index):
    """The six equilibrium equations of each shaft, from its load list (forces, torques and the bending moment
    of the axial force) and the bearing reactions of the FEM: sum of forces zero in X, Y and Z, sum of moments
    about the origin of the axis zero about X, Y and Z (``_applied``; the bearings take no moment and no torque).
    The reactions are those of a hand solution of the two-support shaft from the ``MeshForces`` alone
    (``_hand_reactions``), whose radial bearing loads are the vault statics (physics review of block 3):
    shaft 1 865.581 / 934.053 N, shaft 2 1946.975 / 990.089 N, shaft 3 2095.100 / 598.600 N.
    Tolerance: 1e-9 of the largest load (linear solve of a small system)."""
    shaft = two_stage[1].shafts[index]
    result = _solve(shaft)
    x_b = np.asarray(result.bearing_positions)
    assert list(x_b) == [10.0, 190.0]
    F, M = _applied(shaft)
    R = np.array([np.asarray(result.R_axial), np.asarray(result.R_xy), np.asarray(result.R_xz)])   # (X, Y, Z) x bearing
    scale = max(ld.magnitude for ld in shaft.radial_loads)                       # [N] largest tooth force
    assert F + R.sum(axis=1) == pytest.approx(np.zeros(3), abs=1e-9 * scale)
    M_R = sum(np.cross([x, 0.0, 0.0], R[:, k]) for k, x in enumerate(x_b))
    assert M + M_R == pytest.approx(np.zeros(3), abs=1e-9 * 200.0 * scale)
    R_y, R_z = _hand_reactions(two_stage, index, x_b)
    assert np.asarray(result.R_xy) == pytest.approx(R_y, rel=1e-9, abs=1e-9 * scale)
    assert np.asarray(result.R_xz) == pytest.approx(R_z, rel=1e-9, abs=1e-9 * scale)
    expected = ((865.581, 934.053), (1946.975, 990.089), (2095.100, 598.600))[index]
    assert np.hypot(R_y, R_z) == pytest.approx(expected, abs=5e-4)
    # the axial thrust of the helical stage goes to the locating (ball) bearing alone
    assert result.R_axial[1] == 0.0 and (F[0] == 0.0) == (index == 2)
    # torsion [N m]: the largest torque of the shaft is the torque that enters it
    T_in = abs(two_stage[1].torque_Nm[index])
    assert np.max(np.abs(result.T)) == pytest.approx(T_in, rel=1e-9)
    assert result.R[0] > 0.0 and result.R[1] > 0.0 and result.v_max > 0.0


def test_bearing_loads_of_the_shaft_with_two_meshes(two_stage):
    """Q05, Q06: shaft 2 carries the wheel of the helical stage (with the bending moment F_a r_w = 24663.1 N mm)
    and the pinion of the spur stage. Radial bearing loads 1946.975 N (ball, x = 10 mm) and 990.089 N (roller,
    x = 190 mm), the values of the vault statics on two supports; without the moment they would be 2061.2 /
    944.1 N, with the loads before the rulings 1825.8 / 793.6 N."""
    result = _solve(two_stage[1].shafts[1])
    assert np.asarray(result.R) == pytest.approx([1946.9751042864077, 990.0892813513829], rel=1e-9)
    assert two_stage[1].shafts[1].external_moments[0].magnitude == pytest.approx(24663.121101417102, rel=1e-9)
