"""Tests of ``axisforge.solvers.machine_elements.gears.contact`` (``load_sharing``, ``contact_inputs``; moved
from ``CylindricalFamily`` by Q31, José 2026-10-15): closed forms of ISO_21771_2007 (5.4.7, 5.4.8, 5.6), bad
inputs, the load of ``MeshForces``, and GEARpie's load per unit length as LITERAL values (copied by
``scripts/questions_values_step4.py``, vault). The pairs are those of the geometry tests of the family
(``core/machine_elements/gears/tests/test_contact_inputs.py``).

Position along the path: g_y [mm] from the pitch point C, positive toward the tip of gear 1.
Units: lengths [mm], speeds [m/s], forces [N], load per unit length [N/mm], time [s].
"""
import math
import warnings

import numpy as np
import pytest

from axisforge.core.machine_elements.gears.tests import baseline_cases as bc
from axisforge.core.machine_elements.gears.tests.test_contact_inputs import (
    CASES,
    FAMILY,
    N1,
    OMEGA1,
    pair,
    same,
)
from axisforge.solvers.machine_elements.gears import contact_inputs, load_sharing

F_BN_CASE2 = 1847.1085622260473            # [N] F_bt / cos(beta_b) of case 2 for 60.75 N m (the alternative of Q07b)


# ----------------------------------------------------------------------------------------------
# load sharing
# ----------------------------------------------------------------------------------------------
def test_spur_pair_shares_one_half_in_double_contact_and_all_in_single_contact():
    p = CASES["case1"]
    g = FAMILY.path_points(p, "gear1")
    ls = load_sharing(p, np.array([g["A"], (g["A"] + g["B"]) / 2, g["C"], (g["D"] + g["E"]) / 2, g["E"]]))
    assert list(ls["share"]) == [0.5, 0.5, 1.0, 0.5, 0.5]
    assert list(ls["n_pairs"]) == [2, 2, 1, 2, 2]
    assert list(ls["l_mm"]) == [p.b] * 5 and list(ls["l_sum_mm"]) == [2 * p.b, 2 * p.b, p.b, 2 * p.b, 2 * p.b]
    outside = load_sharing(p, np.array([g["A"] - 0.01, g["E"] + 0.01]))
    assert list(outside["share"]) == [0.0, 0.0] and list(outside["l_mm"]) == [0.0, 0.0]


def test_single_contact_zone_length_gives_the_transverse_contact_ratio():
    """1 < eps_alpha < 2: the zone B..D of single contact is (2 - eps_alpha) p_bt long (ISO_21771_2007, Eq. (90))."""
    p = CASES["case1"]
    g = FAMILY.path_points(p, "gear1")
    assert same(2.0 - (g["D"] - g["B"]) / p.gear1.pbt, p.epslon_alpha)
    grid = np.linspace(g["A"], g["E"], 200001)
    single = load_sharing(p, grid)["n_pairs"] == 1
    assert grid[single].min() == pytest.approx(g["B"], abs=1e-3) and grid[single].max() == pytest.approx(g["D"], abs=1e-3)


def test_pair_with_a_contact_ratio_above_two_shares_between_two_and_three_tooth_pairs():
    """Internal spur pair z 40/-120, mn 2 mm, alpha_n 15 deg (no baseline case), eps_alpha 2.48: no zone of
    single contact."""
    p = pair(bc.gear(2.0, 40, b=20.0, alpha_n_deg=15.0), bc.gear(2.0, -120, b=20.0, alpha_n_deg=15.0))
    assert 2.0 < p.epslon_alpha < 3.0 and p.T1A > 0.0
    g = FAMILY.path_points(p, "gear1")
    ls = load_sharing(p, np.linspace(g["A"], g["E"], 2001))
    assert set(ls["n_pairs"]) == {2, 3} and set(ls["share"]) == {0.5, 1.0 / 3.0}


@pytest.mark.parametrize("case", CASES)
def test_shares_of_the_tooth_pairs_in_contact_add_up_to_one(case):
    p = CASES[case]
    g = FAMILY.path_points(p, "gear1")
    g_y = np.linspace(g["A"], g["E"], 997)
    shares = sum(load_sharing(p, g_y + k * p.gear1.pbt)["share"] for k in range(-6, 7))
    assert np.all(shares == pytest.approx(1.0, rel=1e-12))


@pytest.mark.parametrize("case", CASES)
def test_path_length_and_contact_lines_agree_with_the_contact_ratio(case):
    """eps_alpha = g_alpha / p_bt (ISO_21771_2007, Eq. (90)); the sum of the contact lines averaged over one base
    pitch is eps_alpha b / cos(beta_b); one tooth pair carries the whole force over one base pitch of travel."""
    p = CASES[case]
    pbt = p.gear1.pbt
    assert same(p.AE / pbt, p.epslon_alpha)
    one_pitch = np.linspace(0.0, pbt, 100000, endpoint=False)      # periodic in p_bt: the plain mean is the integral
    assert load_sharing(p, one_pitch)["l_sum_mm"].mean() == pytest.approx(
        p.epslon_alpha * p.b / math.cos(p.gear1.betab), rel=1e-4)
    reach = p.AE + p.b * math.tan(p.gear1.betab)                   # a helical line touches before A and after E
    travel = np.linspace(-p.AC - reach, -p.AC + 2 * reach, 300001)
    share = load_sharing(p, travel)["share"]
    assert share.sum() * (travel[1] - travel[0]) == pytest.approx(pbt, rel=1e-4)


def test_helical_contact_line_is_never_longer_than_l_max():
    """ISO_21771_2007, Eq. (98)-(99): l_max = min(g_alpha / sin(beta_b), b / cos(beta_b)). Units guard [mm], case 2."""
    p = CASES["case2"]
    g = FAMILY.path_points(p, "gear1")
    ls = load_sharing(p, np.array([g["A"], g["C"], g["E"]]))
    l_max = min(p.AE / math.sin(p.gear1.betab), p.b / math.cos(p.gear1.betab))
    assert same(ls["l_mm"][1], l_max) and same(l_max, 23.711986287751117)
    # Q36: reference section at mid face width, so half of the line is already in contact at A and still at E
    assert same(ls["l_mm"][0], 11.855993143875557) and same(ls["l_mm"][2], ls["l_mm"][0])
    assert same(ls["share"][1], 0.9415836882820195) and same(ls["l_sum_mm"][1], 25.18308949363298)


# ----------------------------------------------------------------------------------------------
# one record per step
# ----------------------------------------------------------------------------------------------
def test_contact_inputs_gives_one_record_per_step_from_A_to_E():
    p = CASES["case2"]
    out = contact_inputs(p, "gear1", N1, F_BN_CASE2, n_points=5)
    assert list(out) == ["s_mm", "t_s", "g_y_mm", "rho1_t_mm", "rho2_t_mm", "rho_red_t_mm", "rho1_n_mm", "rho2_n_mm",
                         "rho_red_n_mm", "u1_m_per_s", "u2_m_per_s", "vg_m_per_s", "ue_m_per_s", "share", "l_mm", "l_sum_mm", "n_pairs",
                         "w_N_per_mm"]
    assert all(len(v) == 5 for v in out.values())
    g = FAMILY.path_points(p, "gear1")
    assert out["s_mm"][0] == 0.0 and same(out["s_mm"][-1], p.AE)
    assert same(out["g_y_mm"][0], g["A"]) and same(out["g_y_mm"][-1], g["E"])
    for name, function in (("rho_red_n_mm", FAMILY.curvature_radii), ("share", load_sharing)):
        assert np.array_equal(out[name], function(p, out["g_y_mm"])[name])
    assert np.array_equal(out["ue_m_per_s"], FAMILY.surface_speeds(p, out["g_y_mm"], N1)["ue_m_per_s"])
    # load per unit length = share x F_bn / length of the contact line of this tooth pair
    assert np.all(same(out["w_N_per_mm"] * out["l_mm"], out["share"] * F_BN_CASE2))


def test_contact_inputs_pinned_on_case_2():
    """Units guard: time [s] of the travel A..E at 1450 rpm, load per unit length [N/mm] for F_bn = 1847.1 N."""
    p = CASES["case2"]
    out = contact_inputs(p, "gear1", N1, F_BN_CASE2, n_points=5)
    assert same(out["t_s"][-1], 0.0030445125489303704)
    # ISO_21771_2007, Eq. (105): the contact point moves along the line of action at v_n = omega1 rb1
    assert same(out["t_s"][-1], p.AE / (OMEGA1 * p.gear1.rb))
    assert same(out["w_N_per_mm"][0], 51.93178223060986) and same(out["w_N_per_mm"][1], 46.05440498753953)
    assert np.all(same(contact_inputs(p, "gear1", N1, 2.0 * F_BN_CASE2, n_points=5)["w_N_per_mm"], 2.0 * out["w_N_per_mm"]))


def test_contact_inputs_at_given_times_are_those_at_the_same_positions():
    p = CASES["case1"]
    by_position = contact_inputs(p, "gear1", N1, 1000.0, n_points=7)
    by_time = contact_inputs(p, "gear1", N1, 1000.0, t_s=by_position["t_s"])
    for name in by_position:
        assert np.all(same(by_time[name], by_position[name])), name
    # twice the speed: the same positions in half the time, twice the surface speeds
    fast = contact_inputs(p, "gear1", 2 * N1, 1000.0, t_s=by_position["t_s"] / 2)
    assert np.all(same(fast["g_y_mm"], by_position["g_y_mm"])) and np.all(same(fast["u1_m_per_s"], 2 * by_position["u1_m_per_s"]))


def test_times_with_gear_2_driving_start_at_the_tip_of_gear_1():
    """t = 0 at A, which is on the tip circle of the driven gear 1; the travel takes the same time."""
    p = CASES["case2"]
    t_AE = p.AE / (OMEGA1 * p.gear1.rb)
    out = contact_inputs(p, "gear2", N1, F_BN_CASE2, t_s=[0.0, t_AE / 2, t_AE])
    g = FAMILY.path_points(p, "gear2")
    assert same(out["g_y_mm"][0], g["A"]) and same(out["g_y_mm"][2], g["E"]) and g["A"] > 0.0 > g["E"]
    assert same(math.hypot(out["rho1_t_mm"][0], p.gear1.rb), p.da1_w / 2.0)
    assert same(out["s_mm"][1], p.AE / 2) and same(out["g_y_mm"][1], g["A"] - p.AE / 2)
    assert out["vg_m_per_s"][0] > 0.0 > out["vg_m_per_s"][2]            # flank 1 the faster at its own tip
    scalar = contact_inputs(p, "gear2", N1, F_BN_CASE2, t_s=t_AE / 2)
    assert scalar["g_y_mm"].shape == (1,) and same(scalar["w_N_per_mm"][0], out["w_N_per_mm"][1])


def test_unchecked_pair_beyond_a_base_tangent_point_is_returned_as_it_is():
    """Q39: ``validate_pair`` refuses a path that passes a base-tangent point, so ``contact_inputs`` gives no
    warning. Old case 4, an equivalence record built with check=False, starts 0.075 mm beyond T1: its first step
    has rho1 < 0, where flank 1 has no involute, and the numbers are returned as they are (at A: reduced radius
    -0.0745 mm, u1 -0.0114 m/s)."""
    p = pair(*bc.make_internal(0.0), check=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = contact_inputs(p, "gear1", N1, 1000.0, n_points=51)
    assert out["rho1_t_mm"][0] == pytest.approx(-0.07492456399709191, rel=1e-9) and same(out["rho1_t_mm"][0], p.T1A)
    assert out["rho_red_t_mm"][0] < 0.0 and out["u1_m_per_s"][0] < 0.0
    assert out["rho_red_t_mm"][0] == pytest.approx(-0.07451423068780436, rel=1e-9)
    assert out["u1_m_per_s"][0] == pytest.approx(-0.011376818891622977, rel=1e-9)
    assert same(out["w_N_per_mm"][0], 25.0)                            # two tooth pairs at A, b = 20 mm
    assert (out["rho1_t_mm"] <= 0.0).sum() == 1                        # the first step only


@pytest.mark.filterwarnings("ignore:contact_inputs. the path of contact passes a base-tangent point")
@pytest.mark.parametrize("case", CASES)
def test_gear_2_driving_runs_the_same_path_the_other_way(case):
    p = CASES[case]
    forward = contact_inputs(p, "gear1", N1, 1000.0, n_points=11)
    backward = contact_inputs(p, "gear2", N1, 1000.0, n_points=11)
    assert backward["g_y_mm"][0] > backward["g_y_mm"][-1]                # from the tip of gear 1 toward its root
    assert np.all(same(backward["s_mm"], forward["s_mm"])) and np.all(same(backward["t_s"], forward["t_s"]))
    for name in ("g_y_mm", "rho1_t_mm", "rho2_n_mm", "u1_m_per_s", "vg_m_per_s", "l_sum_mm", "w_N_per_mm"):
        assert backward[name] == pytest.approx(forward[name][::-1], rel=1e-9, abs=1e-9), name


def test_nothing_is_stored_on_the_pair():
    p = pair(*bc.pinion_wheel("H501"))
    before = dict(vars(p))
    contact_inputs(p, "gear2", N1, 1000.0, n_points=3)
    assert vars(p) == before


@pytest.mark.parametrize("kwargs, message", [
    (dict(driver="pinion", n_points=5), "driver must be"),
    (dict(n_points=5, t_s=[0.0]), "exactly one"),
    (dict(), "exactly one"),
    (dict(n_points=1), "n_points must be >= 2"),
    (dict(n1_rpm=0.0, n_points=5), "n1_rpm must be > 0"),
    (dict(t_s=[0.0, 1.0]), "t_s must lie in"),
    (dict(t_s=[-1e-6]), "t_s must lie in"),
    (dict(t_s=[]), "at least one time"),
    (dict(t_s=[0.0, float("nan")]), "all finite"),
    (dict(t_s=[float("inf")]), "all finite"),
    (dict(F_bn_N=-1000.0, n_points=5), "F_bn_N must be >= 0"),
    (dict(n1_rpm=-1450.0, n_points=5), "n1_rpm must be > 0"),
])
def test_contact_inputs_refuses_bad_inputs(kwargs, message):
    arguments = dict(driver="gear1", n1_rpm=N1, F_bn_N=1000.0) | kwargs
    with pytest.raises(ValueError, match=message):
        contact_inputs(CASES["case2"], **arguments)


# GEARpie's load per unit length fnx = fbn / lsum [N/mm] (CLASSES/FORCES_SPEEDS.py:72, ANSWER_LS = "N": the same
# rigid model). It depends on the length of the path, which is GEARpie's since Q04. Rows: (case, fbn [N], R1 [mm], fnx at the column at mid face width, fnx at column 0).
GEARPIE_LOAD = [
    ("case1", 1795.7999910531016, 4.294584993943477, 64.1357139661822, 64.1357139661822),
    ("case1", 1795.7999910531016, 13.99875981828456, 128.2714279323644, 128.2714279323644),
    ("case1", 1795.7999910531016, 23.702934642625642, 64.1357139661822, 64.1357139661822),
    ("case2", 1847.1085622260475, 6.230486984422676, 51.93178223060988, 76.0848373009994),
    ("case2", 1847.1085622260475, 14.064661835850831, 76.0848373009994, 51.50285488294276),
    ("case2", 1847.1085622260475, 21.898836687278987, 51.89171610802854, 41.66134467919347),
]


@pytest.mark.parametrize("case, fbn, R1, fnx_mid, fnx_0", GEARPIE_LOAD)
def test_load_per_unit_length_against_gearpie(case, fbn, R1, fnx_mid, fnx_0):
    """Q04: the tip of the family is GEARpie's (ISO_21771_2007, Eq. (33), no cP), so the same pair has the same path."""
    p = CASES[case]
    g_y = R1 - p.T1C
    assert fbn / load_sharing(p, g_y)["l_sum_mm"] == pytest.approx(fnx_mid, rel=1e-9)
    # Q36: GEARpie's column 0 is the section at the face end where the tooth enters; its line has its mid-face
    # point b tan(beta_b) / 2 further back
    back = 0.5 * p.b * math.tan(p.gear1.betab)
    assert fbn / load_sharing(p, g_y - back)["l_sum_mm"] == pytest.approx(fnx_0, rel=1e-9)


def test_normal_force_feeds_the_contact_inputs():
    """``MeshForces.F_bn_N`` is the load of ``contact_inputs``: at the start of meshing of case 2 the
    load per unit length is 51.932 N/mm (the value pinned for F_bn = 1847.109 N in
    ``test_contact_inputs_pinned_on_case_2``), and the line loads times the summed line lengths give F_bn back."""
    p = CASES["case2"]
    m = FAMILY.mesh_forces(p, bc.T_NM, 1)
    out = contact_inputs(p, "gear1", bc.N_RPM, m.F_bn_N, n_points=5)
    assert out["w_N_per_mm"][0] == pytest.approx(51.93178223060986, rel=1e-12)
    assert out["w_N_per_mm"] * out["l_sum_mm"] == pytest.approx(m.F_bn_N, rel=1e-12)
