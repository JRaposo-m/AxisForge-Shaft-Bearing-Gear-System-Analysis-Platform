"""Tests of the contact inputs along the path of contact (``CylindricalFamily.path_points``,
``curvature_radii``, ``surface_speeds``, ``load_sharing``, ``contact_inputs``).

Four groups: against the values the pair already stores (segments, ``ReqI``, radii to the path
points); closed forms of ISO_21771_2007 (5.4.5.3, 5.4.7, 5.4.8, 5.6); bad inputs; and GEARpie's
arrays as LITERAL values, copied by ``scripts/questions_values_step4.py`` (vault) from
``baseline/arrays/gearpie_case{1,2,3}_GFS_*.csv``. Open items carry their question id. Every GEARpie number
below is a row "LITERAL" of ``scripts/questions_values_step4.csv`` (same folder of the vault).

Position along the path: g_y [mm] from the pitch point C, positive toward the tip of gear 1.
Units: lengths [mm], speeds [m/s], forces [N], load per unit length [N/mm], time [s].
"""
import math
import warnings

import numpy as np
import pytest

from axisforge.core.machine_elements.gears import CylindricalFamily, GearPair
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc

FAMILY = CylindricalFamily()
REL = bc.REL_TOL
N1 = bc.N_RPM                              # [rpm] speed of gear 1
OMEGA1 = bc.OMEGA                          # [rad/s]
F_BN_CASE2 = 1847.1085622260473            # [N] F_bt / cos(beta_b) of case 2 for 60.75 N m (the alternative of Q07b)


def pair(g1, g2):
    return GearPair.assemble(FAMILY, gear1=g1, gear2=g2)


def same(got, expected):
    """Equality at the baseline tolerance, absolute 1e-12 mm (or m/s) near zero."""
    return got == pytest.approx(expected, rel=REL, abs=1e-12)


CASES = {
    "case1": pair(*bc.pinion_wheel("C14")),      # external spur, z 16/24, mn 4.5 mm, eps_alpha 1.462
    "case2": pair(*bc.pinion_wheel("H501")),     # external helical, z 20/30, mn 3.5 mm, beta 15 deg
    "case4": pair(*bc.make_internal(0.0)),       # internal spur, z 20/-60, mn 2 mm, eps_alpha 1.950 (T1A < 0: Q15)
    "case5": pair(*bc.make_internal(15.0)),      # internal helical, beta 15 deg
}


# ----------------------------------------------------------------------------------------------
# path points
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_path_points_are_the_stored_points_measured_from_the_pitch_point(case):
    p = CASES[case]
    g = FAMILY.path_points(p, "gear1")
    assert list(g) == list("ABCDE") and g["C"] == 0.0
    assert same(g["A"], -p.AC) and same(g["E"] - g["A"], p.AE)
    assert same(g["B"] - g["A"], p.AB) and same(g["D"] - g["A"], p.AD)
    # B is one transverse base pitch before E and D one after A (ISO_21771_2007, 5.4.5.1)
    assert same(g["E"] - g["B"], p.gear1.pbt) and same(g["D"] - g["A"], p.gear1.pbt)


@pytest.mark.parametrize("case", CASES)
def test_gear_2_driving_exchanges_A_with_E_and_B_with_D(case):
    """Swap rule after ISO_21771_2007, Eq. (80) and (87): A is on the tip circle of the DRIVEN gear."""
    p = CASES[case]
    g1, g2 = FAMILY.path_points(p, "gear1"), FAMILY.path_points(p, "gear2")
    assert (g2["A"], g2["B"], g2["C"], g2["D"], g2["E"]) == (g1["E"], g1["D"], 0.0, g1["B"], g1["A"])
    for driver, tip_at_A, tip_at_E in (("gear1", 2, 1), ("gear2", 1, 2)):
        rho = FAMILY.curvature_radii(p, [FAMILY.path_points(p, driver)[k] for k in "AE"])
        for i, k in ((tip_at_A, 0), (tip_at_E, 1)):
            radius = math.hypot(rho[f"rho{i}_t_mm"][k], getattr(p, f"gear{i}").rb)
            assert same(radius, getattr(p, f"da{i}_w") / 2.0)


def test_path_points_pinned_on_case_2_and_case_4():
    """Units guard [mm]. Case 4 (eps_alpha 1.95 < 2): B comes before D."""
    g = FAMILY.path_points(CASES["case2"], "gear1")
    assert same(g["A"], -7.548430643298625) and same(g["E"], 8.126598199162627)
    g = FAMILY.path_points(CASES["case4"], "gear1")
    assert same(g["D"], -1.0110645623249095) and same(g["B"], -1.3082715749939684) and g["B"] < g["D"]


# ----------------------------------------------------------------------------------------------
# radii of curvature
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_radii_at_the_path_points_are_the_stored_segments(case):
    """ISO_21771_2007, Eq. (81)-(87): rho_X1 = T1X, rho_X2 = T2X, their sum T1T2 (negative for a ring)."""
    p = CASES[case]
    g = FAMILY.path_points(p, "gear1")
    rho = FAMILY.curvature_radii(p, [g[k] for k in "ABCDE"])
    for i, k in enumerate("ABCDE"):
        assert same(rho["rho1_t_mm"][i], getattr(p, "T1" + k))
        assert same(rho["rho2_t_mm"][i], getattr(p, "T2" + k))      # case 5 at C too: one base-cylinder rule (Q21)
    assert np.all(same(rho["rho1_t_mm"] + rho["rho2_t_mm"], p.T1T2))
    assert np.all(np.sign(rho["rho2_t_mm"]) == np.sign(p.gear2.z))   # D-0010 item 8: concave ring flank negative


def test_helical_ring_radius_at_the_pitch_point_is_the_stored_T2C():
    """Q21: with one base-cylinder rule T1T2 - T1C is the stored T2C = -rb2 tan(alphatw) [mm], case 5."""
    p = CASES["case5"]
    assert same(FAMILY.curvature_radii(p, 0.0)["rho2_t_mm"], p.T2C)
    assert same(p.T2C, -p.gear2.rb * math.tan(p.alphatw))


@pytest.mark.parametrize("case", CASES)
def test_radii_to_the_path_points_follow_from_the_radii_of_curvature(case):
    """rA1, rB1, rD1 at A, B, D of gear 1; rA2, rB2, rD2 are the radii of gear 2 at E, D, B (mirrored names)."""
    p = CASES[case]
    g = FAMILY.path_points(p, "gear1")
    for attr, i, k in (("rA1", 1, "A"), ("rB1", 1, "B"), ("rD1", 1, "D"), ("rA2", 2, "E"), ("rB2", 2, "D"), ("rD2", 2, "B")):
        rho = FAMILY.curvature_radii(p, g[k])[f"rho{i}_t_mm"]
        assert same(math.hypot(rho, getattr(p, f"gear{i}").rb), getattr(p, attr))


def test_normal_plane_radii_are_the_transverse_ones_over_cos_betab():
    """Q34 (no library source). Units guard [mm]: case 2 at C."""
    p = CASES["case2"]
    rho = FAMILY.curvature_radii(p, np.array([-3.0, 0.0, 4.0]))
    for name in ("rho1", "rho2", "rho_red"):
        assert np.all(same(rho[name + "_n_mm"] * math.cos(p.gear1.betab), rho[name + "_t_mm"]))
    assert same(rho["rho_red_t_mm"][1], 8.267350576644626) and same(rho["rho_red_n_mm"][1], 8.523274065627332)
    spur = FAMILY.curvature_radii(CASES["case1"], 0.0)
    assert spur["rho_red_n_mm"] == spur["rho_red_t_mm"]


# ----------------------------------------------------------------------------------------------
# surface speeds
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_no_sliding_at_the_pitch_point(case):
    v = FAMILY.surface_speeds(CASES[case], 0.0, N1)
    assert abs(v["vg_m_per_s"]) < 1e-12 and same(v["u1_m_per_s"], v["ue_m_per_s"])


@pytest.mark.parametrize("case", CASES)
def test_sliding_speed_is_proportional_to_the_distance_from_the_pitch_point(case):
    """ISO_21771_2007, Eq. (108): abs(vg) = abs(omega1 g_alpha_y (1 + 1/u)), u = z2 / z1 signed."""
    p = CASES[case]
    g_y = np.array([FAMILY.path_points(p, "gear1")[k] for k in "ABDE"])
    v = FAMILY.surface_speeds(p, g_y, N1)
    expected = OMEGA1 * g_y * (1.0 + p.gear1.z / p.gear2.z) / 1000.0          # [mm/s] -> [m/s]
    assert np.all(v["vg_m_per_s"] == pytest.approx(expected, rel=1e-10, abs=1e-12))
    assert np.all(np.sign(v["vg_m_per_s"]) == np.sign(g_y))                      # flank 1 faster on its addendum side
    assert np.all(same(v["ue_m_per_s"], (v["u1_m_per_s"] + v["u2_m_per_s"]) / 2.0))      # Q35


def test_surface_speeds_pinned_on_case_2():
    """Units guard [m/s]: at C both flanks move at the pitch-line speed times sin(alphatw)."""
    p = CASES["case2"]
    v = FAMILY.surface_speeds(p, 0.0, N1)
    assert same(v["ue_m_per_s"], 2.0922410756989587)
    assert v["ue_m_per_s"] == pytest.approx(OMEGA1 * p.rl1 / 1000.0 * math.sin(p.alphatw), rel=1e-12)
    v = FAMILY.surface_speeds(p, FAMILY.path_points(p, "gear1")["A"], N1)
    assert same(v["u1_m_per_s"], 0.9460598533682053) and same(v["u2_m_per_s"], 2.856361890586128)
    assert same(v["vg_m_per_s"], -1.9103020372179227)


def test_helical_internal_pair_does_not_slide_at_the_pitch_point():
    """Q21: one base-cylinder rule for both gears. Case 5 at 1450 rpm: no sliding at C (abs 1e-12 m/s), and
    the two base circles unroll the same length of the line of action, omega2 rb2 = omega1 rb1 with
    omega2 = omega1 / u (ISO_21771_2007, Eq. (105): v_n is common to both flanks)."""
    p = CASES["case5"]
    assert abs(FAMILY.surface_speeds(p, 0.0, N1)["vg_m_per_s"]) < 1e-12
    assert abs(OMEGA1 / p.u) * p.gear2.rb == pytest.approx(OMEGA1 * p.gear1.rb, rel=1e-12)


@pytest.mark.parametrize("n1_rpm", [0.0, -1450.0])
def test_surface_speeds_refuses_a_speed_that_is_not_positive(n1_rpm):
    """The speed of gear 1 is a magnitude, as in ``contact_inputs``."""
    with pytest.raises(ValueError, match="n1_rpm must be > 0"):
        FAMILY.surface_speeds(CASES["case2"], 0.0, n1_rpm)


def test_internal_pair_slides_less_than_the_external_pair_of_the_same_teeth():
    """NOTE at ISO_21771_2007, 5.6.1: u < 0 for an internal pair, so the factor (1 + 1/u) is smaller."""
    internal = FAMILY.surface_speeds(CASES["case4"], 2.0, N1)
    assert internal["u2_m_per_s"] > 0.0
    assert same(internal["vg_m_per_s"], OMEGA1 * 2.0 * (1.0 - 20.0 / 60.0) / 1000.0)


# ----------------------------------------------------------------------------------------------
# load sharing
# ----------------------------------------------------------------------------------------------
def test_spur_pair_shares_one_half_in_double_contact_and_all_in_single_contact():
    p = CASES["case1"]
    g = FAMILY.path_points(p, "gear1")
    ls = FAMILY.load_sharing(p, np.array([g["A"], (g["A"] + g["B"]) / 2, g["C"], (g["D"] + g["E"]) / 2, g["E"]]))
    assert list(ls["share"]) == [0.5, 0.5, 1.0, 0.5, 0.5]
    assert list(ls["n_pairs"]) == [2, 2, 1, 2, 2]
    assert list(ls["l_mm"]) == [p.b] * 5 and list(ls["l_sum_mm"]) == [2 * p.b, 2 * p.b, p.b, 2 * p.b, 2 * p.b]
    outside = FAMILY.load_sharing(p, np.array([g["A"] - 0.01, g["E"] + 0.01]))
    assert list(outside["share"]) == [0.0, 0.0] and list(outside["l_mm"]) == [0.0, 0.0]


def test_single_contact_zone_length_gives_the_transverse_contact_ratio():
    """1 < eps_alpha < 2: the zone B..D of single contact is (2 - eps_alpha) p_bt long (ISO_21771_2007, Eq. (90))."""
    p = CASES["case1"]
    g = FAMILY.path_points(p, "gear1")
    assert same(2.0 - (g["D"] - g["B"]) / p.gear1.pbt, p.epslon_alpha)
    grid = np.linspace(g["A"], g["E"], 200001)
    single = FAMILY.load_sharing(p, grid)["n_pairs"] == 1
    assert grid[single].min() == pytest.approx(g["B"], abs=1e-3) and grid[single].max() == pytest.approx(g["D"], abs=1e-3)


def test_pair_with_a_contact_ratio_above_two_shares_between_two_and_three_tooth_pairs():
    """Internal spur pair z 40/-120, mn 2 mm, alpha_n 15 deg (no baseline case), eps_alpha 2.48: no zone of
    single contact."""
    p = pair(bc.gear(2.0, 40, b=20.0, alpha_n_deg=15.0), bc.gear(2.0, -120, b=20.0, alpha_n_deg=15.0))
    assert 2.0 < p.epslon_alpha < 3.0 and p.T1A > 0.0
    g = FAMILY.path_points(p, "gear1")
    ls = FAMILY.load_sharing(p, np.linspace(g["A"], g["E"], 2001))
    assert set(ls["n_pairs"]) == {2, 3} and set(ls["share"]) == {0.5, 1.0 / 3.0}


@pytest.mark.parametrize("case", CASES)
def test_shares_of_the_tooth_pairs_in_contact_add_up_to_one(case):
    p = CASES[case]
    g = FAMILY.path_points(p, "gear1")
    g_y = np.linspace(g["A"], g["E"], 997)
    shares = sum(FAMILY.load_sharing(p, g_y + k * p.gear1.pbt)["share"] for k in range(-6, 7))
    assert np.all(shares == pytest.approx(1.0, rel=1e-12))


@pytest.mark.parametrize("case", CASES)
def test_path_length_and_contact_lines_agree_with_the_contact_ratio(case):
    """eps_alpha = g_alpha / p_bt (ISO_21771_2007, Eq. (90)); the sum of the contact lines averaged over one base
    pitch is eps_alpha b / cos(beta_b); one tooth pair carries the whole force over one base pitch of travel."""
    p = CASES[case]
    pbt = p.gear1.pbt
    assert same(p.AE / pbt, p.epslon_alpha)
    one_pitch = np.linspace(0.0, pbt, 100000, endpoint=False)      # periodic in p_bt: the plain mean is the integral
    assert FAMILY.load_sharing(p, one_pitch)["l_sum_mm"].mean() == pytest.approx(
        p.epslon_alpha * p.b / math.cos(p.gear1.betab), rel=1e-4)
    reach = p.AE + p.b * math.tan(p.gear1.betab)                   # a helical line touches before A and after E
    travel = np.linspace(-p.AC - reach, -p.AC + 2 * reach, 300001)
    share = FAMILY.load_sharing(p, travel)["share"]
    assert share.sum() * (travel[1] - travel[0]) == pytest.approx(pbt, rel=1e-4)


def test_helical_contact_line_is_never_longer_than_l_max():
    """ISO_21771_2007, Eq. (98)-(99): l_max = min(g_alpha / sin(beta_b), b / cos(beta_b)). Units guard [mm], case 2."""
    p = CASES["case2"]
    g = FAMILY.path_points(p, "gear1")
    ls = FAMILY.load_sharing(p, np.array([g["A"], g["C"], g["E"]]))
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
    out = FAMILY.contact_inputs(p, "gear1", N1, F_BN_CASE2, n_points=5)
    assert list(out) == ["s_mm", "t_s", "g_y_mm", "rho1_t_mm", "rho2_t_mm", "rho_red_t_mm", "rho1_n_mm", "rho2_n_mm",
                         "rho_red_n_mm", "u1_m_per_s", "u2_m_per_s", "vg_m_per_s", "ue_m_per_s", "share", "l_mm", "l_sum_mm", "n_pairs",
                         "w_N_per_mm"]
    assert all(len(v) == 5 for v in out.values())
    g = FAMILY.path_points(p, "gear1")
    assert out["s_mm"][0] == 0.0 and same(out["s_mm"][-1], p.AE)
    assert same(out["g_y_mm"][0], g["A"]) and same(out["g_y_mm"][-1], g["E"])
    for name, function in (("rho_red_n_mm", FAMILY.curvature_radii), ("share", FAMILY.load_sharing)):
        assert np.array_equal(out[name], function(p, out["g_y_mm"])[name])
    assert np.array_equal(out["ue_m_per_s"], FAMILY.surface_speeds(p, out["g_y_mm"], N1)["ue_m_per_s"])
    # load per unit length = share x F_bn / length of the contact line of this tooth pair
    assert np.all(same(out["w_N_per_mm"] * out["l_mm"], out["share"] * F_BN_CASE2))


def test_contact_inputs_pinned_on_case_2():
    """Units guard: time [s] of the travel A..E at 1450 rpm, load per unit length [N/mm] for F_bn = 1847.1 N."""
    p = CASES["case2"]
    out = FAMILY.contact_inputs(p, "gear1", N1, F_BN_CASE2, n_points=5)
    assert same(out["t_s"][-1], 0.0030445125489303704)
    # ISO_21771_2007, Eq. (105): the contact point moves along the line of action at v_n = omega1 rb1
    assert same(out["t_s"][-1], p.AE / (OMEGA1 * p.gear1.rb))
    assert same(out["w_N_per_mm"][0], 51.93178223060986) and same(out["w_N_per_mm"][1], 46.05440498753953)
    assert np.all(same(FAMILY.contact_inputs(p, "gear1", N1, 2.0 * F_BN_CASE2, n_points=5)["w_N_per_mm"], 2.0 * out["w_N_per_mm"]))


def test_contact_inputs_at_given_times_are_those_at_the_same_positions():
    p = CASES["case1"]
    by_position = FAMILY.contact_inputs(p, "gear1", N1, 1000.0, n_points=7)
    by_time = FAMILY.contact_inputs(p, "gear1", N1, 1000.0, t_s=by_position["t_s"])
    for name in by_position:
        assert np.all(same(by_time[name], by_position[name])), name
    # twice the speed: the same positions in half the time, twice the surface speeds
    fast = FAMILY.contact_inputs(p, "gear1", 2 * N1, 1000.0, t_s=by_position["t_s"] / 2)
    assert np.all(same(fast["g_y_mm"], by_position["g_y_mm"])) and np.all(same(fast["u1_m_per_s"], 2 * by_position["u1_m_per_s"]))


def test_times_with_gear_2_driving_start_at_the_tip_of_gear_1():
    """t = 0 at A, which is on the tip circle of the driven gear 1; the travel takes the same time."""
    p = CASES["case2"]
    t_AE = p.AE / (OMEGA1 * p.gear1.rb)
    out = FAMILY.contact_inputs(p, "gear2", N1, F_BN_CASE2, t_s=[0.0, t_AE / 2, t_AE])
    g = FAMILY.path_points(p, "gear2")
    assert same(out["g_y_mm"][0], g["A"]) and same(out["g_y_mm"][2], g["E"]) and g["A"] > 0.0 > g["E"]
    assert same(math.hypot(out["rho1_t_mm"][0], p.gear1.rb), p.da1_w / 2.0)
    assert same(out["s_mm"][1], p.AE / 2) and same(out["g_y_mm"][1], g["A"] - p.AE / 2)
    assert out["vg_m_per_s"][0] > 0.0 > out["vg_m_per_s"][2]            # flank 1 the faster at its own tip
    scalar = FAMILY.contact_inputs(p, "gear2", N1, F_BN_CASE2, t_s=t_AE / 2)
    assert scalar["g_y_mm"].shape == (1,) and same(scalar["w_N_per_mm"][0], out["w_N_per_mm"][1])


def test_path_that_starts_beyond_a_base_tangent_point_is_returned_with_a_warning():
    """Q39 (with Q15): the first step of baseline case 4 has rho1 < 0 (A is 0.075 mm beyond T1 with the ISO
    tip, Q27), where flank 1 has no involute; the numbers are returned as they are (at A: reduced radius
    -0.0745 mm, u1 -0.0114 m/s)."""
    p = CASES["case4"]
    with pytest.warns(UserWarning, match="base-tangent point"):
        out = FAMILY.contact_inputs(p, "gear1", N1, 1000.0, n_points=51)
    assert out["rho1_t_mm"][0] == pytest.approx(-0.07492456399709191, rel=1e-9) and same(out["rho1_t_mm"][0], p.T1A)
    assert out["rho_red_t_mm"][0] < 0.0 and out["u1_m_per_s"][0] < 0.0
    assert out["rho_red_t_mm"][0] == pytest.approx(-0.07451423068780436, rel=1e-9)
    assert out["u1_m_per_s"][0] == pytest.approx(-0.011376818891622977, rel=1e-9)
    assert same(out["w_N_per_mm"][0], 25.0)                            # two tooth pairs at A, b = 20 mm
    assert (out["rho1_t_mm"] <= 0.0).sum() == 1                        # the first step only


@pytest.mark.parametrize("case", ["case1", "case2", "case5"])
def test_path_between_the_base_tangent_points_gives_no_warning(case):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = FAMILY.contact_inputs(CASES[case], "gear1", N1, 1000.0, n_points=51)
    # flank 2 is an involute where its radius has the sign of z2 (negative: the concave ring flank)
    assert out["rho1_t_mm"].min() > 0.0 and (np.sign(CASES[case].gear2.z) * out["rho2_t_mm"]).min() > 0.0


@pytest.mark.filterwarnings("ignore:contact_inputs. the path of contact passes a base-tangent point")
@pytest.mark.parametrize("case", CASES)
def test_gear_2_driving_runs_the_same_path_the_other_way(case):
    p = CASES[case]
    forward = FAMILY.contact_inputs(p, "gear1", N1, 1000.0, n_points=11)
    backward = FAMILY.contact_inputs(p, "gear2", N1, 1000.0, n_points=11)
    assert backward["g_y_mm"][0] > backward["g_y_mm"][-1]                # from the tip of gear 1 toward its root
    assert np.all(same(backward["s_mm"], forward["s_mm"])) and np.all(same(backward["t_s"], forward["t_s"]))
    for name in ("g_y_mm", "rho1_t_mm", "rho2_n_mm", "u1_m_per_s", "vg_m_per_s", "l_sum_mm", "w_N_per_mm"):
        assert backward[name] == pytest.approx(forward[name][::-1], rel=1e-9, abs=1e-9), name


def test_nothing_is_stored_on_the_pair():
    p = pair(*bc.pinion_wheel("H501"))
    before = dict(vars(p))
    FAMILY.contact_inputs(p, "gear2", N1, 1000.0, n_points=3)
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
        FAMILY.contact_inputs(CASES["case2"], **arguments)


def test_path_points_refuses_an_unknown_driver():
    with pytest.raises(ValueError, match="driver must be"):
        FAMILY.path_points(CASES["case1"], "wheel")


# ----------------------------------------------------------------------------------------------
# GEARpie's arrays (literal values; the files stay in the vault)
# ----------------------------------------------------------------------------------------------
# Each number of the two tables below is written by scripts/questions_values_step4.py (vault folder
# 80_Inbox/2026-10-13/gears_refactor_steps0-5) into questions_values_step4.csv, rows "LITERAL".
# GEARpie's tip is that of the pair (no clearance cP, Q04); each row still names a point of the line of action by the radius of curvature of flank 1,
# R1 = T1A + xd [mm] (CLASSES/LOAD_SHARING.py:99), which is the same point in both codes: g_y = R1 - T1C.
# Rows: (case, speed of gear 1 [rpm], R1 [mm], vr1, vr2, vr, vg [m/s], gs1, gs2, SRR [-]) at the first, middle
# and last index of the arrays (case 3 = the pair of case 2 with the wheel driving at 1450 rpm).
GEARPIE_KINEMATICS = [
    ("case1", 1450.0, 4.294584993943477, 0.6521054389141734, 3.1007309248006503, 1.876418181857412,
     2.4486254858864767, 3.754953324670478, 0.7896929934492467, 1.3049465783062566),
    ("case1", 1450.0, 13.99875981828456, 2.1256227152170535, 2.118386073932064, 2.122004394574559,
     0.0072366412849893536, 0.0034044805944080246, 0.003416110676915935, 0.0034102857201859047),
    ("case1", 1450.0, 23.702934642625642, 3.5991399915199334, 1.1360412230634773, 2.3675906072917052,
     2.4630987684564563, 0.6843575893852015, 2.168142069540753, 1.0403398124957097),
    ("case2", 1450.0, 6.230486984422676, 0.9753460117539211, 2.9447832166275267, 1.9600646141907239,
     1.9694372048736057, 2.019219006526776, 0.6687885185412993, 1.0047817763838116),
    ("case2", 1450.0, 14.064661835850831, 2.201739906136053, 2.1271872870394386, 2.164463596587746,
     0.07455261909661459, 0.03386077478490673, 0.035047510649790926, 0.03444392375743626),
    ("case2", 1450.0, 21.898836687278987, 3.4281338005181854, 1.3095913574513507, 2.368862578984768,
     2.1185424430668345, 0.6179870933703354, 1.6177126024944275, 0.8943289753746652),
    ("case2", 2175.0, 6.230486984422676, 1.4630190176308817, 4.4171748249412905, 2.940096921286086,
     2.954155807310409, 2.019219006526776, 0.6687885185412993, 1.0047817763838116),
    ("case2", 2175.0, 21.898836687278987, 5.142200700777278, 1.964387036177026, 3.553293868477152,
     3.1778136646002517, 0.6179870933703354, 1.6177126024944275, 0.8943289753746652),
]


@pytest.mark.parametrize("case, n1, R1, vr1, vr2, vr, vg, gs1, gs2, SRR", GEARPIE_KINEMATICS)
def test_speeds_against_gearpie_at_the_same_point_of_the_line_of_action(case, n1, R1, vr1, vr2, vr, vg, gs1, gs2, SRR):
    """GEARpie's rolling speeds are omega R / cos(beta_b) (CLASSES/FORCES_SPEEDS.py:80-81); the family returns
    omega rho in the transverse plane (Q38), so the factor is applied here. gs1, gs2 are the specific sliding
    abs(vg) / vr1, abs(vg) / vr2 (not a load share) and SRR = abs(vg) / vr: ratios, no factor."""
    p = CASES[case]
    g_y = R1 - p.T1C
    assert same(FAMILY.curvature_radii(p, g_y)["rho1_t_mm"], R1)
    v = FAMILY.surface_speeds(p, g_y, n1)
    cos_betab = math.cos(p.gear1.betab)
    gearpie = dict(u1=vr1, u2=vr2, ue=vr, vg=vg)
    for name in ("u1", "u2", "ue", "vg"):
        assert abs(v[name + "_m_per_s"]) / cos_betab == pytest.approx(gearpie[name], rel=1e-9)
    assert abs(v["vg_m_per_s"]) / v["u1_m_per_s"] == pytest.approx(gs1, rel=1e-9)
    assert abs(v["vg_m_per_s"]) / v["u2_m_per_s"] == pytest.approx(gs2, rel=1e-9)
    assert abs(v["vg_m_per_s"]) / v["ue_m_per_s"] == pytest.approx(SRR, rel=1e-9)


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
    assert fbn / FAMILY.load_sharing(p, g_y)["l_sum_mm"] == pytest.approx(fnx_mid, rel=1e-9)
    # Q36: GEARpie's column 0 is the section at the face end where the tooth enters; its line has its mid-face
    # point b tan(beta_b) / 2 further back
    back = 0.5 * p.b * math.tan(p.gear1.betab)
    assert fbn / FAMILY.load_sharing(p, g_y - back)["l_sum_mm"] == pytest.approx(fnx_0, rel=1e-9)


def test_start_of_meshing_is_gearpies():
    """Q04: with the tip of ISO_21771_2007, Eq. (33), GEARpie's start of meshing (R1 = 4.2946 mm, its first index) is
    the stored A (T1A) of case 1."""
    assert CASES["case1"].T1A == pytest.approx(4.294584993943477, rel=1e-9)
