"""Tests of the gear family contract (``base``) and of the cylindrical family (``cylindrical``).

Three groups: the contract (registry, immutable objects, one family for external and internal
gears, bad inputs); closed-form identities of the involute geometry; and one test per entry of
``cylindrical._LEGACY`` and per other unruled item, which pins TODAY's value on a baseline case. Where
the review gives the alternative as a relation between quantities of the code, that relation is a
second test marked ``xfail(strict=True)`` with its question id. The numbers are those of the
baseline CSVs (``tests/data/gears_baseline``) or of ``scripts/questions_values_step3.py`` (vault).
Units: lengths [mm], angles [rad], torques [N m], forces [N].
"""
import dataclasses
import math

import numpy as np
import pytest

from axisforge.core.machine_elements.gears import (
    CylindricalFamily, CylindricalGearData, Gear, GearFamily, GearPair, cylindrical)
from axisforge.core.machine_elements.gears.base import _GEAR_FAMILY_REGISTRY
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc

FAMILY = CylindricalFamily()
REL = bc.REL_TOL


def pair(g1, g2, **kw):
    return GearPair.assemble(FAMILY, gear1=g1, gear2=g2, **kw)


def xfail(question):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=question)


@pytest.fixture(scope="module")
def case2():
    """H501 helical pair (z 20/30, mn 3.5 mm, beta 15 deg)."""
    return pair(*bc.pinion_wheel("H501"))


@pytest.fixture(scope="module")
def case4():
    """Internal spur pair (z 20/-60, mn 2 mm)."""
    return pair(*bc.make_internal(0.0))


@pytest.fixture(scope="module")
def case5():
    """Internal helical pair (z 20/-60, mn 2 mm, beta 15 deg)."""
    return pair(*bc.make_internal(15.0))


# ----------------------------------------------------------------------------------------------
# contract
# ----------------------------------------------------------------------------------------------
def test_family_is_registered_with_its_traits():
    assert _GEAR_FAMILY_REGISTRY["CylindricalFamily"] is CylindricalFamily
    assert FAMILY.name == "cylindrical"
    assert FAMILY.axes == "parallel"


def test_a_family_defines_no_gear_alone_by_default():
    class PairOnly(GearFamily):
        name = "pair_only"
        axes = "intersecting"
        assemble_pair = validate_pair = pair_summary = pair_repr = None

    with pytest.raises(NotImplementedError):
        Gear.assemble(PairOnly(), CylindricalGearData(2.0, 20))


def test_external_and_internal_gears_come_from_the_one_family(case4):
    pinion, ring = case4.gear1, case4.gear2
    assert type(pinion) is type(ring) is Gear
    assert pinion.family is ring.family and case4.family is FAMILY
    assert type(pinion.family) is CylindricalFamily
    assert (pinion.z, ring.z) == (20, -60)
    # diameters are magnitudes; the sign of z puts the tip of a ring inside its reference circle
    assert ring.d == pytest.approx(120.0, rel=REL) and ring.da < ring.d < ring.df
    assert pinion.df < pinion.d < pinion.da


def test_assembled_objects_refuse_writes(case2):
    with pytest.raises(AttributeError, match="immutable"):
        case2.al = 100.0
    with pytest.raises(AttributeError, match="immutable"):
        case2.driver = "gear2"            # the driver is an argument of forces(), not stored geometry
    with pytest.raises(AttributeError, match="immutable"):
        case2.gear1.x = 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        case2.gear1.data.x = 0.5
    assert not hasattr(case2, "driver") and not hasattr(case2, "T")


def test_gear_data_is_keyword_only_after_z():
    with pytest.raises(TypeError):
        CylindricalGearData(2.0, 20, 0.3)     # the third positional argument was Ca in one old class and x in the other
    assert CylindricalGearData(2.0, 20, x=0.3).x == 0.3


def test_summary_and_warnings(case2):
    assert case2.warnings == () and case2.gear1.warnings == ()
    assert "family : cylindrical" in case2.gear1.summary()
    assert case2.summary() == FAMILY.pair_summary(case2)


def test_bad_inputs():
    with pytest.raises(ValueError, match="z must be != 0"):
        bc.gear(2.0, 0)
    bad = CylindricalGearData(-1.0, 20, b=-2.0, alpha_n_deg=60.0, beta_n_deg=50.0, Ra=-1.0, Rq=-1.0, Rz=-1.0, label="bad")
    assert bad.validate() == [
        "bad: mn must be > 0, got -1.0", "bad: b must be >= 0, got -2.0", "bad: alpha_n_deg out of range (0, 45), got 60.0",
        "bad: beta_n_deg out of range [0, 45), got 50.0", "bad: Ra must be >= 0, got -1.0", "bad: Rq must be >= 0, got -1.0",
        "bad: Rz must be >= 0, got -1.0"]
    g = bc.gear(2.0, 20, b=10.0)
    with pytest.raises(ValueError, match="Module mismatch"):
        pair(g, bc.gear(2.5, 30))
    with pytest.raises(ValueError, match="Pressure angle mismatch"):
        pair(g, bc.gear(2.0, 30, alpha_n_deg=25.0))
    with pytest.raises(ValueError, match="Helix angle mismatch"):
        pair(g, bc.gear(2.0, 30, beta_n_deg=10.0))
    with pytest.raises(ValueError, match="Internal gear must have more teeth"):
        pair(g, bc.gear(2.0, -20))
    with pytest.raises(ValueError, match="driver must be 'gear1' or 'gear2'"):
        FAMILY.forces(pair(g, bc.gear(2.0, 30, b=10.0)), 10.0, driver="pinion")


# ----------------------------------------------------------------------------------------------
# closed forms of the involute geometry (no standard needed beyond the definitions)
# ----------------------------------------------------------------------------------------------
def test_spur_gear_closed_forms():
    """Spur gear, mn 2 mm, z 20, default rack: d = mn z; db = d cos(alpha_n); pb = pi mn cos(alpha_n)."""
    g = bc.gear(2.0, 20)
    alpha = math.radians(20.0)
    assert g.d == pytest.approx(40.0, rel=REL)
    assert g.db == pytest.approx(40.0 * math.cos(alpha), rel=REL)
    assert g.pb == pytest.approx(math.pi * 2.0 * math.cos(alpha), rel=REL) and g.pbt == pytest.approx(g.pb, rel=REL)
    assert g.df == pytest.approx(40.0 - 2 * 2.0 * 1.25, rel=REL)        # root: d - 2 mn (haP + cP)
    assert math.isinf(g.pz) and math.isinf(g.px)                       # a spur gear has no lead


def test_pair_without_profile_shift_works_at_the_reference_centre_distance():
    """x1 + x2 = 0: alpha_wt = alpha_t and a_w = a (ISO_21771_2007, Eq. (55) with a zero sum)."""
    p = pair(bc.gear(2.0, 20, b=10.0, beta_n_deg=15.0), bc.gear(2.0, 47, b=10.0, beta_n_deg=15.0))
    assert p.alphatw == pytest.approx(p.gear1.alphat, rel=1e-10)        # through the root solver
    assert p.al == pytest.approx(p.a, rel=1e-10)
    assert p.dl1 == pytest.approx(p.gear1.d, rel=1e-10) and p.dl2 == pytest.approx(p.gear2.d, rel=1e-10)


@pytest.mark.parametrize("beta", (0.0, 15.0))
def test_line_of_action_identities(beta, case4):
    """The pitch point divides the base-tangent segment (T1C + T2C = T1T2, signed for a ring), the
    path points are one transverse base pitch apart, and the working pitch circles roll: dw2/dw1 = z2/z1."""
    ext = pair(*bc.pinion_wheel("H501" if beta else "C14"))
    for p in (ext, case4):
        assert p.T1C + p.T2C == pytest.approx(p.T1T2, rel=1e-12)
        assert p.AD == pytest.approx(p.gear1.pbt, rel=1e-12) and p.AE - p.AB == pytest.approx(p.gear1.pbt, rel=1e-12)
        assert p.dl2 / p.dl1 == pytest.approx(abs(p.gear2.z) / p.gear1.z, rel=1e-12)
        assert p.inv_alphatw == pytest.approx(math.tan(p.alphatw) - p.alphatw, rel=1e-12)
    assert ext.galpha == pytest.approx(ext.AE, rel=1e-12)              # external pair: path of contact = A to E
    assert ext.epslon_alpha == pytest.approx(ext.AE / ext.gear1.pbt, rel=1e-12)


def test_values_at_the_reference_diameter(case5):
    """At dy = d the pressure angle, helix angle and pitch are the reference ones; the radius of
    curvature of a ring flank is negative (concave)."""
    for g in (case5.gear1, case5.gear2):
        v = FAMILY.values_at(g, g.d)
        assert v["alpha_yt"] == pytest.approx(g.alphat, rel=1e-12)
        assert v["beta_y"] == pytest.approx(g.beta, rel=1e-12) and v["p_yt"] == pytest.approx(g.pt, rel=1e-12)
        assert math.copysign(1.0, v["L_y"]) == math.copysign(1.0, g.z)


# ----------------------------------------------------------------------------------------------
# unit guard: one test per public function that takes a torque
# ----------------------------------------------------------------------------------------------
def test_forces_takes_the_torque_in_newton_metre(case2):
    """Baseline case 2, 60.75 N m on gear 1: Ft = 1659.83 N = 2000 T / d_w1. The same number given in
    N mm (60750) gives 1.66e6 N: nothing converts silently."""
    F = FAMILY.forces(case2, bc.T_NM, 0.0, 1)
    assert F["Ft"] == pytest.approx(1659.8313933933089, rel=REL)
    assert F["Ft"] == pytest.approx(2000.0 * bc.T_NM / case2.dl1, rel=REL)          # T [N m], d_w [mm]
    assert F["T_out"] == pytest.approx(91.125, rel=REL)                             # [N m]
    assert FAMILY.forces(case2, 1000.0 * bc.T_NM, 0.0, 1)["Ft"] == pytest.approx(1659831.3933933089, rel=REL)


# ----------------------------------------------------------------------------------------------
# _LEGACY: one test per entry (today's value), and the alternative where it is a relation
# ----------------------------------------------------------------------------------------------
def test_every_legacy_entry_has_both_kinds_and_a_test():
    assert all(set(entry) == {"external", "internal"} for entry in cylindrical._LEGACY.values())
    tested = {name[len("test_legacy_"):] for name in globals() if name.startswith("test_legacy_")}
    assert set(cylindrical._LEGACY) <= tested


def test_legacy_base_cylinder(case5):
    """Q21 (I1): base helix angle, base diameter and transverse pressure angle of a helical ring."""
    ring = case5.gear2
    assert ring.betab == pytest.approx(0.27903642861272426, rel=REL)        # 15.99 deg for beta = 15 deg
    assert ring.db == pytest.approx(117.30013345227457, rel=REL)
    assert ring.alphat == pytest.approx(0.3356586244277489, rel=REL)
    assert ring.betab > ring.beta                                            # larger than the reference helix angle


@xfail("Q21: I1 / row 3, one rule for the base helix angle of both kinds (ISO_21771_2007, Eq. (5)-(7))")
def test_alternative_ring_has_the_base_helix_angle_of_its_pinion(case5):
    """Two gears in mesh have the same base helix angle and transverse pressure angle (clause 5)."""
    assert case5.gear2.betab == pytest.approx(case5.gear1.betab, rel=REL)
    assert case5.gear2.alphat == pytest.approx(case5.gear1.alphat, rel=REL)


def test_legacy_virtual_teeth(case5):
    """Q25 (I5): a ring has no virtual number of teeth, so the Henriot split raises AttributeError."""
    assert case5.gear1.zn == pytest.approx(20 / math.cos(math.radians(15.0)) ** 3, rel=REL)
    assert not hasattr(case5.gear2, "zn") and not hasattr(case5.gear2, "de")
    with pytest.raises(AttributeError):
        FAMILY.correction_for_gs_equilibrium(case5.gear1, case5.gear2)
    with pytest.raises(AttributeError):
        pair(case5.gear1, case5.gear2, equalise_gs=True)


def test_legacy_beta_range():
    """Q55: a negative helix angle is reported for an external gear and accepted for a ring."""
    assert bc.gear(2.0, 60, beta_n_deg=-5.0, label="g").validate() == ["g: beta_n_deg out of range [0, 45), got -5.0"]
    assert bc.gear(2.0, -60, beta_n_deg=-5.0, label="g").validate() == []


@xfail("Q55: the same helix-angle range for both kinds")
def test_alternative_ring_helix_angle_is_checked():
    assert any("beta_n_deg" in m for m in bc.gear(2.0, -60, beta_n_deg=-5.0).validate())


def test_legacy_gear_limits():
    """Q55: undercut is checked on an external gear only; the tip of a ring must lie between its base
    and reference circles."""
    assert bc.gear(2.0, 8, label="g").validate() == ["g: gear is undercut — z=8, mn=2.0, alpha=20.0°, beta=0.0°, x=0.0"]
    assert FAMILY.undercutting(bc.gear(2.0, 8)) and not FAMILY.undercutting(bc.gear(2.0, 20))
    assert bc.gear(2.0, -60, x=-2.0, label="g").validate() == [
        "g: tip diameter da=123.0000 ≥ pitch diameter d=120.0000. Internal gear addendum must reduce da below d."]


def test_legacy_alphatw_from_x_sum(case5):
    """Q21 (I1): the internal pair takes alpha_t of the ring; baseline case 5."""
    assert case5.alphatw == pytest.approx(0.3356586244277506, rel=REL)
    assert case5.alphatw == pytest.approx(case5.gear2.alphat, rel=1e-10)     # x1 + x2 = 0: that of the ring
    assert case5.al == pytest.approx(41.41104721640335, rel=REL)
    assert case5.dl2 - case5.dl1 == pytest.approx(83.19148633657922, rel=REL)   # against 2 al = 82.822 mm


@xfail("Q21: I1, the working pitch circles of a helical internal pair are not tangent (dw2 - dw1 != 2 a_w)")
def test_alternative_internal_working_pitch_circles_are_tangent(case5):
    assert case5.dl2 - case5.dl1 == pytest.approx(2.0 * case5.al, rel=1e-9)


def test_legacy_alphatw_from_al(case2, case5):
    """Q21 (I1): imposing the centre distance the family derived gives the same working pressure angle;
    only an internal pair gets a backlash ``jbn`` (zero here)."""
    again = pair(case5.gear1, case5.gear2, al=case5.al)
    assert again.alphatw == pytest.approx(case5.alphatw, rel=REL)
    assert again.jbn == pytest.approx(0.0, abs=bc.ABS_TOL)
    assert case5.jbn is None
    assert pair(case2.gear1, case2.gear2, al=case2.al).jbn is None
    assert pair(case2.gear1, case2.gear2, al=case2.al).alphatw == pytest.approx(case2.alphatw, rel=1e-10)


def test_legacy_u(case2, case4):
    """Q03 (X1): u = max/min for an external pair whatever the order; z2/z1 (signed) for an internal pair."""
    swapped = pair(case2.gear2, case2.gear1)                # gear 1 = wheel (z 30), gear 2 = pinion (z 20)
    assert case2.u == swapped.u == 1.5
    assert case4.u == -3.0


@xfail("Q03: row 1 / X1, u = z2/z1 (ISO_21771_2007, 5.2.1, Eq. (52))")
def test_alternative_gear_ratio_follows_the_order_of_the_gears(case2):
    swapped = pair(case2.gear2, case2.gear1)
    assert swapped.u == pytest.approx(swapped.gear2.z / swapped.gear1.z, rel=REL)


def test_legacy_working_pitch_diameters(case2):
    """Q03 (X1): gear 1 of an external pair gets the smaller working diameter even when it is the wheel."""
    swapped = pair(case2.gear2, case2.gear1)
    assert swapped.gear1.z > swapped.gear2.z
    assert swapped.dl1 == pytest.approx(case2.dl1, rel=REL) and swapped.dl1 == pytest.approx(73.20020604719922, rel=REL)
    assert swapped.dl1 < swapped.dl2


@xfail("Q03: row 1 / X1, d_w = d_b / cos(alpha_wt) for each gear (ISO_21771_2007, Eq. (56)-(57))")
def test_alternative_working_pitch_diameter_belongs_to_its_gear(case2):
    swapped = pair(case2.gear2, case2.gear1)
    assert swapped.dl1 == pytest.approx(swapped.gear1.db / math.cos(swapped.alphatw), rel=1e-9)


def test_legacy_tip_alteration(case2):
    """Q23 (I3): factor of k and the gear that takes it. External: (z1 + z2)/2, both tips; internal:
    (z2 - z1)/2, tip of gear 1 only."""
    assert cylindrical._LEGACY["tip_alteration"]["external"](20, 30) == (25.0, 1.0)
    assert cylindrical._LEGACY["tip_alteration"]["internal"](20, -60) == (-40.0, 0.0)
    ext = pair(case2.gear1, case2.gear2, addendum_reduction=True)
    assert ext.k == pytest.approx(0.008973779105703877, rel=REL)
    m = ext.gear1.mn
    assert ext.da1_w == pytest.approx(ext.gear1.da - 2 * m * ext.k, rel=REL)
    assert ext.da2_w == pytest.approx(ext.gear2.da - 2 * m * ext.k, rel=REL)
    internal = pair(bc.gear(2.0, 20, x=0.3, b=20.0), bc.gear(2.0, -60, x=0.1, b=20.0), addendum_reduction=True)
    assert internal.k != 0.0
    assert internal.da1_w == pytest.approx(internal.gear1.da - 2 * 2.0 * internal.k, rel=REL)
    assert internal.da2_w == pytest.approx(internal.gear2.da, rel=REL)          # the ring keeps its tip


def test_legacy_galpha(case4):
    """Q22 (I2): the pair ``galpha`` of an internal pair is negative and is not its A-E distance."""
    assert case4.galpha == pytest.approx(-12.595659070395595, rel=REL)
    assert case4.AE == pytest.approx(14.76595239565792, rel=REL)


@xfail("Q22: I2 / row 45, g_alpha of an internal pair = its A-E distance (ISO_21771_2007, Eq. (77))")
def test_alternative_internal_path_of_contact_is_A_to_E(case4):
    assert case4.galpha == pytest.approx(case4.AE, rel=1e-9)


def test_reduced_radius_of_an_internal_pair(case4):
    """Q22 (I2): ``ReqI`` is the plus-sign sum with a positive ring radius; baseline case 4."""
    assert case4.ReqI == pytest.approx(5.130302149885034, rel=REL)


@xfail("Q22: I2 / row 45, rho_C2 of a ring is negative (ISO_21771_2007, Eq. (82)): 10.261 mm in case 4")
def test_alternative_reduced_radius_with_the_signed_ring_radius(case4):
    """With the stored signed segments rho_C1 = T1C > 0 and rho_C2 = T2C < 0."""
    assert case4.ReqI == pytest.approx(1.0 / (1.0 / case4.T1C + 1.0 / case4.T2C), rel=1e-9)


def test_legacy_profile_shift_check(case2, case5):
    """Q26 (I6): the internal check reports the centre distance that the family itself derived."""
    assert case2.profile_shift_check[0] == "x1+x2" and case5.profile_shift_check[0] == "x2-x1"
    assert case5.profile_shift_check[1] == pytest.approx(0.3574235135793648, rel=REL)
    again = pair(case5.gear1, case5.gear2, al=case5.al)
    assert again.validate()[0] == ("Profile shift inconsistency: declared x2-x1=0.0000 but al=41.4110 mm implies "
                                   "x2-x1=0.3574. Adjust x1, x2, or al so they are consistent.")
    assert not any("Profile shift" in m for m in pair(case2.gear1, case2.gear2, al=case2.al).validate())


@xfail("Q26: I6, an internal pair at the centre distance derived from its own profile shifts is consistent")
def test_alternative_internal_pair_accepts_its_own_centre_distance(case5):
    assert not any("Profile shift" in m for m in pair(case5.gear1, case5.gear2, al=case5.al).validate())


def test_legacy_force_input(case2, case5):
    """Q03 (X1): the driver changes the forces of an internal pair and not those of an external pair
    (the alternative for the external pair is ``test_alternative_wheel_driver_takes_the_torque_on_the_wheel``)."""
    assert FAMILY.forces(case2, bc.T_NM, driver="gear2") == FAMILY.forces(case2, bc.T_NM, driver="gear1")
    assert FAMILY.forces(case5, bc.T_NM, driver="gear1")["Ft"] == pytest.approx(2960.4069103569436, rel=REL)
    ring_drives = FAMILY.forces(case5, bc.T_NM, driver="gear2")
    assert ring_drives["Ft"] == pytest.approx(977.9998991176811, rel=REL)
    assert ring_drives["T_out"] == pytest.approx(-20.25, rel=REL)


def test_legacy_interference(case2, case4):
    """Q24 (I4): every internal pair carries the trimming message (and here the trochoid one), so
    ``validate_or_raise`` raises; an external pair is tested by -T1A > T1C."""
    assert case4.validate() == ["ring: trochoid (flank) interference with mating gear (z1=20, x1=0.0).",
                                "ring: trimming interference — tooth difference z2-z1=-80 < 2."]
    with pytest.raises(ValueError, match="trimming interference"):
        case4.validate_or_raise()
    assert case2.validate() == []
    low = pair(bc.gear(2.0, 8, b=10.0, label="p"), bc.gear(2.0, 60, b=10.0, label="w"))
    assert low.validate()[-1] == ("Involute interference on gear1 (pinion): path of approach 3.7124 mm > limit "
                                  "2.7362 mm. Reduce haP, increase x1, or increase z1.")


@xfail("Q24: I4 / row 4, the trimming test (z2 - z1 < 2 with the signed z2) is true for every internal pair")
def test_alternative_internal_pair_without_the_trimming_message(case4):
    assert not any("trimming" in m for m in case4.validate())


def test_legacy_text(case2, case4):
    """Q50: the texts keep the class names of the two pair classes the family replaces."""
    assert repr(case2).startswith("HelicalGearMeshing(z1=20, z2=30, ")
    assert repr(case4).startswith("InternalGearMeshing(z1=20, z2=-60, ")
    assert case2.summary().splitlines()[0].startswith("── HelicalGearMeshing ─")
    assert case4.summary().splitlines()[1] == "  gear1 (ext): z=20, mn=2.0, x=0.0, β=0.0°"


def test_renamed_text_rows_are_counted():
    """Q50: the baseline rows that name a removed class and are compared against the new text."""
    cases = [p.stem for p in sorted(bc.DATA_DIR.glob("*.csv"))]
    got = bc.run_baseline()[0]
    for old, (new, count) in bc.RENAMED_TEXT.items():
        hits = [(c, i) for c in cases for i, (_, _, text) in enumerate(bc.baseline_rows(c)) if text == old]
        assert len(hits) == count
        assert all(got[c][i][2] == new for c, i in hits)
    # the rows `driver` / `driven` of the pairs are the declared argument, not a result of the code
    names, count = bc.ECHOED_ROWS
    echoed = sum(1 for c in cases for obj, name, _ in bc.baseline_rows(c) if name in names and obj != "link")
    renamed = sum(n for _, n in bc.RENAMED_TEXT.values())
    total = sum(len(bc.baseline_rows(c)) for c in cases)
    assert echoed == count == 58
    assert (total, total - echoed - renamed, renamed, echoed) == (7181, 7117, 6, 58)     # rows: computed, renamed text, echoed


# ----------------------------------------------------------------------------------------------
# other unruled items kept as today
# ----------------------------------------------------------------------------------------------
def test_ring_tip_diameter_carries_the_clearance(case4):
    """Q27 (I7): ring of baseline case 4, d = 120 mm, tip 115 mm = d - 2 mn (haP + cP)."""
    assert case4.gear2.da == pytest.approx(115.0, rel=REL)


@xfail("Q27: I7 / row 2, ring tip without the clearance (ISO_21771_2007, 4.5.3, Eq. (33)): 116 mm")
def test_alternative_ring_tip_diameter_without_clearance(case4):
    ring = case4.gear2
    assert ring.da == pytest.approx(ring.d - 2.0 * ring.mn * (ring.haP + ring.x), rel=REL)


def test_assemble_does_not_validate(case4):
    """Q51 / Q15: baseline case 4 assembles although its path starts beyond the base-tangent point
    (T1A < 0) and its ``validate()`` is not empty."""
    assert case4.T1A == pytest.approx(-2.3949738015367448, rel=REL)
    assert case4.validate() != []


@xfail("Q51: validation at assembly (D-0026 item 9)")
def test_alternative_assemble_raises_on_an_invalid_pair():
    try:
        pair(*bc.make_internal(0.0))
    except ValueError:
        return
    raise AssertionError("assemble returned a pair whose validate() is not empty")


def test_gear_messages_appear_twice_in_the_pair():
    """Q56: the pair repeats the messages of its gears (today it validated its working copies too)."""
    low = pair(bc.gear(2.0, 8, b=10.0, label="p"), bc.gear(2.0, 60, b=10.0, label="w"))
    assert low.validate()[:2] == 2 * ["p: gear is undercut — z=8, mn=2.0, alpha=20.0°, beta=0.0°, x=0.0"]


@xfail("Q56: each gear message once")
def test_alternative_gear_messages_appear_once_in_the_pair():
    low = pair(bc.gear(2.0, 8, b=10.0, label="p"), bc.gear(2.0, 60, b=10.0, label="w"))
    assert sum("undercut" in m for m in low.validate()) == 1


def test_second_inverse_involute_of_the_interference_tests():
    """Q19: the Newton inverse involute returns the angle near its start value (20 deg) ..."""
    alpha = math.radians(20.0)
    assert cylindrical._solve_inv(math.tan(alpha) - alpha) == pytest.approx(alpha, rel=1e-10)


@xfail("Q19: the Newton inverse involute does not converge for inv = 0.2 (returns 8.5e22) and does not raise")
def test_alternative_second_inverse_involute_converges():
    alpha = cylindrical._solve_inv(0.2)
    assert math.tan(alpha) - alpha == pytest.approx(0.2, abs=1e-9)


def test_henriot_split_returns_non_physical_shifts_for_some_pairs():
    """Q13: outside the baseline the split does not raise; for z 12/54 (mn 2 mm, x = 0) it returns
    x1 = -x2 = 4.55, a profile shift of 4.5 modules (not usable)."""
    x1, x2 = FAMILY.correction_for_gs_equilibrium(bc.gear(2.0, 12, b=20.0), bc.gear(2.0, 54, b=20.0))
    assert (x1, x2) == (pytest.approx(4.548969365135395, rel=1e-9), pytest.approx(-4.548969365135395, rel=1e-9))
    with pytest.raises(RuntimeError, match="no solution for x2"):
        FAMILY.correction_for_gs_equilibrium(*bc.pinion_wheel("H501"))
