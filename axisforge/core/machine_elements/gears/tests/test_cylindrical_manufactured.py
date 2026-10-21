"""Tests of the manufactured layer of the cylindrical family (``cylindrical``, ISO_21771_2007 clause 7).

Block 1: the tool record ``GeneratingRackProfile``, the input record ``CylindricalGearManufacturingData``,
the four generating helpers against closed forms and hand values, and the ideal undercut check in the
x form (Eq. (135)), which must give the old z-form answer for every baseline gear.
Units: lengths [mm], angles [rad] (inputs in [deg]).
"""
import math

import pytest

from axisforge.core.machine_elements.gears import (
    CylindricalFamily,
    CylindricalGearData,
    CylindricalGearManufacturingData,
    GeneratingRackProfile,
    cylindrical,
)
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc

FAMILY = CylindricalFamily()
RACK_A = GeneratingRackProfile(haP0=1.25, rhoaP0=0.38, alpha_P0_deg=20.0)   # counterpart of ISO 53 rack A


def _baseline_gears():
    """External gears of the baseline cases: C14 and H501 pinion and wheel (cases 1-3, 7), the pinions of the
    internal cases z 20 and 24, spur and helical (cases 4, 5, 4n, 5n), the sun and planet of the planetary
    trains (cases 6, 6b; same inputs as ``bc.run_planetary``)."""
    gears = [*bc.pinion_wheel("C14"), *bc.pinion_wheel("H501")]
    for beta in (0.0, 15.0):
        h = 1 if beta else 0
        gears += [bc.make_internal(beta, z1=z1)[0] for z1 in (20, 24)]
        gears += [bc.gear(2.0, 20, b=20.0, beta_n_deg=beta, hand=hand, label=lab) for hand, lab in ((h, "sun"), (-h, "planet"))]
    return gears


# ----------------------------------------------------------------------------------------------
# records
# ----------------------------------------------------------------------------------------------
def test_counterpart_of_the_basic_rack():
    """ISO_21771_2007 7.1, Fig. 35; ISO_53_1998 3.2, Fig. 1: haP0 = haP + cP, rhoaP0 = rfP, alpha_P0 = alpha_n.
    ISO 53 rack A (1.25, 0.38, 20 deg) and a non-default rack (cP 0.4, rfP 0.39 as ISO 53 rack D, 25 deg)."""
    assert GeneratingRackProfile.from_basic_rack(CylindricalGearData(2.0, 20)) == RACK_A
    other = GeneratingRackProfile.from_basic_rack(CylindricalGearData(2.0, 20, haP=1.0, cP=0.4, rfP=0.39, alpha_n_deg=25.0))
    assert (other.haP0, other.rhoaP0, other.alpha_P0_deg) == (pytest.approx(1.4, rel=1e-15), 0.39, 25.0)
    assert RACK_A.validate() == [] and other.validate() == []


def test_bad_cutter():
    bad = GeneratingRackProfile(haP0=0.0, rhoaP0=-0.1, alpha_P0_deg=45.0)
    assert bad.validate() == ["cutter: haP0 must be > 0, got 0.0", "cutter: rhoaP0 must be >= 0, got -0.1",
                              "cutter: alpha_P0_deg out of range (0, 45), got 45.0"]
    with pytest.raises(ValueError, match="haP0 must be > 0"):
        bad.validate_or_raise()


def test_manufacturing_data_keeps_the_gear_data():
    """mn and z stay positional; the allowances are required; the nominal fields are those of the parent."""
    data = CylindricalGearManufacturingData(3.5, 20, E_sns=-0.05, E_sni=-0.1)
    assert isinstance(data, CylindricalGearData)
    assert (data.mn, data.z, data.x, data.E_sns, data.E_sni, data.cutter) == (3.5, 20, 0.0, -0.05, -0.1, None)
    assert data.validate() == []
    with pytest.raises(TypeError):
        CylindricalGearManufacturingData(3.5, 20)


def test_manufacturing_data_messages():
    """E_sni <= E_sns <= 0 (ISO_21771_2007 7.3; no library source for E_sns <= 0), the cutter messages, the tool
    pressure angle equal to alpha_n (Fig. 39 NOTE)."""
    def msgs(**kw):
        return CylindricalGearManufacturingData(2.0, 20, label="g", **{"E_sns": -0.05, "E_sni": -0.1, **kw}).validate()

    assert msgs(E_sns=-0.1, E_sni=-0.05) == ["g: tooth thickness allowances must satisfy E_sni <= E_sns <= 0, "
                                             "got E_sns=-0.1, E_sni=-0.05"]
    assert msgs(E_sns=0.05) == ["g: tooth thickness allowances must satisfy E_sni <= E_sns <= 0, got E_sns=0.05, E_sni=-0.1"]
    assert msgs(E_sns=0.0, E_sni=0.0) == [] and msgs(cutter=RACK_A) == []
    assert msgs(cutter=GeneratingRackProfile(haP0=1.25, rhoaP0=0.38, alpha_P0_deg=25.0)) == [
        "g: cutter alpha_P0_deg=25.0 differs from alpha_n_deg=20.0; not implemented (ISO_21771_2007, Fig. 39 NOTE)"]
    assert msgs(cutter=GeneratingRackProfile(haP0=-1.0, rhoaP0=0.38, alpha_P0_deg=20.0)) == [
        "g: cutter: haP0 must be > 0, got -1.0"]
    parent_first = CylindricalGearManufacturingData(-2.0, 20, E_sns=0.05, E_sni=-0.1, label="g").validate()
    assert parent_first[0] == "g: mn must be > 0, got -2.0" and parent_first[1].startswith("g: tooth thickness allowances")


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def test_minimum_generating_profile_shift_rack_a_spur():
    """ISO_21771_2007 7.7, Eq. (135), rack A, spur: x_Emin(z 17) = 1.25 - 0.38 (1 - sin 20 deg) - 17 sin^2(20 deg) / 2
    = +0.005657 > 0, so z 17 at x = 0 is slightly undercut (classical z_min = 17.1), and x_Emin < 0 for z 18..40."""
    a = math.radians(20.0)
    assert cylindrical._min_generating_profile_shift(17, 2.0, a, 0.0, RACK_A)[0] == pytest.approx(0.005657, abs=1e-6)
    assert all(cylindrical._min_generating_profile_shift(z, 2.0, a, 0.0, RACK_A)[0] < 0.0 for z in range(18, 41))


def test_undercutting_gives_the_old_answer_for_every_baseline_gear():
    """MG-07: ``undercutting`` is now x < x_Emin (Eq. (135), counterpart of the basic rack). The old criterion was the
    z form z < 2 cos(beta) / sin(alpha_t)**2 (haP - x), reimplemented here as the reference: the two differ only
    by h_FaP0 / m_n against haP (3e-5 for rack A, spur; about 4e-3 at beta 15 deg), so every baseline gear, far from the limit, keeps its answer
    (False, the golden rows ``undercutting()``). A gear well below the limit is True in both forms. A ring: False."""
    def old_z_form(g):
        return g.z < 2.0 * math.cos(g.beta) / math.sin(g.alphat) ** 2 * (g.haP - g.x)

    gears = _baseline_gears() + [bc.gear(2.0, 8), bc.gear(2.0, 12, beta_n_deg=15.0, hand=1)]
    assert len(gears) == 14
    for g in gears:
        assert FAMILY.undercutting(g) == old_z_form(g), g.label
    assert [FAMILY.undercutting(g) for g in gears[-2:]] == [True, True]
    assert all(g.warnings == () for g in gears[:-2]) and all(len(g.warnings) == 1 for g in gears[-2:])
    assert not FAMILY.undercutting(bc.make_internal(0.0)[1])


def test_generated_root_diameter_at_zero_allowance_is_the_nominal_root():
    """Eq. (125) d_fE = d + 2 x_E mn - 2 haP0 mn with x_E = x and haP0 = haP + cP = hfP (counterpart of the basic
    rack) is Eq. (34) of the external gear, d_f = d + 2 mn (x - hfP): equal to the stored ``df`` of every baseline
    gear (Q02 tolerance). Eq. (123)/(124) with E_sn = 0 gives x_E = x."""
    for g in _baseline_gears():
        d_fE = cylindrical._generated_root_diameter(g.d, g.x, g.mn, GeneratingRackProfile.from_basic_rack(g.data))
        assert d_fE == pytest.approx(g.df, rel=bc.REL_TOL), g.label
    assert cylindrical._generating_profile_shift(0.1, 2.0, math.radians(20.0), 0.0) == 0.1


@pytest.mark.parametrize("name", ("C14", "H501"))
@pytest.mark.parametrize("E_sn", (0.0, -0.1))
def test_root_form_diameter_three_forms(name, E_sn):
    """ISO_21771_2007 7.6: Eq. (128) first form (the code) == second form sqrt((d - 2h)**2 + 4 h**2 / tan(alpha_t)**2)
    == Eq. (129) + (130), d_Ff = d_b / cos(alpha_Ff) with tan(alpha_Ff) = tan(alpha_t) - 4 [(h_aP0 - rho_aP0
    (1 - sin alpha_t)) / m_n - x_E] cos(beta) / (z sin(2 alpha_t)). Eq. (130) prints h_aP0 (a length) inside a
    bracket with x_E; it is read with h_aP0 / m_n (José, 2026-10-21, MG-05: dimensionally consistent reading of a
    misprint). h = m_n (h_aP0 - x_E - rho_aP0 (1 - sin alpha_t)). Pinions of C14 (spur) and H501 (beta 15 deg).
    This checks the algebraic identity of the printed forms (and the coded first form), not an independent
    reading of the standard."""
    g = bc.pinion_wheel(name)[0]
    x_E = cylindrical._generating_profile_shift(g.x, g.mn, g.alpha, E_sn)
    d_Ff = cylindrical._root_form_diameter(g.d, g.db, g.alphat, x_E, g.mn, RACK_A)
    h = g.mn * (RACK_A.haP0 - x_E - RACK_A.rhoaP0 * (1.0 - math.sin(g.alphat)))
    second = math.sqrt((g.d - 2.0 * h) ** 2 + 4.0 * h ** 2 / math.tan(g.alphat) ** 2)
    tan_Ff = math.tan(g.alphat) - 4.0 * (RACK_A.haP0 - RACK_A.rhoaP0 * (1.0 - math.sin(g.alphat)) - x_E) \
        * math.cos(g.beta) / (g.z * math.sin(2.0 * g.alphat))
    assert d_Ff == pytest.approx(second, rel=1e-12)
    assert d_Ff == pytest.approx(g.db / math.cos(math.atan(tan_Ff)), rel=1e-12)
    assert g.db < d_Ff < g.d


def test_helpers_reproduce_the_hand_check_of_h501():
    """H501 pinion (z 20, mn 3.5, beta 15 deg, x 0.1809), E_sns = -0.05 mm (illustrative, not from a drawing: MG-19),
    rack A: the values of the lead's independent script ``_evidence/hand_check.py`` (vault, 2026-10-21; math only,
    from the page images): h_FaP0 3.5139682322 mm, x_Emin -0.2831905771, x_Es 0.1612751613, d_fEs 64.8482587577 mm,
    d_Ffs 68.3863388710 mm (printed to 10 decimals)."""
    g = bc.pinion_wheel("H501")[0]
    x_Emin, hFaP0 = cylindrical._min_generating_profile_shift(g.z, g.mn, g.alphat, g.beta, RACK_A)
    assert (hFaP0, x_Emin) == (pytest.approx(3.5139682322, abs=1e-9), pytest.approx(-0.2831905771, abs=1e-9))
    x_Es = cylindrical._generating_profile_shift(g.x, g.mn, g.alpha, -0.05)
    assert x_Es == pytest.approx(0.1612751613, abs=1e-9)
    assert cylindrical._generated_root_diameter(g.d, x_Es, g.mn, RACK_A) == pytest.approx(64.8482587577, abs=1e-9)
    assert cylindrical._root_form_diameter(g.d, g.db, g.alphat, x_Es, g.mn, RACK_A) == pytest.approx(68.3863388710, abs=1e-9)
