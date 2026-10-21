"""Tests of the manufactured layer of the cylindrical family (``cylindrical``, ISO_21771_2007 clause 7).

Block 1: the tool record ``GeneratingRackProfile``, the input record ``CylindricalGearManufacturingData``,
the four generating helpers against closed forms, and the ideal undercut check in the x form (Eq. (135)),
which must give the old z-form answer for every baseline gear.
Block 2: ``ManufacturedCylindricalGear``: zero allowances reproduce the ideal gear, pair and shaft loads;
the hand check of H501; internal gears; undercut at the lower allowance.
Block 3: ``ManufacturedCylindricalGearPair``: backlash, active root diameters, d_Nf >= d_Ff.
Block 4: the placeholders (MG-02, MG-15 to MG-18) refuse to compute.
Units: lengths [mm], angles [rad] (inputs in [deg]).
"""
import dataclasses
import math

import numpy as np
import pytest

from axisforge.core.machine_elements.gears import (
    CylindricalFamily,
    CylindricalGearData,
    CylindricalGearManufacturingData,
    GearPair,
    GeneratingPinionCutter,
    GeneratingRackProfile,
    ManufacturedCylindricalGear,
    ManufacturedCylindricalGearPair,
    cylindrical,
)
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc
from axisforge.core.machine_elements.gears.tests.test_cylindrical_family import PAIRS

FAMILY = CylindricalFamily()
RACK_A = GeneratingRackProfile(haP0=1.25, rhoaP0=0.38, alpha_P0_deg=20.0)   # counterpart of ISO 53 rack A
MANUFACTURED = ("cutter", "hFaP0", "x_Es", "x_Ei", "x_Emin", "s_ns", "s_ni", "s_ats", "s_ati",
                "d_fEs", "d_fEi", "d_Ffs", "d_Ffi")


def _same(got, ref):
    """Equivalence of the baseline (Q02): floats within ``bc._close`` (relative 1e-12, absolute 1e-9 near zero,
    inf equal), containers element by element, anything else by ``==``."""
    if isinstance(ref, (float, np.floating)):
        return isinstance(got, (float, np.floating)) and bc._close(float(got), float(ref))
    if isinstance(ref, (tuple, list)):
        return len(got) == len(ref) and all(_same(a, b) for a, b in zip(got, ref))
    if isinstance(ref, dict):
        return got.keys() == ref.keys() and all(_same(got[k], ref[k]) for k in ref)
    return got == ref


def _rings():
    """Internal gears of the baseline cases: the rings z -60 of cases 4, 5 (spur, helical) and of the planetary
    trains 6, 6b (hand of the planet, ``bc.run_planetary``)."""
    rings = [bc.make_internal(beta)[1] for beta in (0.0, 15.0)]
    return rings + [bc.gear(2.0, -60, b=20.0, beta_n_deg=beta, hand=-1 if beta else 0, label="ring") for beta in (0.0, 15.0)]


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


# ----------------------------------------------------------------------------------------------
# ManufacturedCylindricalGear
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("g", _baseline_gears() + _rings(), ids=lambda g: f"{g.label or 'g'}_z{g.z}_beta{g.beta_n_deg:g}")
def test_zero_allowances_reproduce_the_ideal_gear(g):
    """With E_sns = E_sni = 0 the manufactured gear keeps every nominal attribute (each key of ``assemble_gear``
    and each field of ``CylindricalGearData``, Q02 tolerance), x_Es = x_Ei = x, s_ns = s_ni = s_n of Eq. (49),
    s_ats = s_ati = Eq. (38) at the nominal x (computed here from d_a, d_b, alpha_t, x, z), and for an external
    gear d_fEs = d_fEi = d_f (Eq. (125) at x_E = x with haP0 = hfP is Eq. (34)). The manufactured keys never
    overwrite a NOMINAL key (only ``warnings`` is shared, and joined); ``cutter`` deliberately replaces the copy of
    the data field (the resolved tool; the input stays in ``data.cutter``)."""
    m = ManufacturedCylindricalGear.from_ideal(g, E_sns=0.0, E_sni=0.0)
    nominal, computed = FAMILY.assemble_gear(g.data), FAMILY.assemble_manufactured_gear(m)
    assert nominal.keys() & computed.keys() == {"warnings"}
    assert computed.keys() & {f.name for f in dataclasses.fields(m.data)} == {"cutter"}
    for name in [k for k in nominal if k != "warnings"] + [f.name for f in dataclasses.fields(CylindricalGearData)]:
        assert _same(getattr(m, name), getattr(g, name)), name
    assert m.warnings[:len(g.warnings)] == g.warnings
    assert m.x_Es == m.x_Ei == g.x
    s_n = g.mn * (math.pi / 2 + 2 * g.x * math.tan(g.alpha))
    assert m.s_ns == pytest.approx(s_n, rel=1e-12) and m.s_ni == pytest.approx(s_n, rel=1e-12)
    alpha_at = math.acos(g.db / g.da)
    s_at = g.da * ((math.pi + 4 * g.x * math.tan(g.alpha)) / (2 * abs(g.z))
                   + math.copysign(1.0, g.z) * (math.tan(g.alphat) - g.alphat - math.tan(alpha_at) + alpha_at))
    assert m.s_ats == m.s_ati == pytest.approx(s_at, rel=1e-12)
    if g.z > 0:
        assert m.d_fEs == pytest.approx(g.df, rel=1e-12) and m.d_fEi == pytest.approx(g.df, rel=1e-12)


def _zero(g):
    """The manufactured gear of ``g`` with zero allowances."""
    return ManufacturedCylindricalGear.from_ideal(g, E_sns=0.0, E_sni=0.0)


def test_manufactured_gears_give_the_same_shaft_loads():
    """Two manufactured gears with zero allowances are accepted by ``GearElement`` and a ``Transmission`` (baseline
    case 7a: H501 on two shafts, ``bc.build_case7``) and give identical results to the ideal gears: the torques,
    speeds and senses, the mesh forces and every stored load on the two loaded shafts (the pair itself:
    ``test_zero_allowances_reproduce_the_ideal_pair``)."""
    p, w = bc.pinion_wheel("H501", position=bc.GEAR_POS)
    a = bc.resolve_case7(bc.build_case7("pinion", 1, gears=(p, w)))
    b = bc.resolve_case7(bc.build_case7("pinion", 1, gears=(_zero(p), _zero(w))))
    assert isinstance(b.geP.gear, ManufacturedCylindricalGear)
    for name in ("torque_Nm", "rotation", "speed_rpm"):
        assert _same(getattr(b.resolved, name), getattr(a.resolved, name)), name
    assert _same([dataclasses.asdict(m) for m in b.resolved.meshes], [dataclasses.asdict(m) for m in a.resolved.meshes])

    def loads(shaft):
        return [(type(ld).__name__, ld.label, ld.position, ld.magnitude, getattr(ld, "theta_deg", None)) for ld in shaft.loads]

    for sa, sb in zip(a.resolved.shafts, b.resolved.shafts):
        assert _same(loads(sb), loads(sa))


def test_from_ideal_equals_assemble():
    """``from_ideal`` builds the same gear as ``assemble`` on the same data and allowances (every attribute). From a
    manufactured gear only the nominal fields are copied: the new allowances replace the old ones."""
    g = bc.pinion_wheel("H501")[0]
    a = ManufacturedCylindricalGear.from_ideal(g, E_sns=-0.05, E_sni=-0.10)
    data = CylindricalGearManufacturingData(**{f.name: getattr(g.data, f.name) for f in dataclasses.fields(g.data)},
                                            E_sns=-0.05, E_sni=-0.10)
    b = ManufacturedCylindricalGear.assemble(FAMILY, data)
    assert vars(a).keys() == vars(b).keys()
    for name, value in vars(b).items():
        if name != "_family":
            assert _same(getattr(a, name), value), name
    assert ManufacturedCylindricalGear.from_ideal(a, E_sns=0.0, E_sni=-0.02).data.E_sni == -0.02


# _evidence/hand_check_output.txt (vault 80_Inbox/2026-10-21/cylindrical_manufactured_gear/), printed to 10 decimals
H501_HAND = {
    "pinion": dict(hFaP0=3.5139682322, x_Emin=-0.2831905771, x_Es=0.1612751613, x_Ei=0.1416503226, s_ns=5.9086826514,
                   s_ni=5.8586826514, s_ats=2.2936542217, s_ati=2.2359859127, d_fEs=64.8482587577, d_fEi=64.7108848868,
                   d_Ffs=68.3863388710, d_Ffi=68.3371634111),
    "wheel": dict(hFaP0=3.5139682322, x_Emin=-0.9267813275, x_Es=0.0694751613, x_Ei=0.0498503226, s_ns=5.6747953789,
                  s_ni=5.6247953789, s_ats=2.5852866300, s_ati=2.5298924876, d_fEs=100.4403250721, d_fEi=100.3029512011,
                  d_Ffs=103.6269322828, d_Ffi=103.5532822759),
}


@pytest.mark.parametrize("which", ("pinion", "wheel"))
def test_hand_check_of_h501(which):
    """H501 (baseline case 2: z 20/30, mn 3.5, beta 15 deg, x 0.1809/0.0891, rack A), E_sns = -0.05 mm,
    E_sni = -0.10 mm on both gears: illustrative, not from a drawing (José, 2026-10-21, MG-19). Reference values
    from the lead's independent script ``_evidence/hand_check.py`` (math only, written from the page images of
    ISO_21771_2007; output ``hand_check_output.txt``), not from the code under test: Eq. (123)/(124), (118)/(119)
    with (49), (38) at d_a with x_E, (125), (128), (135)."""
    g = bc.pinion_wheel("H501")[which == "wheel"]
    m = ManufacturedCylindricalGear.from_ideal(g, E_sns=-0.05, E_sni=-0.10)
    for name, value in H501_HAND[which].items():
        assert getattr(m, name) == pytest.approx(value, abs=1e-9), name
    assert m.cutter == RACK_A and m.data.cutter is None and m.validate() == [] and m.warnings == ()


def test_ring_tip_thickness_against_an_independent_route():
    """Ring z -60, mn 2 mm, spur, x 0, default rack, zero allowances: s_ats = s_ati = 1.8331162424 mm at d_a = 116 mm,
    from ``_evidence/hand_check_output.txt`` (vault 80_Inbox/2026-10-21/cylindrical_manufactured_gear/, last line):
    the space width of the external gear z +60 at the same diameter, ISO_21771_2007 4.7.3, Eq. (43), a route
    independent of Eq. (38) with z/abs(z)."""
    m = ManufacturedCylindricalGear.from_ideal(_rings()[0], E_sns=0.0, E_sni=0.0)
    assert m.da == pytest.approx(116.0, rel=1e-12)
    assert m.s_ats == m.s_ati == pytest.approx(1.8331162424, abs=1e-9)


@pytest.mark.parametrize("ring", _rings()[:2], ids=("spur", "helical"))
def test_internal_gear(ring):
    """z < 0: x_E, s_n and the tip thickness (Eq. (38), both kinds) are computed; hFaP0, x_Emin, d_fE, d_Ff are None
    (a rack-type cutter cannot generate an internal gear; Eq. (125), (128), (135) are for external gears; MG-03),
    with one warning; the gear validates."""
    m = ManufacturedCylindricalGear.from_ideal(ring, E_sns=-0.05, E_sni=-0.10)
    assert m.x_Ei < m.x_Es < ring.x and m.s_ni < m.s_ns and 0.0 < m.s_ati < m.s_ats
    assert all(getattr(m, name) is None for name in ("hFaP0", "x_Emin", "d_fEs", "d_fEi", "d_Ffs", "d_Ffi"))
    assert len(m.warnings) == 1 and "a rack-type cutter cannot generate it" in m.warnings[0]
    assert m.validate() == []


def test_undercut_at_the_lower_allowance():
    """MG-10 (José, 2026-10-21): spur z 17, mn 2, x 0.01 (just above x_Emin = 0.005657, no nominal warning),
    E_sns = 0, E_sni = -0.1 mm: x_Ei = 0.01 - 0.1 / (4 tan 20 deg) < x_Emin. validate() has the undercut error
    (assemble with the checks refuses the gear); d_Ffi is None with its warning (Eq. (128) holds only without
    undercut, MG-11), d_Ffs is computed."""
    data = CylindricalGearManufacturingData(2.0, 17, x=0.01, E_sns=0.0, E_sni=-0.1, label="g")
    with pytest.raises(ValueError, match="undercut at the lower allowance"):
        ManufacturedCylindricalGear.assemble(FAMILY, data)
    m = ManufacturedCylindricalGear.assemble(FAMILY, data, check=False)
    assert m.x_Ei < m.x_Emin < m.x_Es
    assert m.validate() == [f"g: undercut at the lower allowance - x_Ei={m.x_Ei:.6f} < x_Emin={m.x_Emin:.6f} "
                            "(ISO_21771_2007 Eq. (135)); d_Ff (Eq. (128)) is not valid"]
    assert m.d_Ffi is None and m.d_Ffs is not None
    assert len(m.warnings) == 1 and m.warnings[0].startswith("g: d_Ffi not computed: x_Ei=")


def test_pointed_tooth_is_an_error():
    """s_at <= 0 at the tip circle is an error (ISO_21771_2007 7.9 prints no minimum; none is invented): spur
    z 10, mn 2, x 1.0, whose tooth is pointed at the nominal tip; both allowances are reported."""
    data = CylindricalGearManufacturingData(2.0, 10, x=1.0, E_sns=-0.05, E_sni=-0.1)
    m = ManufacturedCylindricalGear.assemble(FAMILY, data, check=False)
    assert m.s_ati < m.s_ats <= 0.0
    assert [e.split(" - ")[0] for e in m.validate()] == ["Gear: pointed tooth", "Gear: pointed tooth"]


def test_only_cylindrical_manufacturing_inputs_are_accepted():
    with pytest.raises(TypeError, match="needs CylindricalGearManufacturingData"):
        ManufacturedCylindricalGear.assemble(FAMILY, CylindricalGearData(2.0, 20))
    with pytest.raises(TypeError, match="needs a CylindricalFamily"):
        ManufacturedCylindricalGear.assemble(object(), CylindricalGearManufacturingData(2.0, 20, E_sns=0.0, E_sni=0.0))


# ----------------------------------------------------------------------------------------------
# ManufacturedCylindricalGearPair
# ----------------------------------------------------------------------------------------------
def _made_pair(gears, check=True, **kw):
    return ManufacturedCylindricalGearPair.assemble(FAMILY, gear1=gears[0], gear2=gears[1], check=check, **kw)


@pytest.mark.parametrize("case", PAIRS)
def test_zero_allowances_reproduce_the_ideal_pair(case):
    """Zero allowances, derived a_w, the baseline pairs (``test_cylindrical_family.PAIRS``): a ``GearPair`` and a
    ``ManufacturedCylindricalGearPair`` of the manufactured gears have every nominal pair attribute and the mesh
    forces of the ideal pair (Q02 tolerance), and jbn_max = jbn_min = 0 (Eq. (63) with x_E = x at the
    zero-backlash a_w), for external and internal pairs. Old case 4 fails validate() (Q39, Q24): check=False.

    ``validate()`` of the manufactured pair adds nothing for an external pair; for every internal pair it adds the
    two d_Nf1 < d_Ff1 messages of the pinion (both allowances): the concave ring tip reaches deeper into the pinion
    than a rack would, and with ISO 53 rack A a rack mate already ends at the end of the straight cutter flank
    (h_FaP0 = 0.99997 m_n vs h_aP 1.0 m_n), so the ring tip meets the pinion root fillet (issue MG-24). The
    manufactured pair is therefore built with check=False."""
    ideal = PAIRS[case]
    check = ideal.validate() == []
    made = (_zero(ideal.gear1), _zero(ideal.gear2))
    plain = GearPair.assemble(FAMILY, gear1=made[0], gear2=made[1], check=check)
    manufactured = _made_pair(made, check=False)
    for pair in (plain, manufactured):
        for name, value in vars(ideal).items():
            if name not in ("gear1", "gear2", "_family", "warnings"):
                assert _same(getattr(pair, name), value), name
        assert _same(dataclasses.asdict(FAMILY.mesh_forces(pair, 60.75, 1)), dataclasses.asdict(FAMILY.mesh_forces(ideal, 60.75, 1)))
    assert plain.warnings == ideal.warnings
    assert manufactured.warnings[:len(ideal.warnings)] == ideal.warnings
    assert manufactured.jbn_max == pytest.approx(0.0, abs=1e-12) and manufactured.jbn_min == pytest.approx(0.0, abs=1e-12)
    extra = manufactured.validate()[len(ideal.validate()):]
    if ideal.gear2.z > 0:
        assert extra == []
    else:
        assert len(extra) == 2 and all(m.startswith("gear 1 (pinion): active root diameter d_Nf1=") for m in extra)


def test_zero_allowances_at_an_imposed_centre_distance_give_the_nominal_backlash():
    """Zero allowances, a_w imposed 0.2 % above the derived one (H501): jbn_max = jbn_min = the nominal ``jbn`` of
    the pair (both Eq. (63), with x_E = x)."""
    p, w = bc.pinion_wheel("H501")
    al = GearPair.assemble(FAMILY, gear1=p, gear2=w).al * 1.002
    pair = _made_pair((_zero(p), _zero(w)), al=al)
    assert pair.jbn > 0.0
    assert pair.jbn_max == pytest.approx(pair.jbn, rel=1e-12) and pair.jbn_min == pytest.approx(pair.jbn, rel=1e-12)


def test_hand_check_of_the_h501_pair():
    """H501 (baseline case 2), E_sns = -0.05 mm, E_sni = -0.10 mm on both gears: illustrative, not from a drawing
    (José, 2026-10-21, MG-19). Reference values from ``_evidence/hand_check_output.txt`` (vault
    80_Inbox/2026-10-21/cylindrical_manufactured_gear/; independent script: Eq. (55) with a Newton inverse involute,
    Eq. (63), Eq. (64), (65) with d_Fa = d_a): jbn_max 0.1879385242, jbn_min 0.0939692621, d_Nf1 68.9500671039,
    d_Nf2 104.7691058425 mm; jbn = -(E_sn1 + E_sn2) cos(alpha_n) at the derived a_w. d_Nf >= d_Ff holds."""
    made = [ManufacturedCylindricalGear.from_ideal(g, E_sns=-0.05, E_sni=-0.10) for g in bc.pinion_wheel("H501")]
    pair = _made_pair(made)
    for name, value in dict(jbn_max=0.1879385242, jbn_min=0.0939692621, d_Nf1=68.9500671039, d_Nf2=104.7691058425).items():
        assert getattr(pair, name) == pytest.approx(value, abs=1e-9), name
    assert pair.d_Nf1 >= made[0].d_Ffi and pair.d_Nf2 >= made[1].d_Ffi
    assert pair.validate() == [] and pair.warnings == ()


def test_active_root_below_the_root_form_diameter_is_an_error():
    """Constructed failing pair (hard-coded): pinion z 12, x 0.6, with a long-addendum wheel z 60, x 0, haP 1.1,
    mn 2 mm, zero allowances, derived a_w 73.1358 mm. The pinion is not undercut, the path stays between the
    base-tangent points (T1A > 0) and the wheel tip clears the pinion root circle (+0.236 mm), but it reaches
    d_Nf1 = 22.7525 mm < d_Ff1 = 22.8273 mm (-0.075 mm): contact on the pinion root fillet (ISO_21771_2007 5.4.1,
    Eq. (66), (67)); one error per allowance, and nothing else."""
    def made(z, x, **kw):
        return ManufacturedCylindricalGear.assemble(
            FAMILY, CylindricalGearManufacturingData(2.0, z, x=x, b=10.0, E_sns=0.0, E_sni=0.0, label=f"z{z}", **kw))

    pair = _made_pair((made(12, 0.6), made(60, 0.0, haP=1.1)), check=False)
    assert pair.T1A > 0.0 and pair.al - pair.da2_w / 2 - pair.gear1.rf == pytest.approx(0.2358117, abs=1e-6)
    assert pair.d_Nf1 - pair.gear1.d_Ffs == pytest.approx(-0.0748709, abs=1e-6) and pair.gear1.d_Ffs == pair.gear1.d_Ffi
    msgs = pair.validate()
    assert [m.split(" mm at the ")[1][:5] for m in msgs] == ["upper", "lower"]
    assert all(m.startswith(f"gear 1 (z12): active root diameter d_Nf1={pair.d_Nf1:.4f} mm < root form diameter")
               for m in msgs)
    with pytest.raises(ValueError, match="the mate's tip reaches the root fillet"):
        _made_pair((pair.gear1, pair.gear2))


def test_internal_pair_warns_that_the_ring_is_not_checked():
    """Internal pair 24/-60 (case 4n), zero allowances: the ring has no d_Ff (rack cutter), so d_Nf >= d_Ff is not
    checked for gear 2 (one warning of the pair). check=False: the pinion messages of MG-24."""
    pair = _made_pair((_zero(PAIRS["case4n"].gear1), _zero(PAIRS["case4n"].gear2)), check=False)
    assert pair.warnings == ("d_Nf >= d_Ff not checked for gear 2 (ring): d_Ffs, d_Ffi not computed (internal gear or "
                             "undercut at that allowance).",)


def test_pair_of_plain_gears_is_refused_and_repr_names_the_class():
    p, w = bc.pinion_wheel("H501")
    with pytest.raises(TypeError, match="needs two ManufacturedCylindricalGear, got Gear and Gear"):
        _made_pair((p, w))
    with pytest.raises(TypeError, match="needs a CylindricalFamily"):
        ManufacturedCylindricalGearPair.assemble(object(), gear1=_zero(p), gear2=_zero(w))
    assert repr(_made_pair((_zero(p), _zero(w)))).startswith("ManufacturedCylindricalGearPair(z1=20, z2=30, ")


# ----------------------------------------------------------------------------------------------
# placeholders (block 4)
# ----------------------------------------------------------------------------------------------
ANNEX_A = (("span_measurement", (3,)), ("dimension_over_balls", (7.0,)), ("chordal_tooth_thickness", ()),
           ("chordal_height", ()), ("double_flank_centre_distance", None))


@pytest.mark.parametrize("name, args", ANNEX_A, ids=[n for n, _ in ANNEX_A])
def test_annex_a_test_dimensions_are_placeholders(name, args):
    """MG-15: each Annex A test dimension, on the family and as the shortcut of the gear, raises
    NotImplementedError naming the issue."""
    gear = _zero(bc.pinion_wheel("H501")[0])
    args = (gear,) if args is None else args          # the master gear of A.8
    with pytest.raises(NotImplementedError, match="MG-15"):
        getattr(FAMILY, name)(gear, *args)
    with pytest.raises(NotImplementedError, match="MG-15"):
        getattr(gear, name)(*args)


def test_pinion_type_cutter_is_a_placeholder():
    """MG-16: a pinion-type cutter cannot be built."""
    with pytest.raises(NotImplementedError, match="MG-16"):
        GeneratingPinionCutter(z0=25, x0=0.0, haP0=1.25, rhoaP0=0.38, alpha_P0_deg=20.0)


@pytest.mark.parametrize("target, field, value, issue", (
    ("data", "quality_grade", 6, "MG-18"), ("data", "q_min", 0.05, "MG-02"), ("data", "q_max", 0.1, "MG-02"),
    ("cutter", "prP0", 0.02, "MG-17"), ("cutter", "alpha_prP0_deg", 10.0, "MG-17")))
def test_placeholder_fields_are_refused(target, field, value, issue):
    """MG-02, MG-17, MG-18: a placeholder field of the data or of the cutter set to a value gives one validate()
    message naming the field and the issue, and the gear is not assembled."""
    kw = {"cutter": dataclasses.replace(RACK_A, **{field: value})} if target == "cutter" else {field: value}
    data = CylindricalGearManufacturingData(2.0, 20, E_sns=-0.05, E_sni=-0.1, **kw)
    errors = data.validate()
    assert len(errors) == 1 and field in errors[0] and issue in errors[0]
    with pytest.raises(ValueError, match=issue):
        ManufacturedCylindricalGear.assemble(FAMILY, data)
