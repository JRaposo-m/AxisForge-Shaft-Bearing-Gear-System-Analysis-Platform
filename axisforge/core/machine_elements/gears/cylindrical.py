"""Cylindrical involute gears on parallel axes, external and internal, spur and helical.

One family for both kinds of teeth under the sign convention of ISO_21771_2007, 4.1.5: the number
of teeth of an internal gear is negative, every diameter and the centre distance are positive. The
file follows the order of the standard: basic rack (ISO_53_1998), one gear (clause 4), the pair
(clause 5), limits and checks (clause 7), the tooth forces, then the contact inputs along the path
(5.4-5.6). Lengths in [mm], angles in [rad] (inputs in [deg]), torques in [N m], forces in [N].
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, fields
import numpy as np
from scipy import optimize

from axisforge.core.machine_elements.gears.base import Gear, GearFamily, MeshForces, register_gear_family


def _involute(ang):
    """Involute function inv(ang) = tan(ang) - ang [rad], ang in [rad] (ISO_21771_2007, 4.3.9, Eq. (18))."""
    return np.tan(ang) - ang


def _inv_residual(xx, inv_a):
    """Residual ``inv_a - inv(xx)`` [rad]; zero when the angle xx [rad] has involute inv_a."""
    return inv_a - np.tan(xx) + xx


def _kind(z) -> str:
    """Kind of teeth from the sign of z (ISO_21771_2007, 4.1.5): key of the ``_LEGACY`` table."""
    # Q55 (José, 2026-10-15): keep. The kind is the sign of z (it was the class the caller chose).
    return "internal" if z < 0 else "external"


# ==================================================================================================
# Generation of the teeth by a rack-type cutter (ISO_21771_2007, clause 7): pure functions of numbers
# and of the cutter record, shared by the ideal undercut check and the manufactured gear. External
# gear, rack cutter (or hob) without protuberance, finish cutting.
# ==================================================================================================
def _min_generating_profile_shift(z, mn, alphat, beta, cutter) -> tuple[float, float]:
    """Smallest generating profile shift coefficient that cuts the teeth without undercut.

    Undercut starts when the end of the straight tip flank of the tool passes the interference
    point T (the base-tangent point) on the generating line of action (Fig. 39); the limit is the
    shift that puts the end of that straight flank on the line of action at the base-tangent point.

    Parameters
    ----------
    z : int
        Number of teeth [-], positive (external gear).
    mn : float
        Normal module [mm].
    alphat : float
        Transverse pressure angle of the gear [rad].
    beta : float
        Reference helix angle of the gear [rad].
    cutter : GeneratingRackProfile
        Rack-type tool (coefficients h_aP0 / m_n, rho_aP0 / m_n).

    Returns
    -------
    tuple of float
        ``(x_Emin, hFaP0)``: minimum generating profile shift coefficient x_Emin [-] and the
        straight part of the tip flank of the tool h_FaP0 [mm].

    Notes
    -----
    h_FaP0 = m_n (h_aP0 - rho_aP0 (1 - sin(alpha_t))): the bracket of Eq. (128) / (130), the
    "straight part of tip flank of tool-generating profile" of the Fig. 39 key. It is taken with
    the alpha_t of the GEAR, as printed. Issue MG-01: Fig. 35 draws h_FaP0 in the normal section of
    the tool, where the tip-radius geometry would give alpha_P0 instead; kept as printed (0.004 in
    x_Emin at beta 15 deg, z 17). Valid for an external gear and a non-protuberance rack cutter
    or hob only.

    References
    ----------
    .. [1] ISO_21771_2007, 7.7, Eq. (135), p. 59 (PDF p. 63); 7.6, Fig. 39 key, Eq. (128),
       p. 57-58 (PDF p. 61-62).
    """
    hFaP0 = mn * (cutter.haP0 - cutter.rhoaP0 * (1.0 - np.sin(alphat)))
    x_Emin = hFaP0 / mn - z * np.sin(alphat) ** 2 / (2.0 * np.cos(beta))      # Eq. (135)
    return x_Emin, hFaP0


def _ideal_x_Emin(data, alphat, beta) -> float:
    """Limit of the IDEAL undercut check: x_Emin [-] of Eq. (135) with the counterpart of the basic rack.

    The one criterion of ``assemble_gear`` (warning) and ``undercutting``: the gear is undercut if
    x < x_Emin (x_E = x). External gear only: -inf for an internal gear (Eq. (135) is printed for
    external gears), so that no ring is reported.

    Parameters
    ----------
    data : CylindricalGearData
        Gear data (z, mn and the basic rack).
    alphat, beta : float
        Transverse pressure angle and reference helix angle of the gear [rad].

    Returns
    -------
    float
        x_Emin [-], or -inf for z < 0.
    """
    if data.z < 0:
        return -np.inf
    return _min_generating_profile_shift(data.z, data.mn, alphat, beta, GeneratingRackProfile.from_basic_rack(data))[0]


def _generating_profile_shift(x, mn, alpha_n, E_sn) -> float:
    """Generating profile shift coefficient x_E of a tooth thinned by the allowance E_sn.

    The cutter is fed deeper than the nominal x by the amount that removes E_sn from the normal
    tooth thickness on both flanks: E_sn / (2 tan(alpha_n)) radially.

    Parameters
    ----------
    x : float
        Nominal profile shift coefficient [-].
    mn : float
        Normal module [mm].
    alpha_n : float
        Normal pressure angle [rad].
    E_sn : float
        Tooth thickness allowance [mm], E_sns (upper) or E_sni (lower); negative thins the tooth.

    Returns
    -------
    float
        x_E [-]: x_Es from E_sns (Eq. (123)) or x_Ei from E_sni (Eq. (124)).

    Notes
    -----
    Finish cutting only (machining allowance q = 0); the pre-finish shifts of Eq. (120)-(122) are
    not computed.

    References
    ----------
    .. [1] ISO_21771_2007, 7.4, Eq. (123), (124), p. 56 (PDF p. 60).
    """
    return x + E_sn / (2.0 * mn * np.tan(alpha_n))


def _generated_root_diameter(d, x_E, mn, cutter) -> float:
    """Root diameter cut by a rack-type tool at the generating profile shift x_E.

    Parameters
    ----------
    d : float
        Reference diameter [mm].
    x_E : float
        Generating profile shift coefficient [-] (x_Es or x_Ei).
    mn : float
        Normal module [mm].
    cutter : GeneratingRackProfile
        Rack-type tool (coefficient h_aP0 / m_n).

    Returns
    -------
    float
        Generated root diameter d_fE [mm]: the tool tip line, h_aP0 below the tool datum line,
        which lies x_E m_n outside the reference circle.

    Notes
    -----
    External gear, rack cutter; the standard asks for the values of the process that produces the
    finished tooth.

    References
    ----------
    .. [1] ISO_21771_2007, 7.5, Eq. (125), p. 57 (PDF p. 61).
    """
    return d + 2.0 * x_E * mn - 2.0 * cutter.haP0 * mn


def _root_form_diameter(d, db, alphat, x_E, mn, cutter) -> float:
    """Root form diameter: where the involute generated by the straight tool flank ends.

    Below d_Ff the flank is the fillet (trochoid) cut by the tip radius of the tool.

    Parameters
    ----------
    d, db : float
        Reference and base diameters [mm].
    alphat : float
        Transverse pressure angle [rad].
    x_E : float
        Generating profile shift coefficient [-] (x_Es or x_Ei).
    mn : float
        Normal module [mm].
    cutter : GeneratingRackProfile
        Rack-type tool (coefficients h_aP0 / m_n, rho_aP0 / m_n).

    Returns
    -------
    float
        Root form diameter d_Ff [mm].

    Notes
    -----
    Eq. (128), first form, with h = m_n (h_aP0 - x_E - rho_aP0 (1 - sin(alpha_t))) [mm], the
    height of the end of the straight tool flank below the reference line of the gear:
    d_Ff = sqrt((d sin(alpha_t) - 2 h / sin(alpha_t))**2 + d_b**2). External gear, rack cutter
    with tool tips parallel to the datum line, valid only without undercut (x_E >= x_Emin) and
    without pre-finish root relief.

    References
    ----------
    .. [1] ISO_21771_2007, 7.6, Eq. (128), p. 58 (PDF p. 62).
    """
    h = mn * (cutter.haP0 - x_E - cutter.rhoaP0 * (1.0 - np.sin(alphat)))
    return np.sqrt((d * np.sin(alphat) - 2.0 * h / np.sin(alphat)) ** 2 + db ** 2)


# ==================================================================================================
# _LEGACY: what the external and internal gears do DIFFERENTLY, kept by a ruling (Q25 deferred, Q55
# keep). One entry per quantity, one expression per kind of teeth, carried unchanged from the four
# modules this file replaces; each is reached only through ``_legacy(name, z)``. The texts of the
# pair are one for both kinds (Q50, José 2026-10-15).
# ==================================================================================================
def _virtual_teeth_external(mn, z, beta):
    zn = z / (np.cos(beta) ** 3)    # teeth of the spur gear equivalent in the normal section
    return {"zn": zn, "de": mn * zn}


def _beta_range_external(data, tag):
    if not (0.0 <= data.beta_n_deg < 45.0):
        return [f"{tag}: beta_n_deg out of range [0, 45), got {data.beta_n_deg}"]
    return []


def _gear_limits_internal(gear, tag):
    errors = []
    if gear.da >= gear.d:
        errors.append(
            f"{tag}: tip diameter da={gear.da:.4f} ≥ pitch diameter d={gear.d:.4f}. "
            "Internal gear addendum must reduce da below d."
        )
    if gear.da <= gear.db:   # no involute at the tip: arccos(db/da) of the tip pressure angle has no value
        errors.append(
            f"{tag}: tip circle (da={gear.da:.4f}) ≤ base circle (db={gear.db:.4f}). "
            "No valid involute at tooth tip — increase z or x, or reduce haP."
        )
    return errors


def _internal_interference(pair) -> list[str]:
    """Trochoid and trimming interference of an internal pair (pinion inside the ring).

    Tests of SDP-SI, "Elements of Metric Gear Technology", section 5 (the KHK technical reference),
    each written as "no interference when ...". The third test of that section, involute
    interference, Eq. (5-2): z1 / z2 >= 1 - tan(alpha_a2) / tan(alpha_w) with alpha_a2 =
    arccos(d_b2 / d_a2) (Eq. (5-3)) and d_a2 >= d_b2 (Eq. (5-5), the ring tip limit of
    ``validate_gear``), is the Q39 criterion T1A >= 0 of ``validate_pair``: its margin
    z1/z2 - 1 + tan(alpha_a2)/tan(alpha_w) equals T1A / (r_b2 tan(alpha_w)) (with T1A = r_b2
    tan(alpha_a2) - a_w sin(alpha_w) + ..., a_w sin(alpha_w) = (r_b2 - r_b1) tan(alpha_w) and
    r_b1/r_b2 = z1/z2). It is therefore reported once, by ``validate_pair``, and SDP Eq. (5-2) is
    the library source of that criterion for internal pairs (lead decision, 2026-10-15).

    - trochoid interference, Eq. (5-6)-(5-8): theta1 z1 / z2 + inv(alpha_w) - inv(alpha_a2) >= theta2,
      theta1 = arccos((r_a2**2 - r_a1**2 - a**2) / (2 a r_a1)) + inv(alpha_a1) - inv(alpha_w),
      theta2 = arccos((a**2 + r_a2**2 - r_a1**2) / (2 a r_a2)): the pinion tip, leaving the mesh,
      must not touch the ring tooth tip (its relative path is a trochoid);
    - trimming interference, Eq. (5-9)-(5-10): theta1 + inv(alpha_a1) - inv(alpha_w) >=
      z2 / z1 (theta2 + inv(alpha_a2) - inv(alpha_w)), theta1 = arcsin(sqrt((1 - (cos(alpha_a1) /
      cos(alpha_a2))**2) / (1 - (z1 / z2)**2))), theta2 = arcsin(sqrt(((cos(alpha_a2) /
      cos(alpha_a1))**2 - 1) / ((z2 / z1)**2 - 1))): the pinion cannot be put in radially.

    Parameters
    ----------
    pair : GearPair
        Internal pair of this family (z2 < 0).

    Returns
    -------
    list of str
        One message per failed test (trochoid, trimming), with the two sides of its inequality;
        empty if none fails.

    Notes
    -----
    Symbols of SDP (z2 > 0, x2_KHK = -x2_ISO) and the attributes used here: z1 = ``gear1.z``,
    z2 = abs(``gear2.z``); r_a1, r_a2 = ``da1_w / 2``, ``da2_w / 2`` (working tips, ISO_21771_2007
    Eq. (33), no cP); r_b1, r_b2 = ``gear1.rb``, ``gear2.rb``; a (a_x in Eq. (5-4)) = ``al``, the
    working centre distance; alpha_w = ``alphatw``, the transverse working pressure angle (Eq. (5-4)
    is the same relation as ISO_21771_2007, Eq. (54)). SDP writes the tests for spur gears; for a
    helical pair they are applied in the transverse plane (no published check). Validated at x = 0
    by the 18 limits of the KHK catalogue table "Allowable Mating Pinions" (z2 = 60..200, m 1,
    20 deg); for x != 0 they hold by the symbols of the equations, with no published check. The
    trimming test of a helical pair is an extrapolation of the spur derivation (a helical ring is
    normally assembled axially, where trimming does not arise). nan means that the expression has
    no real value; the test is then counted as failed, as in the scan that reproduced the KHK
    table. For the trochoid test nan means that the two tip circles do not intersect: such a pair
    has eps_alpha < 1 and is refused by ``validate_pair`` anyway.

    References
    ----------
    .. [1] SDP-SI, "Elements of Metric Gear Technology", section 5, Eq. (5-2)-(5-10) (vault
       ``80_Inbox/2026-10-15/internal_gears_validation_khk.md``; José, Q24, 2026-10-15).
    .. [2] KHK catalogue, internal gears, p. 208, "Allowable Mating Pinions and Number of Teeth".
    """
    z1, z2 = pair.gear1.z, abs(pair.gear2.z)
    ra1, ra2 = pair.da1_w / 2.0, pair.da2_w / 2.0
    rb1, rb2 = pair.gear1.rb, pair.gear2.rb
    a, alpha_w = pair.al, pair.alphatw
    errors = []
    with np.errstate(invalid="ignore"):     # no real value -> nan -> the test fails (see Notes)
        alpha_a1 = np.arccos(rb1 / ra1)                                    # Eq. (5-8)
        alpha_a2 = np.arccos(rb2 / ra2)                                    # Eq. (5-3)
        # trochoid, Eq. (5-6)-(5-7)
        theta1 = np.arccos((ra2**2 - ra1**2 - a**2) / (2 * a * ra1)) + _involute(alpha_a1) - _involute(alpha_w)
        theta2 = np.arccos((a**2 + ra2**2 - ra1**2) / (2 * a * ra2))
        lhs, rhs = theta1 * z1 / z2 + _involute(alpha_w) - _involute(alpha_a2), theta2
        if not lhs >= rhs:
            errors.append(f"trochoid interference of the internal pair (SDP-SI Eq. (5-6)): "
                          f"{lhs:.6f} < {rhs:.6f} rad")
        # trimming, Eq. (5-9)-(5-10)
        theta1 = np.arcsin(np.sqrt((1 - (np.cos(alpha_a1) / np.cos(alpha_a2))**2) / (1 - (z1 / z2)**2)))
        theta2 = np.arcsin(np.sqrt(((np.cos(alpha_a2) / np.cos(alpha_a1))**2 - 1) / ((z2 / z1)**2 - 1)))
        lhs = theta1 + _involute(alpha_a1) - _involute(alpha_w)
        rhs = z2 / z1 * (theta2 + _involute(alpha_a2) - _involute(alpha_w))
        if not lhs >= rhs:
            errors.append(f"trimming interference of the internal pair (SDP-SI Eq. (5-9)): "
                          f"{lhs:.6f} < {rhs:.6f} rad")
    return errors


_LEGACY = {
    # QUESTION(Q25): deferred (José, 2026-10-15): no z_n for a ring until the rating step (check the sign ISO 6336 expects, Q29).
    "virtual_teeth": {"external": _virtual_teeth_external, "internal": lambda mn, z, beta: {}},
    # Q55 (José, 2026-10-15): keep. The helix-angle range is checked for an external gear only.
    "beta_range": {"external": _beta_range_external, "internal": lambda data, tag: []},
    # Q55 (José, 2026-10-15): keep. da < d and da > db (ring). None for an external gear: its undercut is
    # a warning, assemble_gear["warnings"] (MG-07, José 2026-10-21).
    "gear_limits": {"external": lambda gear, tag: [], "internal": _gear_limits_internal},
}


def _legacy(name: str, z):
    """Entry of ``_LEGACY`` for the kind of teeth given by the sign of ``z``."""
    return _LEGACY[name][_kind(z)]


# ==================================================================================================
# Input record of one gear (basic rack: ISO_53_1998)
# ==================================================================================================
@dataclass(frozen=True)
class CylindricalGearData:
    """Input data of one cylindrical involute gear: basic rack, size, profile shift, finish.

    The sign of ``z`` gives the kind of teeth: positive for an external gear, negative for an
    internal gear (ring), as in ISO_21771_2007, 4.1.5. Everything else is a magnitude.

    Parameters
    ----------
    mn : float
        Normal module m_n [mm].
    z : int
        Number of teeth [-], signed: z < 0 for an internal gear. Must not be zero.
    Ca, Cf : float, optional
        Tip and root relief. Stored only; no formula uses them and the code states no unit
        (ISO_6336-1_2019, Table 2, gives [um] for C_a and C_f). Default 0.0.
    x : float, optional
        Profile shift coefficient [-] (ISO_21771_2007, 4.2.9). Default 0.0.
    b : float, optional
        Face width [mm]. Default 0.0.
    alpha_n_deg : float, optional
        Normal pressure angle alpha_n of the basic rack [deg]. Default 20.0.
    beta_n_deg : float, optional
        Helix angle beta at the reference cylinder [deg], a magnitude; its hand is ``hand``. The
        "_n" of the name has no meaning in the formulas. Default 0.0 (spur gear).
    hand : int, optional
        Hand of the helix: +1 right-hand, -1 left-hand [-]; required (+1 or -1) when
        beta_n_deg != 0, 0 allowed only for a spur gear (D-0010 item 17; Q08, José 2026-10-15).
        Right-hand: the tooth trace turns in the right-hand sense about the gear axis when
        followed along it (geometric definition of D-0010 / ``MeshForces``; the definition of the
        flank direction in ISO_21771_2007, 4.1.4 was not transcribed). Default 0.
    haP : float, optional
        Addendum coefficient of the basic rack h_aP* [-] (addendum = haP mn). Default 1.0.
    cP : float, optional
        Bottom clearance coefficient of the basic rack c_P* [-]. Default 0.25.
    rfP : float, optional
        Root fillet radius coefficient of the basic rack rho_fP* [-]. Default 0.38.
    Ra, Rq, Rz : float, optional
        Arithmetic mean, root mean square and mean peak-to-valley roughness of the flanks [um].
        Defaults 0.8, 1.0, 4.0 (no library source).
    label : str, optional
        Name used in validation messages.
    material_id : str, optional
        Material key; stored only.
    position : float, optional
        Axial position of the centre of the face width on its shaft [mm]; stored only, read by
        ``ShaftSystem``.

    Notes
    -----
    The defaults alpha_n = 20 deg, haP = 1.0, cP = 0.25, rfP = 0.38 are the standard basic rack
    tooth profile of ISO_53_1998, Table 2 (checked), there as multiples of the module. haP, cP and
    rfP are dimensionless coefficients here, whereas ISO_6336-1_2019, Table 2, lists h_aP and h_fP
    as lengths [mm].

    References
    ----------
    .. [1] ISO_53_1998, Table 2, p. 3 (PDF p. 5).
    .. [2] ISO_21771_2007, 4.1.5 (sign of z), 4.2.9 (profile shift).
    """
    # Q18 (José, 2026-10-15): keep - the rack defaults are those of ISO_53_1998, Table 2 (the edition held).
    mn: float
    z: int
    _: KW_ONLY
    Ca: float = 0.0
    Cf: float = 0.0
    x: float = 0.0
    b: float = 0.0
    alpha_n_deg: float = 20.0
    beta_n_deg: float = 0.0
    hand: int = 0
    haP: float = 1.0
    cP: float = 0.25
    rfP: float = 0.38
    Ra: float = 0.8
    Rq: float = 1.0
    Rz: float = 4.0
    label: str = ""
    material_id: str = ""
    position: float = 0.0

    def validate(self) -> list[str]:
        """Check the domains of the inputs.

        Returns
        -------
        list of str
            One message per violation; empty if valid. Checks: mn > 0, z != 0, b >= 0,
            0 < alpha_n_deg < 45, 0 <= beta_n_deg < 45 (external gear only), hand +1 or -1 for a
            helical gear (-1, 0 or +1 for a spur gear), Ra, Rq, Rz >= 0. The limits 45 deg have no
            library source.
        """
        errors: list[str] = []
        tag = self.label or "Gear"

        if self.mn <= 0:
            errors.append(f"{tag}: mn must be > 0, got {self.mn}")
        if self.z == 0:
            errors.append(f"{tag}: z must be != 0, got {self.z}")
        if self.b < 0:
            errors.append(f"{tag}: b must be >= 0, got {self.b}")
        if not (0.0 < self.alpha_n_deg < 45.0):
            errors.append(
                f"{tag}: alpha_n_deg out of range (0, 45), got {self.alpha_n_deg}"
            )
        errors.extend(_legacy("beta_range", self.z)(self, tag))
        # Q08 (José, 2026-10-15): the axial force of a helical mesh needs the hand of the helix
        if self.hand not in ((1, -1) if self.beta_n_deg != 0.0 else (1, 0, -1)):
            errors.append(f"{tag}: hand must be +1 (right) or -1 (left) for a helical gear "
                          f"(0 only for beta_n_deg = 0), got {self.hand!r}")
        if self.Ra < 0:
            errors.append(f"{tag}: Ra must be >= 0, got {self.Ra}")
        if self.Rq < 0:
            errors.append(f"{tag}: Rq must be >= 0, got {self.Rq}")
        if self.Rz < 0:
            errors.append(f"{tag}: Rz must be >= 0, got {self.Rz}")
        return errors


@dataclass(frozen=True, kw_only=True)
class GeneratingRackProfile:
    """Rack-type cutting tool (rack cutter or hob) that generates the teeth: the counterpart rack.

    The tool is the complement of the basic rack: its teeth fill the tooth spaces of the basic rack,
    so its addendum cuts the root and its tip radius the root fillet of the gear. Coefficients are
    multiples of the normal module m_n, like ``haP``, ``cP``, ``rfP`` of ``CylindricalGearData``.

    Parameters
    ----------
    haP0 : float
        Addendum of the tool h_aP0 / m_n [-], measured from the tool datum line.
    rhoaP0 : float
        Tip radius of the tool rho_aP0 / m_n [-] (Fig. 35 labels it r_aP0, Eq. (128) prints
        rho_aP0).
    alpha_P0_deg : float
        Pressure angle of the tool flank alpha_P0 [deg].

    Notes
    -----
    The straight part of the tip flank h_FaP0 is not a field: it needs the transverse pressure angle
    of the gear (``_min_generating_profile_shift``).

    References
    ----------
    .. [1] ISO_21771_2007, 7.1, Fig. 35, p. 53 (PDF p. 57).
    .. [2] ISO_53_1998, 3.2, Fig. 1, Table 2, p. 2-3 (PDF p. 4-5).
    """
    haP0: float
    rhoaP0: float
    alpha_P0_deg: float

    @classmethod
    def from_basic_rack(cls, data: CylindricalGearData) -> GeneratingRackProfile:
        """Counterpart rack of the basic rack of ``data``.

        haP0 = haP + cP, rhoaP0 = rfP, alpha_P0_deg = alpha_n_deg: the counterpart rack encloses the
        bottom land of the basic rack (ISO_21771_2007, 7.1, Fig. 35), i.e. it is the basic rack
        made symmetrical about the profile reference line P-P and displaced by half a pitch
        (ISO_53_1998, 3.2, Fig. 1, "mating rack"), so its teeth fill the tooth spaces of the basic
        rack: its addendum is the basic rack dedendum h_fP = h_aP + c_P and its tip radius the basic rack
        fillet radius rho_fP. No page prints these three equalities as equations: they follow from
        the figures.

        Parameters
        ----------
        data : CylindricalGearData
            Gear data whose basic rack (``haP``, ``cP``, ``rfP``, ``alpha_n_deg``) is used.

        Returns
        -------
        GeneratingRackProfile
        """
        return cls(haP0=data.haP + data.cP, rhoaP0=data.rfP, alpha_P0_deg=data.alpha_n_deg)

    def validate(self) -> list[str]:
        """Check the domains of the inputs.

        Returns
        -------
        list of str
            One message per violation; empty if valid. Checks: haP0 > 0, rhoaP0 >= 0,
            0 < alpha_P0_deg < 45 (no library source for the 45 deg limit, as in
            ``CylindricalGearData``).
        """
        errors: list[str] = []
        if self.haP0 <= 0:
            errors.append(f"cutter: haP0 must be > 0, got {self.haP0}")
        if self.rhoaP0 < 0:
            errors.append(f"cutter: rhoaP0 must be >= 0, got {self.rhoaP0}")
        if not (0.0 < self.alpha_P0_deg < 45.0):
            errors.append(f"cutter: alpha_P0_deg out of range (0, 45), got {self.alpha_P0_deg}")
        return errors

    def validate_or_raise(self) -> None:
        """Raise ``ValueError`` with all messages of ``validate()`` if any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


@dataclass(frozen=True, kw_only=True)
class CylindricalGearManufacturingData(CylindricalGearData):
    """Input data of one cylindrical gear as made: the nominal data plus the manufacturing inputs.

    All the fields of ``CylindricalGearData`` (``mn`` and ``z`` may still be positional), then
    two kinds of input:

    - design inputs, read from the drawing: the tooth thickness allowances ``E_sns``, ``E_sni``;
    - process inputs, chosen by the workshop: the generating tool ``cutter``.

    Parameters
    ----------
    E_sns, E_sni : float
        Upper and lower tooth thickness allowances [mm], normal section: s_ns = s_n + E_sns,
        s_ni = s_n + E_sni. Required. A negative allowance reduces the tooth thickness.
    cutter : GeneratingRackProfile or None, optional
        Rack-type tool that generates the teeth. None (default): the counterpart of the basic rack
        of this record, ``GeneratingRackProfile.from_basic_rack(self)``.

    References
    ----------
    .. [1] ISO_21771_2007, 7.3, Eq. (118), (119), p. 55 (PDF p. 59) (allowances); 7.1-7.2 (tool).
    """
    E_sns: float
    E_sni: float
    cutter: GeneratingRackProfile | None = None

    def validate(self) -> list[str]:
        """Check the domains of the inputs.

        Returns
        -------
        list of str
            The messages of ``CylindricalGearData.validate``, then: E_sni <= E_sns <= 0; the
            messages of ``cutter.validate()`` when a cutter is given; the tool pressure angle
            alpha_P0_deg equal to alpha_n_deg (relative 1e-6).

        Notes
        -----
        E_sns <= 0: ISO_21771_2007, 7.3 says that "a negative tooth thickness allowance reduces the
        tooth thickness" and prints no sign limit; no library source for the limit E_sns <= 0.
        alpha_P0 = alpha_n: otherwise alpha_wt0 and d_wE replace alpha_t and d in the generating
        equations (ISO_21771_2007, Fig. 39 NOTE, p. 57 / PDF p. 61); not implemented.
        """
        errors = super().validate()
        tag = self.label or "Gear"
        if not (self.E_sni <= self.E_sns <= 0.0):
            errors.append(f"{tag}: tooth thickness allowances must satisfy E_sni <= E_sns <= 0, "
                          f"got E_sns={self.E_sns}, E_sni={self.E_sni}")
        if self.cutter is not None:
            errors.extend(f"{tag}: {m}" for m in self.cutter.validate())
            # same tolerance as the pressure angles of two gears in mesh (assemble_pair)
            if not np.isclose(self.cutter.alpha_P0_deg, self.alpha_n_deg, rtol=1e-6):
                errors.append(f"{tag}: cutter alpha_P0_deg={self.cutter.alpha_P0_deg} differs from "
                              f"alpha_n_deg={self.alpha_n_deg}; not implemented (ISO_21771_2007, Fig. 39 NOTE)")
        return errors


# ==================================================================================================
# The family
# ==================================================================================================
@register_gear_family
class CylindricalFamily(GearFamily):
    """Cylindrical involute gears and gear pairs on parallel axes (ISO_21771_2007), external and internal.

    ``assemble_gear`` gives the geometry of one gear (clause 4), ``assemble_pair`` that of two gears
    in mesh (clause 5), ``validate_gear`` / ``validate_pair`` the limits (clause 7 and checks with
    no library source), ``mesh_forces`` the forces of the mesh for a torque (D-0010,
    ``MeshForces``), ``path_points``, ``curvature_radii``, ``surface_speeds`` the geometry and
    kinematics along the path of contact (the load sharing and the contact inputs are in
    ``axisforge.solvers.machine_elements.gears.contact``). An internal pair is a pair whose gear 2 has z < 0.

    Notes
    -----
    Where the external and the internal classes this family replaces computed a quantity
    differently and no ruling has chosen, both expressions are kept in the module table
    ``_LEGACY``, each with its question id; the docstrings below say "by kind" for those.

    Assumptions: rigid, unmodified involute flanks; aligned gears (the common face width is
    min(b1, b2)); the tip circle is taken as the limit of the active profile (the standard uses
    the active tip diameter d_Na).

    Warnings
    --------
    - Q28 (José, 2026-10-15: keep, note only; finding I8): for an internal pair with a_w > 0, ISO_21771_2007 Eq. (56) first form,
      (64), (65), (68), (69) and Eq. (92) first form do not hold as printed (two readers). None of
      them is coded here: the signed segments ``T1A``..``T2E`` and the form d_b / cos(alpha_wt)
      are used instead.
    - Q29 (José, 2026-10-15: keep, note for the rating step; finding I9): ISO_6336-1_2019,
      Table 2, footnote a, makes a, d, d_a and z_2 negative for internal gearing; here, as in
      ISO 21771, only z is signed. A rating formula that contains a, d, d_a, z2 or u must be
      checked for the sign it expects.

    References
    ----------
    .. [1] ISO_21771_2007, clauses 4, 5 and 7 (equations cited in each method).
    .. [2] ISO_53_1998, Table 2 (basic rack).
    """

    @property
    def name(self) -> str:
        """Family identifier, ``'cylindrical'``."""
        return "cylindrical"

    @property
    def axes(self) -> str:
        """Parallel axes."""
        return "parallel"

    # ----------------------------------------------------------------------------------------------
    # One gear (ISO 21771 clause 4)
    # ----------------------------------------------------------------------------------------------
    def assemble_gear(self, data: CylindricalGearData) -> dict:
        """Reference, base, tip and root cylinders, pitches and angles of one gear.

        Parameters
        ----------
        data : CylindricalGearData
            Input record; z < 0 for an internal gear.

        Returns
        -------
        dict
            Attributes set on the ``Gear`` (all diameters and radii are positive magnitudes):

            alpha, beta : normal pressure angle and reference helix angle [rad].
            hfP : dedendum coefficient of the basic rack, haP + cP [-].
            mt : transverse module [mm] (ISO_21771_2007, 4.2.7, Eq. (2); checked).
            mx : axial module, mt / tan(beta) [mm]; infinite for beta = 0 (4.2.7, Eq. (3)).
            pn, pt : normal and transverse pitch [mm] (4.4.2, Eq. (24), (23)).
            px, pz : axial pitch and lead [mm]; infinite for beta = 0 (4.4, Eq. (27); 4.3.2, Eq. (4)).
            d, r : reference diameter and radius [mm] (4.2.4, Eq. (1), with abs(z)).
            alphat : transverse pressure angle [rad] (4.3.5, Eq. (13)).
            betab : base helix angle, sin(beta_b) = sin(beta) cos(alpha_n) [rad] (4.3.3, Eq. (5)-(7)).
            db, rb : base diameter and radius [mm] (4.3.10, Eq. (19)-(20)).
            tau : angular pitch, 2 pi / abs(z) [rad] (4.4, Eq. (21)).
            rhoF : root fillet radius of the basic rack, rfP mn [mm] (ISO_53_1998, Table 2, times mn).
            pb, pbt : normal and transverse base pitch [mm] (4.4.5, Eq. (29), (28)).
            dv, rv : diameter and radius of the V-circle, d + 2 (z/abs(z)) x mn [mm] (4.5.1, Eq. (32)).
            da, ra : tip diameter and radius, d + 2 (z/abs(z)) mn (haP + x) [mm] (4.5.3, Eq. (33)
            with k = 0; no clearance term: cP is in the root only).
            df, rf : root diameter and radius [mm] (4.5.4, Eq. (34)).
            zn, de : virtual number of teeth z / cos(beta)**3 [-] and mn zn [mm]; external gear
            only (Q25). ISO_6336-1_2019, 9.3.1, Formula (81) (unchecked). ``de`` is not the d_e
            of ISO_6336-1_2019, Table F.1.
            warnings : tuple of str, becomes ``Gear.warnings`` (never raises): "gear is undercut
            ..." for an external gear with x < x_Emin (ISO_21771_2007, 7.7, Eq. (135), x_E = x,
            counterpart rack of the basic rack, ``_ideal_x_Emin``; MG-07, José 2026-10-21); empty
            otherwise.

        Raises
        ------
        ValueError
            If z = 0 (no kind of teeth, and the factor z/abs(z) divides by zero).

        Notes
        -----
        The factor z/abs(z) of the standard (-1 for an internal gear) puts the tip inside and
        the root outside the reference circle of a ring. ``da`` carries no tip alteration k: the
        working tip diameters with k are pair quantities (``assemble_pair``). The base cylinder
        follows one rule for both kinds, so a ring and the pinion it meshes with have the same
        alphat, betab and pbt (the condition of clause 5 for two gears in mesh).

        References
        ----------
        .. [1] ISO_21771_2007, 4.2-4.5, Eq. (1)-(34) (checked by a second reader).
        """
        mn, z, x = data.mn, data.z, data.x
        if z == 0:
            raise ValueError(f"{data.label or 'Gear'}: z must be != 0, got {z}")
        s = z / abs(z)   # ISO_21771_2007, 4.1.5: +1 external, -1 internal (tip inside the reference circle)

        alpha = np.radians(data.alpha_n_deg)
        beta  = np.radians(data.beta_n_deg)
        hfP   = data.haP + data.cP

        mt = mn / np.cos(beta)
        mx = mt / np.tan(beta) if beta != 0 else np.inf      # a spur gear has no axial pitch and no lead
        pn = np.pi * mn
        pt = np.pi * mt
        pz = abs(z) * mt * np.pi / np.tan(beta) if beta != 0 else np.inf
        px = np.pi * mx

        d = mt * abs(z)
        alphat = np.arctan(np.tan(alpha) / np.cos(beta))     # ISO_21771_2007, 4.3.5, Eq. (13)
        betab = np.arcsin(np.sin(beta) * np.cos(alpha))      # ISO_21771_2007, 4.3.3, Eq. (5)-(7)
        db = d * np.cos(alphat)                              # ISO_21771_2007, 4.3.10, Eq. (19)-(20)

        # tip: addendum haP mn (no clearance: cP belongs to the dedendum hfP of the root, Eq. (34))
        da = d + 2 * mn * s * (data.haP + x)                 # ISO_21771_2007, 4.5.3, Eq. (33), k = 0
        df = d + 2 * mn * s * (x - hfP)
        dv = d + 2 * s * x * mn

        # MG-07 (José, 2026-10-21): the ideal undercut is a warning, with x_E = x and the counterpart
        # rack of the basic rack (Eq. (135): external gear, rack cutter)
        warnings_ = ()
        x_Emin = _ideal_x_Emin(data, alphat, beta)
        if x < x_Emin:
            warnings_ = (
                f"{data.label or 'Gear'}: gear is undercut - z={z}, mn={mn}, alpha={data.alpha_n_deg} deg, "
                f"beta={data.beta_n_deg} deg, x={x} < x_Emin={x_Emin:.6f} (ISO_21771_2007 Eq. (135), "
                "counterpart of the basic rack, x_E = x); the manufactured gear (ManufacturedCylindricalGear) "
                "gives the real check.",)

        return dict(
            alpha=alpha, beta=beta, hfP=hfP, mt=mt, mx=mx, pn=pn, pt=pt, pz=pz, px=px,
            d=d, r=d / 2.0, alphat=alphat, betab=betab, db=db, rb=db / 2.0,
            tau=2 * np.pi / abs(z), rhoF=data.rfP * mn,
            pb=pn * np.cos(alpha), pbt=pt * np.cos(alphat),
            dv=dv, rv=dv / 2, da=da, ra=da / 2.0, df=df, rf=df / 2.0,
            **_legacy("virtual_teeth", z)(mn, z, beta),
            warnings=warnings_,
        )

    @staticmethod
    def values_at(gear: Gear, dy: float) -> dict:
        """Involute quantities at an arbitrary diameter dy of the flank.

        Parameters
        ----------
        gear : Gear
            Gear of this family.
        dy : float
            Diameter at which the involute is evaluated [mm]; dy >= db is needed (otherwise
            arccos gives nan).

        Returns
        -------
        dict
            beta_y : helix angle at dy [rad]; alpha_yt : transverse pressure angle at dy [rad]
            (ISO_21771_2007, 4.3.5, Eq. (12)); alpha_yn : normal pressure angle at dy [rad];
            L_y : transverse radius of curvature of the profile at dy [mm], signed with z/abs(z)
            (negative for a ring: concave flank; 4.3.8, Eq. (17)); inv_alpha_yt :
            tan(alpha_yt) - alpha_yt [rad] (4.3.9, Eq. (18)); p_yt, p_yn : transverse and normal
            pitch at dy [mm].
        """
        beta_y       = np.arctan(np.tan(gear.beta) * dy / gear.d)               # helix angle at dy
        alpha_yt     = np.arccos(gear.db / dy)                                  # transverse pressure angle
        alpha_yn     = np.arctan(np.tan(alpha_yt) * np.cos(beta_y))             # normal pressure angle
        L_y          = (gear.z / abs(gear.z)) * gear.db * np.tan(alpha_yt) / 2  # radius of curvature
        inv_alpha_yt = np.tan(alpha_yt) - alpha_yt
        p_yt         = dy * gear.pt / gear.d
        p_yn         = p_yt * np.cos(beta_y)

        return {
            "beta_y": beta_y,
            "alpha_yt": alpha_yt,
            "alpha_yn": alpha_yn,
            "L_y": L_y,
            "inv_alpha_yt": inv_alpha_yt,
            "p_yt": p_yt,
            "p_yn": p_yn,
        }

    def assemble_manufactured_gear(self, gear: Gear) -> dict:
        """Sizes of one gear as cut: generating profile shifts, tooth thicknesses and root cylinders.

        Parameters
        ----------
        gear : Gear
            Gear whose data is a ``CylindricalGearManufacturingData`` and whose nominal attributes
            (``assemble_gear``) are already set; not yet frozen (``ManufacturedCylindricalGear``).

        Returns
        -------
        dict
            The 13 computed attributes, ``cutter`` to ``d_Ffi``, documented in
            ``ManufacturedCylindricalGear`` (Attributes), and
            ``warnings`` (tuple of str): one per d_Ff not computed because of undercut at that
            allowance (Eq. (128) holds only without undercut; MG-11); for an internal gear one
            message (rack cutter, see Notes).

        Notes
        -----
        A rack-type cutter cannot generate an internal gear, and Eq. (125), (128), (135) are
        printed for external gears: for z < 0 hFaP0, x_Emin, d_fE and d_Ff are None (pinion-type
        cutter, Eq. (126), (131)-(134), not implemented: MG-03). The tip thickness uses the
        nominal d_a, whereas 7.9 asks for the upper-limit tip diameter (MG-06): when MG-06 is
        done, the critical limit of d_a for each kind, the largest for an external gear and the
        smallest for a ring (a ring tooth thins toward smaller diameters). Finish cutting only
        (q = 0).

        References
        ----------
        .. [1] ISO_21771_2007, 4.7.1, Eq. (38); 4.7.2, Eq. (41), p. 25 (PDF p. 29); 4.7.5, Eq. (49);
           7.3-7.7, Eq. (118)-(135), p. 55-59 (PDF p. 59-63).
        """
        data = gear.data
        cutter = GeneratingRackProfile.from_basic_rack(data) if data.cutter is None else data.cutter
        tag = gear.label or "Gear"
        s = gear.z / abs(gear.z)   # ISO_21771_2007, 4.1.5: +1 external, -1 internal
        s_n = gear.mn * (np.pi / 2 + 2 * gear.x * np.tan(gear.alpha))     # Eq. (49), nominal x
        # transverse pressure angle at the tip circle, through its involute (Eq. (18))
        inv_alpha_at = self.values_at(gear, gear.da)["inv_alpha_yt"]
        external = gear.z > 0
        if external:
            x_Emin, hFaP0 = _min_generating_profile_shift(gear.z, gear.mn, gear.alphat, gear.beta, cutter)
            warnings_ = []
        else:
            x_Emin = hFaP0 = None
            warnings_ = [f"{tag}: internal gear: a rack-type cutter cannot generate it; hFaP0, x_Emin, d_fE and "
                         "d_Ff are not computed (ISO_21771_2007 Eq. (125), (128), (135) are for external gears; "
                         "pinion-type cutter not implemented, MG-03)."]
        out = dict(cutter=cutter, hFaP0=hFaP0, x_Emin=x_Emin)
        for k, E_sn, which in (("s", data.E_sns, "upper"), ("i", data.E_sni, "lower")):
            x_E = out[f"x_E{k}"] = _generating_profile_shift(gear.x, gear.mn, gear.alpha, E_sn)
            out[f"s_n{k}"] = s_n + E_sn                                                # Eq. (118), (119)
            # Eq. (41): half tooth angle at the reference circle, generated size with x_E
            psi = (np.pi + 4 * x_E * np.tan(gear.alpha)) / (2 * abs(gear.z))
            # Eq. (38) at d_y = d_a: AxisForge stores ring diameters positive (D-0010), so d_a enters without
            # the sign of ISO's signed diameters; z/abs(z) makes the ring tooth thinner toward its tip (d_a < d)
            out[f"s_at{k}"] = gear.da * (psi + s * (_involute(gear.alphat) - inv_alpha_at))
            out[f"d_fE{k}"] = _generated_root_diameter(gear.d, x_E, gear.mn, cutter) if external else None
            out[f"d_Ff{k}"] = None
            if external and x_E >= x_Emin:
                out[f"d_Ff{k}"] = _root_form_diameter(gear.d, gear.db, gear.alphat, x_E, gear.mn, cutter)
            elif external:
                warnings_.append(f"{tag}: d_Ff{k} not computed: x_E{k}={x_E:.6f} < x_Emin={x_Emin:.6f}, undercut "
                                 f"at the {which} allowance (ISO_21771_2007 Eq. (128) holds only without undercut).")
        out["warnings"] = tuple(warnings_)
        return out

    # ----------------------------------------------------------------------------------------------
    # The pair (ISO 21771 clause 5)
    # ----------------------------------------------------------------------------------------------
    def assemble_pair(self, gear1: Gear, gear2: Gear, *, label: str = "", al: float | None = None,
                      addendum_reduction: bool = False) -> tuple[Gear, Gear, dict]:
        """Working centre distance and pressure angle, contact ratios and path of contact of two gears in mesh.

        Parameters
        ----------
        gear1 : Gear
            The pinion: external, abs(z1) <= abs(z2) (indices of ISO_21771_2007: 1 = pinion,
            2 = wheel).
        gear2 : Gear
            The wheel, external, or the ring, internal (z < 0); same module, normal pressure angle
            and helix angle as ``gear1``.
        label : str, optional
            Name used in the summary.
        al : float or None, optional
            Working centre distance a_w [mm], positive. None: derived from x1 + x2 (zero
            backlash). A value: imposed; the working pressure angle is back-calculated and the
            backlash ``jbn`` that it leaves is stored.
        addendum_reduction : bool, optional
            Apply the tip alteration k to the working tip diameters of both gears. Default False
            (k = 0).

        Returns
        -------
        tuple
            ``(gear1, gear2, attributes)``; the attributes set on the ``GearPair`` are:

            label; al_imposed : whether ``al`` was given.
            a : reference centre distance, (d2 + (z2/abs(z2)) d1) / 2 [mm], positive (not printed
            in ISO 21771 clause 5).
            al : working centre distance a_w [mm], positive for both kinds (5.2.5, Eq. (58)).
            alphatw, inv_alphatw : transverse working pressure angle alpha_wt and its involute
            [rad] (5.2.3-5.2.4, Eq. (54), (55)).
            jbn : normal backlash [mm] when ``al`` is imposed, else None (5.3, Eq. (63) with
            x_E = the declared x). jbn > 0: clearance between the non-working flanks; jbn < 0: the
            flanks would overlap at that a_w (the teeth must be thinned). An a_w larger than the
            zero-backlash one gives jbn > 0 for an external pair and jbn < 0 for an internal pair.
            x1, x2 : profile shift coefficients used [-].
            u : gear ratio z2 / z1, signed (negative for an internal pair) [-] (5.2.1, Eq. (52)).
            k : tip alteration coefficient [-] (4.5.2; the value from a_w, see Notes).
            dl1, dl2, rl1, rl2 : working pitch diameters d_w = d_b / cos(alpha_wt) and radii [mm]
            (Eq. (56)-(57), third form).
            da1_w, da2_w : working tip diameters, d + 2 (z/abs(z)) mn (haP + x + k) [mm] (Eq. (33)).
            alpha_a1, alpha_a2 : transverse pressure angles at the tips [rad].
            epslon_a1, epslon_a2 : addendum contact ratios of gear 1 and gear 2 [-]; that of
            a ring is negative as stored and is subtracted in epslon_alpha.
            epslon_alpha, epslon_beta, epslon_gamma : transverse contact ratio, overlap ratio
            and total contact ratio [-] (5.4.7, Eq. (88)-(97)).
            b : common face width, min(b1, b2) [mm].
            galpha1, galpha2 : addendum parts of the path of contact, rb (tan(alpha_a) -
            tan(alpha_wt)) [mm]; galpha2 of a ring is negative as stored.
            galpha : length of the path of contact [mm], positive, equal to ``AE`` (5.4.5.2,
            Eq. (77)).
            ReqI : reduced radius of curvature at the pitch point, transverse plane,
            1 / (1/rho_C1 + 1/rho_C2) with rho_C2 < 0 for a ring [mm] (Eq. (81)-(82) and NOTE).
            T1T2, T1E, T2A, T1A, T1B, T1C, T1D, T2E, T2B, T2C, T2D : distances along the line of
            action from the base-tangent points T1, T2 to the points A..E of the path [mm]. They
            are the transverse radii of curvature of the flanks at those points: 5.4.5.3,
            Eq. (81)-(87) print T1C, T2C, T2A, T1E, T1B, T2D and T1T2.
            AE, AB, AC, AD : distances from A [mm]; rA1, rB1, rD1, rA2, rB2, rD2 : radii from
            the gear axis to the path points [mm] (derived: sqrt(segment**2 + rb**2); no numbered
            equation, no library source).
            HV : gear loss factor [-] (ISO_TR_14179-2_2001, 5.2.4, Eq. (14); unchecked, Q53).
            Ram, Rrms, RzS : (Ra1 + Ra2)/2, sqrt(Rq1**2 + Rq2**2), Rz1 + Rz2 [um] (no source).
            profile_shift_check : (x1 + x2 implied by a_w, declared x1 + x2) [-] (5.3, Eq. (62),
            signed z1 + z2).
            warnings : advisory messages, they become ``GearPair.warnings`` and never raise:
            "profile shifts do not match the centre distance ..." with the sum implied by the
            imposed a_w, the declared sum and jbn, when abs(jbn) >= 1e-9 mm (Q26 + Q57, José
            2026-10-15); "transverse contact ratio ... is low (< 1.2)" when 1 <= epslon_alpha <
            1.2 (answer 6, D-0026 item 9; no library source for 1.2).

        Raises
        ------
        ValueError
            If the gears differ in module, normal pressure angle or helix angle (relative
            tolerance 1e-6), if gear 1 is internal (z1 < 0), if gear 2 is internal and
            abs(z2) <= z1, if abs(z1) > abs(z2) (gear 1 must be the pinion), or if a helical
            external pair has equal hands or a helical internal pair opposite hands (Q08).

        Notes
        -----
        One expression per quantity for both kinds, with the sign factor s2 = z2/abs(z2) and the
        signed sum z1 + z2 (negative for an internal pair). The stored centre distances a, a_w
        are positive; ISO 21771 makes them negative for an internal pair, hence the factor s2
        wherever their signed value enters (k).

        The profile shifts are those of the gear data: the pair does not split a sum x1 + x2
        between its gears (Q13, José 2026-10-15: the Henriot split is deleted; ISO_21771_2007, 5.3
        prints no rule). FUTURE: a profile-shift split from a source (DIN 3992 / Niemann) if needed.

        Working pressure angle: with a_w derived, inv(alpha_wt) = inv(alpha_t) + 2 tan(alpha_n)
        (x1 + x2) / (z1 + z2) (Eq. (55)) and a_w = (d_b2 + s2 d_b1) / (2 cos(alpha_wt))
        (Eq. (58)); with a_w imposed, cos(alpha_wt) = a cos(alpha_t) / a_w (Eq. (54)). The
        inverse involute has no printed method (Eq. (18) is the definition): brentq on
        [1e-6, 1.5] rad (José, 2026-10-15).

        Tip alteration (Q23, José 2026-10-15): k = s2 (a_w - a) / mn - (x1 + x2), the same on
        both gears, when ``addendum_reduction``; it makes the bottom clearance at a_w equal to
        cP mn on both sides (4.5.2: k < 0 shortens the addendum; no equation for k is printed
        there). For an internal pair k can be positive and then lengthens both tips (z 20/-60,
        x 0.2/0.2: k = +0.040, T1A 0.94 -> 0.55 mm). DIN 3960 / Niemann practice not checked (no
        library source yet).

        For an internal pair (s2 = -1) the segments measured from T2 and T1T2 are negative, as
        the NOTE of ISO_21771_2007, 5.4.5.3 states for T1T2 and the radii of curvature of the
        ring. A is the start of the path on the tip circle of gear 2 and E its end on the tip
        circle of gear 1, i.e. the names are those of gear 1 driving; they are stored once,
        whatever gear drives. A negative ``T1A`` means that the path starts beyond the
        base-tangent point of gear 1 (interference); it is stored here and refused by
        ``validate_pair`` (Q39).

        ``rA1``, ``rB1``, ``rD1`` are the radii of gear 1 at A, B, D; ``rA2``, ``rB2``, ``rD2``
        are the radii of gear 2 at E, D, B (mirrored names).

        References
        ----------
        .. [1] ISO_21771_2007, 4.5.2-4.5.3, Eq. (33); 5.2-5.4, Eq. (52)-(97) (checked by a second
           reader).
        .. [2] ISO_TR_14179-2_2001, 5.2.4, Eq. (14) (clause map only).
        """
        # --- the two gears must have the same basic rack size and helix (ISO 21771, clause 5) ---
        if not np.isclose(gear1.mn, gear2.mn, rtol=1e-6):
            raise ValueError(
                f"Module mismatch: gear1 mn={gear1.mn}, gear2 mn={gear2.mn}. "
                "Both gears must have the same module."
            )
        if not np.isclose(gear1.alpha_n_deg, gear2.alpha_n_deg, rtol=1e-6):
            raise ValueError(
                f"Pressure angle mismatch: gear1 α={gear1.alpha_n_deg}°, "
                f"gear2 α={gear2.alpha_n_deg}°. "
                "Both gears must have the same pressure angle."
            )
        if not np.isclose(gear1.beta_n_deg, gear2.beta_n_deg, rtol=1e-6):
            raise ValueError(
                f"Helix angle mismatch: gear1 β={gear1.beta_n_deg}°, "
                f"gear2 β={gear2.beta_n_deg}°. "
                "Both gears must have the same helix angle magnitude."
            )

        z1, z2 = gear1.z, gear2.z
        # Q03 (José, 2026-10-15), D-0010 item 10: gear 1 is the pinion, an external gear
        if z1 < 0:
            raise ValueError(f"gear 1 must be the external gear (pinion), got z1={z1}; give the ring as gear2.")
        s2 = z2 / abs(z2)   # +1 external pair, -1 internal pair
        if s2 < 0 and abs(z2) <= z1:   # the pinion must fit inside the ring
            raise ValueError(
                f"Internal gear must have more teeth than the external gear "
                f"(z2={z2} <= z1={z1})."
            )
        # geometric indices, 1 = pinion, 2 = wheel; the driver is not the pair's
        if abs(z1) > abs(z2):
            raise ValueError(
                f"gear 1 must be the pinion: |z1| <= |z2|, got z1={z1}, z2={z2}. "
                "Give the smaller gear as gear1; which gear drives is set by the power path."
            )

        # D-0010 item 17 (Q08): the helices of an external pair are of opposite hands, those of an
        # internal pair of the same hand (ISO_21771_2007, 5.1.3, 5.1.4)
        if gear1.beta != 0.0 and (gear1.hand == gear2.hand) != (s2 < 0):
            raise ValueError(
                f"Hand of helix mismatch: gear1 hand={gear1.hand}, gear2 hand={gear2.hand}; an external pair "
                "needs opposite hands, an internal pair the same hand."
            )

        # --- reference centre distance: sum of the reference radii, difference for an internal pair ---
        a = (gear2.d + s2 * gear1.d) / 2.0

        # the two gears share alpha_n, beta and (one base-cylinder rule) alpha_t, beta_b and p_bt
        alpha  = gear1.alpha     # normal pressure angle
        alphat = gear1.alphat    # transverse pressure angle
        betab  = gear1.betab     # base helix angle
        m      = gear1.mn

        # --- working centre distance and transverse working pressure angle ---
        al_imposed = al is not None
        if al_imposed:
            # the working pitch circles roll on the same base circles: a_w cos(alpha_wt) = a cos(alpha_t) (Eq. (54))
            alphatw = np.arccos(a * np.cos(alphat) / al)
            x1, x2 = gear1.x, gear2.x
        else:
            x1, x2 = gear1.x, gear2.x
            # zero backlash, Eq. (55) with the signed tooth sum (z1 + z2 < 0 for a ring)
            inv_alphatw = _involute(alphat) + 2 * np.tan(alpha) * (x1 + x2) / (z1 + z2)
            # inverse involute: no method is printed (Eq. (18) is the definition only); 1.5 rad bounds any working angle
            alphatw = optimize.brentq(_inv_residual, 1e-6, 1.5, args=(inv_alphatw,))
            # Eq. (58): base radii added for an external pair, subtracted for a ring; a_w > 0
            al = (gear2.db + s2 * gear1.db) / (2 * np.cos(alphatw))
        inv_alphatw = _involute(alphatw)

        # sum of profile shifts that a_w needs for zero backlash (Eq. (62), signed z1 + z2)
        x_sum_from_al = (inv_alphatw - _involute(alphat)) * (z1 + z2) / (2 * np.tan(alpha))
        jbn = None
        warnings_ = []
        if al_imposed:
            # normal backlash left by the declared shifts at the imposed a_w: Eq. (63) solved for j_bn with
            # x_E = the declared x; jbn > 0: clearance between the non-working flanks, jbn < 0: the flanks
            # would overlap at that a_w (the teeth must be thinned)
            jbn = (x_sum_from_al - (x1 + x2)) * 2 * m * np.sin(alpha)
            # Q26 + Q57 (José, 2026-10-15): zero backlash for now, a mismatch is ONE warning (not a validate error);
            # 1e-9 mm is far above the rounding of a derived a_w (1e-14)
            if abs(jbn) >= 1e-9:
                warnings_.append(
                    f"profile shifts do not match the centre distance: declared x1 + x2 = {x1 + x2:.4f}, "
                    f"al = {al:.4f} mm implies x1 + x2 = {x_sum_from_al:.4f} for zero backlash "
                    f"(ISO_21771_2007 Eq. (62)); normal backlash jbn = {jbn:.6f} mm (Eq. (63)).")

        u = z2 / z1   # Eq. (52), signed

        # --- tip alteration coefficient k (Eq. (33): + k mn on both tips, k < 0 shortens) ---
        if addendum_reduction:
            # (a_w - a) / mn - (x1 + x2) with the signed centre distances of ISO (negative for a ring): factor s2
            k = s2 * (al - a) / m - (x1 + x2)
        else:
            k = 0.0

        # working pitch circles: the circles on which the two gears roll, d_w = d_b / cos(alpha_wt) (Eq. (56)-(57))
        dl1, dl2 = gear1.db / np.cos(alphatw), gear2.db / np.cos(alphatw)
        rl1, rl2 = dl1 / 2, dl2 / 2

        # --- working tip diameters, Eq. (33) with k (the clearance cP is in the root only) ---
        da1 = gear1.d + 2 * m * (gear1.haP + x1 + k)
        da2 = gear2.d + s2 * 2 * m * (gear2.haP + x2 + k)
        ra1, ra2 = da1 / 2.0, da2 / 2.0
        db1, db2 = gear1.db, gear2.db
        rb1, rb2 = gear1.rb, gear2.rb
        pbt = gear1.pbt   # transverse base pitch

        # transverse pressure angle at the tooth tip (the tip circle taken as the active tip circle)
        alpha_a1 = np.arccos(db1 / da1)
        alpha_a2 = np.arccos(db2 / da2)

        # addendum contact ratios: roll angle from the pitch point to the tip over the angular pitch.
        # For a ring tan(alpha_a2) < tan(alpha_wt), so its part is subtracted through s2.
        epslon_a1 = z1 * (np.tan(alpha_a1) - np.tan(alphatw)) / (2 * np.pi)
        epslon_a2 = abs(z2) * (np.tan(alpha_a2) - np.tan(alphatw)) / (2 * np.pi)
        epslon_alpha = epslon_a1 + s2 * epslon_a2

        # overlap ratio: only non-zero for helical gears; b = common face width of ALIGNED gears
        b = min(gear1.b, gear2.b)
        epslon_beta = b * np.tan(betab) / pbt
        epslon_gamma = epslon_alpha + epslon_beta
        # answer 6 (José, 2026-10-15), D-0026 item 9: a low contact ratio is advisory, not an error
        if 1.0 <= epslon_alpha < 1.2:
            warnings_.append(f"transverse contact ratio εα={epslon_alpha:.4f} is low (< 1.2) — "
                             "consider increasing tooth count or face width.")

        galpha1 = rb1 * (np.tan(alpha_a1) - np.tan(alphatw))
        galpha2 = rb2 * (np.tan(alpha_a2) - np.tan(alphatw))
        # Eq. (77) with the stored (positive) a_w: s2 on the ring tip term AND on 2 a_w sin(alpha_wt);
        # for an external pair it is galpha1 + galpha2
        galpha = 1/2 * ((da1**2 - db1**2)**(1/2) + s2 * ((da2**2 - db2**2)**(1/2) - 2 * al * np.sin(alphatw)))

        # reduced radius of curvature at the pitch point C, transverse plane: rho_C = r_w sin(alpha_wt),
        # negative for the concave flank of a ring (Eq. (82) NOTE, D-0010 item 8)
        ReqI = 1 / (1 / (rl1 * np.sin(alphatw))
                    + 1 / (s2 * rl2 * np.sin(alphatw)))

        # --- path of contact: distances along the line of action (signed with s2 on the side of gear 2) ---
        T1T2 = s2 * al * np.sin(alphatw)
        T1E = (ra1**2 - rb1**2) ** (1 / 2)          # end of contact E: tip circle of gear 1
        T2A = s2 * ((ra2**2 - rb2**2) ** (1 / 2))   # start of contact A: tip circle of gear 2
        # a negative T1A (start of contact beyond T1) is stored; validate_pair refuses it (Q39); old case 4: -0.0749 mm
        T1A = T1T2 - T2A
        T1B = T1E - pbt                             # B, D: one transverse base pitch from E and from A
        T1C = rb1 * np.tan(alphatw)                 # pitch point C
        T1D = T1A + pbt
        T2E = T1T2 - T1E
        T2B = T2E + pbt
        T2C = s2 * rb2 * np.tan(alphatw)
        T2D = T2A - pbt
        AE = T1E - T1A
        AB = T1B - T1A
        AC = T1C - T1A
        AD = T1D - T1A
        # radii to the points of the path (gear 2: mirrored names, at E, D, B)
        rA1 = (T1A**2 + rb1**2) ** (1 / 2)
        rB1 = (T1B**2 + rb1**2) ** (1 / 2)
        rD1 = (T1D**2 + rb1**2) ** (1 / 2)
        rA2 = ((T2A - AE)**2 + rb2**2) ** (1 / 2)
        rB2 = ((T2A - AD)**2 + rb2**2) ** (1 / 2)
        rD2 = ((T2A - AB)**2 + rb2**2) ** (1 / 2)

        # QUESTION(Q53): deferred (José, 2026-10-15): gear loss factor "according to Ohlendorf", unchecked until the thermal/efficiency step; then transcribe ISO_TR_14179-2_2001, 5.2.4, Eq. (14) and check it (also for u < 0 and z1 > z2).
        HV = (np.pi * (u + 1) / (z1 * u * np.cos(betab)) *
              (1 - epslon_alpha + epslon_a1**2 +
               epslon_a2**2))

        return gear1, gear2, dict(
            label=label, al_imposed=al_imposed, a=a, al=al, alphatw=alphatw, inv_alphatw=inv_alphatw,
            jbn=jbn, x1=x1, x2=x2, u=u, k=k, dl1=dl1, dl2=dl2, rl1=rl1, rl2=rl2, da1_w=da1, da2_w=da2,
            alpha_a1=alpha_a1, alpha_a2=alpha_a2, epslon_a1=epslon_a1, epslon_a2=epslon_a2,
            epslon_alpha=epslon_alpha, b=b, epslon_beta=epslon_beta, epslon_gamma=epslon_gamma,
            galpha1=galpha1, galpha2=galpha2, galpha=galpha, ReqI=ReqI,
            T1T2=T1T2, T1E=T1E, T2A=T2A, T1A=T1A, T1B=T1B, T1C=T1C, T1D=T1D, T2E=T2E, T2B=T2B,
            T2C=T2C, T2D=T2D, AE=AE, AB=AB, AC=AC, AD=AD,
            rA1=rA1, rB1=rB1, rD1=rD1, rA2=rA2, rB2=rB2, rD2=rD2, HV=HV,
            Ram=(gear1.Ra + gear2.Ra) / 2, Rrms=(gear1.Rq**2 + gear2.Rq**2) ** (1 / 2),
            RzS=(gear1.Rz + gear2.Rz),
            profile_shift_check=(x_sum_from_al, x1 + x2),
            warnings=warnings_,
        )

    # ----------------------------------------------------------------------------------------------
    # Limits and checks (ISO 21771 clause 7; most have no library source)
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def undercutting(gear: Gear) -> bool:
        """Undercut of the root of the IDEAL gear: x < x_Emin (MG-07, José 2026-10-21).

        The ideal check: x_E = x and the counterpart of the gear's basic rack; an explicit
        ``cutter`` of ``CylindricalGearManufacturingData`` is ignored on purpose. With a negative
        allowance the real x_E < x, so the manufactured gear gives the real check.

        Parameters
        ----------
        gear : Gear
            Gear of this family.

        Returns
        -------
        bool
            External gear: True if its profile shift x is below the minimum generating profile
            shift x_Emin of ISO_21771_2007, 7.7, Eq. (135) (``_ideal_x_Emin``, the criterion of the
            warning of ``assemble_gear`` too). Internal gear: False (Eq. (135) is for external
            gears).

        Notes
        -----
        For ISO 53 rack A, spur, the limit equals the classical z form z < 2 cos(beta) /
        sin(alpha_t)**2 (haP - x) to 3e-5 in x; at beta 15 deg to about 4e-3, because alpha_t
        enters h_FaP0.
        """
        return bool(gear.x < _ideal_x_Emin(gear.data, gear.alphat, gear.beta))

    def validate_gear(self, gear: Gear) -> list[str]:
        """Check one gear.

        Returns
        -------
        list of str
            The messages of ``CylindricalGearData.validate``, then by kind (Q55, keep): da < d and
            da > db for a ring; none for an external gear (its undercut is a warning of
            ``assemble_gear``, MG-07, José 2026-10-21).
        """
        tag = gear.label or "Gear"
        return gear.data.validate() + _legacy("gear_limits", gear.z)(gear, tag)

    @staticmethod
    def validate_manufactured_gear(gear: Gear) -> list[str]:
        """Checks of a gear as cut, added to ``validate_gear`` by ``ManufacturedCylindricalGear``.

        Returns
        -------
        list of str
            One message per violation; empty if valid:

            (1) external gear with x_Ei < x_Emin: undercut at the lower allowance (MG-10).
            Eq. (128) holds only without undercut, so d_Ff and the pair check d_Nf >= d_Ff are
            invalid when x_Ei < x_Emin; x_Es >= x_Ei, so one message covers both.
            (2) s_ats or s_ati not > 0: pointed tooth at the tip circle. ISO_21771_2007, 7.9,
            p. 59 (PDF p. 63) prints no minimum, so none is invented; the nominal d_a is used,
            whereas 7.9 asks for the upper-limit tip diameter (MG-06).
        """
        tag = gear.label or "Gear"
        errors: list[str] = []
        if gear.x_Emin is not None and gear.x_Ei < gear.x_Emin:
            errors.append(f"{tag}: undercut at the lower allowance - x_Ei={gear.x_Ei:.6f} < x_Emin={gear.x_Emin:.6f} "
                          "(ISO_21771_2007 Eq. (135)); d_Ff (Eq. (128)) is not valid")
        for k in ("s", "i"):
            s_at = getattr(gear, f"s_at{k}")
            # "not > 0" also catches nan: d_a < d_b has no tip pressure angle (reachable only with check=False)
            if not s_at > 0.0:
                errors.append(f"{tag}: pointed tooth - s_at{k}={s_at:.6f} mm <= 0 at the tip circle "
                              f"da={gear.da:.4f} mm (ISO_21771_2007 Eq. (38))")
        return errors

    def validate_pair(self, pair) -> list[str]:
        """Check the pair geometry.

        Returns
        -------
        list of str
            One message per violation; empty if the pair is valid:

            (1) the messages of both gears, once each (Q56);
            (2) epslon_alpha < 1.0: the next tooth pair is not in contact when the present one
            leaves (no library source for the limit; a value below 1.2 is a warning of the pair,
            see ``assemble_pair``);
            (3) start or end of contact beyond a base-tangent point, T1A < 0 or s2 T2E < 0 (Q39,
            both kinds): there the flank of the other gear has no involute, i.e. involute
            interference (geometry of the line of action, ISO_21771_2007, 5.4.5.3). For an
            internal pair T1A >= 0 is the involute-interference test of SDP-SI, Eq. (5-2) (the
            same inequality, see ``_internal_interference``), its library source; one message;
            (4) internal pair only: the trochoid and trimming interference tests of SDP-SI
            section 5 (``_internal_interference``, Q24). ISO_21771_2007, 5.4.6 leaves
            interference of internal pairs outside its scope.
        """
        # Q56 (José, 2026-10-15): each gear message once
        errors: list[str] = pair.gear1.validate() + pair.gear2.validate()

        if pair.epslon_alpha < 1.0:
            errors.append(
                f"Transverse contact ratio εα={pair.epslon_alpha:.4f} < 1.0 — "
                "gears will lose contact during meshing."
            )

        # Q39 (José, 2026-10-15): the path A..E must lie between the base-tangent points T1 and T2;
        # T1A < 0: A beyond T1 (start of contact on the pinion below its base circle), s2 T2E < 0:
        # E beyond T2 (s2 makes the ring segment, negative as stored, comparable)
        s2 = np.sign(pair.gear2.z)
        if pair.T1A < 0.0 or s2 * pair.T2E < 0.0:
            # internal pair: T1A < 0 is the same inequality as SDP-SI Eq. (5-2), reported once (lead decision)
            sdp = (" For an internal pair T1A < 0 is the involute interference of SDP-SI Eq. (5-2)."
                   if s2 < 0 and pair.T1A < 0.0 else "")
            errors.append(
                f"Interference: the path of contact passes a base-tangent point (T1A = {pair.T1A:.4f} mm, "
                f"s2 T2E = {s2 * pair.T2E:.4f} mm; both must be >= 0).{sdp}"
            )
        if s2 < 0:
            errors.extend(_internal_interference(pair))
        return errors

    # ----------------------------------------------------------------------------------------------
    # Text
    # ----------------------------------------------------------------------------------------------
    # Q50 (José, 2026-10-15): one text for external and internal pairs; the kind is the sign of z2
    # (ISO_21771_2007, 4.1.5). ASCII only, as the Gear text, so that it prints on a cp1252 console.
    def pair_summary(self, pair) -> str:
        """Multi-line text with the main data of the pair (lengths [mm], angles [deg])."""
        header = f"-- {pair.label or 'GearPair'} "
        lines = [header + "-" * max(0, 52 - len(header))]
        lines += [f"  gear{i}     : z={g.z}, mn={g.mn}, x={g.x}, beta={g.beta_n_deg} deg"
                  for i, g in ((1, pair.gear1), (2, pair.gear2))]
        lines += [
            f"  u         : {pair.u:.4f}",
            f"  a         : {pair.a:.4f} mm   (reference centre distance)",
            f"  al        : {pair.al:.4f} mm   (working centre distance)",
            f"  alpha_tw  : {np.degrees(pair.alphatw):.4f} deg",
            f"  dl1       : {pair.dl1:.4f} mm",
            f"  dl2       : {pair.dl2:.4f} mm",
            f"  eps_alpha : {pair.epslon_alpha:.4f}",
            f"  eps_beta  : {pair.epslon_beta:.4f}",
            f"  eps_gamma : {pair.epslon_gamma:.4f}",
            "-" * 52,
        ]
        return "\n".join(lines)

    def pair_repr(self, pair) -> str:
        """One-line representation: tooth numbers, module, helix angle, al, alphatw, eps_gamma."""
        return (
            f"GearPair(z1={pair.gear1.z}, z2={pair.gear2.z}, "
            f"mn={pair.gear1.mn}, beta={pair.gear1.beta_n_deg} deg, al={pair.al:.4f} mm, "
            f"alpha_tw={np.degrees(pair.alphatw):.4f} deg, eps_gamma={pair.epslon_gamma:.4f})"
        )

    # ----------------------------------------------------------------------------------------------
    # Forces (the pair holds no driver and no torque: Q03, José 2026-10-15)
    # ----------------------------------------------------------------------------------------------
    def mesh_forces(self, pair, T1_Nm: float, rot1: int) -> MeshForces:
        """Forces of the mesh on the two gears, for a torque on gear 1 (pitch-cylinder set of D-0010).

        The single normal force between the flanks is resolved at the pitch point C: tangential
        and radial components for the shafts, the force along the line of action and the force
        normal to the flanks for the contact, and the nominal tangential force of ISO 6336 at
        the reference cylinder. ``MeshForces`` holds the mesh frame and the sign table.

        Parameters
        ----------
        pair : GearPair
            Pair of this family.
        T1_Nm : float
            Torque applied to gear 1 by its own shaft, about +X [N m], signed (NOT N mm: 60.75 N m
            on gear 1 of baseline case 2 gives F_tw = 1659.83 N; 60750 would give 1.66e6 N).
        rot1 : int
            Sense of rotation of gear 1 about +X, +1 or -1 [-]. Gear 1 drives when
            ``T1_Nm * rot1 > 0``. With the symmetric flanks of a cylindrical gear the forces
            follow from the signed torque alone, so ``rot1`` is checked and not used; it belongs
            to the family contract (a bevel pair needs it for the loaded flank).

        Returns
        -------
        MeshForces
            ``F_tw_N``, ``F_r_N``, ``F_a_N``, ``F_bt_N``, ``F_bn_N``, ``F_t_nominal_N`` [N]
            (magnitudes); ``on_1``, ``on_2`` = (F_c, F_t, F_x) [N], signed, ``on_2 = -on_1``;
            ``rho_mm`` [mm]; ``x_offset_mm`` = (0, 0).

        Raises
        ------
        ValueError
            If ``rot1`` is not +1 or -1.

        Notes
        -----
        With d_w1 = d_b1 / cos(alpha_wt) the working pitch diameter, d_b1 the base diameter, d_1
        the reference diameter and beta_b the base helix angle of gear 1 [mm, rad]::

            F_tw = 2000 abs(T_1) / d_w1          F_r  = F_tw tan(alpha_wt)
            F_bt = 2000 abs(T_1) / d_b1          F_a  = F_bt tan(beta_b)
            F_bn = F_bt / cos(beta_b)            F_t (nominal) = 2000 abs(T_1) / d_1
            rho_1 = sign(z_2) d_w1 / 2           rho_2 = rho_1 - a_w
            F_1,t = -1000 T_1 / rho_1            F_1,c = -sign(z_1 z_2) F_r
            F_1,x = -h_1 sign(rho_1) sign(F_1,t) F_a     h_1 = ``pair.gear1.hand`` (Q08)

        The factor 2000 converts N m and mm to N (D-0010 item 5). Baseline case 2 (z 20/30,
        mn 3.5 mm, beta 15 deg), T_1 = 60.75 N m: F_tw 1659.831, F_r 674.505, F_a 449.236,
        F_bt 1791.646, F_bn 1847.109, F_t nominal 1676.571 N.

        This is the only force calculation of the family; the shaft loads are built from it by
        ``mesh_forces_to_shaft_loads`` (Q05, Q07a, Q07b, José 2026-10-15). A pure function of the pair and the torque: the driver follows from the sign
        of T_1 rot1 (D-0010 item 11).

        Assumptions: ideal mesh (no friction, no losses), rigid involute flanks, steady rotation,
        one normal force through the pitch point at mid face width.

        References
        ----------
        .. [1] ISO_6336-1_2019, 4.2.2, Formula (1), p. 19 (F_t nominal; checked).
        .. [2] ISO_21771_2007, 5.2.4, Eq. (56) third form (d_w = d_b / cos(alpha_wt); checked).
        .. [3] D-0010 items 5, 14-18; ``units_and_conventions.md``, C.4, C.5: no library source
               for F_tw, F_r, F_a, F_bt, F_bn and the directions (statics and helicoid geometry).
        """
        if rot1 not in (1, -1):
            raise ValueError(f"rot1 must be +1 or -1, got {rot1!r}")
        g1, z2 = pair.gear1, pair.gear2.z
        d_w1 = pair.dl1                        # working pitch diameter of gear 1, d_b1 / cos(alpha_wt)
        T = abs(T1_Nm)
        # 2000 = 2 (diameter -> radius) x 1000 (mm -> m): T [N m] / r [m] gives [N]
        F_tw = 2000.0 * T / d_w1
        F_r = F_tw * np.tan(pair.alphatw)
        F_bt = 2000.0 * T / g1.db
        F_a = F_bt * np.tan(g1.betab)          # = F_tw tan(beta_w): one normal force, three cylinders
        F_bn = F_bt / np.cos(g1.betab)
        F_t_nominal = 2000.0 * T / g1.d        # reference cylinder: ISO 6336-1 Formula (1)

        # pitch point C on the line of centres: between the axes for an external pair (rho_1 > 0),
        # beyond the pinion on the side away from the ring centre for an internal pair (rho_1 < 0)
        rho_1 = np.sign(z2) * d_w1 / 2.0
        rho_2 = rho_1 - pair.al
        F_1t = -1000.0 * T1_Nm / rho_1         # moment of the mesh force about the axis balances T_1
        F_1c = -np.sign(g1.z * z2) * F_r       # compressive contact: external gear pushed toward its axis
        # axial component normal to the tooth trace at C, with the hand of gear 1 (D-0010 item 16; 0 for a spur gear)
        F_1x = -pair.gear1.hand * np.sign(rho_1) * np.sign(F_1t) * F_a
        on_1 = (float(F_1c), float(F_1t), float(F_1x))
        return MeshForces(F_tw_N=float(F_tw), F_r_N=float(F_r), F_a_N=float(F_a), F_bt_N=float(F_bt),
                          F_bn_N=float(F_bn), F_t_nominal_N=float(F_t_nominal), on_1=on_1,
                          on_2=(-on_1[0], -on_1[1], -on_1[2]),    # action = reaction at C
                          rho_mm=(float(rho_1), float(rho_2)))

    # ----------------------------------------------------------------------------------------------
    # Contact inputs along the path of contact (ISO 21771, 5.4.3-5.4.8 and 5.6; the load has no library source)
    # ----------------------------------------------------------------------------------------------
    # Q31, Q37 (José, 2026-10-15): the geometry of the path stays here; load_sharing and contact_inputs, which take a
    # force or a time, are functions of axisforge.solvers.machine_elements.gears.contact (no split of this file by clause)
    @staticmethod
    def path_points(pair, driver: str) -> dict[str, float]:
        """Points A, B, C, D, E of the path of contact for a given driving gear, as positions g_y.

        The path of contact is the part of the line of action (the common tangent of the two base
        circles, transverse plane) on which a tooth pair touches. Meshing starts at A, on the tip
        circle of the DRIVEN gear, and ends at E, on the tip circle of the DRIVING gear; C is the
        pitch point; B lies one transverse base pitch before E and D one after A. The pair stores
        these points once, with the names of gear 1 driving; this function only relabels them.

        Parameters
        ----------
        pair : GearPair
            Pair of this family.
        driver : str
            "gear1" or "gear2": the gear that drives (an operating quantity, not stored).

        Returns
        -------
        dict
            Keys "A", "B", "C", "D", "E": position g_y of each point [mm].

        Raises
        ------
        ValueError
            If ``driver`` is not "gear1" or "gear2".

        Notes
        -----
        Position along the path, used by every function of this group: g_y = rho_y1 - rho_C1 [mm],
        the distance from the pitch point C along the line of action, POSITIVE toward the tip of
        gear 1 (the transverse radius of curvature of flank 1 grows with g_y). It does not depend
        on the driver; abs(g_y) is the g_alpha_y of ISO_21771_2007, Eq. (107). With gear 1 driving
        the contact point travels from A (g_y < 0) to E (g_y > 0); with gear 2 driving A and E, and
        B and D, are exchanged and the contact point travels toward decreasing g_y.

        B and D bound the zone of single tooth contact of ONE transverse section (a spur pair)
        for 1 < eps_alpha < 2; a helical pair with eps_gamma >= 2 (baseline case 2) never has a
        single tooth pair in contact. For eps_alpha >= 2 B and D are still E
        minus and A plus one base pitch, and D comes before B.

        Assumptions: rigid, unmodified involute flanks; the tip circle is the end of the active
        profile.

        References
        ----------
        .. [1] ISO_21771_2007, 5.4.3 and 5.4.5.1, p. 34-35 (definitions of A..E by driven and
           driving gear); swap rule in the text after Eq. (80) and after Eq. (87), p. 36 (checked).
        """
        if driver not in ("gear1", "gear2"):
            raise ValueError(f"driver must be 'gear1' or 'gear2', got '{driver}'.")
        # the stored segments T1A..T1E are the radii of curvature of flank 1 at the points named for gear 1 driving
        g = {name: getattr(pair, "T1" + name) - pair.T1C for name in "ABCDE"}
        if driver == "gear2":
            # start of meshing on the tip circle of the driven gear, now gear 1: the stored E
            g = {"A": g["E"], "B": g["D"], "C": g["C"], "D": g["B"], "E": g["A"]}
        return g

    @staticmethod
    def curvature_radii(pair, g_y) -> dict[str, np.ndarray]:
        """Radii of curvature of the two flanks, and their reduced radius, at positions g_y of the path.

        At the contact point the two involute flanks are replaced by two cylinders with these
        radii (the local equivalent cylinders of the line contact). The transverse radius of an
        involute is the distance from the contact point to the base-tangent point along the line of
        action, so it varies linearly along the path and the two radii add up to T1T2.

        Parameters
        ----------
        pair : GearPair
            Pair of this family.
        g_y : float or array_like
            Position along the path [mm], from the pitch point C, positive toward the tip of gear 1
            (see ``path_points``). Not limited to A..E.

        Returns
        -------
        dict
            rho1_t_mm, rho2_t_mm : radius of curvature of flank 1 and of flank 2 in the
            transverse plane [mm]. SIGNED: positive for a convex (external) flank, negative for
            the concave flank of a ring (D-0010 item 8).
            rho_red_t_mm : reduced radius 1 / (1/rho1 + 1/rho2) with those signs, transverse
            plane [mm].
            rho1_n_mm, rho2_n_mm, rho_red_n_mm : the same in the plane normal to the contact
            line (the principal radii of the flanks), each transverse value divided by
            cos(beta_b) [mm].

        Notes
        -----
        rho_y1 = T1C + g_y and rho_y2 = T1T2 - rho_y1. At the stored points they are the stored
        segments T1A..T1E and T2A, T2B, T2D, T2E. A radius is zero at its base-tangent point
        (start of the involute) and changes sign beyond it, where there is no involute: the
        values returned there are not radii of a flank. ``validate_pair`` refuses such a pair (Q39);
        old case 4, kept as an equivalence record with ``check=False``, starts 0.075 mm beyond T1.

        At C the reduced radius is the stored ``ReqI`` and rho_y2 the stored ``T2C``, for both kinds.

        Normal plane: an involute helicoid is a developable surface, with zero curvature along
        the contact line. The transverse profile is an oblique section of it, inclined at beta_b
        to the flank normal, so its radius rho_t is the radius of curvature of that section and
        not a principal radius; by Meusnier's theorem the principal radius, in the plane normal
        to the contact line, is rho_n = rho_t / cos(beta_b) (beta_b is the same for both gears).
        The tangent of the transverse profile is the principal direction across the contact line,
        so Euler's factor is 1 and only Meusnier's cos(beta_b) acts: the relation is exact, not an
        approximation.

        Assumptions: rigid, unmodified involute flanks; no deflection.

        References
        ----------
        .. [1] ISO_21771_2007, 4.3.8, Eq. (17), p. 21 (radius of curvature at any point Y, with
           the factor z/abs(z)); 5.4.5.3, Eq. (81)-(87) and NOTE, p. 36 (sum of the two radii, negative
           values of an internal gear) (checked).
        .. [2] Normal plane, rho_t / cos(beta_b): derived (Notes), no library page.
        """
        g_y = np.asarray(g_y, dtype=float)
        rho1 = pair.T1C + g_y      # Eq. (81) at C, plus the distance travelled along the line of action
        # Eq. (87): rho_y1 + rho_y2 = T1T2 at every point; T1T2 < 0 for an internal pair, so the ring radius is negative
        rho2 = pair.T1T2 - rho1
        # reduced radius with the SIGNED ring radius (ISO_21771_2007, Eq. (82) NOTE)
        rho_red = 1.0 / (1.0 / rho1 + 1.0 / rho2)
        # Q34 (José, 2026-10-15): keep - normal-plane radii rho_n = rho_t / cos(beta_b), derived in the Notes (developable helicoid, Meusnier)
        cos_betab = np.cos(pair.gear1.betab)
        return {"rho1_t_mm": rho1, "rho2_t_mm": rho2, "rho_red_t_mm": rho_red,
                "rho1_n_mm": rho1 / cos_betab, "rho2_n_mm": rho2 / cos_betab, "rho_red_n_mm": rho_red / cos_betab}

    @staticmethod
    def surface_speeds(pair, g_y, n1_rpm: float) -> dict[str, np.ndarray]:
        """Tangential speeds of the two flanks, sliding speed and entrainment speed at positions g_y.

        Each flank moves through the contact along the common tangent with the speed
        omega rho_y (rolling). The two speeds are equal only at the pitch point; elsewhere their
        difference is the sliding speed, which grows linearly with the distance from C. Their mean
        is the speed at which lubricant is drawn into the contact.

        Parameters
        ----------
        pair : GearPair
            Pair of this family.
        g_y : float or array_like
            Position along the path [mm], from C, positive toward the tip of gear 1.
        n1_rpm : float
            Rotational speed of GEAR 1 [rpm], > 0 (a magnitude: the sense of rotation is not an
            input), whichever gear drives.

        Returns
        -------
        dict
            u1_m_per_s, u2_m_per_s : tangential speed of flank 1 and of flank 2 at the contact
            point [m/s]; positive where the flank is an involute (rho_y1 > 0; rho_y2 > 0 for an
            external gear 2, rho_y2 < 0 for a ring).
            vg_m_per_s : sliding speed u1 - u2 [m/s], SIGNED: positive where flank 1 is the faster
            (g_y > 0, addendum of gear 1), zero at C.
            ue_m_per_s : entrainment speed (u1 + u2) / 2 [m/s], the mean speed (GEARpie's ``vsumc``
            is the sum u1 + u2, twice this value; Q35).

        Notes
        -----
        u1 = omega1 rho_y1 and u2 = omega2 rho_y2 = omega1 rho_y2 / u with u = z2 / z1 signed, so
        that vg = omega1 g_y (1 + 1/u). For an internal pair u and rho_y2 are both negative: u2
        is positive and the sliding speed is smaller than that of the external pair.

        Helical pair, which component this is (derived from rigid-body kinematics, no library
        source; ISO 21771, 5.6.3 gives only "rho_y omega" for the transverse profile). At the
        contact point take e_L along the line of action, e_z along the axes and e_p normal to the
        plane of action. The velocity of the point of flank i is omega_i r_bi e_L +
        omega_i rho_yi e_p; the contact line lies along cos(beta_b) e_z + sin(beta_b) e_L and the
        flank normal along cos(beta_b) e_L - sin(beta_b) e_z. So: (a) omega rho_y, returned here,
        is the component PERPENDICULAR to the contact line in the tangent plane, the rolling speed
        across the line; (b) the component ALONG the contact line is omega1 r_b1 sin(beta_b), the
        same on both flanks (no sliding along the line), and is NOT returned; (c) GEARpie's
        omega rho_y / cos(beta_b) is omega times the normal-plane radius, not a component of
        the flank velocity (Q38).

        Driver: ISO 21771 writes 5.6 for the pinion driving. The speeds themselves do not depend
        on the driver; only the names A and E of the two ends do (``path_points``).

        Assumptions: rigid involute kinematics, constant speed, no modifications, no deflection.

        References
        ----------
        .. [1] ISO_21771_2007, 5.6.1, Eq. (106)-(108), p. 42 (sliding speed; printed with a
           plus-minus sign); 5.6.3, p. 44 (tangential velocity "rho_y omega") (checked).
        .. [2] Entrainment speed: no numbered equation, no library source.
        """
        if n1_rpm <= 0:
            raise ValueError(f"n1_rpm must be > 0, got {n1_rpm}.")
        rho = CylindricalFamily.curvature_radii(pair, g_y)
        omega1 = n1_rpm * 2.0 * np.pi / 60.0   # [rpm] -> [rad/s]
        u = pair.u                             # z2 / z1, signed (ISO_21771_2007, Eq. (52))
        # Q38 (José, 2026-10-15): keep - omega rho_y, across the contact line (case 2 at C: 2.0922 m/s); the along-line
        # component omega1 rb1 sin(beta_b) (1.2522 m/s) becomes an extra key only if a model needs it
        u1 = omega1 * rho["rho1_t_mm"] / 1000.0        # [rad/s] [mm] -> [m/s]
        u2 = omega1 / u * rho["rho2_t_mm"] / 1000.0    # omega2 = omega1 / u; ring: u < 0 and rho_y2 < 0, u2 > 0
        # Q35 (José, 2026-10-15): keep - ue = (u1 + u2) / 2 (GEARpie's vsumc is the sum u1 + u2, a factor 2), vg = u1 - u2 signed
        return {"u1_m_per_s": u1, "u2_m_per_s": u2, "vg_m_per_s": u1 - u2, "ue_m_per_s": (u1 + u2) / 2.0}


# ==================================================================================================
# The gear as manufactured (ISO 21771 clause 7)
# ==================================================================================================
class ManufacturedCylindricalGear(Gear):
    """One cylindrical gear as cut: the nominal (IDEAL) gear plus its manufactured sizes.

    Two layers: the nominal geometry of ``CylindricalFamily.assemble_gear`` (clause 4, profile
    shift x, zero backlash), unchanged, and the sizes that the tooth thickness allowances and the
    generating tool give (clause 7, ``CylindricalFamily.assemble_manufactured_gear``). The class
    ONLY ADDS: no nominal attribute is redefined, so the gear is accepted wherever a ``Gear`` is
    (``GearPair.assemble``, ``GearElement``, ``Transmission``) with identical results. The one
    exception is an external gear with x < x_Emin: on the ideal gear that is a warning (MG-07),
    but as cut it is an error even with zero allowances (x_Ei = x < x_Emin, MG-10), so
    ``from_ideal(g, E_sns=0, E_sni=0)`` raises unless ``check=False``.

    Attributes
    ----------
    E_sns, E_sni : float
        Upper and lower tooth thickness allowances [mm] (data fields; 7.3, Eq. (118), (119)).
    cutter : GeneratingRackProfile
        The tool used: ``data.cutter``, or the counterpart of the basic rack when it is None
        (``data.cutter`` keeps the input).
    hFaP0 : float or None
        Straight part of the tip flank of the tool [mm] (Fig. 39 key).
    x_Es, x_Ei : float
        Generating profile shift coefficients [-] (7.4, Eq. (123), (124)).
    x_Emin : float or None
        Minimum generating profile shift coefficient [-] (7.7, Eq. (135)).
    s_ns, s_ni : float
        Normal tooth thicknesses at the reference circle [mm] (7.3, Eq. (118), (119)).
    s_ats, s_ati : float
        Transverse tooth thicknesses at the tip circle [mm] (4.7.1, Eq. (38) with x_E).
    d_fEs, d_fEi : float or None
        Generated root diameters [mm] (7.5, Eq. (125)).
    d_Ffs, d_Ffi : float or None
        Root form diameters [mm] (7.6, Eq. (128)); None under undercut at that allowance.

    Notes
    -----
    Rack-type cutter, external gear: for an internal gear (z < 0) hFaP0, x_Emin, d_fE and d_Ff
    are None and the gear carries one warning (pinion-type cutter not implemented, MG-03).
    ``validate`` adds ``CylindricalFamily.validate_manufactured_gear`` to the nominal checks.

    References
    ----------
    .. [1] ISO_21771_2007, clause 7, p. 53-59 (PDF p. 57-63).
    """

    @classmethod
    def assemble(cls, family: GearFamily, data: CylindricalGearManufacturingData, *,
                 check: bool = True) -> ManufacturedCylindricalGear:
        """Build an assembled, immutable manufactured gear.

        Parameters
        ----------
        family : CylindricalFamily
            Family instance.
        data : CylindricalGearManufacturingData
            Nominal data plus allowances and tool.
        check : bool, optional
            If True (default), run ``validate()`` and refuse an invalid gear (as ``Gear.assemble``).

        Returns
        -------
        ManufacturedCylindricalGear
            The nominal attributes, then the manufactured ones; ``warnings`` joins the nominal
            warnings and the manufactured ones.

        Raises
        ------
        TypeError
            If ``family`` is not a ``CylindricalFamily`` or ``data`` not a
            ``CylindricalGearManufacturingData``.
        ValueError
            As ``Gear.assemble``.
        """
        if not isinstance(family, CylindricalFamily):
            raise TypeError(f"{cls.__name__} needs a CylindricalFamily, got {type(family).__name__}")
        if not isinstance(data, CylindricalGearManufacturingData):
            raise TypeError(f"{cls.__name__} needs CylindricalGearManufacturingData, got {type(data).__name__}")
        gear = cls(family, data)
        nominal = family.assemble_gear(data)
        nominal_warnings = nominal.pop("warnings")
        for attr, value in nominal.items():    # the manufactured sizes are computed from the nominal ones
            setattr(gear, attr, value)
        computed = family.assemble_manufactured_gear(gear)
        gear._freeze({**computed, "warnings": (*nominal_warnings, *computed["warnings"])})
        if check:
            gear.validate_or_raise()
        return gear

    @classmethod
    def from_ideal(cls, gear: Gear, *, E_sns: float, E_sni: float,
                   cutter: GeneratingRackProfile | None = None, check: bool = True) -> ManufacturedCylindricalGear:
        """Manufactured gear with the nominal data of ``gear`` and the given allowances and tool.

        Parameters
        ----------
        gear : Gear
            Assembled gear of the cylindrical family. Only the ``CylindricalGearData`` fields of
            its data are copied: the allowances and tool of a manufactured gear given here are
            replaced by the arguments.
        E_sns, E_sni : float
            Upper and lower tooth thickness allowances [mm].
        cutter : GeneratingRackProfile or None, optional
            Tool; None: counterpart of the basic rack.
        check : bool, optional
            Passed to ``assemble``. Default True.

        Returns
        -------
        ManufacturedCylindricalGear
            ``assemble(gear.family, data, check=check)``.
        """
        nominal = {f.name: getattr(gear.data, f.name) for f in fields(CylindricalGearData)}
        data = CylindricalGearManufacturingData(**nominal, E_sns=E_sns, E_sni=E_sni, cutter=cutter)
        return cls.assemble(gear.family, data, check=check)

    def validate(self) -> list[str]:
        """Check the gear: ``validate_gear`` (incl. the allowances of the data), then
        ``validate_manufactured_gear``.

        Returns
        -------
        list of str
            One message per violation; empty if valid.
        """
        return super().validate() + self.family.validate_manufactured_gear(self)
