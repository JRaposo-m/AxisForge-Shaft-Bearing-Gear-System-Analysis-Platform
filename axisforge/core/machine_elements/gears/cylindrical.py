"""Cylindrical involute gears on parallel axes, external and internal, spur and helical.

Both kinds of teeth follow the sign convention of ISO_21771_2007, 4.1.5: the number of teeth of an
internal gear is negative, every diameter and the centre distance are positive. Lengths in [mm],
angles in [rad] (inputs in [deg]), torques in [N m], forces in [N].

Two families, in two sections of this file; the dependency goes one way only (MANUFACTURED ->
IDEAL, the IDEAL section never refers to the MANUFACTURED one):

- IDEAL, ``CylindricalFamily`` (clauses 4-5 of ISO_21771_2007, nominal profile shift, zero
  backlash): basic rack (ISO_53_1998), one gear (clause 4), the pair (clause 5), its checks, the
  tooth forces, the contact inputs along the path (5.4-5.6). No undercut check: undercut (7.7,
  Eq. (135)) needs a generating tool (José, 2026-10-21, ruling B).
  ``Gear.assemble(CylindricalFamily(), CylindricalGearData(...))``.
- MANUFACTURED, ``ManufacturedCylindricalFamily(CylindricalFamily)`` (clause 7: tooth thickness
  allowances and generating tool; it only adds attributes and checks to the ideal ones, undercut
  included). ``Gear.assemble(ManufacturedCylindricalFamily(), CylindricalGearManufacturingData(...))``.

Pairs: ``GearPair.assemble(family, gear1=..., gear2=...)`` with the same family as the gears.

Examples
--------
>>> fam = CylindricalFamily()
>>> g1, g2 = (Gear.assemble(fam, CylindricalGearData(3.5, z, x=0.18)) for z in (20, 30))
>>> pair = GearPair.assemble(fam, gear1=g1, gear2=g2)
>>> made_fam = ManufacturedCylindricalFamily()
>>> m1, m2 = (Gear.assemble(made_fam, CylindricalGearManufacturingData(
...     3.5, z, x=0.18, E_sns=-0.05, E_sni=-0.10)) for z in (20, 30))
>>> made = GearPair.assemble(made_fam, gear1=m1, gear2=m2)
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, fields
from typing import NamedTuple
import numpy as np
from scipy import optimize

from axisforge.core.machine_elements.gears.base import Gear, GearFamily, MeshForces, register_gear_family


# ##################################################################################################
# IDEAL: the nominal gear and pair (ISO_21771_2007, clauses 4-5). Refers to nothing below.
# ##################################################################################################
def _involute(ang):
    """Involute function inv(ang) = tan(ang) - ang [rad], ang in [rad] (ISO_21771_2007, 4.3.9, Eq. (18))."""
    return np.tan(ang) - ang


def _inv_residual(xx, inv_a):
    """Residual ``inv_a - inv(xx)`` [rad]; zero when the angle xx [rad] has involute inv_a."""
    return inv_a - np.tan(xx) + xx


def _kind(z) -> str:
    """Kind of teeth from the sign of z (ISO_21771_2007, 4.1.5): key of the ``_LEGACY`` table."""
    return "internal" if z < 0 else "external"


def _normal_backlash(x_sum_from_al, x_sum, mn, alpha) -> float:
    """Normal backlash j_bn [mm] of two gears whose profile shifts sum to ``x_sum`` [-] at a working
    centre distance that needs ``x_sum_from_al`` [-] for zero backlash (Eq. (62)); ``mn`` [mm],
    normal pressure angle ``alpha`` [rad]. > 0: clearance between the non-working flanks; < 0: the
    flanks would overlap. ISO_21771_2007, 5.3, Eq. (63), p. 32 (PDF p. 36), solved for j_bn; x_sum
    is the declared x1 + x2 or the generating x_E1 + x_E2."""
    return (x_sum_from_al - x_sum) * 2 * mn * np.sin(alpha)


# ==================================================================================================
# One gear: helpers for both kinds of teeth (the kind is the sign of z, ISO_21771_2007, 4.1.5)
# ==================================================================================================
def _virtual_teeth(mn, z, beta):
    """Virtual number of teeth zn [-], signed like z (negative for a ring), and the diameter of the
    virtual gear de = mn abs(zn) [mm], positive as every diameter of this file. Spur gear equivalent
    in the normal section (Tredgold, z / cos(beta)**3); ISO_6336 uses z / (cos(beta_b)**2 cos(beta))."""
    zn = z / np.cos(beta) ** 3
    return {"zn": zn, "de": mn * abs(zn)}


def _gear_limits(gear, tag):
    """Tip circle on the tooth side of the reference circle (da > d external, da < d ring, i.e. a
    positive addendum haP + x > 0) and above the base circle (an involute at the tip). No library source."""
    errors = []
    s = gear.z / abs(gear.z)
    if s * (gear.da - gear.d) <= 0:
        errors.append(f"{tag}: tip circle da={gear.da:.4f} on the wrong side of the reference circle "
                      f"d={gear.d:.4f} (addendum (haP + x) mn = {(gear.haP + gear.x) * gear.mn:.4f} mm must be > 0).")
    if gear.da <= gear.db:   # no involute at the tip: arccos(db/da) of the tip pressure angle has no value
        errors.append(f"{tag}: tip circle da={gear.da:.4f} <= base circle db={gear.db:.4f}: "
                      "no involute at the tooth tip.")
    return errors


# ==================================================================================================
# Internal pair: interference tests of SDP-SI section 5 (called by validate_pair when z2 < 0)
# ==================================================================================================

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
        # involute, Eq. (5.2)
        lhs = z1 / z2
        rhs = 1 - np.tan(alpha_a2)/np.tan(alpha_w)
        if not lhs >= rhs:
            errors.append(f"involute interference of the internal pair (SDP-SI Eq. (5.2)): " 
                          f"{lhs:.6f} < {rhs:.6f}")
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
    k: float = 0.0
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
            0 < alpha_n_deg < 45, 0 <= beta_n_deg < 45, hand +1 or -1 for a
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
        if not (0.0 <= self.beta_n_deg < 45.0):
            errors.append(f"{tag}: beta_n_deg out of range [0, 45), got {self.beta_n_deg}")
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


# ==================================================================================================
# The family
# ==================================================================================================
@register_gear_family
class CylindricalFamily(GearFamily):
    """Cylindrical involute gears and gear pairs on parallel axes (ISO_21771_2007), external and internal.

    ``assemble_gear`` gives the geometry of one gear (clause 4), ``assemble_pair`` that of two gears
    in mesh (clause 5), ``validate_gear`` / ``validate_pair`` the limits of the ideal gear and
    pair (no undercut check: it needs a generating tool, see the module docstring),
    ``mesh_forces`` the forces of the mesh for a torque (D-0010, ``MeshForces``), ``path_points``, ``curvature_radii``, ``surface_speeds`` the geometry and
    kinematics along the path of contact (the load sharing and the contact inputs are in
    ``axisforge.solvers.machine_elements.gears.contact``). An internal pair is a pair whose gear 2 has z < 0.

    Notes
    -----
    Where the external and the internal classes this family replaces computed a quantity
    differently and no ruling has chosen, both expressions are kept in the module table
    one expression per quantity for both kinds, with z/abs(z) where the kind matters

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
        alphat, betab and pbt (the condition of clause 5 for two gears in mesh). No undercut
        check and no warning: undercut (ISO_21771_2007, 7.7, Eq. (135)) needs a generating tool
        (ruling B, José 2026-10-21; module docstring).

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
        da = d + 2 * mn * s * (data.haP + x + data.k)                 # ISO_21771_2007, 4.5.3, Eq. (33), k = 0
        df = d - 2 * mn * s * (hfP - x)
        dv = d + 2 * s * x * mn

        return dict(
            alpha=alpha, beta=beta, hfP=hfP, mt=mt, mx=mx, pn=pn, pt=pt, pz=pz, px=px,
            d=d, r=d / 2.0, alphat=alphat, betab=betab, db=db, rb=db / 2.0,
            tau=2 * np.pi / abs(z), rhoF=data.rfP * mn,
            pb=pn * np.cos(alpha), pbt=pt * np.cos(alphat),
            dv=dv, rv=dv / 2, da=da, ra=da / 2.0, df=df, rf=df / 2.0,
            **_virtual_teeth(mn, z, beta),
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
            jbn = _normal_backlash(x_sum_from_al, x1 + x2, m, alpha)
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
    # Limits and checks (no undercut check: it needs a generating tool; most have no library source)
    # ----------------------------------------------------------------------------------------------
    def validate_gear(self, gear: Gear) -> list[str]:
        """Check one gear.

        Returns
        -------
        list of str
            The messages of ``CylindricalGearData.validate``, then: da < d and
            da > db for a ring. No undercut check: it needs a
            generating tool (ruling B, José 2026-10-21; module docstring).
        """
        tag = gear.label or "Gear"
        return gear.data.validate() + _gear_limits(gear, tag)

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
        """One-line representation: class name, tooth numbers, module, helix angle, al, alphatw, eps_gamma."""
        return (
            f"{type(pair).__name__}(z1={pair.gear1.z}, z2={pair.gear2.z}, "
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


# ##################################################################################################
# MANUFACTURED: the gear as cut (ISO_21771_2007, clause 7). Depends on the IDEAL section above; the
# IDEAL section never refers to this one.
# ##################################################################################################

# ==================================================================================================
# Tool and input records
# ==================================================================================================
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
    prP0 : None
        PLACEHOLDER - not implemented (MG-17): protuberance of the tool pr / m_n [-]. Symbol "pr"
        of ISO_6336-3_2006, Fig. 2, p. 9 (PDF p. 17), with the tool subscripts P0 of ISO_701_1998,
        Tables 2 and 4, p. 2 (PDF p. 10). Must be None (refused by ``validate``).
    alpha_prP0_deg : None
        PLACEHOLDER - not implemented (MG-17): protuberance angle of the tool [deg]. Subscript
        "pr" = protuberance, ISO_701_1998, Tables 3 and 5, p. 2 (PDF p. 10); no library page prints
        this symbol: constructed by the ISO 701 rules. Must be None (refused by ``validate``).

    Notes
    -----
    The straight part of the tip flank h_FaP0 is not a field: it needs the transverse pressure angle
    of the gear (``ManufacturedCylindricalFamily._min_generating_profile_shift``). A protuberance
    cutter leaves an undercut of the rack profile, U_FP at the angle alpha_FP (ISO_53_1998, A.2,
    p. 5 / PDF p. 7); Eq. (135) is printed for non-protuberance cutters only.

    References
    ----------
    .. [1] ISO_21771_2007, 7.1, Fig. 35, p. 53 (PDF p. 57).
    .. [2] ISO_53_1998, 3.2, Fig. 1, Table 2, p. 2-3 (PDF p. 4-5).
    """
    haP0: float
    rhoaP0: float
    alpha_P0_deg: float
    prP0: float | None = None
    alpha_prP0_deg: float | None = None

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
            ``CylindricalGearData``); prP0 and alpha_prP0_deg None (placeholders, MG-17).
        """
        errors: list[str] = []
        if self.haP0 <= 0:
            errors.append(f"cutter: haP0 must be > 0, got {self.haP0}")
        if self.rhoaP0 < 0:
            errors.append(f"cutter: rhoaP0 must be >= 0, got {self.rhoaP0}")
        if not (0.0 < self.alpha_P0_deg < 45.0):
            errors.append(f"cutter: alpha_P0_deg out of range (0, 45), got {self.alpha_P0_deg}")
        for name in ("prP0", "alpha_prP0_deg"):
            if getattr(self, name) is not None:
                errors.append(f"cutter: {name} is not implemented (MG-17), got {getattr(self, name)!r}")
        return errors

    def validate_or_raise(self) -> None:
        """Raise ``ValueError`` with all messages of ``validate()`` if any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


@dataclass(frozen=True, kw_only=True)
class GeneratingPinionCutter:
    """Pinion-type cutter (shaper cutter) that generates the teeth.

    PLACEHOLDER - not implemented (MG-16). It would also generate internal gears, which a rack
    cutter cannot (MG-03). Subscript 0 = tool (ISO_701_1998). The generating equations need, from
    these fields and the gear: z_0, d_b0, the tool tip diameter d_a0 (Eq. (126)), the form
    diameter d_Fa0 of the tool tip (Eq. (131), (134); it also needs a tool tip-rounding rule, not
    stated here), the centre distance a_0 and the working pressure angle alpha_wt0 of cutter and
    gear. The helix angle of the tool is implied: beta_0 = beta.

    Parameters
    ----------
    z0 : int
        Number of teeth of the cutter [-].
    x0 : float
        Profile shift coefficient of the cutter [-].
    haP0 : float
        Addendum of the cutter h_aP0 / m_n [-].
    rhoaP0 : float
        Tip radius of the cutter rho_aP0 / m_n [-].
    alpha_P0_deg : float
        Pressure angle of the cutter alpha_P0 [deg].

    Raises
    ------
    NotImplementedError
        Always, at construction.

    References
    ----------
    .. [1] ISO_21771_2007, 7.5, Eq. (126); 7.6, Eq. (131)-(134), p. 57-58 (PDF p. 61-62).
    .. [2] ISO_701_1998, subscript 0 (tool), Tables 2 and 4, p. 2 (PDF p. 10).
    """
    z0: int
    x0: float
    haP0: float
    rhoaP0: float
    alpha_P0_deg: float

    def __post_init__(self):
        raise NotImplementedError("MG-16: pinion-type cutter (ISO_21771_2007 Eq. (126), (131)-(134)) not implemented")


class InspectionLimits(NamedTuple):
    """Values of one tooth thickness test dimension at the two allowances [mm].

    Returned by the Annex A test dimensions (placeholders, MG-15): ISO_21771_2007, Annex A,
    p. 60-82 (PDF p. 64-86). The order follows the allowance, not the magnitude: for an internal
    gear (span over spaces, A.17, A.22-A.23) the value at x_Es can be the smaller. These are the
    values of the nominal system (planned test dimensions); the allowable limits of the
    functional system need the ISO 1328 deviations (F_r, F_pk for the span, A.18-A.19; F_r for
    balls, A.49-A.50; F_rc and d_a max / min for the chord, A.30-A.35), not in the library
    (MG-18).

    Parameters
    ----------
    upper : float
        Value at the upper tooth thickness allowance, from x_Es [mm].
    lower : float
        Value at the lower tooth thickness allowance, from x_Ei [mm].
    """
    upper: float
    lower: float


@dataclass(frozen=True, kw_only=True)
class CylindricalGearManufacturingData(CylindricalGearData):
    """Input data of one cylindrical gear as made: the nominal data plus the manufacturing inputs.

    All the fields of ``CylindricalGearData`` (``mn`` and ``z`` may still be positional), then
    two kinds of input:

    - design inputs, read from the drawing: the tooth thickness allowances ``E_sns``, ``E_sni``
      and the accuracy grade ``quality_grade``;
    - process inputs, chosen by the workshop: the generating tool ``cutter`` and the machining
      allowances ``q_min``, ``q_max``.

    Parameters
    ----------
    E_sns, E_sni : float
        Upper and lower tooth thickness allowances [mm], normal section: s_ns = s_n + E_sns,
        s_ni = s_n + E_sni. Required. A negative allowance reduces the tooth thickness.
    quality_grade : None
        PLACEHOLDER - not implemented (MG-18): accuracy grade of ISO 1328 [-] (ISO 1328 is not in
        the library: source to find). Must be None (refused by ``validate``).
    cutter : GeneratingRackProfile or None, optional
        Rack-type tool that generates the teeth. None (default): the counterpart of the basic rack
        of this record, ``GeneratingRackProfile.from_basic_rack(self)``. A
        ``GeneratingPinionCutter`` is a placeholder (MG-16) that cannot be built.
    q_min, q_max : None
        PLACEHOLDER - not implemented (MG-02): smallest and largest machining allowance for the
        finishing process, normal to the flank [mm] (ISO_21771_2007, 7.2, p. 54 / PDF p. 58;
        pre-finish shifts 7.4, Eq. (120)-(122), p. 56 / PDF p. 60). Must be None (refused by
        ``validate``): only finish cutting (q = 0) is computed.

    References
    ----------
    .. [1] ISO_21771_2007, 7.3, Eq. (118), (119), p. 55 (PDF p. 59) (allowances); 7.1-7.2 (tool).
    """
    E_sns: float
    E_sni: float
    quality_grade: int | None = None
    cutter: GeneratingRackProfile | GeneratingPinionCutter | None = None
    q_min: float | None = None
    q_max: float | None = None

    def validate(self) -> list[str]:
        """Check the domains of the inputs.

        Returns
        -------
        list of str
            The messages of ``CylindricalGearData.validate``, then: E_sni <= E_sns <= 0; the
            messages of ``cutter.validate()`` when a cutter is given; the tool pressure angle
            alpha_P0_deg equal to alpha_n_deg (relative 1e-6); quality_grade, q_min, q_max None
            (placeholders, MG-18, MG-02).

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
        for name, issue in (("quality_grade", "MG-18"), ("q_min", "MG-02"), ("q_max", "MG-02")):
            if getattr(self, name) is not None:
                errors.append(f"{tag}: {name} is not implemented ({issue}), got {getattr(self, name)!r}")
        return errors


# ==================================================================================================
# The manufactured family
# ==================================================================================================
@register_gear_family
class ManufacturedCylindricalFamily(CylindricalFamily):
    """Cylindrical involute gears as cut: the IDEAL family plus the sizes of clause 7 (ISO_21771_2007).

    Two layers: the nominal geometry of ``CylindricalFamily`` (clause 4, profile shift x, zero
    backlash), unchanged, and the sizes that the tooth thickness allowances and the generating
    tool give (clause 7). The family ONLY ADDS: ``assemble_gear`` and ``assemble_pair`` return the
    nominal attributes of ``CylindricalFamily`` plus the manufactured ones, ``validate_gear`` and
    ``validate_pair`` the nominal messages plus the manufactured ones; every other method is
    inherited. A gear of this family is a ``Gear`` and is accepted wherever a ``Gear`` is
    (``GearPair.assemble`` with ``CylindricalFamily``, ``GearElement``, ``Transmission``) with
    identical nominal results. The one exception is an external gear with x < x_Emin: the ideal
    gear says nothing (no undercut check, ruling B, José 2026-10-21), but as cut it is an error
    even with zero allowances (x_Ei = x < x_Emin, MG-10), so ``from_ideal(g, E_sns=0, E_sni=0)``
    raises unless ``check=False``.

    Build with ``Gear.assemble(ManufacturedCylindricalFamily(), CylindricalGearManufacturingData(...))``
    and ``GearPair.assemble(ManufacturedCylindricalFamily(), gear1=..., gear2=...)``.

    Attributes
    ----------
    Added to the ``Gear`` (``assemble_gear``):

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

    Added to the ``GearPair`` (``assemble_pair``):

    jbn_max, jbn_min : float
        Largest and smallest normal backlash [mm] at the working centre distance.
    d_Nf1, d_Nf2 : float
        Active root diameters of gear 1 and gear 2 [mm].

    Notes
    -----
    Rack-type cutter, external gear: for an internal gear (z < 0) hFaP0, x_Emin, d_fE and d_Ff
    are None and the gear carries one warning (pinion-type cutter not implemented, MG-03).

    References
    ----------
    .. [1] ISO_21771_2007, clause 7, p. 53-59 (PDF p. 57-63).
    """

    # ----------------------------------------------------------------------------------------------
    # Generation of the teeth by a rack-type cutter (ISO_21771_2007, clause 7): pure functions of
    # numbers and of the cutter record. External gear, rack cutter (or hob) without protuberance,
    # finish cutting.
    # ----------------------------------------------------------------------------------------------
    @staticmethod
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


    @staticmethod
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


    @staticmethod
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


    @staticmethod
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

    # ----------------------------------------------------------------------------------------------
    # One gear as cut (ISO 21771 clause 7)
    # ----------------------------------------------------------------------------------------------
    def assemble_gear(self, data: CylindricalGearManufacturingData) -> dict:
        """Nominal geometry of one gear, then its sizes as cut: generating profile shifts, tooth
        thicknesses and root cylinders.

        Parameters
        ----------
        data : CylindricalGearManufacturingData
            Nominal data plus allowances and tool.

        Returns
        -------
        dict
            The attributes of ``CylindricalFamily.assemble_gear``, unchanged, then the 13 computed
            attributes, ``cutter`` to ``d_Ffi``, documented in ``ManufacturedCylindricalFamily``
            (Attributes), and
            ``warnings`` (tuple of str): one per d_Ff not computed because of undercut at that
            allowance (Eq. (128) holds only without undercut; MG-11); for an internal gear one
            message (rack cutter, see Notes).

        Raises
        ------
        TypeError
            If ``data`` is not a ``CylindricalGearManufacturingData``.
        ValueError
            As ``CylindricalFamily.assemble_gear``.

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
        if not isinstance(data, CylindricalGearManufacturingData):
            raise TypeError(f"{type(self).__name__} needs CylindricalGearManufacturingData, got {type(data).__name__}")
        nominal = super().assemble_gear(data)   # the manufactured sizes are computed from the nominal ones
        z, mn, x = data.z, data.mn, data.x
        alpha, alphat, beta, d, db, da = (nominal[k] for k in ("alpha", "alphat", "beta", "d", "db", "da"))
        cutter = GeneratingRackProfile.from_basic_rack(data) if data.cutter is None else data.cutter
        tag = data.label or "Gear"
        s = z / abs(z)   # ISO_21771_2007, 4.1.5: +1 external, -1 internal
        s_n = mn * (np.pi / 2 + 2 * x * np.tan(alpha))     # Eq. (49), nominal x
        # transverse pressure angle at the tip circle, through its involute (Eq. (18); as values_at)
        inv_alpha_at = _involute(np.arccos(db / da))
        external = z > 0
        if external:
            x_Emin, hFaP0 = self._min_generating_profile_shift(z, mn, alphat, beta, cutter)
            warnings_ = []
        else:
            x_Emin = hFaP0 = None
            warnings_ = [f"{tag}: internal gear: a rack-type cutter cannot generate it; hFaP0, x_Emin, d_fE and "
                         "d_Ff are not computed (ISO_21771_2007 Eq. (125), (128), (135) are for external gears; "
                         "pinion-type cutter not implemented, MG-03)."]
        out = dict(nominal, cutter=cutter, hFaP0=hFaP0, x_Emin=x_Emin)
        for k, E_sn, which in (("s", data.E_sns, "upper"), ("i", data.E_sni, "lower")):
            x_E = out[f"x_E{k}"] = self._generating_profile_shift(x, mn, alpha, E_sn)
            out[f"s_n{k}"] = s_n + E_sn                                                # Eq. (118), (119)
            # Eq. (41): half tooth angle at the reference circle, generated size with x_E
            psi = (np.pi + 4 * x_E * np.tan(alpha)) / (2 * abs(z))
            # Eq. (38) at d_y = d_a: AxisForge stores ring diameters positive (D-0010), so d_a enters without
            # the sign of ISO's signed diameters; z/abs(z) makes the ring tooth thinner toward its tip (d_a < d)
            out[f"s_at{k}"] = da * (psi + s * (_involute(alphat) - inv_alpha_at))
            out[f"d_fE{k}"] = self._generated_root_diameter(d, x_E, mn, cutter) if external else None
            out[f"d_Ff{k}"] = None
            if external and x_E >= x_Emin:
                out[f"d_Ff{k}"] = self._root_form_diameter(d, db, alphat, x_E, mn, cutter)
            elif external:
                warnings_.append(f"{tag}: d_Ff{k} not computed: x_E{k}={x_E:.6f} < x_Emin={x_Emin:.6f}, undercut "
                                 f"at the {which} allowance (ISO_21771_2007 Eq. (128) holds only without undercut).")
        out["warnings"] = (*nominal.get("warnings", ()), *warnings_)
        return out

    def validate_gear(self, gear: Gear) -> list[str]:
        """Check one gear as cut: ``CylindricalFamily.validate_gear`` (incl. the allowances of the
        data), then the checks of the gear as cut.

        Returns
        -------
        list of str
            The messages of ``CylindricalFamily.validate_gear``, then one message per violation:

            (1) external gear with x_Ei < x_Emin: undercut at the lower allowance (MG-10).
            Eq. (128) holds only without undercut, so d_Ff and the pair check d_Nf >= d_Ff are
            invalid when x_Ei < x_Emin; x_Es >= x_Ei, so one message covers both.
            (2) s_ats or s_ati not > 0: pointed tooth at the tip circle. ISO_21771_2007, 7.9,
            p. 59 (PDF p. 63) prints no minimum, so none is invented; the nominal d_a is used,
            whereas 7.9 asks for the upper-limit tip diameter (MG-06).
        """
        tag = gear.label or "Gear"
        errors = super().validate_gear(gear)
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

    def from_ideal(self, gear: Gear, *, E_sns: float, E_sni: float,
                   cutter: GeneratingRackProfile | None = None, check: bool = True) -> Gear:
        """Gear of this family with the nominal data of ``gear`` and the given allowances and tool.

        Parameters
        ----------
        gear : Gear
            Assembled gear of ``CylindricalFamily`` or of this family. Only the
            ``CylindricalGearData`` fields of its data are copied: the allowances and tool of a
            manufactured gear given here are replaced by the arguments.
        E_sns, E_sni : float
            Upper and lower tooth thickness allowances [mm].
        cutter : GeneratingRackProfile or None, optional
            Tool; None: counterpart of the basic rack.
        check : bool, optional
            Passed to ``Gear.assemble``. Default True.

        Returns
        -------
        Gear
            ``Gear.assemble(self, data, check=check)``.
        """
        nominal = {f.name: getattr(gear.data, f.name) for f in fields(CylindricalGearData)}
        data = CylindricalGearManufacturingData(**nominal, E_sns=E_sns, E_sni=E_sni, cutter=cutter)
        return Gear.assemble(self, data, check=check)

    # ----------------------------------------------------------------------------------------------
    # The pair as cut (ISO 21771 clause 5 with the generating profile shifts of clause 7)
    # ----------------------------------------------------------------------------------------------
    def assemble_pair(self, gear1: Gear, gear2: Gear, *, label: str = "", al: float | None = None,
                      addendum_reduction: bool = False) -> tuple[Gear, Gear, dict]:
        """Nominal geometry of two gears in mesh, then the backlash and active root diameters of
        the pair as cut.

        Parameters
        ----------
        gear1, gear2 : Gear
            Gears assembled with ``ManufacturedCylindricalFamily``.
        label, al, addendum_reduction
            As ``CylindricalFamily.assemble_pair``.

        Returns
        -------
        tuple
            ``(gear1, gear2, attributes)``: the attributes of ``CylindricalFamily.assemble_pair``,
            unchanged, then:

            jbn_max, jbn_min : normal backlash [mm] at the pair's working centre distance a_w
            (derived or imposed), from the lower (x_Ei1 + x_Ei2) and the upper (x_Es1 + x_Es2)
            allowances (``_normal_backlash``, Eq. (63), with the sum implied by a_w,
            ``profile_shift_check[0]``, signed z1 + z2). Same sign as ``jbn``: > 0 clearance
            between the non-working flanks.
            d_Nf1, d_Nf2 : active root diameters [mm], 2 ``rA1`` and 2 ``rA2`` (the existing radii
            of the start of the active profile of each gear; equal to 5.4.1, Eq. (64), (65),
            p. 33 (PDF p. 37) with d_Na = d_a, with a_w taken negative for an internal pair
            (AxisForge stores a_w > 0)).
            warnings : tuple of str: one per gear with an allowance whose d_Ff is None (ring, or
            undercut at that allowance): d_Nf >= d_Ff is not checked there.

        Notes
        -----
        j_bn is defined in 5.5.1, p. 40 (PDF p. 44); Eq. (100)-(104) only convert it to other
        backlashes (not computed). Centre-distance tolerances are out of scope: a_w is the one
        value of the pair.

        Raises
        ------
        TypeError
            If a gear was not assembled with ``ManufacturedCylindricalFamily``.
        ValueError
            As ``CylindricalFamily.assemble_pair``.
        """
        if not (isinstance(gear1.family, ManufacturedCylindricalFamily)
                and isinstance(gear2.family, ManufacturedCylindricalFamily)):
            raise TypeError(f"{type(self).__name__} needs two gears assembled with ManufacturedCylindricalFamily, "
                            f"got {type(gear1.family).__name__} and {type(gear2.family).__name__}")
        g1, g2, nominal = super().assemble_pair(gear1, gear2, label=label, al=al,
                                                addendum_reduction=addendum_reduction)
        x_sum_from_al = nominal["profile_shift_check"][0]   # Sum x implied by a_w (Eq. (62)), reused
        warnings_ = []
        for i, g in ((1, g1), (2, g2)):
            missing = [k for k in ("s", "i") if getattr(g, f"d_Ff{k}") is None]
            if missing:
                warnings_.append(f"d_Nf >= d_Ff not checked for gear {i} ({g.label or 'Gear'}): "
                                 f"{', '.join(f'd_Ff{k}' for k in missing)} not computed (internal gear or undercut "
                                 "at that allowance).")
        return g1, g2, dict(
            nominal,
            # largest backlash from the thinnest teeth (lower allowances), smallest from the upper
            jbn_max=_normal_backlash(x_sum_from_al, g1.x_Ei + g2.x_Ei, g1.mn, g1.alpha),
            jbn_min=_normal_backlash(x_sum_from_al, g1.x_Es + g2.x_Es, g1.mn, g1.alpha),
            d_Nf1=2 * nominal["rA1"], d_Nf2=2 * nominal["rA2"],
            warnings=(*nominal["warnings"], *warnings_),
        )

    def validate_pair(self, pair) -> list[str]:
        """Check the pair as cut: ``CylindricalFamily.validate_pair`` (incl. the checks of the two
        gears as cut), then d_Nf >= d_Ff.

        Returns
        -------
        list of str
            The messages of ``CylindricalFamily.validate_pair``, then one message per gear and
            allowance whose root form diameter is computed and lies
            above the active root diameter, d_Nf < d_Ff: the tip of the mate would reach below the
            start of the involute, i.e. contact on the root fillet. ISO_21771_2007, 5.4.1, p. 33
            (PDF p. 37): "d_Nf may be governed by the start of involute", Eq. (66), (67).
            Gears or allowances without d_Ff are a warning of the pair (``assemble_pair``).

        Notes
        -----
        ISO 21771 5.4.1 does not call d_Nf < d_Ff invalid: it sets d_Nf = d_Ff (Eq. (66), (67)) and
        limits the active tip of the mate, d_Na <= d_Fa (Eq. (68), (69), i.e. tip relief). AxisForge
        has no d_Fa (d_Fa = d_a), so this error is stricter than the standard (ruling MG-10:
        Eq. (128) and this check). The rack-A internal baseline pairs fail it: the ring tip
        (h_a2 = 1.0 m_n) reaches the pinion fillet (issue MG-24).
        """
        errors = super().validate_pair(pair)
        for i, gear in ((1, pair.gear1), (2, pair.gear2)):
            d_Nf = getattr(pair, f"d_Nf{i}")
            for k, which in (("s", "upper"), ("i", "lower")):
                d_Ff = getattr(gear, f"d_Ff{k}")
                # "not >=" also catches nan
                if d_Ff is not None and not d_Nf >= d_Ff:
                    errors.append(f"gear {i} ({gear.label or 'Gear'}): active root diameter d_Nf{i}={d_Nf:.4f} mm "
                                  f"< root form diameter d_Ff{k}={d_Ff:.4f} mm at the {which} allowance: the mate's "
                                  "tip reaches the root fillet (ISO_21771_2007 5.4.1, Eq. (66), (67))")
        return errors

    # ----------------------------------------------------------------------------------------------
    # Tooth thickness test dimensions of a manufactured gear (ISO 21771 Annex A): placeholders (MG-15).
    # Not placeholders here: A.4 constant chord, A.5-A.6 radial single-ball/cylinder M_rK, M_rZ, A.9.
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def span_measurement(gear: Gear, k: int) -> InspectionLimits:
        """Span measurement W_k over k teeth (external gear) or tooth spaces (internal spur gear).

        PLACEHOLDER - not implemented (MG-15).

        Parameters
        ----------
        gear : Gear
            Gear as cut.
        k : int
            Number of teeth spanned (tooth spaces for an internal gear) [-].

        Returns
        -------
        InspectionLimits
            W_k at x_Es (upper) and at x_Ei (lower) [mm].

        References
        ----------
        .. [1] ISO_21771_2007, A.2, Eq. (A.1)-(A.23), p. 61-67 (PDF p. 65-71).
        """
        raise NotImplementedError("MG-15: span measurement W_k (ISO_21771_2007 A.2) not implemented")

    @staticmethod
    def dimension_over_balls(gear: Gear, D_M: float) -> InspectionLimits:
        """Diametral dimension over two balls M_dK, or over two cylinders M_dZ.

        PLACEHOLDER - not implemented (MG-15).

        Cylinders (A.7.1) are allowed on external gears and internal spur gears; for helical gears
        with an odd number of teeth the standard asks for "special calculations".

        Parameters
        ----------
        gear : Gear
            Gear as cut.
        D_M : float
            Diameter of the measuring ball or cylinder [mm].

        Returns
        -------
        InspectionLimits
            M_dK (M_dZ) at x_Es (upper) and at x_Ei (lower) [mm].

        References
        ----------
        .. [1] ISO_21771_2007, A.5-A.7, Eq. (A.38)-(A.50), p. 75-81 (PDF p. 79-85): the ball
           centre circle of A.5, Eq. (A.38)-(A.46), on which A.7 (Eq. (A.47)-(A.50)) and A.7.1
           build.
        """
        raise NotImplementedError("MG-15: dimension over balls M_dK (ISO_21771_2007 A.5-A.7) not implemented")

    @staticmethod
    def chordal_tooth_thickness(gear: Gear, d_y: float | None = None) -> InspectionLimits:
        """Normal chordal tooth thickness s_cy on the Y-cylinder of diameter d_y.

        PLACEHOLDER - not implemented (MG-15).

        Parameters
        ----------
        gear : Gear
            Gear as cut.
        d_y : float or None, optional
            Diameter of the Y-cylinder [mm]; None: d_a - 2 m_n, "often used as the Y-cylinder
            diameter" (A.3.1).

        Returns
        -------
        InspectionLimits
            s_cy at x_Es (upper) and at x_Ei (lower) [mm].

        References
        ----------
        .. [1] ISO_21771_2007, A.3, Eq. (A.24)-(A.35), p. 68-70 (PDF p. 72-74).
        """
        raise NotImplementedError("MG-15: chordal tooth thickness s_cy (ISO_21771_2007 A.3) not implemented")

    @staticmethod
    def chordal_height(gear: Gear, d_y: float | None = None) -> float:
        """Height above the chord (chordal addendum) h_ccy at which s_cy is measured.

        PLACEHOLDER - not implemented (MG-15).

        One float, not ``InspectionLimits``: A.3.2 uses ONE chordal height for both thickness
        limits (the Delta s term of the minimum thickness makes the same height serve both).

        Parameters
        ----------
        gear : Gear
            Gear as cut.
        d_y : float or None, optional
            Diameter of the Y-cylinder [mm]; None: as ``chordal_tooth_thickness``.

        Returns
        -------
        float
            h_ccy [mm].

        References
        ----------
        .. [1] ISO_21771_2007, A.3.2, Eq. (A.30)-(A.35), p. 69-70 (PDF p. 73-74).
        """
        raise NotImplementedError("MG-15: chordal height h_ccy (ISO_21771_2007 A.3.2) not implemented")

    @staticmethod
    def double_flank_centre_distance(gear: Gear,
                                     master: Gear) -> InspectionLimits:
        """Centre distance a_L of zero-backlash (double-flank) mesh with a master gear.

        PLACEHOLDER - not implemented (MG-15).

        Parameters
        ----------
        gear : Gear
            Gear as cut.
        master : Gear
            Master gear: subscript L in ISO_21771_2007, 3.2 (p. 8 / PDF p. 12), 3 in ISO_701_1998,
            Table 4 (p. 2 / PDF p. 10). Eq. (A.52) needs its z_L, x_L and ONE actual allowance
            E_snL, so the master must have E_sns == E_sni (= E_snL) and the same m_n, alpha_n and
            beta as ``gear`` (the implementation will refuse it otherwise).

        Returns
        -------
        InspectionLimits
            a_L at x_Es (upper) and at x_Ei (lower) [mm]; alpha_L from Eq. (A.52).

        References
        ----------
        .. [1] ISO_21771_2007, A.8, Eq. (A.51), (A.52), p. 81 (PDF p. 85).
        """
        raise NotImplementedError("MG-15: double-flank centre distance a_L (ISO_21771_2007 A.8) not implemented")
