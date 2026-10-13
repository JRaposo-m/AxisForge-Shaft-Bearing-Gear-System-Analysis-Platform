"""Cylindrical involute gears on parallel axes, external and internal, spur and helical.

One family for both kinds of teeth under the sign convention of ISO_21771_2007, 4.1.5: the number
of teeth of an internal gear is negative, every diameter and the centre distance are positive. The
file follows the order of the standard: basic rack (ISO_53_1998), one gear (clause 4), the pair
(clause 5), limits and checks (clause 7), the tooth forces, then the contact inputs along the path
(5.4-5.6). Lengths in [mm], angles in [rad] (inputs in [deg]), torques in [N m], forces in [N].
"""
from __future__ import annotations

import warnings
from dataclasses import KW_ONLY, dataclass
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
    # QUESTION(Q55): the kind is the sign of z, where it was the CLASS the caller chose: a positive z given to the old ring class, or a negative z to the old external class, now gives another gear (no baseline or design-studies case does it).
    return "internal" if z < 0 else "external"


# ==================================================================================================
# _LEGACY: what today's external and internal classes do DIFFERENTLY and no ruling has unified.
# One entry per quantity, one expression per kind of teeth, carried unchanged from the four
# modules this file replaces; each is reached only through ``_legacy(name, z)``. An entry is
# deleted (and its baseline rows become exception rows) when José rules its question.
# ==================================================================================================
def _base_cylinder_external(mn, z, alpha, beta, d):
    alphat = np.arctan(np.tan(alpha) / np.cos(beta))      # ISO_21771_2007, 4.3.5, Eq. (13)
    betab = np.arcsin(np.sin(beta) * np.cos(alpha))       # ISO_21771_2007, 4.3.3, Eq. (5)-(7)
    db = d * np.cos(alphat)                               # ISO_21771_2007, 4.3.10, Eq. (19)-(20)
    return alphat, betab, db


def _base_cylinder_internal(mn, z, alpha, beta, d):
    # QUESTION(Q21): I1 / row 3, entry `base_cylinder`: not the rule of ISO_21771_2007 Eq. (5)-(7), (13), (19) above.
    betab = np.arcsin(np.sin(beta) / np.cos(alpha))       # quotient, not the product of Eq. (5)-(7): beta_b > beta
    db = abs(z) * mn * np.cos(alpha) / np.cos(betab)
    alphat = np.arccos(db / d)
    return alphat, betab, db


def _virtual_teeth_external(mn, z, beta):
    zn = z / (np.cos(beta) ** 3)    # teeth of the spur gear equivalent in the normal section
    return {"zn": zn, "de": mn * zn}


def _alphatw_from_x_sum_external(gear1, gear2, x_sum):
    alpha  = gear1.alpha     # normal pressure angle
    alphat = gear1.alphat    # transverse pressure angle
    z_sum  = gear1.z + gear2.z
    inv_alphatw = (
        np.tan(alphat) - alphat
        + 2 * np.tan(alpha) * x_sum / z_sum
    )
    # inverse involute: no method is printed (ISO_21771_2007, Eq. (18) is the definition only).
    # Bracket 1e-6 .. 0.7 rad (0.7 rad = 40.1 deg bounds the working pressure angle).
    alphatw = optimize.brentq(_inv_residual, 1e-6, 0.7, args=(inv_alphatw,))
    al = (
        gear1.db + gear2.db
    ) / (2 * np.cos(alphatw))
    return alphatw, al


def _alphatw_from_x_sum_internal(gear1, gear2, x_sum):
    # QUESTION(Q21): I1, entry `alphatw_from_x_sum`: alpha_t and beta of the ring, and another expression of a_w than the external pair.
    alphat = gear2.alphat    # of the RING (differs from that of gear 1 when beta != 0: Q21)
    beta   = gear2.beta
    z_sum  = gear2.z + gear1.z   # signed: -(abs(z2) - z1)
    alpha  = gear1.alpha
    inv_alphatw = (
        np.tan(alphat) - alphat
        + 2 * np.tan(alpha) * x_sum / z_sum
    )
    alphatw = optimize.brentq(_inv_residual, 1e-6, 0.7, args=(inv_alphatw,))
    al = gear2.mn * np.cos(alphat) * abs(z_sum) / (2 * np.cos(beta) * np.cos(alphatw))
    return alphatw, al


def _alphatw_from_al_external(gear1, gear2, a, al):
    # QUESTION(Q21): I1, entry `alphatw_from_al`: alpha_t of gear 1 (the internal entry below takes that of the ring).
    return np.arccos(a * np.cos(gear1.alphat) / al), None   # no backlash value for an external pair


def _alphatw_from_al_internal(gear1, gear2, a, al):
    # QUESTION(Q21): I1, entry `alphatw_from_al`: alpha_t and beta of the ring.
    z_sum = gear2.z + gear1.z
    alphatw = np.arccos(abs(z_sum) * gear2.mn * np.cos(gear2.alphat) / (
        2 * al * np.cos(gear2.beta)))
    inv_alphat = np.tan(gear2.alphat) - gear2.alphat
    inv_alphatw = np.tan(alphatw) - alphatw
    sum_x = (gear1.z + gear2.z) * (inv_alphatw - inv_alphat) / (2 * np.tan(gear1.alpha))
    # normal backlash [mm] from the profile shift left over by the imposed centre distance
    # QUESTION(Q57): jbn = (x1 + x2 - sum_x) 2 mn sin(alpha_n); ISO_21771_2007, 5.3, Eq. (63) as transcribed gives (sum_x - (x1 + x2)) 2 mn sin(alpha_n), the opposite sign (case 5 with al x 1.002: +0.0529 mm here).
    jbn = (gear1.x + gear2.x - sum_x) * (2 * gear1.mn * np.sin(gear1.alpha))
    return alphatw, jbn


def _working_pitch_diameters_external(al, u, db1, db2, alphatw):
    return 2 * al / (u + 1), 2 * u * al / (u + 1)   # by position: gear 1 always gets the smaller one


def _working_pitch_diameters_internal(al, u, db1, db2, alphatw):
    return db1 / np.cos(alphatw), db2 / np.cos(alphatw)   # ISO_21771_2007, Eq. (56)-(57), third form


def _galpha_internal(galpha1, galpha2, da1, db1, da2, db2, z2, al, alphatw):
    # QUESTION(Q22): I2 / row 45, entry `galpha`: the sign factor is on the tip term of the ring only, not on 2 a_w sin(alpha_wt).
    return 1/2 * ((da1**2 - db1**2)**(1/2) + z2/abs(z2) * (da2**2 - db2**2)**(1/2) - 2 * al * np.sin(alphatw))


def _profile_shift_check_external(gear1, gear2, inv_alphatw):
    # QUESTION(Q26): I6, entry `profile_shift_check`: the sum x1 + x2 with z1 + z2 (the internal entry below uses differences).
    # inv_alphatw = inv_alphat + 2*tan(alpha)*(x1+x2)/(z1+z2)   (ISO_21771_2007, 5.3, Eq. (62))
    z_sum = gear1.z + gear2.z
    inv_alphat = np.tan(gear1.alphat) - gear1.alphat
    return "x1+x2", (inv_alphatw - inv_alphat) * z_sum / (2 * np.tan(gear1.alpha)), gear1.x + gear2.x


def _profile_shift_check_internal(gear1, gear2, inv_alphatw):
    # QUESTION(Q26): I6, entry `profile_shift_check`.
    # compares a DIFFERENCE built with z2 - z1, while the centre distance comes from x1 + x2 and z2 + z1
    z_diff = gear2.z - gear1.z
    inv_alphat = np.tan(gear1.alphat) - gear1.alphat
    return "x2-x1", (inv_alphatw - inv_alphat) * z_diff / (2 * np.tan(gear1.alpha)), gear2.x - gear1.x


def _force_input_external(pair, driver):
    return pair.rl1, pair.gear1.beta, pair.u            # the torque is always taken on gear 1


def _force_input_internal(pair, driver):
    if driver == "gear1":
        return pair.rl1, pair.gear1.beta, pair.u
    return pair.rl2, pair.gear2.beta, 1.0 / pair.u


def _undercutting(gear) -> bool:
    """True if z < z_min = 2 cos(beta) / sin(alphat)**2 * (haP - x), i.e. the root is undercut.

    Equals ISO_21771_2007, 7.7, Eq. (135) (minimum generating profile shift, rack-type tool,
    external gears) solved for z with the default rack, to 3e-5 (one reader); no z_min is printed.
    """
    z_min = 2.0 * (np.cos(gear.beta) / (np.sin(gear.alphat) ** 2)) * (gear.haP - gear.x)
    return bool(gear.z < z_min)


def _beta_range_external(data, tag):
    if not (0.0 <= data.beta_n_deg < 45.0):
        return [f"{tag}: beta_n_deg out of range [0, 45), got {data.beta_n_deg}"]
    return []


def _gear_limits_external(gear, tag):
    if _undercutting(gear):
        return [f"{tag}: gear is undercut — z={gear.z}, mn={gear.mn}, "
                f"alpha={gear.alpha_n_deg}°, beta={gear.beta_n_deg}°, x={gear.x}"]
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


def _interference_external(pair):
    """Involute interference of an external pair: -T1A > T1C on gear 1 and its mirror on gear 2 [mm]. No library source."""
    errors = []
    ra1, ra2 = pair.da1_w / 2.0, pair.da2_w / 2.0
    approach = (
        np.sqrt(max(ra2**2 - pair.gear2.rb**2, 0.0))
        - pair.al * np.sin(pair.alphatw)
    )
    limit1 = pair.gear1.rb * np.tan(pair.alphatw)
    if approach > limit1:
        errors.append(
            f"Involute interference on gear1 (pinion): path of approach "
            f"{approach:.4f} mm > limit {limit1:.4f} mm. "
            "Reduce haP, increase x1, or increase z1."
        )
    recess = (
        np.sqrt(max(ra1**2 - pair.gear1.rb**2, 0.0))
        - pair.al * np.sin(pair.alphatw)
    )
    limit2 = pair.gear2.rb * np.tan(pair.alphatw)
    if recess > limit2:
        errors.append(
            f"Involute interference on gear2 (wheel): path of recess "
            f"{recess:.4f} mm > limit {limit2:.4f} mm. "
            "Reduce haP, increase x2, or increase z2."
        )
    return errors


def _solve_inv(inv_target: float, alpha0: float = 0.3491) -> float:
    """Inverse involute by Newton-Raphson from alpha0 = 0.3491 rad (20 deg): at most 50 steps, stop at abs(f) < 1e-12.

    No convergence test after the loop: for a target of 0.2 it returns 8.5e22 without raising.
    """
    # QUESTION(Q19): this second inverse involute is kept only because the three interference tests below use it and their messages are recorded; the alternative is to delete it with them (plan 2.1) or to use the brentq of the pair.
    alpha = alpha0
    for _ in range(50):
        f  = np.tan(alpha) - alpha - inv_target
        df = np.tan(alpha) ** 2          # d/d(alpha) [tan(alpha) - alpha] = tan(alpha)**2
        alpha -= f / df
        if abs(f) < 1e-12:
            break
    return alpha


def _involute_interference(ring, z1: int, x1: float = 0.0) -> bool:
    """True when the tip pressure angle of the ring exceeds the working angle (alphaa2 > alphaw).

    Uses the NORMAL pressure angle in inv(alpha) and the signed z of the ring:
    inv(alphaw) = inv(alpha) + 2 tan(alpha) (x1 - x) / (z - z1). No library source (KHK named).
    """
    inv_alpha = np.tan(ring.alpha) - ring.alpha
    inv_alphaw = inv_alpha + 2 * np.tan(ring.alpha) * (x1 - ring.x) / (ring.z - z1)
    alphaw = _solve_inv(inv_alphaw)
    alphaa2 = np.arccos(ring.db / ring.da)          # pressure angle at tip of internal gear
    return alphaa2 > alphaw


def _trochoid_interference(ring, z1: int, x1: float = 0.0) -> bool:
    """True when the tip pressure angle of the mating gear exceeds the working angle (alphaa1 > alphaw).

    The mating gear is taken as a SPUR gear (d1 = mn z1) with a tip without clearance,
    da1 = d1 + 2 mn (haP + x1). No library source (KHK named).
    """
    d1  = ring.mn * z1
    db1 = d1 * np.cos(ring.alpha)
    inv_alpha  = np.tan(ring.alpha) - ring.alpha
    inv_alphaw = inv_alpha + 2 * np.tan(ring.alpha) * (x1 - ring.x) / (ring.z - z1)
    alphaw     = _solve_inv(inv_alphaw)
    da1    = d1 + 2 * ring.mn * (ring.haP + x1)    # tip diameter of external gear
    alphaa1 = np.arccos(db1 / da1)
    return alphaa1 > alphaw


def _trimming_interference(ring, z1: int) -> bool:
    """True when (z - z1) < 2 with the SIGNED z of the ring: always true for z < 0. No library source."""
    return (ring.z - z1) < 2


def _validate_mesh(ring, z1: int, x1: float = 0.0) -> list[str]:
    """Messages of the three interference tests of a ring against a mating external gear (z1, x1)."""
    errors: list[str] = []
    tag = ring.label or "Gear"
    if _involute_interference(ring, z1, x1):
        errors.append(
            f"{tag}: involute interference with mating gear (z1={z1}, x1={x1})."
        )
    if _trochoid_interference(ring, z1, x1):
        errors.append(
            f"{tag}: trochoid (flank) interference with mating gear (z1={z1}, x1={x1})."
        )
    if _trimming_interference(ring, z1):
        errors.append(
            f"{tag}: trimming interference — tooth difference z2-z1={ring.z - z1} < 2."
        )
    return errors


def _summary_head_external(pair):
    g1, g2 = pair.gear1, pair.gear2
    return "HelicalGearMeshing", [
        f"  gear1 : z={g1.z}, mn={g1.mn}, x={g1.x}, β={g1.beta_n_deg}°",
        f"  gear2 : z={g2.z}, mn={g2.mn}, x={g2.x}, β={g2.beta_n_deg}°",
    ]


def _summary_head_internal(pair):
    g1, g2 = pair.gear1, pair.gear2
    return "InternalGearMeshing", [
        f"  gear1 (ext): z={g1.z}, mn={g1.mn}, x={g1.x}, β={g1.beta_n_deg}°",
        f"  gear2 (int): z={g2.z}, mn={g2.mn}, x={g2.x}, β={g2.beta_n_deg}°",
        f" da2 : {pair.da2_w:.4f} mm",
        f" T1C : {pair.T1C:.4f} mm   (pinion tip contact point)",
    ]


_LEGACY = {
    # QUESTION(Q21): I1 / row 3. A helical ring gets beta_b = arcsin(sin beta / cos alpha_n), d_b from it and alpha_t = arccos(d_b/d); ISO_21771_2007 Eq. (5)-(7), (13), (19) print one rule for both kinds (the external one).
    "base_cylinder": {"external": _base_cylinder_external, "internal": _base_cylinder_internal},
    # QUESTION(Q25): I5 / row 5. A ring has no virtual number of teeth, so equalise_gs raises AttributeError for an internal pair (Q13).
    "virtual_teeth": {"external": _virtual_teeth_external, "internal": lambda mn, z, beta: {}},
    # QUESTION(Q55): the helix-angle range is checked for an external gear only; a ring accepts any beta.
    "beta_range": {"external": _beta_range_external, "internal": lambda data, tag: []},
    # QUESTION(Q55): limits of one gear differ by kind: undercut (external only) against da < d and da > db (ring only).
    "gear_limits": {"external": _gear_limits_external, "internal": _gear_limits_internal},
    # QUESTION(Q21): I1. The internal pair takes alpha_t and beta of the RING (the external pair those of gear 1) and another expression of a_w; the two agree at beta = 0.
    "alphatw_from_x_sum": {"external": _alphatw_from_x_sum_external, "internal": _alphatw_from_x_sum_internal},
    # QUESTION(Q21): I1, same choice of gear with an imposed centre distance; the internal pair also stores a backlash jbn (Q57).
    "alphatw_from_al": {"external": _alphatw_from_al_external, "internal": _alphatw_from_al_internal},
    # QUESTION(Q03): row 1 / X1. External u = max/min >= 1 whatever the order of the gears; ISO_21771_2007 Eq. (52) is u = z2/z1, signed (what the internal pair does).
    "u": {"external": lambda z1, z2: max(z1, z2) / min(z1, z2), "internal": lambda z1, z2: z2 / z1},
    # QUESTION(Q03): row 1 / X1. External working pitch diameters by position (gear 1 the smaller); under d_b/cos(alpha_wt) they are exchanged when z1 > z2 (cases 3b, 3c).
    "working_pitch_diameters": {"external": _working_pitch_diameters_external,
                                "internal": _working_pitch_diameters_internal},
    # QUESTION(Q23): I3. Factor of the tip alteration k, and the share of k taken by gear 2: (z1+z2)/2 on both gears (external) against (z2-z1)/2 on gear 1 only (internal). No library source for either.
    "tip_alteration": {"external": lambda z1, z2: ((z1 + z2) / 2.0, 1.0),
                       "internal": lambda z1, z2: ((z2 - z1) / 2.0, 0.0)},
    # QUESTION(Q22): I2 / row 45. Internal galpha has an unsigned 2 a_w sin(alpha_wt) term (negative result); ISO_21771_2007 Eq. (77) puts the sign factor on the whole bracket (= the stored AE).
    "galpha": {"external": lambda galpha1, galpha2, *_: galpha1 + galpha2, "internal": _galpha_internal},
    # QUESTION(Q26): I6. The internal check compares x2 - x1 with a value built on z2 - z1, so it reports an al that the family itself derived (case 5).
    "profile_shift_check": {"external": _profile_shift_check_external, "internal": _profile_shift_check_internal},
    # QUESTION(Q03): row 1 / X1. The external pair ignores the driver and takes the torque on gear 1; the internal pair takes it on the driver.
    "force_input": {"external": _force_input_external, "internal": _force_input_internal},
    # QUESTION(Q24): I4 / row 4. Interference tests with no library source; trimming is always true for z2 < 0 and trochoid fires for most pairs, so validate_or_raise() raises for every internal pair.
    "interference": {"external": _interference_external,
                     "internal": lambda pair: _validate_mesh(pair.gear2, pair.gear1.z, pair.gear1.x)},
    # QUESTION(Q50): presentation text of the two classes this family replaces, recorded by the baseline (class names in __repr__ and summary(), two extra lines for an internal pair).
    "text": {"external": _summary_head_external, "internal": _summary_head_internal},
}


def _legacy(name: str, z):
    """Entry of ``_LEGACY`` for the kind of teeth given by the sign of ``z``."""
    return _LEGACY[name][_kind(z)]


def _henriot_shifts(gear1, gear2, a):
    """Profile shifts (x1, x2) [-] that equalise the specific sliding at the two ends of the path (Henriot).

    Carried verbatim from the external pair. Returns shifts for some pairs (x1 = 4.5 for
    z = 12/54), raises ``RuntimeError`` for others (every external pair of the baseline) and
    ``AttributeError`` for an internal pair (first line: a ring has no ``zn``). Known defects, not
    repaired: ``rb`` is written without the factor 1/2 and the pitch radius uses mt, not mn; the
    unknown is added to tip radii that already hold the declared shift. No library source
    (ISO_21771_2007, 5.3: the split of the sum "may be decided", no rule).
    """
    # QUESTION(Q13): rows 5, 46. Kept as it is today (raises for every baseline pair); the alternative is to delete equalise_gs or to repair it against a source.
    z1, z2   = gear1.zn, gear2.zn
    ra1, ra2 = gear1.mt * z1 / 2.0 + gear1.mn * (gear1.haP + gear1.cP + gear1.x), gear2.mt * z2 / 2.0 + gear2.mn * (gear2.haP + gear2.cP + gear2.x)
    rb1, rb2 = gear1.mt * z1 * np.cos(gear1.alphat), gear2.mt * z2 * np.cos(gear2.alphat)
    mn       = gear1.mn
    alphat   = gear1.alphat

    def _gs1B(ra2_v, rb2_v, a_v, alphat_v):
        sq  = np.sqrt(max(ra2_v**2 - rb2_v**2, 0.0))
        den = a_v * np.sin(alphat_v) - sq
        if abs(den) < 1e-12:
            return np.inf
        return abs(1.0 - (z1 / z2) * sq / den)

    def _gs2A(ra1_v, rb1_v, a_v, alphat_v):
        sq  = np.sqrt(max(ra1_v**2 - rb1_v**2, 0.0))
        den = a_v * np.sin(alphat_v) - sq
        if abs(den) < 1e-12:
            return np.inf
        return abs((z2 / z1) * sq / den - 1.0)

    def _solve_direct(z1_v, z2_v, ra1_v, ra2_v, rb1_v, rb2_v, a_v, alphat_v):
        def equation(x):
            return (_gs1B(ra2_v - x * mn, rb2_v, a_v, alphat_v)
                  - _gs2A(ra1_v + x * mn, rb1_v, a_v, alphat_v))

        x_lim = min(
            (ra2_v - rb2_v) / mn * 0.9,
            (a_v * np.sin(alphat_v) - 1e-3) / mn
        )
        xs   = np.linspace(-x_lim * 0.8, x_lim * 0.8, 400)
        vals = [equation(xi) for xi in xs]
        bracket = [(xs[i], xs[i+1])
                   for i in range(len(vals)-1)
                   if vals[i] * vals[i+1] < 0]
        if not bracket:
            raise RuntimeError(
                "correction_for_gs_equilibrium: no solution found. "
                "Check gear geometry."
            )
        return optimize.brentq(equation, bracket[0][0], bracket[0][1], xtol=1e-10)

    # --- z1 + z2 > 60: x2 = -x1, centre distance kept ---
    if (z1 + z2) > 60:
        x = _solve_direct(z1, z2, ra1, ra2, rb1, rb2, a, alphat)
        return x, -x

    # --- z1 + z2 <= 60: x1 against a hypothetical mate with 60 - z1 teeth, then x2 ---
    z2_h  = 60 - z1
    a_h   = gear1.mt / 2.0 * (z1 + z2_h)
    ra2_h = gear1.mt / 2.0 * z2_h + mn
    rb2_h = gear1.mt / 2.0 * z2_h * np.cos(alphat)

    x1 = _solve_direct(z1, z2_h, ra1, ra2_h, rb1, rb2_h, a_h, alphat)

    inv_alphat = np.tan(alphat) - alphat
    alpha = gear1.alpha

    def equation_x2(x2_v):
        inv_aw = inv_alphat + 2.0 * np.tan(alpha) * (x1 + x2_v) / (z1 + z2)
        alphatw_v = optimize.brentq(_inv_residual, 1e-6, 0.7, args=(inv_aw,))
        al_v      = a * np.cos(alphat) / np.cos(alphatw_v)
        return (_gs1B(ra2 + x2_v * mn, rb2, al_v, alphatw_v)
              - _gs2A(ra1 + x1  * mn, rb1, al_v, alphatw_v))

    xs2   = np.linspace(-1.5, 1.5, 600)
    vals2 = [equation_x2(xi) for xi in xs2]
    bracket2 = [(xs2[i], xs2[i+1])
                for i in range(len(vals2)-1)
                if vals2[i] * vals2[i+1] < 0]
    if not bracket2:
        raise RuntimeError(
            "correction_for_gs_equilibrium (z≤60): no solution for x2."
        )
    x2 = optimize.brentq(equation_x2, bracket2[0][0], bracket2[0][1], xtol=1e-10)
    return x1, x2


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
        Helix angle beta at the reference cylinder [deg], a magnitude: there is no hand of helix.
        The "_n" of the name has no meaning in the formulas. Default 0.0 (spur gear).
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
    .. [1] ISO_53_1998, Table 2.
    .. [2] ISO_21771_2007, 4.1.5 (sign of z), 4.2.9 (profile shift).
    """
    # QUESTION(Q18): the classes this record replaces cited "ISO 53:2013" for the rack defaults; the edition held and checked is ISO_53_1998, Table 2 (text only, same numbers).
    mn: float
    z: int
    _: KW_ONLY
    Ca: float = 0.0
    Cf: float = 0.0
    x: float = 0.0
    b: float = 0.0
    alpha_n_deg: float = 20.0
    beta_n_deg: float = 0.0
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
            0 < alpha_n_deg < 45, 0 <= beta_n_deg < 45 (external gear only), Ra, Rq, Rz >= 0.
            The limits 45 deg have no library source.
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
    in mesh (clause 5), ``validate_gear`` / ``validate_pair`` the limits (clause 7 and checks with
    no library source), ``forces`` the tooth forces for a torque as the shaft loads still use them,
    ``mesh_forces`` the same mesh under D-0010 (``MeshForces``), ``contact_inputs`` the contact data
    along the path of contact. An internal pair is a pair whose gear 2 has z < 0.

    Notes
    -----
    Where the external and the internal classes this family replaces computed a quantity
    differently and no ruling has chosen, both expressions are kept in the module table
    ``_LEGACY``, each with its question id; the docstrings below say "by kind" for those.

    Assumptions: rigid, unmodified involute flanks; aligned gears (the common face width is
    min(b1, b2)); the tip circle is taken as the limit of the active profile (the standard uses
    the active tip diameter d_Na); no hand of helix.

    Warnings
    --------
    - Q28 (finding I8): for an internal pair with a_w > 0, ISO_21771_2007 Eq. (56) first form,
      (64), (65), (68), (69) and Eq. (92) first form do not hold as printed (two readers). None of
      them is coded here: the signed segments ``T1A``..``T2E`` and the form d_b / cos(alpha_wt)
      are used instead.
    - Q29 (finding I9): ISO_6336-1_2019, Table 2, footnote a, makes a, d, d_a and z_2 negative
      for internal gearing; here, as in ISO 21771, only z is signed. A rating formula that
      contains a, d or u must be checked for the sign it expects.

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
            alphat : transverse pressure angle [rad] (4.3.5, Eq. (13); by kind, Q21).
            betab : base helix angle [rad] (4.3.3, Eq. (5)-(7); by kind, Q21).
            db, rb : base diameter and radius [mm] (4.3.10, Eq. (19)-(20); by kind, Q21).
            tau : angular pitch, 2 pi / abs(z) [rad] (4.4, Eq. (21)).
            rhoF : root fillet radius of the basic rack, rfP mn [mm] (ISO_53_1998, Table 2, times mn).
            pb, pbt : normal and transverse base pitch [mm] (4.4.5, Eq. (29), (28)).
            dv, rv : diameter and radius of the V-circle, d + 2 (z/abs(z)) x mn [mm] (4.5.1, Eq. (32)).
            da, ra : tip diameter and radius [mm] (4.5.3, Eq. (33), PLUS the clearance cP: Q04).
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
        working tip diameters with k are pair quantities (``assemble_pair``).

        Warnings
        --------
        - Q04 / Q27: da = d + 2 (z/abs(z)) mn (haP + cP + x). Eq. (33) has no clearance term, so
          the teeth are longer than the standard ones by cP mn and a pair at its reference centre
          distance has no bottom clearance.
        - Q21: for a helical ring the base helix angle is larger than the reference helix angle
          and alphat, db, pbt differ from those of the external gear it meshes with.

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
        alphat, betab, db = _legacy("base_cylinder", z)(mn, z, alpha, beta, d)

        # QUESTION(Q04): row 2 / X2. The tip carries the clearance cP; ISO_21771_2007, 4.5.3, Eq. (33) is d + 2 (z/|z|) (x mn + haP mn + k mn), no clearance term.
        # QUESTION(Q27): I7. Same for a ring (case 4: tip 115 mm against 116 mm), which puts the start of contact 2.39 mm beyond the base-tangent point instead of 0.07 mm.
        da = d + 2 * mn * s * (data.haP + data.cP + x)
        df = d + 2 * mn * s * (x - hfP)
        dv = d + 2 * s * x * mn

        return dict(
            alpha=alpha, beta=beta, hfP=hfP, mt=mt, mx=mx, pn=pn, pt=pt, pz=pz, px=px,
            d=d, r=d / 2.0, alphat=alphat, betab=betab, db=db, rb=db / 2.0,
            tau=2 * np.pi / abs(z), rhoF=data.rfP * mn,
            pb=pn * np.cos(alpha), pbt=pt * np.cos(alphat),
            dv=dv, rv=dv / 2, da=da, ra=da / 2.0, df=df, rf=df / 2.0,
            **_legacy("virtual_teeth", z)(mn, z, beta),
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
                      equalise_gs: bool = False, addendum_reduction: bool = False) -> tuple[Gear, Gear, dict]:
        """Working centre distance and pressure angle, contact ratios and path of contact of two gears in mesh.

        Parameters
        ----------
        gear1 : Gear
            External gear (the pinion in the formulas).
        gear2 : Gear
            Mating gear, external or internal (z < 0); same module, normal pressure angle and
            helix angle as ``gear1``.
        label : str, optional
            Name used in the summary.
        al : float or None, optional
            Working centre distance a_w [mm]. None: derived from x1 + x2 (zero backlash). A
            value: imposed; the working pressure angle is back-calculated.
        equalise_gs : bool, optional
            If True and ``al`` is None, the profile shifts are first taken from
            ``correction_for_gs_equilibrium`` (Q13: it raises for every pair of the baseline).
        addendum_reduction : bool, optional
            Apply the tip alteration k to the working tip diameters. Default False (k = 0).

        Returns
        -------
        tuple
            ``(gear1, gear2, attributes)``; the attributes set on the ``GearPair`` are:

            label; al_imposed : whether ``al`` was given.
            a : reference centre distance, (d2 + (z2/abs(z2)) d1) / 2 [mm] (not printed in
            ISO 21771 clause 5).
            al : working centre distance a_w [mm], positive for both kinds (5.2.5, Eq. (58)).
            alphatw, inv_alphatw : transverse working pressure angle alpha_wt and its involute
            [rad] (5.2.3-5.2.4, Eq. (54), (55); by kind, Q21).
            jbn : normal backlash [mm]; internal pair with ``al`` imposed only, else None. Its
            sign is opposite to ISO_21771_2007, 5.3, Eq. (63) (Q57).
            x1, x2 : profile shift coefficients used [-].
            u : gear ratio [-] (5.2.1, Eq. (52); by kind, Q03).
            k : tip alteration coefficient [-] (4.5.2 gives no value; no library source).
            dl1, dl2, rl1, rl2 : working pitch diameters d_w and radii [mm] (Eq. (56)-(57); by
            kind, Q03).
            da1_w, da2_w : working tip diameters, with k [mm] (Eq. (33) plus cP: Q04).
            alpha_a1, alpha_a2 : transverse pressure angles at the tips [rad].
            epslon_a1, epslon_a2 : addendum contact ratios of gear 1 and gear 2 [-]; that of
            a ring is negative as stored (case 4: -1.564) and is subtracted in epslon_alpha.
            epslon_alpha, epslon_beta, epslon_gamma : transverse contact ratio, overlap ratio
            and total contact ratio [-] (5.4.7, Eq. (88)-(97)).
            b : common face width, min(b1, b2) [mm].
            galpha1, galpha2 : addendum parts of the path of contact, rb (tan(alpha_a) -
            tan(alpha_wt)) [mm]; galpha2 of a ring is negative as stored (case 4: -9.235 mm).
            galpha : length of the path of contact [mm] (5.4.5.2,
            Eq. (77)-(80); by kind, Q22).
            ReqI : reduced radius of curvature at the pitch point, transverse plane [mm].
            T1T2, T1E, T2A, T1A, T1B, T1C, T1D, T2E, T2B, T2C, T2D : distances along the line of
            action from the base-tangent points T1, T2 to the points A..E of the path [mm]. They
            are the transverse radii of curvature of the flanks at those points: 5.4.5.3,
            Eq. (81)-(87) print T1C, T2C, T2A, T1E, T1B, T2D and T1T2.
            AE, AB, AC, AD : distances from A [mm]; rA1, rB1, rD1, rA2, rB2, rD2 : radii from
            the gear axis to the path points [mm] (derived: sqrt(segment**2 + rb**2); no numbered
            equation, no library source).
            HV : gear loss factor [-] (ISO_TR_14179-2_2001, 5.2.4, Eq. (14); unchecked, Q53).
            Ram, Rrms, RzS : (Ra1 + Ra2)/2, sqrt(Rq1**2 + Rq2**2), Rz1 + Rz2 [um] (no source).
            profile_shift_check : (wording, value implied by al, declared value) of the
            consistency check of ``validate_pair`` (5.3, Eq. (62); by kind, Q26).

        Raises
        ------
        ValueError
            If the gears differ in module, normal pressure angle or helix angle (relative
            tolerance 1e-6), or if gear 2 is internal and abs(z2) <= z1.
        RuntimeError, AttributeError
            From ``correction_for_gs_equilibrium`` when ``equalise_gs`` is True.

        Notes
        -----
        Sign factor s2 = z2/abs(z2). For an internal pair (s2 = -1) the segments measured from
        T2 and T1T2 are negative, as the NOTE of ISO_21771_2007, 5.4.5.3 states for T1T2 and the
        radii of curvature of the ring. A is the start of the path on the tip circle of gear 2
        and E its end on the tip circle of gear 1, i.e. the names are those of gear 1 driving;
        they are stored once, whatever gear drives. A negative ``T1A`` means that the path
        starts beyond the base-tangent point of gear 1 (interference); it is stored, not refused.

        ``rA1``, ``rB1``, ``rD1`` are the radii of gear 1 at A, B, D; ``rA2``, ``rB2``, ``rD2``
        are the radii of gear 2 at E, D, B (mirrored names).

        Warnings
        --------
        - Q03: for an external pair u = max(z)/min(z) and gear 1 gets the smaller working
          diameter whatever its number of teeth; ``HV`` uses z1 and u in the same way.
        - Q22: ``ReqI`` is 1 / (1/rho_C1 + 1/rho_C2) with a POSITIVE radius for a ring
          (ISO_21771_2007, Eq. (82) makes rho_C2 negative), and ``galpha`` of an internal pair is
          not its ``AE``.
        - Q23: the factor of k and the gear that takes it differ by kind.

        References
        ----------
        .. [1] ISO_21771_2007, 5.2-5.4, Eq. (52)-(97) (checked by a second reader).
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
        s2 = z2 / abs(z2)   # +1 external pair, -1 internal pair
        if s2 < 0 and abs(z2) <= z1:   # the pinion must fit inside the ring
            raise ValueError(
                f"Internal gear must have more teeth than the external gear "
                f"(z2={z2} <= z1={z1})."
            )

        # --- reference centre distance: sum of the reference radii, difference for an internal pair ---
        a = (gear2.d + s2 * gear1.d) / 2.0

        # --- working centre distance and transverse working pressure angle ---
        al_imposed = al is not None
        jbn = None
        if al_imposed:
            alphatw, jbn = _legacy("alphatw_from_al", z2)(gear1, gear2, a, al)
            x1, x2 = gear1.x, gear2.x
        else:
            if equalise_gs:
                x1, x2 = self.correction_for_gs_equilibrium(gear1, gear2)
            else:
                x1, x2 = gear1.x, gear2.x
            alphatw, al = _legacy("alphatw_from_x_sum", z2)(gear1, gear2, x1 + x2)
        inv_alphatw = np.tan(alphatw) - alphatw

        u = _legacy("u", z2)(z1, z2)

        # QUESTION(Q21): I1. alphat, betab and pbt of GEAR 1 are used from here on, while alphatw and al of a helical internal pair came from alphat of the ring: its two gears have different base pitches (case 5: 6.0870 against 6.1418 mm).
        alpha  = gear1.alpha     # normal pressure angle
        alphat = gear1.alphat    # transverse pressure angle
        beta   = gear1.beta      # helix angle
        betab  = gear1.betab     # base helix angle
        m      = gear1.mn

        # --- tip alteration coefficient k (the code names the MAAG Gear Book; no library source) ---
        k_factor, k_share_2 = _legacy("tip_alteration", z2)(z1, z2)
        if addendum_reduction:
            k = k_factor * (
                (_involute(alphatw) - _involute(alphat)) / np.tan(alpha)
                - 1.0 / np.cos(beta) * (np.cos(alphat) / np.cos(alphatw) - 1.0)
            )
        else:
            k = 0.0

        dl1, dl2 = _legacy("working_pitch_diameters", z2)(al, u, gear1.db, gear2.db, alphatw)
        rl1, rl2 = dl1 / 2, dl2 / 2

        # --- working tip diameters, with k; the clearance cP is in them as in Gear.da (Q04) ---
        da1 = gear1.d + 2 * m * (gear1.haP + gear1.cP + x1 - k)
        da2 = gear2.d + s2 * 2 * m * (gear2.haP + gear2.cP + x2 - k_share_2 * k)
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

        galpha1 = rb1 * (np.tan(alpha_a1) - np.tan(alphatw))
        galpha2 = rb2 * (np.tan(alpha_a2) - np.tan(alphatw))
        galpha = _legacy("galpha", z2)(galpha1, galpha2, da1, db1, da2, db2, z2, al, alphatw)

        # reduced radius of curvature at the pitch point C, transverse plane: rho_C = r_w sin(alpha_wt)
        # QUESTION(Q22): I2 / row 45. Plus-sign sum with a positive ring radius; with rho_C2 negative (ISO_21771_2007, Eq. (82)) the internal value is larger by a factor that depends on u (case 4: 5.130 against 10.261 mm, factor 2.000; case 5: 1.987).
        ReqI = 1 / (1 / (rl1 * np.sin(alphatw))
                    + 1 / (rl2 * np.sin(alphatw)))

        # --- path of contact: distances along the line of action (signed with s2 on the side of gear 2) ---
        T1T2 = s2 * al * np.sin(alphatw)
        T1E = (ra1**2 - rb1**2) ** (1 / 2)          # end of contact E: tip circle of gear 1
        T2A = s2 * ((ra2**2 - rb2**2) ** (1 / 2))   # start of contact A: tip circle of gear 2
        # QUESTION(Q15): a negative T1A (start of contact beyond the base-tangent point T1: interference) is stored, not refused; baseline case 4 has T1A = -2.395 mm.
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

        # QUESTION(Q53): gear loss factor "according to Ohlendorf"; the source ISO_TR_14179-2_2001, 5.2.4, Eq. (14) is a clause-map citation, the equation was not transcribed or checked.
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
            profile_shift_check=_legacy("profile_shift_check", z2)(gear1, gear2, inv_alphatw),
        )

    @staticmethod
    def correction_for_gs_equilibrium(gear1: Gear, gear2: Gear) -> tuple[float, float]:
        """Profile shifts x1, x2 that equalise the maximum specific sliding (Henriot procedure).

        Parameters
        ----------
        gear1, gear2 : Gear
            The two gears of the pair.

        Returns
        -------
        tuple of float
            (x1, x2) [-].

        Raises
        ------
        RuntimeError
            If no sign change of the sliding difference is found (every external pair of the
            baseline).
        AttributeError
            For an internal pair: a ring has no virtual number of teeth ``zn``.

        Warnings
        --------
        Q13: not usable as it is; see ``_henriot_shifts``. No library source.
        """
        return _henriot_shifts(gear1, gear2, (gear1.d + gear2.d) / 2.0)

    # ----------------------------------------------------------------------------------------------
    # Limits and checks (ISO 21771 clause 7; most have no library source)
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def undercutting(gear: Gear) -> bool:
        """Undercut of the root of an external gear: True if z < z_min (see ``_undercutting``)."""
        return _undercutting(gear)

    def validate_gear(self, gear: Gear) -> list[str]:
        """Check one gear.

        Returns
        -------
        list of str
            The messages of ``CylindricalGearData.validate``, then by kind (Q55): undercut for an
            external gear; da < d and da > db for a ring.
        """
        tag = gear.label or "Gear"
        return gear.data.validate() + _legacy("gear_limits", gear.z)(gear, tag)

    def validate_pair(self, pair) -> list[str]:
        """Check the pair geometry.

        Returns
        -------
        list of str
            One message per violation; empty if the pair is valid. (1) The messages of both gears,
            each twice. (2) Only when ``al`` was imposed: profile shift implied by ``al`` against
            the declared one, tolerance 1e-4 (by kind, Q26). (3) epslon_alpha < 1.0; a value
            below 1.2 adds a message starting "Warning:" to the same list, so
            ``validate_or_raise`` raises on it. (4) Interference, by kind (Q24): external pair,
            -T1A > T1C and its mirror; internal pair, the involute, trochoid and trimming tests.

        Warnings
        --------
        The list is never empty for an internal pair (Q24): ``validate_or_raise`` raises for all
        of them. None of the limits of (3) and (4) has a library source; ISO_21771_2007, 5.4.6
        states that interference of internal pairs is outside its scope.
        """
        errors: list[str] = []

        # QUESTION(Q56): each gear message appears twice (today the pair validated the gears and then its working copies of them); the alternative is once.
        gear_errors = pair.gear1.validate() + pair.gear2.validate()
        errors.extend(gear_errors + gear_errors)

        if pair.al_imposed:
            wording, from_al, declared = pair.profile_shift_check
            if abs(from_al - declared) > 1e-4:
                errors.append(
                    f"Profile shift inconsistency: declared {wording}={declared:.4f} "
                    f"but al={pair.al:.4f} mm implies {wording}={from_al:.4f}. "
                    "Adjust x1, x2, or al so they are consistent."
                )

        if pair.epslon_alpha < 1.0:
            errors.append(
                f"Transverse contact ratio εα={pair.epslon_alpha:.4f} < 1.0 — "
                "gears will lose contact during meshing."
            )
        elif pair.epslon_alpha < 1.2:
            errors.append(
                f"Warning: transverse contact ratio εα={pair.epslon_alpha:.4f} is low (< 1.2) — "
                "consider increasing tooth count or face width."
            )

        errors.extend(_legacy("interference", pair.gear2.z)(pair))
        return errors

    # ----------------------------------------------------------------------------------------------
    # Text
    # ----------------------------------------------------------------------------------------------
    def pair_summary(self, pair) -> str:
        """Multi-line text with the main data of the pair (lengths [mm], angles [deg])."""
        name, head = _legacy("text", pair.gear2.z)(pair)
        lines = [
            f"── {pair.label or name} ─────────────────────────────────",
            *head,
            f"  u     : {pair.u:.4f}",
            f"  a     : {pair.a:.4f} mm   (reference centre distance)",
            f"  al    : {pair.al:.4f} mm   (working centre distance)",
            f"  αtw   : {np.degrees(pair.alphatw):.4f}°",
            f"  dl1   : {pair.dl1:.4f} mm",
            f"  dl2   : {pair.dl2:.4f} mm",
            f"  εα    : {pair.epslon_alpha:.4f}",
            f"  εβ    : {pair.epslon_beta:.4f}",
            f"  εγ    : {pair.epslon_gamma:.4f}",
            "────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def pair_repr(self, pair) -> str:
        """One-line representation: tooth numbers, module, helix angle, al, alphatw, eps_gamma."""
        name, _ = _legacy("text", pair.gear2.z)(pair)
        return (
            f"{name}(z1={pair.gear1.z}, z2={pair.gear2.z}, "
            f"mn={pair.gear1.mn}, β={pair.gear1.beta_n_deg}°, al={pair.al:.4f} mm, "
            f"αtw={np.degrees(pair.alphatw):.4f}°, εγ={pair.epslon_gamma:.4f})"
        )

    # ----------------------------------------------------------------------------------------------
    # Forces
    # ----------------------------------------------------------------------------------------------
    # QUESTION(Q05) / QUESTION(Q07a) / QUESTION(Q07b): this dictionary is kept until rows 17-19 are ruled: the golden rows of cases 1-5 record it and Transmission builds today's shaft loads from it. mesh_forces() below is the same mesh under D-0010.
    @staticmethod
    def forces(pair, T: float, phi_deg: float = 0.0, rotation_dir: int = 1,
               driver: str = "gear1") -> dict[str, float]:
        """Tooth force components and output torque of the pair for an input torque T.

        The driver is an operating quantity, given here and not stored on the pair. For an
        internal pair the torque is taken on the gear named by ``driver``; for an external pair
        it is always taken on gear 1 and ``driver`` is not used (Q03).

        Parameters
        ----------
        pair : GearPair
            Pair of this family.
        T : float
            Torque on the driving gear [N m] (NOT N mm: 60.75 N m on gear 1 of baseline case 2
            gives Ft = 1659.83 N).
        phi_deg : float, optional
            Angle of the line of centres gear1 -> gear2 in the shaft cross-section, from +Y
            toward +Z [deg] (for an internal pair whose ring drives the meaning is open: Q20).
        rotation_dir : int, optional
            +1 or -1, sense of rotation of the driving gear about +X [-].
        driver : str, optional
            "gear1" or "gear2": the gear that delivers the input torque.

        Returns
        -------
        dict
            Ft, Fr, Fa, Fn : tangential, radial, axial and "normal" force [N], magnitudes;
            T_out : torque on the driven gear [N m]; theta_Fr_driver, theta_Ft_driver,
            theta_Fr_driven, theta_Ft_driven : angles [deg, 0-360) of the radial and tangential
            forces on the driving and on the driven gear, from +Y toward +Z; rotation_dir_out :
            sense of rotation of the driven gear [-].

        Raises
        ------
        ValueError
            If ``driver`` is not "gear1" or "gear2".

        Notes
        -----
        As coded, with rl_in the WORKING pitch radius [mm] of the gear that takes the torque,
        beta_in its reference helix angle and u_eff = u (gear 1 drives) or 1/u (gear 2 drives)::

            Ft = T / (rl_in / 1000)          T [N m], rl_in [mm] -> [m]
            Fr = Ft tan(alphatw)
            Fa = Ft tan(beta_in)
            Fn = Ft / cos(alphatw)
            T_out = T u_eff                  (ideal mesh, no losses)
            rotation_dir_out = -sign(u_eff) rotation_dir
            theta_Fr_driver = phi;        theta_Ft_driver = phi + 90 rotation_dir
            theta_Fr_driven = phi + 180;  theta_Ft_driven = phi + 180 + 90 rotation_dir_out

        For an internal pair u < 0: T_out has the sign opposite to T and the two gears turn the
        same way. Fa has no direction: there is no hand of helix (Q08).

        Warnings
        --------
        - ISO_6336-1_2019, 4.2.2, Formula (1) writes the NOMINAL F_t = 2000 T / d on the
          reference diameter; here Ft is on the working pitch cylinder.
        - No library source for Fr, Fa, Fn or for any direction (statics only). Open items: Q05
          (both radial forces of an external pair point toward the mesh and its two tangential
          forces point the same way), Q07a (Fa), Q07b (``Fn`` is the transverse base force F_bt,
          not the force normal to the flank), Q03, Q20.

        References
        ----------
        .. [1] ISO_6336-1_2019, 4.2.2, Formula (1)-(2) (checked).
        """
        if driver not in ("gear1", "gear2"):
            raise ValueError(
                f"driver must be 'gear1' or 'gear2', got '{driver}'. "
                "This selects which gear delivers input torque — it is independent "
                "of gear1/gear2 being external/internal."
            )
        # QUESTION(Q03): row 1 / X1. For an external pair the torque is put on gear 1 (the smaller working diameter) and T_out = T u with u = max/min, whatever `driver` says (cases 3b, 3c: 20755 W out for 9224.5 W in).
        rl_in, beta_in, u_eff = _legacy("force_input", pair.gear2.z)(pair, driver)

        # F_t on the WORKING pitch radius of the gear that takes the torque; T [N m] / r [m] gives [N]
        Ft = T / (rl_in / 1000)
        Fr = Ft * np.tan(pair.alphatw)
        # QUESTION(Q07a): row 19 / X4. Working tangential force with the REFERENCE helix angle; the alternative is F_bt tan(beta_b) (case 2: 444.750 against 449.236 N).
        # QUESTION(Q08): row 19 / X4. A magnitude: no hand of helix, so the thrust cannot reverse with the sense of rotation.
        Fa = Ft * np.tan(beta_in)
        # QUESTION(Q07b): row 19 / X4. Ft/cos(alphatw) is the transverse base force F_bt; the force normal to the flank is F_bn = F_bt/cos(beta_b) (case 2: 1791.646 against 1847.109 N).
        Fn = Ft / np.cos(pair.alphatw)
        T_out = T * u_eff

        # QUESTION(Q05): row 17 / X3. Radial force of the driver at phi (toward the mesh for an external pair, not separating); tangential force by the sense of rotation.
        # QUESTION(Q20): for an internal pair whose ring drives, phi is ambiguous (driver -> driven or gear1 -> gear2) and neither reading gives both a separating Fr and a moment of Ft equal to -rot T.
        theta_Fr_driver = phi_deg % 360.0
        theta_Ft_driver = (phi_deg + 90.0 * rotation_dir) % 360.0

        rotation_dir_out = - u_eff/abs(u_eff) * rotation_dir     # external pair: reversed; internal (u < 0): same sense

        # QUESTION(Q05): row 17 / X3. For an external pair theta_Ft_driven equals theta_Ft_driver: the two tangential forces are not an action-reaction pair.
        theta_Fr_driven = (phi_deg + 180.0) % 360.0
        theta_Ft_driven = (phi_deg + 180.0 + 90.0 * rotation_dir_out) % 360.0

        return {
            "Ft": Ft, "Fr": Fr, "Fa": Fa, "Fn": Fn, "T_out": T_out,
            "theta_Fr_driver": theta_Fr_driver, "theta_Ft_driver": theta_Ft_driver,
            "theta_Fr_driven": theta_Fr_driven, "theta_Ft_driven": theta_Ft_driven,
            "rotation_dir_out": rotation_dir_out,
        }

    def mesh_forces(self, pair, T1_Nm: float, rot1: int, *, hand1: int = 1) -> MeshForces:
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
        hand1 : int, optional
            Hand of the helix of gear 1: +1 right-hand (default), -1 left-hand [-]. Gear 2 has the
            opposite hand in an external pair and the same hand in an internal pair. No effect on
            a spur pair. The gear data hold no hand (Q08): when it is not given, gear 1 is taken
            as RIGHT-hand, so a driving gear 1 turning in the +1 sense is pushed along +X.

        Returns
        -------
        MeshForces
            ``F_tw_N``, ``F_r_N``, ``F_a_N``, ``F_bt_N``, ``F_bn_N``, ``F_t_nominal_N`` [N]
            (magnitudes); ``on_1``, ``on_2`` = (F_c, F_t, F_x) [N], signed, ``on_2 = -on_1``;
            ``rho_mm`` [mm]; ``x_offset_mm`` = (0, 0).

        Raises
        ------
        ValueError
            If ``rot1`` or ``hand1`` is not +1 or -1.

        Notes
        -----
        With d_w1 = d_b1 / cos(alpha_wt) the working pitch diameter, d_b1 the base diameter, d_1
        the reference diameter and beta_b the base helix angle of gear 1 [mm, rad]::

            F_tw = 2000 abs(T_1) / d_w1          F_r  = F_tw tan(alpha_wt)
            F_bt = 2000 abs(T_1) / d_b1          F_a  = F_bt tan(beta_b)
            F_bn = F_bt / cos(beta_b)            F_t (nominal) = 2000 abs(T_1) / d_1
            rho_1 = sign(z_2) d_w1 / 2           rho_2 = rho_1 - a_w
            F_1,t = -1000 T_1 / rho_1            F_1,c = -sign(z_1 z_2) F_r
            F_1,x = -h_1 sign(rho_1) sign(F_1,t) F_a

        The factor 2000 converts N m and mm to N (D-0010 item 5). Baseline case 2 (z 20/30,
        mn 3.5 mm, beta 15 deg), T_1 = 60.75 N m: F_tw 1659.831, F_r 674.505, F_a 449.236,
        F_bt 1791.646, F_bn 1847.109, F_t nominal 1676.571 N.

        Assumptions: ideal mesh (no friction, no losses), rigid involute flanks, steady rotation,
        one normal force through the pitch point at mid face width.

        Warnings
        --------
        These are the D-0010 definitions, NOT the numbers of ``forces()``, which the shaft loads
        of today still use: ``F_a_N`` is 449.236 N where ``forces()`` gives ``Fa`` 444.750 N
        (Q07a), ``F_bn_N`` 1847.109 N where ``Fn`` is 1791.646 N = ``F_bt_N`` (Q07b), and for an
        external pair built with z1 > z2 the torque is taken on the working diameter of gear 1
        itself (Q03). For a helical internal pair the base cylinder of gear 1 is used; the ring
        has another one today (Q21), so ``rho_mm`` and the stored working radii disagree and the
        torque balance of the ring is off: case 5, 60.75 N m on the pinion, rho_2 = -61.932 mm
        against the stored r_w2 = 62.117 mm, moment of the mesh force on the ring -183.344 N m
        against T z2/z1 = -182.25 N m (0.6 %). A spur internal pair (case 4) is exact.

        References
        ----------
        .. [1] ISO_6336-1_2019, 4.2.2, Formula (1), p. 19 (F_t nominal; checked).
        .. [2] ISO_21771_2007, 5.2.4, Eq. (56) third form (d_w = d_b / cos(alpha_wt); checked).
        .. [3] D-0010 items 5, 14-18; ``units_and_conventions.md``, C.4, C.5: no library source
               for F_tw, F_r, F_a, F_bt, F_bn and the directions (statics and helicoid geometry).
        """
        if rot1 not in (1, -1):
            raise ValueError(f"rot1 must be +1 or -1, got {rot1!r}")
        if hand1 not in (1, -1):
            raise ValueError(f"hand1 must be +1 (right-hand) or -1 (left-hand), got {hand1!r}")
        g1, z2 = pair.gear1, pair.gear2.z
        # QUESTION(Q03): working pitch diameter of gear 1 from ITS base circle (ISO_21771_2007 Eq. (56), third form), not the stored pair.dl1, which is the smaller of the two diameters whatever the order of an external pair (case 3b: 109.800 against 73.200 mm).
        d_w1 = g1.db / np.cos(pair.alphatw)
        T = abs(T1_Nm)
        # 2000 = 2 (diameter -> radius) x 1000 (mm -> m): T [N m] / r [m] gives [N]
        F_tw = 2000.0 * T / d_w1
        F_r = F_tw * np.tan(pair.alphatw)
        F_bt = 2000.0 * T / g1.db
        # QUESTION(Q21): base helix angle of gear 1; a helical ring has another beta_b today (case 5: 14.08 against 15.99 deg).
        F_a = F_bt * np.tan(g1.betab)          # = F_tw tan(beta_w): one normal force, three cylinders
        F_bn = F_bt / np.cos(g1.betab)
        F_t_nominal = 2000.0 * T / g1.d        # reference cylinder: ISO 6336-1 Formula (1)

        # pitch point C on the line of centres: between the axes for an external pair (rho_1 > 0),
        # beyond the pinion on the side away from the ring centre for an internal pair (rho_1 < 0)
        rho_1 = np.sign(z2) * d_w1 / 2.0
        rho_2 = rho_1 - pair.al
        F_1t = -1000.0 * T1_Nm / rho_1         # moment of the mesh force about the axis balances T_1
        F_1c = -np.sign(g1.z * z2) * F_r       # compressive contact: external gear pushed toward its axis
        # QUESTION(Q08): the axial sign needs the hand of the helix, which the gear data do not hold; hand1 = +1 (right-hand gear 1) unless given.
        F_1x = -hand1 * np.sign(rho_1) * np.sign(F_1t) * F_a
        on_1 = (float(F_1c), float(F_1t), float(F_1x))
        return MeshForces(F_tw_N=float(F_tw), F_r_N=float(F_r), F_a_N=float(F_a), F_bt_N=float(F_bt),
                          F_bn_N=float(F_bn), F_t_nominal_N=float(F_t_nominal), on_1=on_1,
                          on_2=(-on_1[0], -on_1[1], -on_1[2]),    # action = reaction at C
                          rho_mm=(float(rho_1), float(rho_2)))

    # ----------------------------------------------------------------------------------------------
    # Contact inputs along the path of contact (ISO 21771, 5.4.3-5.4.8 and 5.6; the load has no library source)
    # ----------------------------------------------------------------------------------------------
    # QUESTION(Q31): surface_speeds and contact_inputs take a speed or a force, which D-0026 item 7 puts in solvers/; all five are kept here as pure functions of the pair that store nothing. The plan (section 2.4) puts path_points, curvature_radii, surface_speeds here; load_sharing and contact_inputs come from card G5 items 1 and 3, and the plan names the slot solvers/machine_elements/gears/contact.py for load sharing.
    # QUESTION(Q37): with them this file passes the 1200 lines at which D-0026 keeps "one file per clause group" as the fallback; it is not split (alternative: load_sharing and contact_inputs as module functions of the solvers slot of Q31).
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
        single tooth pair in contact. For eps_alpha >= 2 (baseline case 4) B and D are still E
        minus and A plus one base pitch, and D comes before B.

        Assumptions: rigid, unmodified involute flanks; the tip circle is the end of the active
        profile (Q04: with the clearance cP in the tip diameter).

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
        values returned there are not radii of a flank (baseline case 4 starts 2.39 mm beyond T1,
        Q15: rho1 < 0 and a negative reduced radius, returned without a warning by this function).

        Internal pair: with the negative radius of the ring the reduced radius at C is NOT the
        stored ``ReqI``, which takes the ring radius as positive (Q22): case 4 gives 10.261 mm here
        against ``ReqI`` = 5.130 mm = 1 / (1/rho1 + 1/abs(rho2)). For an external pair the two agree.
        Helical internal pair (Q21): rho_y2 at C differs from the stored ``T2C``, because the base
        circles of the two gears come from different transverse pressure angles.

        Normal plane: an involute helicoid is a developable surface, with zero curvature along
        the contact line. The transverse profile is an oblique section of it, inclined at beta_b
        to the flank normal, so its radius rho_t is the radius of curvature of that section and
        not a principal radius; by Meusnier's theorem the principal radius, in the plane normal
        to the contact line, is rho_n = rho_t / cos(beta_b). The base helix angle is that of
        gear 1 (Q21: the ring of a helical internal pair has another one today).

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
        # QUESTION(Q21): I1. For a helical internal pair T1T2 - T1C is not the stored T2C (case 5: -20.3998 against -20.4606 mm).
        rho2 = pair.T1T2 - rho1
        # QUESTION(Q22): I2 / row 45. Reduced radius with the SIGNED ring radius (ISO_21771_2007, Eq. (82) NOTE); the stored ReqI takes it positive (case 4 at C: 10.261 here, ReqI 5.130 mm).
        rho_red = 1.0 / (1.0 / rho1 + 1.0 / rho2)
        # QUESTION(Q34): normal-plane radii as rho_t / cos(beta_b): derived (developable helicoid, Meusnier), no library page; the alternative is to return the transverse radii only.
        # QUESTION(Q21): I1. beta_b of gear 1; the ring of a helical internal pair has another one today (case 5: 14.08 against 15.99 deg).
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
            ue_m_per_s : entrainment speed (u1 + u2) / 2 [m/s].

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

        Helical internal pair (Q21): the base circles of the two gears follow different rules
        today, so the sliding speed at C is not zero (case 5 at 1450 rpm: -0.0062 m/s).

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
        # QUESTION(Q03): u = z2 / z1, signed (ISO_21771_2007, Eq. (52)); the stored pair.u of an external pair is max(z)/min(z).
        u = pair.gear2.z / pair.gear1.z
        # QUESTION(Q38): returned: omega rho_y, the flank speed perpendicular to the contact line (case 2 at C: 2.0922 m/s); not returned: the component along the line, omega1 rb1 sin(beta_b) (1.2522 m/s, equal on both flanks); GEARpie's vr1, vr2 are omega rho_y / cos(beta_b) (2.1570 m/s), not a velocity component.
        u1 = omega1 * rho["rho1_t_mm"] / 1000.0        # [rad/s] [mm] -> [m/s]
        u2 = omega1 / u * rho["rho2_t_mm"] / 1000.0    # omega2 = omega1 / u; ring: u < 0 and rho_y2 < 0, u2 > 0
        # QUESTION(Q35): entrainment speed taken as the mean (u1 + u2) / 2 and the sliding speed signed as u1 - u2 (Eq. (106) prints plus-minus); no numbered equation for the first.
        return {"u1_m_per_s": u1, "u2_m_per_s": u2, "vg_m_per_s": u1 - u2, "ue_m_per_s": (u1 + u2) / 2.0}

    @staticmethod
    def load_sharing(pair, g_y) -> dict[str, np.ndarray]:
        """Share of the normal tooth force carried by one tooth pair at positions g_y of the path.

        Several tooth pairs, one transverse base pitch apart along the line of action, are in
        contact at the same time. The model: the teeth are rigid and equal, so the normal force
        is spread with ONE load per unit length over all the contact lines present. A tooth pair
        then carries the fraction (length of its contact line) / (sum of the lengths). For a
        spur pair this is 1 in the zone of single contact and 1/2 in the two zones of double
        contact (1/2 and 1/3 for 2 <= eps_alpha < 3, baseline case 4).

        Parameters
        ----------
        pair : GearPair
            Pair of this family.
        g_y : float or array_like
            Position [mm] of the contact point of the tooth pair in the transverse section at
            MID FACE WIDTH, from C, positive toward the tip of gear 1.

        Returns
        -------
        dict
            share : fraction of the normal force on this tooth pair [-]; zero where it has no
            contact.
            l_mm : length of the contact line of this tooth pair [mm].
            l_sum_mm : sum of the lengths of all contact lines present at that instant [mm].
            n_pairs : number of tooth pairs in contact at that instant [-].

        Notes
        -----
        Contact lines: in the plane of action (length g_alpha along the path, width b) the
        contact lines are straight, inclined at beta_b to the gear axes. With s the distance from
        the stored point A of the mid-face point of a line, the line covers the path interval
        [s - b tan(beta_b)/2, s + b tan(beta_b)/2]; the part of it inside [0, g_alpha], divided by
        sin(beta_b), is its length. A spur pair has l = b for 0 <= s <= g_alpha. The largest value
        is the l_max of ISO_21771_2007, Eq. (98)-(99). The neighbouring tooth pairs are at
        s + k p_bt. With the mid-face section as the reference the result depends neither on the
        driver nor on the hand of helix (there is none: Q08).

        A helical tooth pair is in contact before its mid-face point reaches A and after it leaves
        E (by b tan(beta_b)/2): ``share`` is not zero just outside the path. The sum of ``share``
        over the tooth pairs k is 1 at every position.

        Assumptions: teeth of equal, constant stiffness per unit length (rigid limit), no
        modifications, no misalignment, no friction; common face width b = min(b1, b2) of aligned gears; transverse base pitch of
        gear 1 (Q21).

        References
        ----------
        .. [1] ISO_21771_2007, 5.4.8, p. 39-40, Eq. (98)-(99): contact lines at beta_b, their
           largest length; the sum of the lengths is defined in words only (checked).
        .. [2] Uniform load per unit length and the lengths along the path: derived, no library
           source. GEARpie uses the same model (CLASSES/LOAD_SHARING.py:62-90,
           CLASSES/FORCES_SPEEDS.py:72), with the section at the face end as its first column.
        """
        # QUESTION(Q30): load-sharing model = one load per unit length on all contact lines (rigid, equal teeth); alternatives: mesh stiffness (ISO_6336-1_2019, clause 9, not written) or a measured/FE share.
        g_y = np.asarray(g_y, dtype=float)
        # QUESTION(Q21): I1. Base pitch and base helix angle of gear 1, as the stored path points; the ring of a helical internal pair has others today (case 5: 6.087 against 6.142 mm, 14.08 against 15.99 deg).
        pbt = pair.gear1.pbt
        betab = pair.gear1.betab
        s = g_y + pair.AC                          # distance from the stored A (C is at AC from A)
        # QUESTION(Q36): helical pair, the contact-line length is that of the line whose MID-FACE point is at g_y; the alternative reference is the face end where the tooth enters (GEARpie column 0), which needs the driver.
        half = 0.5 * pair.b * np.tan(betab)        # half of the path interval covered by one contact line [mm]
        tol = 1e-9 * pbt                           # the ends A and E count as in contact: sums of stored segments round at 1e-15 relative

        def line_length(s):
            if betab == 0.0:
                return np.where((s >= -tol) & (s <= pair.AE + tol), pair.b, 0.0)   # spur: the whole face width
            inside = np.clip(np.minimum(s + half, pair.AE) - np.maximum(s - half, 0.0), 0.0, None)
            return inside / np.sin(betab)

        # every tooth pair that can touch while this one is within reach of the path
        n_max = int(np.ceil((pair.AE + 2.0 * half) / pbt))
        lengths = np.array([line_length(s + k * pbt) for k in range(-n_max, n_max + 1)])
        l_own, l_sum = lengths[n_max], lengths.sum(axis=0)
        share = np.divide(l_own, l_sum, out=np.zeros_like(l_sum), where=l_sum > 0.0)
        return {"share": share, "l_mm": l_own, "l_sum_mm": l_sum, "n_pairs": (lengths > 0.0).sum(axis=0)}

    @staticmethod
    def contact_inputs(pair, driver: str, n1_rpm: float, F_bn_N: float, *, n_points: int | None = None,
                       t_s=None) -> dict[str, np.ndarray]:
        """Everything a contact calculation needs at each step of a contact point along the path.

        One record per step, from the start of meshing A to its end E for the given driver, of
        the contact point in ONE transverse section: where it is and when, the two equivalent
        cylinders, how fast the surfaces move, and the load per unit length of contact line.
        These are the inputs of a line contact (Hertz, EHL) that changes along the path. For a
        spur pair this is the whole life of a tooth pair; for a helical pair it is the section at
        mid face width only (Notes).

        Parameters
        ----------
        pair : GearPair
            Pair of this family.
        driver : str
            "gear1" or "gear2": the driving gear (fixes A, E and the direction of travel).
        n1_rpm : float
            Rotational speed of GEAR 1 [rpm], > 0 (a magnitude, as in ``surface_speeds``),
            whichever gear drives.
        F_bn_N : float
            Total force normal to the flanks transmitted by the mesh [N], >= 0 (F_bn = F_bt /
            cos(beta_b)). An argument: no force is derived from a torque here (Q07b: the ``Fn`` of
            ``forces`` is F_bt).
        n_points : int, optional
            Number of equally spaced steps from A to E, both included (>= 2).
        t_s : array_like, optional
            Times of the steps [s], t = 0 at A; finite, at least one, all within the time of
            travel from A to E.
            Exactly one of ``n_points`` and ``t_s`` is given.

        Returns
        -------
        dict
            Arrays of equal length, one entry per step:

            s_mm : distance travelled from A along the path [mm].
            t_s : time since A [s].
            g_y_mm : position from the pitch point C, positive toward the tip of gear 1 [mm].
            rho1_t_mm, rho2_t_mm, rho_red_t_mm, rho1_n_mm, rho2_n_mm, rho_red_n_mm : radii of
            curvature, signed (``curvature_radii``) [mm].
            u1_m_per_s, u2_m_per_s, vg_m_per_s, ue_m_per_s : surface, sliding and entrainment speeds
            (``surface_speeds``) [m/s].
            share, l_mm, l_sum_mm, n_pairs : load share and contact-line lengths
            (``load_sharing``) [-], [mm], [mm], [-].
            w_N_per_mm : load per unit length of contact line, share F_bn / l [N/mm].

        Raises
        ------
        ValueError
            If ``driver`` is not "gear1" or "gear2", if n1_rpm <= 0, if F_bn_N < 0, if not
            exactly one of ``n_points`` and ``t_s`` is given, if n_points < 2, or if ``t_s`` is
            empty, not finite or outside the travel from A to E.

        Warns
        -----
        UserWarning
            If the radius of curvature of an external flank is zero or negative on a returned
            step: the path starts beyond a base-tangent point, where there is no involute
            (baseline cases 4 and 5, Q15 / Q27). The numbers are returned unchanged (Q39).

        Notes
        -----
        Time and position: the contact point moves along the line of action at the normal speed
        of the flanks, v_n = omega1 r_b1 (ISO_21771_2007, Eq. (105)), so s = omega1 r_b1 t. The
        standard gives v_n; the relation s(t) is derived from it (no library sentence).

        Helical pair: a step is the contact point in the transverse section at MID face width.
        A tooth pair is in contact over g_alpha + b tan(beta_b) of travel (case 2: 24.78 mm); the
        record covers g_alpha (19.01 mm, 77 %). ``share`` and ``l_mm`` describe the whole contact
        line, the radii and speeds only its mid point, and the load per unit length is uniform by
        the model. Along the line the path position spans +-b tan(beta_b)/2: in case 2 at C,
        +-2.88 mm, rho1 from 10.90 to 16.66 mm (+-21 %) and a sliding speed of +-0.73 m/s at the
        line ends against 0 recorded; at A and E the recorded point is the END of a half-length
        line. The record is fair for a two-dimensional slice at mid face, not for the line (Q36).

        Helical internal pair (Q21): the sliding speed recorded at C is not zero (case 5:
        -0.0062 m/s at 1450 rpm) and the normal speeds of the two flanks differ by 0.9 %.

        Assumptions: those of the four functions above; steady speed and constant F_bn.

        References
        ----------
        .. [1] ISO_21771_2007, 5.6.1, Eq. (105), p. 42 (checked).
        .. [2] Load per unit length: no library source.
        """
        if (n_points is None) == (t_s is None):
            raise ValueError("give exactly one of n_points and t_s.")
        if n1_rpm <= 0:
            raise ValueError(f"n1_rpm must be > 0, got {n1_rpm}.")
        if F_bn_N < 0:
            raise ValueError(f"F_bn_N must be >= 0, got {F_bn_N}.")
        points = CylindricalFamily.path_points(pair, driver)
        # QUESTION(Q21): I1. Base radius of gear 1; in a helical internal pair omega2 rb2 is not omega1 rb1 today (case 5: ratio 1.0090).
        v_n = n1_rpm * 2.0 * np.pi / 60.0 * pair.gear1.rb   # [rad/s] [mm] = [mm/s], speed of the contact point along the path
        if t_s is None:
            if n_points < 2:
                raise ValueError(f"n_points must be >= 2, got {n_points}.")
            s = np.linspace(0.0, pair.AE, n_points)
        else:
            s = v_n * np.atleast_1d(np.asarray(t_s, dtype=float))
            if s.size == 0 or not np.all(np.isfinite(s)):
                raise ValueError("t_s must hold at least one time, all finite.")
            if s.min() < 0.0 or s.max() > pair.AE * (1.0 + 1e-12):   # 1e-12: rounding of t = AE / v_n given back as a time
                raise ValueError(f"t_s must lie in [0, {pair.AE / v_n:.6g}] s, the travel from A to E.")
        # from A toward E: increasing g_y when gear 1 drives (root to tip of the driving flank), decreasing when gear 2 drives
        g_y = points["A"] + np.sign(points["E"] - points["A"]) * s

        radii = CylindricalFamily.curvature_radii(pair, g_y)
        # QUESTION(Q39): steps where an external flank has no involute (rho <= 0: the path starts beyond a base-tangent point, Q15 / Q27) are returned with a warning; the alternatives are to refuse the pair or to cut the path at the base-tangent point.
        if np.any(radii["rho1_t_mm"] <= 0.0) or (pair.gear2.z > 0 and np.any(radii["rho2_t_mm"] <= 0.0)):
            warnings.warn("contact_inputs: the path of contact passes a base-tangent point; the radii of curvature, "
                          "speeds and reduced radius of the steps beyond it do not describe an involute flank.",
                          stacklevel=2)
        sharing = CylindricalFamily.load_sharing(pair, g_y)
        # QUESTION(Q32): the Hertz half-width b is not computed here; it belongs to slippy_bridge (D-0019), from w_N_per_mm, rho_red_n_mm and E'.
        # QUESTION(Q33): the flanks are given as local equivalent cylinders (the radii), not as involute surfaces.
        return {"s_mm": s, "t_s": s / v_n, "g_y_mm": g_y,
                **radii,
                **CylindricalFamily.surface_speeds(pair, g_y, n1_rpm),
                **sharing,
                # share F_bn / l with share = l / l_sum: written without l, which is zero at the ends of a helical line
                "w_N_per_mm": F_bn_N / sharing["l_sum_mm"]}
