"""
core/mechanical_elements/Gears/Parallel_Axis_gears/internal_gear_meshing.py

Internal gear pair: working geometry, interference checks, and meshing forces.

Mates an external gear (SpurGear or HelicalGear — both are expected to share
the same attribute interface: mn, z, x, b, alpha, alphat, beta, betab, db,
rb, d, ra, haP, Ra, Rq, Rz, alpha_n_deg, beta_n_deg, pbt) with an
InternalGear (ring gear, z > 0, conventional KHK definition — see
internal_gear.py's module docstring).

References:
  - ISO 21771:2007  — cylindrical involute gear geometry
  - KHK Gear Technical Reference §4.2 — Internal gear pair calculations
  - Gear Solutions: "Internal ring gears – design and considerations"
  - MAAG Gear Book — working geometry, contact ratio

Sign / convention notes
------------------------
Unlike the external (helical-helical) pair, where most quantities use the
SUM z1+z2 / x1+x2, an internal pair uses the DIFFERENCE z2-z1 / x2-x1
throughout (z2 = internal/ring teeth, always taken here as a positive
magnitude, z2 > z1):

    a   = (d2 - d1) / 2                                    (reference)
    inv(alphawt) = inv(alphat) + 2*tan(alpha)*(x2-x1)/(z2-z1)
    al  = a * cos(alphat) / cos(alphawt)                   (working)
    dl1 = 2*al / (u - 1)        dl2 = 2*al*u / (u - 1)      where u = z2/z1

This module does NOT negate x2 or z2 internally — both external and
internal gears are expected to be built with positive z, exactly as
InternalGear's "z>0 conventional" case in internal_gear.py.
"""

from __future__ import annotations
import numpy as np
from scipy import optimize
import copy

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.internal_gear import InternalGear
from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import SpurHelicalGear


class InternalGearMeshing:
    """
    Defines an internal gear pair (external gear1 + internal/ring gear2)
    and computes working centre distance, transverse working pressure
    angle, contact ratio, interference checks and meshing forces.

    Parameters
    ----------
    gear1 : SpurGear | HelicalGear   — external gear (pinion), driving
    gear2 : InternalGear              — internal (ring) gear, driven
    label : str
    al    : float | None
        Working centre distance [mm].
        If None → computed from reference geometry and profile shifts.
        If provided → gear1_w / gear2_w are updated accordingly, and
        consistency with declared x1, x2 is validated.
    addendum_reduction : bool
        Adapted from the MAAG/KHK external-pair addendum-reduction
        coefficient k (see gear_geometry docstring) — applies only to the
        external gear's tip circle. Off by default; verify against
        KHK §4.2 before relying on it for tight-tolerance designs.
    """

    def __init__(self,
             gear1: SpurHelicalGear,
             gear2: InternalGear,
             label: str = "",
             al: float | None = None,
             equalise_gs: bool = False,
             addendum_reduction: bool = False,
             driver: str = "gear1"):

        self._check_compatibility(gear1, gear2)

        if driver not in ("gear1", "gear2"):
            raise ValueError(
                f"driver must be 'gear1' or 'gear2', got '{driver}'. "
                "This selects which gear delivers input torque — it is independent "
                "of gear1/gear2 being external/internal."
            )
        self.driver = driver
        self.driven = "gear2" if driver == "gear1" else "gear1"

        self.label = label
        self.gear1 = gear1
        self.gear2 = gear2

        z1, z2 = gear1.z, gear2.z
        if abs(z2) <= z1:
            raise ValueError(
                f"Internal gear must have more teeth than the external gear "
                f"(z2={z2} <= z1={z1})."
            )

        # --- reference centre distance: difference, not sum ---
        self.a = (gear2.d - gear1.d) / 2.0

        self._al_overridden = al is not None

        if al is not None:
            self.al = al
            self._set_alphatw_from_al()
            self.x1, self.x2 = gear1.x, gear2.x

        elif equalise_gs:
            # 2 — Henriot: compute x1, x2 first, then derive al from them
            self.correction_for_gs_equilibrium()
            self._set_al_from_x_diff(self.x1 + self.x2)

        else:
            # 3 — derive al from x1+x2 as declared on the gears
            self._set_al_from_x_diff(gear1.x + gear2.x)
            self.x1, self.x2 = gear1.x, gear2.x

        self.inv_alphatw = np.tan(self.alphatw) - self.alphatw

        # --- gear ratio ---
        self.u = z2 / z1

        # --- working geometry ---
        self.gear_geometry(addendum_reduction=addendum_reduction)

        # --- profile shift difference (for consistency check in validate) ---
        z_diff = z2 - z1
        inv_alphat = np.tan(gear1.alphat) - gear1.alphat
        self.x_diff_from_al = (
            (self.inv_alphatw - inv_alphat) * z_diff / (2 * np.tan(gear1.alpha))
        )
        self.x_diff_declared = gear2.x - gear1.x

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    def _set_al_from_x_diff(self, x_diff: float) -> None:
        """
        Derive al and alphatw from a profile shift difference (x2 - x1).
        Sets self.al, self.alphatw, self.inv_alphatw.
        """
        alphat  = self.gear2.alphat
        beta    = self.gear2.beta
        z_sum = self.gear2.z + self.gear1.z
        alpha   = self.gear1.alpha

        self.inv_alphatw = (
            np.tan(alphat) - alphat
            + 2 * np.tan(alpha) * x_diff / z_sum
        )

        def funOPT(xx, inv_a):
            return inv_a - np.tan(xx) + xx

        self.alphatw = optimize.brentq(funOPT, 1e-6, 0.7, args=(self.inv_alphatw,))
        self.al = self.gear2.mn * np.cos(alphat) * abs(z_sum) / (2 * np.cos(beta) * np.cos(self.alphatw))

    def _set_alphatw_from_al(self) -> None:
        """Back-calculate alphatw from an imposed al."""
        z_sum = self.gear2.z + self.gear1.z
        self.alphatw = np.arccos(abs(z_sum) * self.gear2.mn * np.cos(self.gear2.alphat) / (
            2 * self.al * np.cos(self.gear2.beta)))
        inv_alphat = np.tan(self.gear2.alphat) - self.gear2.alphat
        inv_alphatw = np.tan(self.alphatw) - self.alphatw
        sum_x = (self.gear1.z + self.gear2.z) * (inv_alphatw - inv_alphat) / (2 * np.tan(self.gear1.alpha))
        self.jbn = (self.gear1.x + self.gear2.x - sum_x) * (2 * self.gear1.mn * np.sin(self.gear1.alpha)) # backlash
        

    # ------------------------------------------------------------------
    # Equalise specific sliding speeds (Henriot)
    # ------------------------------------------------------------------

    def correction_for_gs_equilibrium(self) -> tuple[float, float]:
        """
        Compute the profile shift coefficients x1, x2 that equalise the
        maximum specific sliding speeds at both gears (g_s1B' = g_s2A').

        Same Henriot procedure as for spur gears, but using the transverse
        pressure angle αt (instead of α) and the transverse reference
        centre distance a = (d1+d2)/2. Profile-shift radial increments
        still use the normal module mn (addendum is defined in the
        normal plane).

        Returns
        -------
        tuple (x1, x2)
        """
        beta     = self.gear1.beta 
        z1, z2   = self.gear1.zn, self.gear2.zn
        ra1, ra2 = self.gear1.mt * z1 / 2.0 + self.gear1.mn * (self.gear1.haP + self.gear1.cP + self.gear1.x), self.gear2.mt * z2 / 2.0 + self.gear2.mn * (self.gear2.haP + self.gear2.cP + self.gear2.x)
        rb1, rb2 = self.gear1.mt * z1 * np.cos(self.gear1.alphat), self.gear2.mt * z2 * np.cos(self.gear2.alphat)
        mn       = self.gear1.mn
        alphat   = self.gear1.alphat
        a        = self.a

        # --- helpers ---
        def _gs1B(ra2_v, rb2_v, a_v, alphat_v, z2):
            sq  = z2 / abs(z2) * np.sqrt(max(ra2_v**2 - rb2_v**2, 0.0))
            den = a_v * np.sin(alphat_v) - sq
            if abs(den) < 1e-12:
                return np.inf
            return abs(1.0 - (z1 / z2) * sq / den)

        def _gs2A(ra1_v, rb1_v, a_v, alphat_v, z2):
            sq  = np.sqrt(max(ra1_v**2 - rb1_v**2, 0.0))
            den = z2 * abs(z2) * a_v * np.sin(alphat_v) - sq
            if abs(den) < 1e-12:
                return np.inf
            return abs((z2 / z1) * sq / den - 1.0)

        def _solve_direct(z1_v, z2_v, ra1_v, ra2_v, rb1_v, rb2_v, a_v, alphat_v):
            """Solve gs1B' = gs2A' with x2=-x1, a'=a."""
            def equation(x):
                return (_gs1B(ra2_v - x * mn, rb2_v, a_v, alphat_v, z2_v)
                      - _gs2A(ra1_v + x * mn, rb1_v, a_v, alphat_v, z2_v))

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

        # --- Case 1: z1+z2 > 60 ---
        if (z1 + z2) > 60:
            x = _solve_direct(z1, z2, ra1, ra2, rb1, rb2, a, alphat)
            self.x1 = x
            self.x2 = -x
            return x, -x

        # --- Case 2: z1+z2 ≤ 60 (Henriot hypothetical gear method) ---
        z2_h  = 60 - z1
        a_h   = self.gear2.mt / 2.0 * (z1 + z2_h)
        ra2_h = self.gear2.mt / 2.0 * z2_h + mn
        rb2_h = self.gear2.mt / 2.0 * z2_h * np.cos(alphat)

        x1 = _solve_direct(z1, z2_h, ra1, ra2_h, rb1, rb2_h, a_h, alphat)

        inv_alphat = np.tan(alphat) - alphat
        alpha = self.gear1.alpha

        def _alphatw_from_x2(x2_v):
            inv_aw = inv_alphat + 2.0 * np.tan(alpha) * (x1 + x2_v) / (z1 + z2)
            def funOPT(xx, inv_a):
                return inv_a - np.tan(xx) + xx
            return optimize.brentq(funOPT, 1e-6, 0.7, args=(inv_aw,))

        def _al_from_alphatw(alphatw_v):
            return self.gear2.mn * np.cos(self.gear2.alphat) * abs(self.gear1.z + self.gear2.z) / (
                2 * np.cos(beta) * np.cos(alphatw_v))

        def equation_x2(x2_v):
            alphatw_v = _alphatw_from_x2(x2_v)
            al_v      = _al_from_alphatw(alphatw_v)
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

        self.x1 = x1
        self.x2 = x2
        return x1, x2


    # ------------------------------------------------------------------
    # Working geometry
    # ------------------------------------------------------------------

    def gear_geometry(self, addendum_reduction: bool = False) -> None:
        """
        Builds gear1_w / gear2_w: independent copies of the reference
        gears with working geometry applied, and computes the contact
        ratios (εα, εβ, εγ).

        """
        self.gear1_w = copy.copy(self.gear1)
        self.gear2_w = copy.copy(self.gear2)

        z1, z2 = self.gear1.z, self.gear2.z
        alpha  = self.gear1.alpha
        alphat = self.gear1.alphat
        beta   = self.gear1.beta
        betab  = self.gear1.betab
        m      = self.gear1.mn

        def involute(ang):
            return np.tan(ang) - ang

        # --- addendum reduction coefficient k (external gear only) ---
        if addendum_reduction:
            self.k = (z2 - z1) / 2.0 * (
                (involute(self.alphatw) - involute(alphat)) / np.tan(alpha)
                - 1.0 / np.cos(beta) * (np.cos(alphat) / np.cos(self.alphatw) - 1.0)
            )
        else:
            self.k = 0.0

        self.gear1_w.k = self.k
        self.gear2_w.k = self.k

        # --- working centre distance / transverse working pressure angle ---
        self.gear1_w.al = self.gear2_w.al = self.al
        self.gear1_w.alphatw = self.gear2_w.alphatw = self.alphatw

        # --- working pitch diameters (difference form) ---
        self.gear1_w.dl = self.gear1_w.db / np.cos(self.gear1_w.alphatw)
        self.gear2_w.dl = self.gear2_w.db / np.cos(self.gear2_w.alphatw)
        self.gear1_w.rl, self.gear2_w.rl = self.gear1_w.dl / 2, self.gear2_w.dl / 2
        rl1, rl2 = self.gear1_w.rl, self.gear2_w.rl

        # --- tip diameters ---
        d1, d2   = self.gear1.d, self.gear2.d
        haP1, haP2 = self.gear1.haP, self.gear2.haP
        cP1, cP2 = self.gear1.cP, self.gear2.cP

        self.gear1_w.da = d1 + 2 * m * (haP1 + cP1 + self.x1 - self.k)
        self.gear1_w.ra = self.gear1_w.da / 2.0

        # ring gear tip circle: unchanged, taken straight from reference
        self.gear2_w.da = d2 + z2/abs(z2) * 2 * m * (haP2 + cP2 + self.x2)
        self.gear2_w.ra = self.gear2_w.da / 2.0

        db1, db2 = self.gear1_w.db, self.gear2_w.db
        rb1, rb2 = db1/2, db2/2
        da1, da2 = self.gear1_w.da, self.gear2_w.da
        ra1, ra2 = da1/2, da2/2

        self.gear1_w.alpha_a = np.arccos(db1 / da1)
        self.gear2_w.alpha_a = np.arccos(db2 / da2)

        alpha_a1 = self.gear1_w.alpha_a
        alpha_a2 = self.gear2_w.alpha_a

        # transverse contact ratio (KHK §4.2 — note the MINUS, not plus,
        # which is the internal-pair analogue of the external z1+z2 sum)
        self.gear1_w.epslon_a1 = z1 * (np.tan(alpha_a1) - np.tan(self.alphatw)) / (2 * np.pi)
        self.gear2_w.epslon_a2 = abs(z2) * (np.tan(alpha_a2) - np.tan(self.alphatw)) / (2 * np.pi)

        self.epslon_alpha = self.gear1_w.epslon_a1 - self.gear2_w.epslon_a2

        # overlap (face) contact ratio — non-zero only if beta != 0
        pbt = self.gear1_w.pbt
        self.b = min(self.gear1.b, self.gear2.b)
        self.epslon_beta = self.b * np.tan(betab) / pbt

        self.epslon_gamma = self.epslon_alpha + self.epslon_beta


        self.gear1_w.galpha = rb1 * (np.tan(alpha_a1) - np.tan(self.alphatw))
        self.gear2_w.galpha = rb2 * (np.tan(alpha_a2) - np.tan(self.alphatw))

        self.galpha = 1/2 * ((da1**2 - db1**2)**(1/2) + z2/abs(z2) * (da2**2 - db2**2)**(1/2) - 2 * self.al * np.sin(self.alphatw))



        # equivalent curvature radius on pitch point
        self.ReqI = 1 / (1 / (rl1 * np.sin(self.alphatw))
                        + 1 / (rl2 * np.sin(self.alphatw)))

        self.T1T2 = z2/abs(z2) * self.al * np.sin(self.alphatw)
        # positions along line of action length (pinion)
        self.T1E = (ra1**2 - rb1**2) ** (1 / 2)
        # positions along line of action length (wheel)
        self.T2A = z2/abs(z2) * ((ra2**2 - rb2**2) ** (1 / 2))
        # positions of pinion and wheel along AE
        self.T1A = self.T1T2 - self.T2A
        self.T1B = self.T1E - pbt
        self.T1C = rb1 * np.tan(self.alphatw)
        self.T1D = self.T1A + pbt
        self.T2E = self.T1T2 - self.T1E
        self.T2B = self.T2E + pbt
        self.T2C = z2/abs(z2) * rb2 * np.tan(self.alphatw)
        self.T2D = self.T2A - pbt
        # positions along path of contact
        self.AE = self.T1E - self.T1A
        self.AB = self.T1B - self.T1A
        self.AC = self.T1C - self.T1A
        self.AD = self.T1D - self.T1A
        # radius along the path of contact
        self.rA1 = (self.T1A**2 + rb1**2) ** (1 / 2)
        self.rB1 = (self.T1B**2 + rb1**2) ** (1 / 2)
        self.rD1 = (self.T1D**2 + rb1**2) ** (1 / 2)
        self.rA2 = ((self.T2A - self.AE)**2 + rb2**2) ** (1 / 2)
        self.rB2 = ((self.T2A - self.AD)**2 + rb2**2) ** (1 / 2)
        self.rD2 = ((self.T2A - self.AB)**2 + rb2**2) ** (1 / 2)

        # gear loss factor according to Ohlendorf
        self.HV = (np.pi * (self.u + 1) / (z1 * self.u * np.cos(betab)) *
                   (1 - self.epslon_alpha + self.gear1_w.epslon_a1**2 +
                    self.gear2_w.epslon_a2**2))

        # gear finishing
        self.Ram = (self.gear1_w.Ra + self.gear2_w.Ra) / 2
        self.Rrms = (self.gear1_w.Rq**2 + self.gear2_w.Rq**2) ** (1 / 2)
        self.RzS = (self.gear1.Rz + self.gear2.Rz)

    # ------------------------------------------------------------------
    # Forces (ISO 6336-1) — same definitions as the external pair, using
    # the working pitch radius of the EXTERNAL gear (gear1) as reference
    # ------------------------------------------------------------------

    def forces(self, T_in: float, phi_deg: float = 0.0,
            rotation_dir_in: int = 1) -> dict[str, float]:
        """
        ...same docstring pattern as external pair...

        Note: internal meshing does NOT reverse rotation sense
        (ring and pinion rotate in the same direction) — rotation_dir_out
        equals rotation_dir_in, unlike external meshing.
        """
        if self.driver == "gear1":
            rl_in, beta_in, u_eff = self.gear1_w.rl, self.gear1.beta, self.u
        else:
            rl_in, beta_in, u_eff = self.gear2_w.rl, self.gear2.beta, 1.0 / self.u

        Ft = T_in / (rl_in / 1000)
        Fr = Ft * np.tan(self.alphatw)
        Fa = Ft * np.tan(beta_in)
        Fn = Ft / np.cos(self.alphatw)
        T_out = T_in * u_eff

        theta_Fr_driver = phi_deg % 360.0
        theta_Ft_driver = (phi_deg + 90.0 * rotation_dir_in) % 360.0

        rotation_dir_out = - u_eff/abs(u_eff) * rotation_dir_in     # = +rotation_dir_in

        theta_Fr_driven = (phi_deg + 180.0) % 360.0
        theta_Ft_driven = (phi_deg + 180.0 + 90.0 * rotation_dir_out) % 360.0

        return {
            "Ft": Ft, "Fr": Fr, "Fa": Fa, "Fn": Fn, "T_out": T_out,
            "theta_Fr_driver": theta_Fr_driver, "theta_Ft_driver": theta_Ft_driver,
            "theta_Fr_driven": theta_Fr_driven, "theta_Ft_driven": theta_Ft_driven,
            "rotation_dir_out": rotation_dir_out,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Validate the gear pair geometry. Returns a list of error strings;
        an empty list means the pair is geometrically valid.

        Checks:
          1. Individual gear validation
          2. Profile shift consistency with al (only when al is overridden)
          3. Contact ratio εα ≥ 1.0
          4. Involute / trochoid / trimming interference, reusing
             InternalGear.validate_mesh() directly
        """
        errors: list[str] = []

        errors.extend(self.gear1.validate())
        errors.extend(self.gear2.validate())
        errors.extend(self.gear1_w.validate())
        errors.extend(self.gear2_w.validate())

        if self._al_overridden:
            if abs(self.x_diff_from_al - self.x_diff_declared) > 1e-4:
                errors.append(
                    f"Profile shift inconsistency: declared x2-x1={self.x_diff_declared:.4f} "
                    f"but al={self.al:.4f} mm implies x2-x1={self.x_diff_from_al:.4f}. "
                    "Adjust x1, x2, or al so they are consistent."
                )

        if self.epslon_alpha < 1.0:
            errors.append(
                f"Transverse contact ratio εα={self.epslon_alpha:.4f} < 1.0 — "
                "gears will lose contact during meshing."
            )
        elif self.epslon_alpha < 1.2:
            errors.append(
                f"Warning: transverse contact ratio εα={self.epslon_alpha:.4f} is low (< 1.2) — "
                "consider increasing tooth count or face width."
            )

        # interference: delegate to InternalGear's own mesh-pair checks
        errors.extend(self.gear2.validate_mesh(self.gear1.z, self.gear1.x))

        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        tag = self.label or "InternalGearMeshing"
        lines = [
            f"── {tag} ─────────────────────────────────",
            f"  gear1 (ext): z={self.gear1_w.z}, mn={self.gear1_w.mn}, x={self.gear1_w.x}, β={self.gear1.beta_n_deg}°",
            f"  gear2 (int): z={self.gear2_w.z}, mn={self.gear2_w.mn}, x={self.gear2_w.x}, β={self.gear2.beta_n_deg}°",
            f" da2 : {self.gear2_w.da:.4f} mm",
            f" T1C : {self.T1C:.4f} mm   (pinion tip contact point)",
            f"  u     : {self.u:.4f}",
            f"  a     : {self.a:.4f} mm   (reference centre distance)",
            f"  al    : {self.al:.4f} mm   (working centre distance)",
            f"  αtw   : {np.degrees(self.alphatw):.4f}°",
            f"  dl1   : {self.gear1_w.dl:.4f} mm",
            f"  dl2   : {self.gear2_w.dl:.4f} mm",
            f"  εα    : {self.epslon_alpha:.4f}",
            f"  εβ    : {self.epslon_beta:.4f}",
            f"  εγ    : {self.epslon_gamma:.4f}",
            "────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"InternalGearMeshing(z1={self.gear1_w.z}, z2={self.gear2_w.z}, "
            f"mn={self.gear1_w.mn}, β={self.gear1.beta_n_deg}°, al={self.al:.4f} mm, "
            f"αtw={np.degrees(self.alphatw):.4f}°, εγ={self.epslon_gamma:.4f})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_compatibility(g1, g2: InternalGear) -> None:
        """Raise ValueError if the two gears cannot mesh."""
        if not np.isclose(g1.mn, g2.mn, rtol=1e-6):
            raise ValueError(
                f"Module mismatch: gear1 mn={g1.mn}, gear2 mn={g2.mn}. "
                "Both gears must have the same module."
            )
        if not np.isclose(g1.alpha_n_deg, g2.alpha_n_deg, rtol=1e-6):
            raise ValueError(
                f"Pressure angle mismatch: gear1 α={g1.alpha_n_deg}°, "
                f"gear2 α={g2.alpha_n_deg}°. "
                "Both gears must have the same pressure angle."
            )
        if not np.isclose(g1.beta_n_deg, g2.beta_n_deg, rtol=1e-6):
            raise ValueError(
                f"Helix angle mismatch: gear1 β={g1.beta_n_deg}°, "
                f"gear2 β={g2.beta_n_deg}°. "
                "Both gears must have the same helix angle magnitude."
            )