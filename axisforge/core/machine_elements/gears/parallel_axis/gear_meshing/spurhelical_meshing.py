"""
core/mechanical_elements/Gears/Parallel_Axis_gears/helical_gear_meshing.py

Helical gear pair: working geometry, interference checks, and meshing forces.

References:
  - ISO 21771:2007  — cylindrical involute gear geometry
  - KHK Technical Reference §4.1–4.2 — gear pair calculations & interference
  - ISO 6336-1:2019 — load on gear teeth (force definitions)
  - MAAG Gear Book — working geometry, contact ratio, Ohlendorf loss factor
"""

from __future__ import annotations
import numpy as np
from scipy import optimize
import copy

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear


class SpurHelicalGearMeshing:
    """
    Defines a helical gear pair and computes:
      - Working centre distance and transverse working pressure angle
      - Transverse, overlap and total contact ratios (εα, εβ, εγ)
      - Interference checks (involute, undercutting)
      - Meshing forces: Ft, Fr, Fa (tangential, radial, axial)

    Parameters
    ----------
    gear1 : HelicalGear   — driving gear (pinion)
    gear2 : HelicalGear   — driven gear (wheel)
    label : str
    al    : float | None
        Working centre distance [mm].
        If None → computed from reference geometry and profile shifts.
        If provided → gear1_w / gear2_w are updated accordingly,
        and consistency with declared x1+x2 is validated.
    """

    def __init__(self,
                 gear1: SpurHelicalGear,
                 gear2: SpurHelicalGear,
                 label: str = "",
                 al: float | None = None,
                 equalise_gs: bool = False,
                 addendum_reduction: bool = False,
                 driver: str = "gear1"):
        """
        Parameters
        ----------
        gear1, gear2  : HelicalGear
        label         : str
        al            : float | None
            Working centre distance [mm].
            - None + equalise_gs=False → derived from x1, x2 via involute function
            - None + equalise_gs=True  → x1, x2 computed by Henriot method first,
                                         then al derived from those
            - float                    → imposed directly; alphatw back-calculated
        equalise_gs   : bool
            If True, call correction_for_gs_equilibrium() to find x1, x2 before
            computing al. Ignored when al is explicitly provided.
        """
        # --- compatibility check before anything else ---
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

        # --- store original (reference) gears ---
        self.gear1 = gear1
        self.gear2 = gear2

        # --- reference centre distance ---
        # a = (d1 + d2) / 2   [transverse pitch diameters, x=0 baseline]
        self.a = (gear1.d + gear2.d) / 2.0

        # --- resolve al and alphatw ---
        # Three possible sources, in priority order:
        #   1. al explicitly imposed by the user
        #   2. Henriot gs equalisation (equalise_gs=True)
        #   3. Derived from x1+x2 on the gears as defined

        self._al_overridden = al is not None

        if al is not None:
            # 1 — user imposed al directly
            self.al = al
            self._set_alphatw_from_al()
            self.x1, self.x2 = gear1.x, gear2.x

        elif equalise_gs:
            # 2 — Henriot: compute x1, x2 first, then derive al from them
            self.correction_for_gs_equilibrium()
            self._set_al_from_x_sum(self.x1 + self.x2)

        else:
            # 3 — derive al from x1+x2 as declared on the gears
            self._set_al_from_x_sum(gear1.x + gear2.x)
            self.x1, self.x2 = gear1.x, gear2.x

        self.inv_alphatw = np.tan(self.alphatw) - self.alphatw

        # --- gear ratio ---
        self.u = max(self.gear1.z, self.gear2.z) / min(self.gear1.z, self.gear2.z)

        # --- working geometry, calculada uma vez, automaticamente ---
        self.gear_geometry(addendum_reduction=addendum_reduction)

        # --- profile shift sum (for consistency check in validate) ---
        # inv_alphatw = inv_alphat + 2*tan(alpha)*(x1+x2)/(z1+z2)
        z_sum = gear1.z + gear2.z
        inv_alphat = np.tan(gear1.alphat) - gear1.alphat
        self.x_sum_from_al  = (
            (self.inv_alphatw - inv_alphat) * z_sum / (2 * np.tan(gear1.alpha))
        )
        self.x_sum_declared = gear1.x + gear2.x

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    def _set_al_from_x_sum(self, x_sum: float) -> None:
        """
        Derive al and alphatw from a profile shift sum (x1 + x2).
        Sets self.al, self.alphatw, self.inv_alphatw.
        """
        alpha  = self.gear1.alpha     # normal pressure angle
        alphat = self.gear1.alphat    # transverse pressure angle
        z_sum  = self.gear1.z + self.gear2.z

        self.inv_alphatw = (
            np.tan(alphat) - alphat
            + 2 * np.tan(alpha) * x_sum / z_sum
        )

        def funOPT(xx, inv_a):
            return inv_a - np.tan(xx) + xx

        self.alphatw = optimize.brentq(funOPT, 1e-6, 0.7, args=(self.inv_alphatw,))
        # al = (db1 + db2) / (2*cos(alphatw))
        self.al = (
            self.gear1.db + self.gear2.db
        ) / (2 * np.cos(self.alphatw))

    def _set_alphatw_from_al(self) -> None:
        """
        Back-calculate alphatw and inv_alphatw from an imposed al.
        Sets self.alphatw, self.inv_alphatw.
        """
        self.alphatw = np.arccos(self.a * np.cos(self.gear1.alphat) / self.al)

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
        z1, z2   = self.gear1.zn, self.gear2.zn
        ra1, ra2 = self.gear1.mt * z1 / 2.0 + self.gear1.mn * (self.gear1.haP + self.gear1.cP + self.gear1.x), self.gear2.mt * z2 / 2.0 + self.gear2.mn * (self.gear2.haP + self.gear2.cP + self.gear2.x)
        rb1, rb2 = self.gear1.mt * z1 * np.cos(self.gear1.alphat), self.gear2.mt * z2 * np.cos(self.gear2.alphat)
        mn       = self.gear1.mn
        alphat   = self.gear1.alphat
        a        = self.a

        # --- helpers ---
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
            """Solve gs1B' = gs2A' with x2=-x1, a'=a."""
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

        # --- Case 1: z1+z2 > 60 ---
        if (z1 + z2) > 60:
            x = _solve_direct(z1, z2, ra1, ra2, rb1, rb2, a, alphat)
            self.x1 = x
            self.x2 = -x
            return x, -x

        # --- Case 2: z1+z2 ≤ 60 (Henriot hypothetical gear method) ---
        z2_h  = 60 - z1
        a_h   = self.gear1.mt / 2.0 * (z1 + z2_h)
        ra2_h = self.gear1.mt / 2.0 * z2_h + mn
        rb2_h = self.gear1.mt / 2.0 * z2_h * np.cos(alphat)

        x1 = _solve_direct(z1, z2_h, ra1, ra2_h, rb1, rb2_h, a_h, alphat)

        inv_alphat = np.tan(alphat) - alphat
        alpha = self.gear1.alpha

        def _alphatw_from_x2(x2_v):
            inv_aw = inv_alphat + 2.0 * np.tan(alpha) * (x1 + x2_v) / (z1 + z2)
            def funOPT(xx, inv_a):
                return inv_a - np.tan(xx) + xx
            return optimize.brentq(funOPT, 1e-6, 0.7, args=(inv_aw,))

        def _al_from_alphatw(alphatw_v):
            return a * np.cos(alphat) / np.cos(alphatw_v)

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
    # Update gear data
    # ------------------------------------------------------------------

    def gear_geometry(self, addendum_reduction: bool = True) -> None:
        """
        Builds gear1_w / gear2_w: independent copies of the reference gears
        with working geometry applied (k, da, ra, dl, αl, αtw), and computes
        the contact ratios (εα, εβ, εγ) and Ohlendorf loss factor HV.

        Parameters
        ----------
        addendum_reduction : bool
            If True, computes the addendum reduction coefficient k
            (MAAG / KHK §4.1) and reduces da accordingly.
            If False, k = 0 and da follows the plain x-only formula.
        """
        # --- working gears: independent copies, originals untouched ---
        self.gear1_w = copy.copy(self.gear1)
        self.gear2_w = copy.copy(self.gear2)

        z1, z2 = self.gear1.z, self.gear2.z
        alpha  = self.gear1.alpha     # normal pressure angle
        alphat = self.gear1.alphat    # transverse pressure angle
        beta   = self.gear1.beta      # helix angle
        betab  = self.gear1.betab     # base helix angle
        m      = self.gear1.mn

        def involute(ang):
            return np.tan(ang) - ang

        # --- addendum reduction coefficient k (MAAG) ---
        if addendum_reduction:
            self.k = (z1 + z2) / 2.0 * (
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

        # working pitch diameter
        self.gear1_w.dl = 2 * self.al / (self.u + 1)
        self.gear2_w.dl = 2 * self.u * self.al / (self.u + 1)
        self.gear1_w.rl, self.gear2_w.rl = self.gear1_w.dl / 2, self.gear2_w.dl / 2

        rl1, rl2 = self.gear1_w.rl, self.gear2_w.rl

        # --- tip (addendum) diameter / radius, corrected for k ---
        d1, d2 = self.gear1.d, self.gear2.d
        haP1, haP2 = self.gear1.haP, self.gear2.haP
        cP1, cP2 = self.gear1.cP, self.gear2.cP

        self.gear1_w.da = d1 + 2 * m * (haP1 + cP1 + self.x1 - self.k)
        self.gear2_w.da = d2 + 2 * m * (haP2 + cP2 + self.x2 - self.k)
        self.gear1_w.ra = self.gear1_w.da / 2.0
        self.gear2_w.ra = self.gear2_w.da / 2.0

        ra1, rb1 = self.gear1_w.ra, self.gear1_w.rb
        ra2, rb2 = self.gear2_w.ra, self.gear2_w.rb
        pbt = self.gear1_w.pbt   # transverse base pitch

        # transverse profile angle at tooth tip
        db1, db2 = self.gear1_w.db, self.gear2_w.db
        da1, da2 = self.gear1_w.da, self.gear2_w.da

        self.gear1_w.alpha_a = np.arccos(db1 / da1)
        self.gear2_w.alpha_a = np.arccos(db2 / da2)

        alpha_a1 = self.gear1_w.alpha_a
        alpha_a2 = self.gear2_w.alpha_a

        # addendum (transverse) contact ratio
        self.gear1_w.epslon_a1 = z1 * (np.tan(alpha_a1) - np.tan(self.alphatw)) / (2 * np.pi)
        self.gear2_w.epslon_a2 = z2 * (np.tan(alpha_a2) - np.tan(self.alphatw)) / (2 * np.pi)

        self.epslon_alpha = self.gear1_w.epslon_a1 + self.gear2_w.epslon_a2

        # overlap (face) contact ratio — only non-zero for helical (beta != 0)
        self.b = min(self.gear1.b, self.gear2.b)
        self.epslon_beta = self.b * np.tan(betab) / pbt

        # total contact ratio
        self.epslon_gamma = self.epslon_alpha + self.epslon_beta

        self.gear1_w.galpha = rb1 * (np.tan(alpha_a1) - np.tan(self.alphatw))
        self.gear2_w.galpha = rb2 * (np.tan(alpha_a2) - np.tan(self.alphatw))

        self.galpha = self.gear1_w.galpha + self.gear2_w.galpha

        # equivalent curvature radius on pitch point
        self.ReqI = 1 / (1 / (rl1 * np.sin(self.alphatw))
                        + 1 / (rl2 * np.sin(self.alphatw)))

        self.T1T2 = self.al * np.sin(self.alphatw)
        # positions along line of action length (pinion)
        self.T1E = (ra1**2 - rb1**2) ** (1 / 2)
        # positions along line of action length (wheel)
        self.T2A = (ra2**2 - rb2**2) ** (1 / 2)
        # positions of pinion and wheel along AE
        self.T1A = self.T1T2 - self.T2A
        self.T1B = self.T1E - pbt
        self.T1C = rb1 * np.tan(self.alphatw)
        self.T1D = self.T1A + pbt
        self.T2E = self.T1T2 - self.T1E
        self.T2B = self.T2E + pbt
        self.T2C = rb2 * np.tan(self.alphatw)
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
    # Forces  (ISO 6336-1)
    # ------------------------------------------------------------------

    def forces(self, T1: float, phi_deg: float = 0.0,
            rotation_dir: int = 1) -> dict[str, float]:
        """
        Compute meshing forces given the driving torque T1 on gear1.

        For helical gears, Fa ≠ 0 (axial thrust due to the helix angle):
            Ft = T1 / rl1                  tangential force [N]
            Fr = Ft·tan(αtw)                radial force     [N]
            Fa = Ft·tan(β)                  axial force      [N]  (sign TBD — see helix_hand note)
            Fn = Ft/(cos(αtw)·cos(β))       normal force     [N]
            T_out = T1·u                    output torque    [N·m] (ideal, no losses)

        Parameters
        ----------
        T1           : driving torque on gear1 [N·m]
        phi_deg      : angular position of the line of centres (gear1 → gear2)
                    in the shaft cross-section [deg], from +Y toward +Z.
                    Supplied by GearSystem from shaft_position geometry.
        rotation_dir : +1 or -1, sign of gear1's rotation about +X.
                    Determines which side of the line of centres Ft acts on.

        Returns
        -------
        dict with Ft, Fr, Fa, Fn, T_out, and theta_Ft/theta_Fr for BOTH gears
        (driver=gear1, driven=gear2). Fa has no theta — apply directly as
        AxialLoad; sign pending HelicalGear.helix_hand (not yet implemented).
        """
        Ft = T1 / (self.gear1_w.rl / 1000)
        Fr = Ft * np.tan(self.alphatw)
        Fa = Ft * np.tan(self.gear1.beta)
        Fn = Ft / np.cos(self.alphatw)
        T_out = T1 * self.u

        theta_Fr_driver = phi_deg % 360.0
        theta_Ft_driver = (phi_deg + 90.0 * rotation_dir) % 360.0

        rotation_dir_out = -self.u / abs(self.u) * rotation_dir   # external: always -1 * rotation_dir

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
          1. Individual gear validation (undercutting, geometry bounds)
          2. Profile shift consistency with al (only when al is overridden)
          3. Contact ratio εα ≥ 1.0
          4. Involute interference (path of approach / recess vs base circle)
        """
        errors: list[str] = []

        # 1 — individual gears
        errors.extend(self.gear1.validate())
        errors.extend(self.gear2.validate())

        errors.extend(self.gear1_w.validate())
        errors.extend(self.gear2_w.validate())

        # 2 — profile shift consistency (only when al explicitly set)
        if self._al_overridden:
            if abs(self.x_sum_from_al - self.x_sum_declared) > 1e-4:
                errors.append(
                    f"Profile shift inconsistency: declared x1+x2={self.x_sum_declared:.4f} "
                    f"but al={self.al:.4f} mm implies x1+x2={self.x_sum_from_al:.4f}. "
                    "Adjust x1, x2, or al so they are consistent."
                )

        # 3 — contact ratio (transverse)
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

        # 4 — involute interference
        approach = (
            np.sqrt(max(self.gear2_w.ra**2 - self.gear2_w.rb**2, 0.0))
            - self.al * np.sin(self.alphatw)
        )
        limit1 = self.gear1_w.rb * np.tan(self.alphatw)
        if approach > limit1:
            errors.append(
                f"Involute interference on gear1 (pinion): path of approach "
                f"{approach:.4f} mm > limit {limit1:.4f} mm. "
                "Reduce haP, increase x1, or increase z1."
            )

        recess = (
            np.sqrt(max(self.gear1_w.ra**2 - self.gear1_w.rb**2, 0.0))
            - self.al * np.sin(self.alphatw)
        )
        limit2 = self.gear2_w.rb * np.tan(self.alphatw)
        if recess > limit2:
            errors.append(
                f"Involute interference on gear2 (wheel): path of recess "
                f"{recess:.4f} mm > limit {limit2:.4f} mm. "
                "Reduce haP, increase x2, or increase z2."
            )

        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        tag = self.label or "HelicalGearMeshing"
        lines = [
            f"── {tag} ─────────────────────────────────",
            f"  gear1 : z={self.gear1_w.z}, mn={self.gear1_w.mn}, x={self.gear1_w.x}, β={self.gear1.beta_n_deg}°",
            f"  gear2 : z={self.gear2_w.z}, mn={self.gear2_w.mn}, x={self.gear2_w.x}, β={self.gear2.beta_n_deg}°",
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
            f"HelicalGearMeshing(z1={self.gear1_w.z}, z2={self.gear2_w.z}, "
            f"mn={self.gear1_w.mn}, β={self.gear1.beta_n_deg}°, al={self.al:.4f} mm, "
            f"αtw={np.degrees(self.alphatw):.4f}°, εγ={self.epslon_gamma:.4f})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_compatibility(g1: SpurHelicalGear, g2: SpurHelicalGear) -> None:
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