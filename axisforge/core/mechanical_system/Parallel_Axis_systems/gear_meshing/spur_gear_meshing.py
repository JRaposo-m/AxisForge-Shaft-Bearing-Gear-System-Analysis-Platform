"""
core/mechanical_elements/Gears/Parallel_Axis_gears/spur_gear_meshing.py

Spur gear pair: working geometry, interference checks, and meshing forces.

References:
  - ISO 21771:2007  — cylindrical involute gear geometry
  - KHK Technical Reference §4.1–4.2 — gear pair calculations & interference
  - ISO 6336-1:2019 — load on gear teeth (force definitions)
"""

from __future__ import annotations
import numpy as np
from scipy import optimize
import copy

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_gear import SpurGear


class SpurGearMeshing:
    """
    Defines a spur gear pair and computes:
      - Working centre distance and pressure angle
      - Contact ratio εα
      - Interference checks (involute, undercutting)
      - Meshing forces: Ft (tangential) and Fr (radial)

    Parameters
    ----------
    gear1 : SpurGear   — driving gear (pinion)
    gear2 : SpurGear   — driven gear (wheel)
    label : str
    al    : float | None
        Working centre distance [mm].
        If None → computed from reference geometry and profile shifts.
        If provided → gear1_w / gear2_w are updated accordingly,
        and consistency with declared x1+x2 is validated.
    """

    def __init__(self,
                 gear1: SpurGear,
                 gear2: SpurGear,
                 label: str = "",
                 al: float | None = None,
                 equalise_gs: bool = False,
                 addendum_reduction: bool = False):
        """
        Parameters
        ----------
        gear1, gear2  : SpurGear
        label         : str
        al            : float | None
            Working centre distance [mm].
            - None + equalise_gs=False → derived from x1, x2 via involute function
            - None + equalise_gs=True  → x1, x2 computed by Henriot method first,
                                         then al derived from those
            - float                    → imposed directly; alphaw back-calculated
        equalise_gs   : bool
            If True, call correction_for_gs_equilibrium() to find x1, x2 before
            computing al. Ignored when al is explicitly provided.
        """
        # --- compatibility check before anything else ---
        self._check_compatibility(gear1, gear2)

        self.label = label

        # --- store original (reference) gears ---
        self.gear1 = gear1
        self.gear2 = gear2
        

        # --- reference centre distance ---
        # a = mn/2 * (z1 + z2)   [standard, x=0 baseline]
        self.a = gear1.mn / 2.0 * (gear1.z + gear2.z)

        # --- resolve al and alphaw ---
        # Three possible sources, in priority order:
        #   1. al explicitly imposed by the user
        #   2. Henriot gs equalisation (equalise_gs=True)
        #   3. Derived from x1+x2 on the gears as defined

        self._al_overridden = al is not None

        if al is not None:
            # 1 — user imposed al directly
            self.al = al
            self._set_alphaw_from_al()
            self.x1, self.x2 = gear1.x, gear2.x
            
        elif equalise_gs:
            # 2 — Henriot: compute x1, x2 first, then derive al from them
            #     correction_for_gs_equilibrium() sets self.x1, self.x2
            #     but gear1/gear2 objects stay unchanged (x is on self, not on gears)
            #     al is derived from x1+x2 via the involute function
            self.correction_for_gs_equilibrium()
            self._set_al_from_x_sum(self.x1 + self.x2)

        else:
            # 3 — derive al from x1+x2 as declared on the gears
            self._set_al_from_x_sum(gear1.x + gear2.x)
            self.x1, self.x2 = gear1.x, gear2.x

        self.inv_alphaw = np.tan(self.alphaw) - self.alphaw

        # --- gear ratio ---
        self.u = gear2.z / gear1.z

        # --- working geometry, calculada uma vez, automaticamente ---
        self.gear_geometry(addendum_reduction=addendum_reduction)


        # --- profile shift sum (for consistency check in validate) ---
        self.x_sum_from_al  = (self.al - self.a) / gear1.mn
        self.x_sum_declared = gear1.x + gear2.x

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    def _set_al_from_x_sum(self, x_sum: float) -> None:
        """
        Derive al and alphaw from a profile shift sum (x1 + x2).
        Sets self.al, self.alphaw, self.inv_alphaw.
        """
        alpha = self.gear1.alpha
        z_sum = self.gear1.z + self.gear2.z

        self.inv_alphaw = (
            np.tan(alpha) - alpha
            + 2 * np.tan(alpha) * x_sum / z_sum
        )

        def funOPT(xx, inv_a):
            return inv_a - np.tan(xx) + xx

        self.alphaw = optimize.brentq(funOPT, 1e-6, 0.7, args=(self.inv_alphaw,))
        self.al = (
            self.gear1.mn * z_sum * np.cos(alpha) / (2 * np.cos(self.alphaw))
        )

    def _set_alphaw_from_al(self) -> None:
        """
        Back-calculate alphaw and inv_alphaw from an imposed al.
        Sets self.alphaw, self.inv_alphaw.
        """
        self.alphaw = np.arccos(self.a*np.cos(self.gear1.alpha)/self.al)


    # ------------------------------------------------------------------
    # Equalise specific sliding speeds (Henriot)
    # ------------------------------------------------------------------

    def correction_for_gs_equilibrium(self) -> tuple[float, float]:
        """
        Compute the profile shift coefficients x1, x2 that equalise the
        maximum specific sliding speeds at both gears (g_s1B' = g_s2A').

        The procedure is due to Henriot and has two branches:

        Case z1 + z2 > 60  (direct solution)
        ─────────────────
        Centre distance kept standard (a' = a), x2 = -x1 = x.
        Solve numerically for x:

            gs1B'(x) = gs2A'(x)

        Case z1 + z2 ≤ 60  (hypothetical gear method)
        ───────────────────────────────────────────────
        1. Build hypothetical pair: z1_h = z1, z2_h = 60 - z1.
        2. Solve the z>60 problem on that pair → x1.
        3. On the real pair, with x1 fixed, solve for x2 using the
           full system (gs equality + involute relation + base circle):
             · gs1B'(x1, x2, a') = gs2A'(x1, x2, a')
             · a'·cos α' = a·cos α
             · inv α' = inv α + 2·tan α·(x1+x2)/(z1+z2)

        Returns
        -------
        tuple (x1, x2)
        """
        z1, z2   = self.gear1.z, self.gear2.z
        ra1, ra2 = self.gear1.ra, self.gear2.ra
        rb1, rb2 = self.gear1.rb, self.gear2.rb
        mn       = self.gear1.mn
        alpha    = self.gear1.alpha
        a        = self.a

        # --- helpers ---
        def _gs1B(ra2_v, rb2_v, a_v, alpha_v):
            sq  = np.sqrt(max(ra2_v**2 - rb2_v**2, 0.0))
            den = a_v * np.sin(alpha_v) - sq
            if abs(den) < 1e-12:
                return np.inf
            return abs(1.0 - (z1 / z2) * sq / den)

        def _gs2A(ra1_v, rb1_v, a_v, alpha_v):
            sq  = np.sqrt(max(ra1_v**2 - rb1_v**2, 0.0))
            den = a_v * np.sin(alpha_v) - sq
            if abs(den) < 1e-12:
                return np.inf
            return abs((z2 / z1) * sq / den - 1.0)

        def _solve_direct(z1_v, z2_v, ra1_v, ra2_v, rb1_v, rb2_v, a_v, alpha_v):
            """Solve gs1B' = gs2A' with x2=-x1, a'=a."""
            def equation(x):
                return (_gs1B(ra2_v - x * mn, rb2_v, a_v, alpha_v)
                      - _gs2A(ra1_v + x * mn, rb1_v, a_v, alpha_v))

            x_lim = min(
                (ra2_v - rb2_v) / mn * 0.9,
                (a_v * np.sin(alpha_v) - 1e-3) / mn
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
            x = _solve_direct(z1, z2, ra1, ra2, rb1, rb2, a, alpha)
            self.x1 = x
            self.x2 = -x
            return x, -x

        # --- Case 2: z1+z2 ≤ 60 (Henriot hypothetical gear method) ---
        # Step 1: hypothetical pair with z2_h = 60 - z1
        z2_h  = 60 - z1
        a_h   = mn / 2.0 * (z1 + z2_h)
        ra2_h = mn / 2.0 * z2_h + mn
        rb2_h = mn / 2.0 * z2_h * np.cos(alpha)

        # Step 2: solve on hypothetical pair → x1
        x1 = _solve_direct(z1, z2_h, ra1, ra2_h, rb1, rb2_h, a_h, alpha)

        # Step 3: solve for x2 on real pair, a' varies with x1+x2
        inv_alpha = np.tan(alpha) - alpha

        def _alphaw_from_x2(x2_v):
            inv_aw = inv_alpha + 2.0 * np.tan(alpha) * (x1 + x2_v) / (z1 + z2)
            def funOPT(xx, inv_a):
                return inv_a - np.tan(xx) + xx
            return optimize.brentq(funOPT, 1e-6, 0.7, args=(inv_aw,))

        def _al_from_alphaw(alphaw_v):
            return a * np.cos(alpha) / np.cos(alphaw_v)

        def equation_x2(x2_v):
            alphaw_v = _alphaw_from_x2(x2_v)
            al_v     = _al_from_alphaw(alphaw_v)
            return (_gs1B(ra2 + x2_v * mn, rb2, al_v, alphaw_v)
                  - _gs2A(ra1 + x1  * mn, rb1, al_v, alphaw_v))

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
    # Update Gear data
    # ------------------------------------------------------------------


    def gear_geometry(self, addendum_reduction: bool = True) -> None:
        """
        Builds gear1_w / gear2_w: independent copies of the reference gears
        with working geometry applied (k, da, ra, dw, αl, αw).

        Parameters
        ----------
        addendum_reduction : bool
            If True, computes the addendum reduction coefficient k
            (ISO 21771 / KHK §4.1) and reduces da accordingly.
            If False, k = 0 and da follows the plain x-only formula.
        """
        # --- working gears: independent copies, originals untouched ---
        self.gear1_w = copy.copy(self.gear1)
        self.gear2_w = copy.copy(self.gear2)

        z1, z2 = self.gear1.z, self.gear2.z
        alpha  = self.gear1.alpha      # transverse = normal pressure angle (spur, beta=0)
        m      = self.gear1.mn

        def involute(ang):
            return np.tan(ang) - ang

        # --- addendum reduction coefficient k (ISO 21771 / KHK) ---
        if addendum_reduction:
            self.k = (z1 + z2) / 2.0 * (
                (involute(self.alphaw) - involute(alpha)) / np.tan(alpha)
                - (np.cos(alpha) / np.cos(self.alphaw) - 1.0)
            )
        else:
            self.k = 0.0

        self.gear1_w.k = self.k
        self.gear2_w.k = self.k

        # --- working centre distance / pressure angle ---
        self.gear1_w.al = self.gear2_w.al = self.al
        self.gear1_w.alphaw = self.gear2_w.alphaw = self.alphaw

        # working pitch diameter
        self.gear1_w.dl = 2*self.al/(self.u + 1)
        self.gear2_w.dl = 2*self.u*self.al/(self.u + 1)
        self.gear1_w.rl, self.gear2_w.rl = self.gear1_w.dl/2, self.gear2_w.dl/2

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
        pb = self.gear1_w.pb

        # transverse profile angle at tooth tip
        db1, db2 = self.gear1_w.db, self.gear2_w.db
        da1, da2 = self.gear1_w.da, self.gear2_w.da

        self.gear1_w.alpha_a = np.arccos(db1/da1)
        self.gear2_w.alpha_a = np.arccos(db2/da2)

        alpha_a1 = self.gear1_w.alpha_a
        alpha_a2 = self.gear2_w.alpha_a

        # addendum contact ratio
        self.gear1_w.epslon_a1 = z1*(np.tan(alpha_a1)
                                  - np.tan(self.alphaw))/(2*np.pi)
        self.gear2_w.epslon_a2 = z2*(np.tan(alpha_a2)
                                  - np.tan(self.alphaw))/(2*np.pi)   

        self.epslon_alpha = self.gear1_w.epslon_a1 + self.gear2_w.epslon_a2

        self.gear1_w.galpha = rb1*(np.tan(alpha_a1) - np.tan(self.alphaw))
        self.gear2_w.galpha = rb2*(np.tan(alpha_a2) - np.tan(self.alphaw))   

        self.galpha = self.gear1_w.galpha + self.gear2_w.galpha

        # equivalent curvature radius on pitch point
        self.ReqI = 1/(1/(rl1*np.sin(self.alphaw))
                       + 1/(rl2*np.sin(self.alphaw)))
        
        self.T1T2 = self.al*np.sin(self.alphaw)
        # positions along line of action length (pinion)
        self.T1E = (ra1**2 - rb1**2)**(1/2)
        # positions along line of action length (wheel)
        self.T2A = (ra2**2 - rb2**2)**(1/2)
        # positions of pinion and wheel along AE
        self.T1A = self.T1T2 - self.T2A
        self.T1B = self.T1E - pb
        self.T1C = rb1*np.tan(self.alphaw)
        self.T1D = self.T1A + pb
        self.T2E = self.T1T2 - self.T1E
        self.T2B = self.T2E + pb
        self.T2C = rb2*np.tan(self.alphaw)
        self.T2D = self.T2A - pb
        # positions along path of contact
        self.AE = self.T1E - self.T1A
        self.AB = self.T1B - self.T1A
        self.AC = self.T1C - self.T1A
        self.AD = self.T1D - self.T1A
        # radius along the path of contact
        self.rA1 = (self.T1A**2 + rb1**2)**(1/2)
        self.rB1 = (self.T1B**2 + rb1**2)**(1/2)
        self.rD1 = (self.T1D**2 + rb1**2)**(1/2)
        self.rA2 = ((self.T2A - self.AE)**2 + rb2**2)**(1/2)
        self.rB2 = ((self.T2A - self.AD)**2 + rb2**2)**(1/2)
        self.rD2 = ((self.T2A - self.AB)**2 + rb2**2)**(1/2)
        # gear loss factor according to Ohlendorf
        self.HV = (np.pi*(self.u+1)/(z1*self.u) *
                   (1-self.epslon_alpha + self.gear1_w.epslon_a1**2 +
                    self.gear2_w.epslon_a2**2))
        # gear finishing
        self.Ram = (self.gear1_w.Ra + self.gear2_w.Ra)/2
        self.Rrms = (self.gear1_w.Rq**2 + self.gear2_w.Ra**2)**(1/2)
        self.RzS = (self.gear1.Rz + self.gear2_w.Rz)
        

    # ------------------------------------------------------------------
    # Forces  (ISO 6336-1)
    # ------------------------------------------------------------------

    def forces(self, T1: float, phi_deg: float = 0.0,
            rotation_dir: int = 1) -> dict[str, float]:
        """
        Compute meshing forces given the driving torque T1 on gear1.

        For spur gears Fa = 0:
            Ft = T1 / rl1        tangential force [N]
            Fr = Ft·tan(αw)      radial force     [N]
            Fn = Ft/cos(αw)      normal force     [N]
            T2 = T1·u            output torque    [N·m] (ideal, no losses)

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
        dict with Ft, Fr, Fn, T_out, and theta_Ft/theta_Fr for BOTH gears
        (driver=gear1, driven=gear2), consistent with InternalGearMeshing/
        HelicalGearMeshing output — GearSystem consumes all three polymorphically.
        """
        Ft = T1 / (self.gear1_w.rl / 1000)
        Fr = Ft * np.tan(self.alphaw)
        Fn = Ft / np.cos(self.alphaw)
        T_out = T1 * self.u

        theta_Fr_driver = phi_deg % 360.0
        theta_Ft_driver = (phi_deg + 90.0 * rotation_dir) % 360.0

        rotation_dir_out = -self.u / abs(self.u) * rotation_dir   # external: always -1 * rotation_dir

        theta_Fr_driven = (phi_deg + 180.0) % 360.0
        theta_Ft_driven = (phi_deg + 180.0 + 90.0 * rotation_dir_out) % 360.0

        return {
            "Ft": Ft, "Fr": Fr, "Fn": Fn, "T_out": T_out,
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

        # 3 — contact ratio
        if self.epslon_alpha < 1.0:
            errors.append(
                f"Contact ratio εα={self.epslon_alpha:.4f} < 1.0 — "
                "gears will lose contact during meshing."
            )
        elif self.epslon_alpha < 1.2:
            errors.append(
                f"Warning: contact ratio εα={self.epslon_alpha:.4f} is low (< 1.2) — "
                "consider increasing tooth count or face width."
            )

        # 4 — involute interference
        approach = (
            np.sqrt(max(self.gear2_w.ra**2 - self.gear2_w.rb**2, 0.0))
            - self.al * np.sin(self.alphaw)
        )
        limit1 = self.gear1_w.rb * np.tan(self.alphaw)
        if approach > limit1:
            errors.append(
                f"Involute interference on gear1 (pinion): path of approach "
                f"{approach:.4f} mm > limit {limit1:.4f} mm. "
                "Reduce haP, increase x1, or increase z1."
            )

        recess = (
            np.sqrt(max(self.gear1_w.ra**2 - self.gear1_w.rb**2, 0.0))
            - self.al * np.sin(self.alphaw)
        )
        limit2 = self.gear2_w.rb * np.tan(self.alphaw)
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
        tag = self.label or "SpurGearMeshing"
        lines = [
            f"── {tag} ─────────────────────────────────",
            f"  gear1 : z={self.gear1_w.z}, mn={self.gear1_w.mn}, x={self.gear1_w.x}",
            f"  gear2 : z={self.gear2_w.z}, mn={self.gear2_w.mn}, x={self.gear2_w.x}",
            f"  u     : {self.u:.4f}",
            f"  a     : {self.a:.4f} mm   (reference centre distance)",
            f"  al    : {self.al:.4f} mm   (working centre distance)",
            f"  αw    : {np.degrees(self.alphaw):.4f}°",
            f"  dl1   : {self.gear1_w.dl:.4f} mm",
            f"  dl2   : {self.gear2_w.dl:.4f} mm",
            f"  εα    : {self.epslon_alpha:.4f}",
            "────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SpurGearMeshing(z1={self.gear1_w.z}, z2={self.gear2_w.z}, "
            f"mn={self.gear1_w.mn}, al={self.al:.4f} mm, "
            f"αw={np.degrees(self.alphaw):.4f}°, εα={self.epslon_alpha:.4f})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_compatibility(g1: SpurGear, g2: SpurGear) -> None:
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