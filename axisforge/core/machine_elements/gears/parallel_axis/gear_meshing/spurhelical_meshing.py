"""
Pair of external spur or helical involute gears: working centre distance and
pressure angle, contact ratios, path of contact, interference checks and the
tooth force components transmitted to the shafts.
"""

from __future__ import annotations
import numpy as np
from scipy import optimize
import copy

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear


class SpurHelicalGearMeshing:
    """
    External spur or helical gear pair (gear1 meshing with gear2).

    From two ``SpurHelicalGear`` objects it computes the working centre
    distance and transverse working pressure angle, the working copies
    ``gear1_w`` / ``gear2_w``, the transverse, overlap and total contact ratios,
    the points of the path of contact, the Ohlendorf loss factor, and (in
    ``forces``) the tooth force components. Lengths are in [mm], angles stored
    in [rad], torques in [N m], forces in [N].

    Parameters
    ----------
    gear1 : SpurHelicalGear
        First gear (the pinion in the formulas of this class).
    gear2 : SpurHelicalGear
        Second gear; same module, normal pressure angle and helix angle as
        ``gear1`` (checked, ``ValueError`` otherwise).
    label : str, optional
        Name used in the summary.
    al : float or None, optional
        Working centre distance [mm]. None: derived from x1 + x2 (or from the
        Henriot shifts if ``equalise_gs``). A value: imposed; alphatw is
        back-calculated and x1, x2 are those of the gears.
    equalise_gs : bool, optional
        If True and ``al`` is None, x1 and x2 are first computed by
        ``correction_for_gs_equilibrium``. Ignored when ``al`` is given.
    addendum_reduction : bool, optional
        Apply the addendum reduction k to the working tip diameters.
        Default False.
    driver : str, optional
        "gear1" or "gear2": which gear delivers the input torque. Stored in
        ``driver`` / ``driven``; ``forces`` does not use it.

    Attributes
    ----------
    a : float
        Reference centre distance, (d1 + d2) / 2 [mm].
    al, alphatw, inv_alphatw : float
        Working centre distance [mm], transverse working pressure angle [rad],
        inv(alphatw) [rad].
    u : float
        Gear ratio max(z1, z2) / min(z1, z2) >= 1 [-].
    x1, x2 : float
        Profile shift coefficients used [-].
    gear1_w, gear2_w : SpurHelicalGear
        Shallow copies of the gears with the working data added (see
        ``gear_geometry``); the originals are not modified.
    epslon_alpha, epslon_beta, epslon_gamma : float
        Transverse, overlap and total contact ratios [-].
    x_sum_from_al, x_sum_declared : float
        x1 + x2 implied by ``al`` and declared by the gears [-].

    Warnings
    --------
    - u is max(z)/min(z), and the working pitch diameters are assigned in the
      order dl1 = 2 al / (u + 1) (smaller), dl2 = 2 u al / (u + 1) (larger):
      ``gear1_w`` always gets the smaller working diameter. If z1 > z2 the
      working diameters, and what is derived from them (rl, Ft, speeds), are
      exchanged between the gears. The loss factor HV and T_out use z1 and u in the same
      way, which is consistent only if gear1 is the smaller gear.
    - The helix angle is a magnitude (no hand of helix).

    References
    ----------
    .. [1] ISO_6336-1_2019, Table 2: u = z2/z1 >= 1; eps_gamma = eps_alpha +
       eps_beta; Table F.1: eps_1 = CE/p_bt, eps_2 = AC/p_bt; 10.5.1, Formulae
       (104)-(112).
    .. [2] ISO 21771:2007, named in the code; not in the library; not checked
       against the source (source to find).
    .. [3] KHK Technical Reference sections 4.1-4.2, named in the code; not in
       the library; not checked against the source (source to find).
    .. [4] MAAG Gear Book (working geometry, contact ratio, Ohlendorf loss
       factor), named in the code; not in the library; not checked against the
       source (source to find).
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
        Resolve al and alphatw, then build the working geometry.

        The arguments are described in the class docstring. Order of
        precedence for the working centre distance: an imposed ``al``, the
        Henriot shifts (``equalise_gs``), the shifts declared on the gears.

        Raises
        ------
        ValueError
            If the gears differ in module, normal pressure angle or helix
            angle, or if ``driver`` is not "gear1" or "gear2".
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

        # --- working geometry, computed once, automatically ---
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
        Working centre distance and pressure angle from the profile shift sum.

        Parameters
        ----------
        x_sum : float
            x1 + x2 [-].

        Notes
        -----
        Sets ``inv_alphatw`` [rad], ``alphatw`` [rad] and ``al`` [mm]::

            inv(alphatw) = inv(alphat) + 2*tan(alpha)*(x1 + x2)/(z1 + z2)  (F09)
            alphatw      = inverse involute, scipy brentq on [1e-6, 0.7] rad  (F10)
            al           = (db1 + db2) / (2*cos(alphatw))                 (F11)

        alpha and alphat are those of gear1 (the pair has equal angles).
        The bracket limits alphatw to below 0.7 rad (40.1 deg).

        References
        ----------
        .. [1] ISO 21771:2007 and "MAAG Gear Book", named in the code; not in
           the library; not checked against the source (source to find).
        """
        alpha  = self.gear1.alpha     # normal pressure angle
        alphat = self.gear1.alphat    # transverse pressure angle
        z_sum  = self.gear1.z + self.gear2.z

        self.inv_alphatw = (
            np.tan(alphat) - alphat
            + 2 * np.tan(alpha) * x_sum / z_sum
        )

        def funOPT(xx, inv_a):
            """Residual ``inv_a - inv(xx)`` [rad]; zero when the angle xx [rad] has involute inv_a."""
            return inv_a - np.tan(xx) + xx

        self.alphatw = optimize.brentq(funOPT, 1e-6, 0.7, args=(self.inv_alphatw,))
        # al = (db1 + db2) / (2*cos(alphatw))
        self.al = (
            self.gear1.db + self.gear2.db
        ) / (2 * np.cos(self.alphatw))

    def _set_alphatw_from_al(self) -> None:
        """
        Transverse working pressure angle from an imposed centre distance.

        Notes
        -----
        Sets ``alphatw`` [rad] only (``inv_alphatw`` is set later by the
        constructor)::

            alphatw = arccos(a * cos(alphat) / al)                     (F12)

        with a = (d1 + d2) / 2 the reference centre distance [mm].
        """
        self.alphatw = np.arccos(self.a * np.cos(self.gear1.alphat) / self.al)

    # ------------------------------------------------------------------
    # Equalise specific sliding speeds (Henriot)
    # ------------------------------------------------------------------

    def correction_for_gs_equilibrium(self) -> tuple[float, float]:
        """
        Profile shifts x1, x2 that equalise the maximum specific sliding.

        Looks for x1, x2 such that the specific sliding at the two ends of the
        path of contact is equal, g_s1B = g_s2A, by the Henriot procedure
        (spur-gear procedure applied to the virtual spur gears of the helical
        pair). Used when ``equalise_gs`` is True.

        Returns
        -------
        tuple of float
            (x1, x2) [-]; also stored in ``self.x1``, ``self.x2``.

        Raises
        ------
        RuntimeError
            If no sign change of the sliding difference is found.

        Notes
        -----
        As coded, with z1, z2 the virtual tooth numbers ``zn`` of the gears,
        sq the root term and a the reference centre distance [mm] (F30)::

            sq_2  = sqrt(ra2**2 - rb2**2);   sq_1 = sqrt(ra1**2 - rb1**2)
            g_s1B = | 1 - (z1/z2) * sq_2 / (a*sin(alphat) - sq_2) |
            g_s2A = | (z2/z1) * sq_1 / (a*sin(alphat) - sq_1) - 1 |
            ra = mt*zn/2 + mn*(haP + cP + x);   rb = mt*zn*cos(alphat)

        If z1 + z2 > 60: x2 = -x1 and a is kept; x1 is the root of
        g_s1B - g_s2A found by a scan of 400 points and brentq. If z1 + z2 <= 60:
        x1 is first found for a hypothetical mate with 60 - z1 teeth (Henriot),
        then x2 is found with alphatw and al recomputed from x1 + x2 through
        the involute equation (600-point scan, brentq).

        Warnings
        --------
        rb is written without the factor 1/2 (mt*zn*cos(alphat) is a diameter)
        while ra is a radius, and the pitch radius uses mt*zn/2, not mn*zn/2
        (to be checked against the source).

        The unknown of the solver (x) is added to tip radii that already
        contain the declared x of the gears (ra uses ``gear.x``), and the root
        is then stored as the total x1 (``self.x1 = x``, ``self.x2 = -x``).
        The stored value is therefore an increment over the declared shift
        stored as if it were the total shift; the two coincide only when the
        declared x of both gears is zero.

        References
        ----------
        .. [1] Henriot, "Traite theorique et pratique des engrenages", named in
           the code; not in the library; not checked against the source (source
           to find). The 60-tooth hypothetical gear rule has no other source in
           the code.
        """
        z1, z2   = self.gear1.zn, self.gear2.zn
        ra1, ra2 = self.gear1.mt * z1 / 2.0 + self.gear1.mn * (self.gear1.haP + self.gear1.cP + self.gear1.x), self.gear2.mt * z2 / 2.0 + self.gear2.mn * (self.gear2.haP + self.gear2.cP + self.gear2.x)
        rb1, rb2 = self.gear1.mt * z1 * np.cos(self.gear1.alphat), self.gear2.mt * z2 * np.cos(self.gear2.alphat)
        mn       = self.gear1.mn
        alphat   = self.gear1.alphat
        a        = self.a

        # --- helpers ---
        def _gs1B(ra2_v, rb2_v, a_v, alphat_v):
            """Specific sliding g_s1B [-] at the pinion (formula in the Notes); inf if the denominator is ~0."""
            sq  = np.sqrt(max(ra2_v**2 - rb2_v**2, 0.0))
            den = a_v * np.sin(alphat_v) - sq
            if abs(den) < 1e-12:
                return np.inf
            return abs(1.0 - (z1 / z2) * sq / den)

        def _gs2A(ra1_v, rb1_v, a_v, alphat_v):
            """Specific sliding g_s2A [-] at the wheel (formula in the Notes); inf if the denominator is ~0."""
            sq  = np.sqrt(max(ra1_v**2 - rb1_v**2, 0.0))
            den = a_v * np.sin(alphat_v) - sq
            if abs(den) < 1e-12:
                return np.inf
            return abs((z2 / z1) * sq / den - 1.0)

        def _solve_direct(z1_v, z2_v, ra1_v, ra2_v, rb1_v, rb2_v, a_v, alphat_v):
            """Solve gs1B' = gs2A' with x2=-x1, a'=a."""
            def equation(x):
                """Sliding difference g_s1B - g_s2A [-] for the shift x [-] (x1 = x, x2 = -x)."""
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
            """Transverse working pressure angle alphatw [rad] for the shifts x1 and x2_v [-]."""
            inv_aw = inv_alphat + 2.0 * np.tan(alpha) * (x1 + x2_v) / (z1 + z2)
            def funOPT(xx, inv_a):
                """Residual ``inv_a - inv(xx)`` [rad]; zero when the angle xx [rad] has involute inv_a."""
                return inv_a - np.tan(xx) + xx
            return optimize.brentq(funOPT, 1e-6, 0.7, args=(inv_aw,))

        def _al_from_alphatw(alphatw_v):
            """Working centre distance al = a cos(alphat) / cos(alphatw_v) [mm], alphatw_v in [rad]."""
            return a * np.cos(alphat) / np.cos(alphatw_v)

        def equation_x2(x2_v):
            """Sliding difference g_s1B - g_s2A [-] for the wheel shift x2_v [-] (x1 fixed)."""
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
        Working geometry, contact ratios, path of contact and loss factor.

        Builds ``gear1_w`` / ``gear2_w`` (shallow copies of the reference gears
        with k, al, alphatw, dl, rl, da, ra, alpha_a, epslon_a1/2, galpha added)
        and the pair attributes epslon_alpha/beta/gamma, galpha, ReqI, the
        points T1T2 ... rD2 of the path of contact, HV and the roughness
        combinations. Lengths [mm], angles [rad].

        Parameters
        ----------
        addendum_reduction : bool, optional
            True: compute the addendum reduction coefficient k and subtract it
            from the tip diameters. False: k = 0. The default of this method is
            True; the constructor always passes its own flag (default False).

        Notes
        -----
        Equations as coded (m = mn of gear1; x1, x2 the resolved shifts)::

            k = (z1+z2)/2 * ((inv(atw) - inv(at))/tan(a)
                - 1/cos(b) * (cos(at)/cos(atw) - 1))                   (F16)
            dl1 = 2*al/(u+1);  dl2 = 2*u*al/(u+1);  rl = dl/2          (F15)
            da_w = d + 2*m*(haP + cP + x - k)                           (F17)
            alpha_a = arccos(db / da_w)                                 (F18)
            eps_a1 = z1*(tan(alpha_a1) - tan(atw))/(2 pi), same for 2   (F19)
            eps_alpha = eps_a1 + eps_a2                                 (F19)
            eps_beta  = b*tan(betab)/pbt,  b = min(b1, b2)              (F20)
            eps_gamma = eps_alpha + eps_beta                            (F21)
            galpha = rb1*(tan(alpha_a1)-tan(atw)) + rb2*(tan(alpha_a2)-tan(atw))
                                                                        (F22)
            ReqI = 1/(1/(rl1 sin(atw)) + 1/(rl2 sin(atw)))              (F23)
            T1T2 = al*sin(atw)                                          (F24)
            T1E = sqrt(ra1**2 - rb1**2);  T2A = sqrt(ra2**2 - rb2**2)   (F25)
            T1A = T1T2 - T2A;  T1B = T1E - pbt;  T1C = rb1*tan(atw)
            T1D = T1A + pbt;   T2E = T1T2 - T1E; T2B = T2E + pbt
            T2C = rb2*tan(atw); T2D = T2A - pbt                         (F26)
            AE = T1E - T1A;  AB, AC, AD = T1B, T1C, T1D minus T1A       (F26)
            rA1 = sqrt(T1A**2 + rb1**2), rB1, rD1; rA2 = sqrt((T2A-AE)**2
                + rb2**2), rB2 (uses AD), rD2 (uses AB)                 (F27)
            (rA2, rB2, rD2: radii of gear2 at E, D, B; see Warnings)
            HV = pi*(u+1)/(z1*u*cos(betab))
                 *(1 - eps_alpha + eps_a1**2 + eps_a2**2)               (F28)
            Ram = (Ra1+Ra2)/2;  Rrms = sqrt(Rq1**2 + Rq2**2);
            RzS = Rz1 + Rz2   [um]                                      (F29)

        Here a, b, at, atw are alpha, beta, alphat, alphatw of gear1; pbt and
        betab are those of gear1_w / gear1; b is the face width [mm] (unrelated
        to the helix angle beta).

        eps_a1 and eps_a2 are the pinion and wheel addendum contact ratios
        eps_1 = CE/p_bt and eps_2 = AC/p_bt of ISO_6336-1_2019, Table F.1
        (p_bt = 2 pi rb / z); eps_gamma is the sum of Table 2. The points
        A..E are those of 10.5.1, Formulae (104)-(110), and rA1, rB1, rD1 are
        the radii of gear1 at A, B, D, Formula (111); ReqI is Formulae
        (96)-(97) at the pitch point.

        Warnings
        --------
        - Working pitch diameters: dl1 (smaller) and dl2 (larger) are assigned
          to gear1_w and gear2_w in that order whatever the tooth numbers; for
          z1 > z2 they are exchanged (class docstring).
        - The tip diameter da_w is used where ISO_6336-1_2019, 10.5.1 uses the
          active tip diameter d_Na, and pbt where the standard writes p_et.
        - ReqI is the transverse reduced radius at the working pitch point,
          Formulae (96)-(97) with d_CP = d_w, without the 1/cos(beta_b) of
          Formula (95).
        - Not computed: radius of curvature along the line of action (G2);
          load per unit length and load sharing along the path of contact (G3);
          sliding and rolling speeds (G4).
        - rA1, rB1, rD1 are the radii of gear1 at the points A, B, D. For gear2
          the names are mirrored: rA2 is the radius of gear2 at the point E
          (sqrt(T2E**2 + rb2**2), the code uses the distance AE), rB2 at the
          point D (distance AD) and rD2 at the point B (distance AB), i.e.
          counted from the wheel's own start of active profile.
          ISO_6336-1_2019, 10.5.1, Formula (112), gives the wheel diameter at
          A, B, D with g_CP = 0, AB, AD; that corresponds to ra2, the coded rD2
          and the coded rB2 (not coded as such).

        References
        ----------
        .. [1] ISO_6336-1_2019, Table 2 (eps_gamma), Table F.1 (eps_1, eps_2),
           10.1 Formulae (95)-(97), 10.5.1 Formulae (104)-(112).
        .. [2] MAAG Gear Book (k, contact ratio, Ohlendorf loss factor), named
           in the code; not in the library; not checked against the source
           (source to find).
        .. [3] KHK Technical Reference section 4.1 (k), named in the code; not
           in the library; not checked against the source (source to find).
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
            """Involute function inv(ang) = tan(ang) - ang [rad], ang in [rad]."""
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
        Tooth force components and output torque of the pair for a torque T1.

        The torque T1 is taken on gear1 (whatever ``driver`` says). Ft, Fr, Fa
        are the components loading the shafts; the system layer applies them
        to the two gears.

        Parameters
        ----------
        T1 : float
            Torque on gear1 [N m].
        phi_deg : float, optional
            Angle of the line of centres gear1 -> gear2 in the shaft cross-
            section, from +Y toward +Z [deg].
        rotation_dir : int, optional
            +1 or -1, sense of rotation of gear1 about +X [-].

        Returns
        -------
        dict
            Ft, Fr, Fa, Fn : tangential, radial, axial and normal force [N];
            T_out : output torque on gear2 [N m]; theta_Fr_driver,
            theta_Ft_driver, theta_Fr_driven, theta_Ft_driven : angles
            [deg, 0-360) of the radial and tangential forces of gear1 (driver)
            and gear2 (driven), from +Y toward +Z; rotation_dir_out : sense of
            rotation of gear2 [-].

        Notes
        -----
        As coded (rl1 = gear1_w.rl, the working pitch radius [mm]; u >= 1)::

            Ft = T1 / (rl1/1000)                                       (F32)
            Fr = Ft * tan(alphatw)                                     (F33)
            Fa = Ft * tan(beta),  beta = reference helix angle of gear1  (F34)
            Fn = Ft / cos(alphatw)                                     (F35)
            T_out = T1 * u                                             (F36)
            theta_Fr_driver = phi mod 360
            theta_Ft_driver = (phi + 90*rotation_dir) mod 360
            rotation_dir_out = -rotation_dir
            theta_Fr_driven = (phi + 180) mod 360
            theta_Ft_driven = (phi + 180 + 90*rotation_dir_out) mod 360  (F37)

        The reference helix angle is used in Fa, not the base or the working
        helix angle. Fa has no angle: there is no hand of helix (beta < 0 is
        rejected by the gear), so its direction is fixed by convention in the
        system layer (+Fa on the driver, -Fa on the driven gear). T_out is
        ideal (no losses). Fn is returned and read by no caller in axisforge.
        ``driver`` is not used here: the torque is always taken on gear1.

        Warnings
        --------
        - Differs from ISO_6336-1_2019, 4.2.2, Formula (1): F_t = 2000 T / d is
          written on the reference diameter; here Ft uses the working pitch
          radius rl1 (the PlanetaryGearTrainMeshing nominal Ft uses the
          reference diameter).
        - Fn: the coded form is Ft / cos(alphatw); the form
          Ft / (cos(alphatw) cos(beta)) differs for helical gears; neither is
          printed in a library source.
        - For z1 > z2, rl1 is the radius of the smaller gear (class docstring).
          T_out = T1*u is then also that of the opposite order: u = max/min
          does not depend on the order, so Ft and T_out are the same as for
          the pair with z1 and z2 exchanged and the same T1.
        - Directions as coded: theta_Fr_driver = phi points from the centre of
          gear1 toward gear2 (toward the mesh) and theta_Fr_driven = phi + 180
          from gear2 toward gear1, i.e. opposite to the separating force on
          each gear; theta_Ft_driven equals theta_Ft_driver for both senses of
          rotation, so the two tangential forces are not an action-reaction
          pair. Open review item (see ``SpurHelicalGearSystem._forces_to_loads``,
          Warnings); no gear source in the library defines these angles.

        References
        ----------
        .. [1] ISO_6336-1_2019, 4.2.2, Formula (1)-(2): F_t = 2000 T / d,
           T [N m], d [mm]; Formula (2) gives T = F_t d / 2000.
        .. [2] F_r, F_a, F_n: the code names "ISO 6336-1", which prints none of
           them in the pages read (source to find).
        """
        # F_t on the WORKING pitch radius; T [N m] / r [m] gives [N] (rl is in mm)
        Ft = T1 / (self.gear1_w.rl / 1000)
        Fr = Ft * np.tan(self.alphatw)
        Fa = Ft * np.tan(self.gear1.beta)   # reference helix angle, not the base or working one
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
        Check the pair geometry.

        Returns
        -------
        list of str
            One message per violation; empty if the pair is valid. Checks:
            (1) ``validate()`` of gear1, gear2 and the working copies;
            (2) |x1+x2 implied by al - declared| <= 1e-4, only when al was
            imposed; (3) epsilon_alpha >= 1.0 (a value below 1.2 adds a message
            starting "Warning:" to the list, so ``validate_or_raise`` raises); (4) involute interference, F31:
            sqrt(ra2**2 - rb2**2) - al*sin(alphatw) > rb1*tan(alphatw) for
            gear1, and the mirrored test for gear2 [mm].

        Notes
        -----
        The interference criterion (F31) has no source in the code or in the
        library (source to find). ISO_6336-1_2019, clause 1, states that its
        methods do not apply with transverse contact ratio below 1,0 or with
        interference between tooth tips and root fillets.

        The left side of the test in (4) equals -T1A (T1A = T1T2 - T2A), so
        the test reads -T1A > T1C. A negative T1A (point A beyond the tangent
        point T1) is therefore reported only when abs(T1A) exceeds T1C.
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
        """Raise ``ValueError`` with all messages of ``validate()`` if any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Multi-line text with the main data of the pair (lengths [mm], angles [deg])."""
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
        """One-line representation: tooth numbers, module, helix angle, al, alphatw, eps_gamma."""
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
        """
        Raise ``ValueError`` if the two gears cannot mesh.

        Compares the normal module [mm], the normal pressure angle [deg] and
        the helix angle [deg], each with relative tolerance 1e-6.
        """
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