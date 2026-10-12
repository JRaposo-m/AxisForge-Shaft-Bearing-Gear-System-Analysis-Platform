"""
Internal gear pair: an external spur or helical gear (gear1) meshing with an
internal (ring) gear (gear2). Computes working centre distance and pressure
angle, contact ratios, path of contact, interference checks and the tooth
force components. The ring tooth number is signed (z2 < 0); see the class
docstring.
"""

from __future__ import annotations
import numpy as np
from scipy import optimize
import copy

from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.internal_gear import InternalGear
from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear


class InternalGearMeshing:
    """
    Internal gear pair (external gear1 + internal/ring gear2).

    Computes the working centre distance and transverse working pressure
    angle, the working copies ``gear1_w`` / ``gear2_w``, the contact ratios,
    the points of the path of contact, interference checks and (in ``forces``)
    the tooth force components. Lengths [mm], angles stored in [rad], torques
    [N m], forces [N].

    Parameters
    ----------
    gear1 : SpurHelicalGear
        External gear (pinion).
    gear2 : InternalGear
        Internal (ring) gear with signed tooth number. The code expects
        z2 < 0; this is not checked here (z2 > 0 is accepted and changes the
        formulas, see Notes). abs(z2) > z1 is checked (``ValueError``
        otherwise). Same module, normal pressure angle and helix angle as
        ``gear1`` (checked, ``ValueError`` otherwise).
    label : str, optional
        Name used in the summary.
    al : float or None, optional
        Working centre distance [mm]. None: derived from x1 + x2. A value:
        imposed; alphatw is back-calculated and the backlash ``jbn`` is set.
    equalise_gs : bool, optional
        If True and ``al`` is None, call ``correction_for_gs_equilibrium``
        first (see its Warnings: it fails for an ``InternalGear``).
    addendum_reduction : bool, optional
        Subtract the coefficient k from the tip diameter of the external gear
        only. Default False.
    driver : str, optional
        "gear1" or "gear2": which gear delivers the input torque (used by
        ``forces``; independent of which gear is external).

    Attributes
    ----------
    a : float
        Reference centre distance, (d2 - d1) / 2 [mm].
    al, alphatw, inv_alphatw : float
        Working centre distance [mm], transverse working pressure angle [rad],
        inv(alphatw) [rad].
    u : float
        z2 / z1 [-]; negative for z2 < 0.
    x1, x2 : float
        Profile shift coefficients used [-].
    gear1_w, gear2_w : object
        Shallow copies of the gears with the working data added.
    epslon_alpha, epslon_beta, epslon_gamma : float
        Transverse, overlap and total contact ratios [-].
    x_diff_from_al, x_diff_declared : float
        Quantities of the consistency check in ``validate`` [-].
    jbn : float
        Set only when ``al`` is imposed (see ``_set_alphatw_from_al``) [mm].

    Notes
    -----
    Sign convention: z2 is negative for the ring, and the code relies on it.
    The signed sum z_sum = z2 + z1 = -(abs(z2) - z1) then plays the role of the
    tooth-number difference; u = z2/z1 is negative; the ring quantities carry
    sgn(z2) = z2/abs(z2) (path-of-contact points, tip diameter); the rotation
    sense is not reversed (rotation_dir_out = +rotation_dir_in). With z2 > 0
    the same code adds the tooth numbers (z_sum = z2 + z1).

    Warnings
    --------
    - ISO_6336-1_2019, Table 2, footnote a: for internal gearing a, d, d_a and
      z2 are negative and z1 positive; here d, db, da of the ring are positive
      magnitudes and a = (d2 - d1)/2 > 0.
    - The profile-shift sum is used in the involute equation (the constructor
      passes x1 + x2 to ``_set_al_from_x_diff``, which divides by the signed
      sum z2 + z1, I02), while ``validate`` compares the declared difference
      x2 - x1 with an implied value computed with (z2 - z1) = -(abs(z2) + z1)
      and alphat of gear1 (I05). The two do not match: an al derived by the
      class and imposed again is in general reported as inconsistent.
    - Helical pairs: alphat of the ring (``InternalGear`` formulas) is used for
      alphatw and al, while pbt, betab and db1 are those of gear1. The two
      gears then have different transverse base pitches, and dl2 - dl1 is
      in general not equal to 2 al. At beta = 0 the two agree.
    - ``equalise_gs=True`` raises ``AttributeError`` (InternalGear has no
      ``zn``).

    References
    ----------
    .. [1] ISO_6336-1_2019, Table 2 footnote a (sign convention); 10.5.1,
       Formulae (104)-(112) (path of contact; one form printed, nothing on
       those pages says how it applies to internal gearing).
    .. [2] ISO 21771:2007, named in the code; not in the library; not checked
       against the source (source to find).
    .. [3] KHK Gear Technical Reference section 4.2, named in the code; not in
       the library; not checked against the source (source to find).
    .. [4] "Internal ring gears - design and considerations" (Gear Solutions),
       named in the code; not in the library (source to find).
    .. [5] MAAG Gear Book, named in the code; not in the library; not checked
       against the source (source to find).
    """

    def __init__(self,
             gear1: SpurHelicalGear,
             gear2: InternalGear,
             label: str = "",
             al: float | None = None,
             equalise_gs: bool = False,
             addendum_reduction: bool = False,
             driver: str = "gear1"):
        """
        Resolve al and alphatw, then build the working geometry.

        The arguments are described in the class docstring.

        Raises
        ------
        ValueError
            If the gears differ in module, normal pressure angle or helix
            angle, if ``driver`` is not "gear1" or "gear2", or if
            abs(z2) <= z1.
        """

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
        Working centre distance and pressure angle from a profile shift value.

        Parameters
        ----------
        x_diff : float
            Profile shift quantity entering the involute equation [-]. The
            constructor passes x1 + x2, although the argument is named for a
            difference.

        Notes
        -----
        Sets ``inv_alphatw`` [rad], ``alphatw`` [rad] and ``al`` [mm] (I02,
        I03)::

            z_sum = z2 + z1                (signed; = -(abs(z2) - z1) for z2 < 0)
            inv(alphatw) = inv(alphat) + 2*tan(alpha)*x_diff/z_sum
            alphatw = inverse involute, scipy brentq on [1e-6, 0.7] rad  (F10)
            al = mn*cos(alphat)*abs(z_sum) / (2*cos(beta)*cos(alphatw))

        alphat and beta are those of gear2 (the ring), alpha is that of gear1.
        For beta != 0 the ring alphat comes from the ``InternalGear`` formulas
        (see the class Warnings).
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
            """Residual ``inv_a - inv(xx)`` [rad]; zero when the angle xx [rad] has involute inv_a."""
            return inv_a - np.tan(xx) + xx

        self.alphatw = optimize.brentq(funOPT, 1e-6, 0.7, args=(self.inv_alphatw,))
        self.al = self.gear2.mn * np.cos(alphat) * abs(z_sum) / (2 * np.cos(beta) * np.cos(self.alphatw))

    def _set_alphatw_from_al(self) -> None:
        """
        Transverse working pressure angle and backlash from an imposed al.

        Notes
        -----
        Sets ``alphatw`` [rad] and ``jbn`` [mm] (``inv_alphatw`` is set later by
        the constructor). With z_sum = z2 + z1 (signed) (I03, I04)::

            alphatw = arccos(abs(z_sum)*mn*cos(alphat) / (2*al*cos(beta)))
            sum_x   = z_sum*(inv(alphatw) - inv(alphat)) / (2*tan(alpha))
            jbn     = (x1 + x2 - sum_x) * 2*mn*sin(alpha)

        ``jbn`` is labelled "backlash" in the code; its formula has no named
        source (source to find).
        """
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
        Profile shifts x1, x2 that equalise the maximum specific sliding.

        Adaptation of the Henriot procedure of the external pair
        (``SpurHelicalGearMeshing.correction_for_gs_equilibrium``) to the
        internal pair, with the reference centre distance a = (d2 - d1)/2.
        Not usable as written: see Warnings.

        Returns
        -------
        tuple of float
            (x1, x2) [-]; also stored in ``self.x1``, ``self.x2``.

        Raises
        ------
        AttributeError
            Always for an ``InternalGear`` as gear2: it has no ``zn``.
        RuntimeError
            If no sign change of the sliding difference is found.

        Notes
        -----
        As coded, with z1, z2 the ``zn`` of the gears and z2 signed (F30)::

            g_s1B = | 1 - (z1/z2) * sq / (a*sin(alphat) - sq) |,
                    sq = sgn(z2)*sqrt(ra2**2 - rb2**2)
            g_s2A = | (z2/z1) * sq1 / (z2*abs(z2)*a*sin(alphat) - sq1) - 1 |,
                    sq1 = sqrt(ra1**2 - rb1**2)
            ra = mt*zn/2 + mn*(haP + cP + x);   rb = mt*zn*cos(alphat)

        The structure of the two branches (z1 + z2 above or not above 60;
        scans of 400 and 600 points with brentq) is that of the external pair.

        Warnings
        --------
        - ``self.gear2.zn`` does not exist for ``InternalGear``.
        - The factor z2*abs(z2) multiplies a*sin(alphat) in g_s2A, and the
          calls of the helper functions in the z1 + z2 <= 60 branch pass four
          arguments to functions that take five (would raise ``TypeError``).
          Both are review items; neither is reached today.
        - As for the external pair, rb is written without the factor 1/2.

        References
        ----------
        .. [1] Henriot, "Traite theorique et pratique des engrenages", named in
           the code; not in the library; not checked against the source (source
           to find).
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
            """Specific sliding g_s1B [-] at the pinion (formula in the Notes); inf if the denominator is ~0."""
            sq  = z2 / abs(z2) * np.sqrt(max(ra2_v**2 - rb2_v**2, 0.0))
            den = a_v * np.sin(alphat_v) - sq
            if abs(den) < 1e-12:
                return np.inf
            return abs(1.0 - (z1 / z2) * sq / den)

        def _gs2A(ra1_v, rb1_v, a_v, alphat_v, z2):
            """Specific sliding g_s2A [-] at the ring (formula in the Notes); inf if the denominator is ~0."""
            sq  = np.sqrt(max(ra1_v**2 - rb1_v**2, 0.0))
            den = z2 * abs(z2) * a_v * np.sin(alphat_v) - sq
            if abs(den) < 1e-12:
                return np.inf
            return abs((z2 / z1) * sq / den - 1.0)

        def _solve_direct(z1_v, z2_v, ra1_v, ra2_v, rb1_v, rb2_v, a_v, alphat_v):
            """Solve gs1B' = gs2A' with x2=-x1, a'=a."""
            def equation(x):
                """Sliding difference g_s1B - g_s2A [-] for the shift x [-] (x1 = x, x2 = -x)."""
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
            """Transverse working pressure angle alphatw [rad] for the shifts x1 and x2_v [-]."""
            inv_aw = inv_alphat + 2.0 * np.tan(alpha) * (x1 + x2_v) / (z1 + z2)
            def funOPT(xx, inv_a):
                """Residual ``inv_a - inv(xx)`` [rad]; zero when the angle xx [rad] has involute inv_a."""
                return inv_a - np.tan(xx) + xx
            return optimize.brentq(funOPT, 1e-6, 0.7, args=(inv_aw,))

        def _al_from_alphatw(alphatw_v):
            """Working centre distance al [mm] for the working pressure angle alphatw_v [rad]."""
            return self.gear2.mn * np.cos(self.gear2.alphat) * abs(self.gear1.z + self.gear2.z) / (
                2 * np.cos(beta) * np.cos(alphatw_v))

        def equation_x2(x2_v):
            """Sliding difference g_s1B - g_s2A [-] for the ring shift x2_v [-] (x1 fixed)."""
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
        Working geometry, contact ratios, path of contact and loss factor.

        Builds ``gear1_w`` / ``gear2_w`` (shallow copies of the reference gears
        with k, al, alphatw, dl, rl, da, ra, alpha_a, epslon_a1/2, galpha added)
        and the pair attributes epslon_alpha/beta/gamma, galpha, ReqI, the
        points T1T2 ... rD2 of the path of contact, HV and the roughness
        combinations. Lengths [mm], angles [rad]. s2 = sgn(z2) = z2/abs(z2).

        Parameters
        ----------
        addendum_reduction : bool, optional
            True: compute k and subtract it from the tip diameter of gear1
            only. False (default): k = 0.

        Notes
        -----
        Equations as coded (m = mn of gear1; x1, x2 the resolved shifts)::

            k = (z2 - z1)/2 * ((inv(atw) - inv(at))/tan(a)
                - 1/cos(b) * (cos(at)/cos(atw) - 1))                   (I07)
            dl = db / cos(atw) for both gears;  rl = dl/2              (I06)
            da1 = d1 + 2*m*(haP1 + cP1 + x1 - k)                       (F17)
            da2 = d2 + s2*2*m*(haP2 + cP2 + x2)   (no k)               (I08)
            alpha_a = arccos(db/da) for both gears                     (F18)
            eps_a1 = z1*(tan(alpha_a1) - tan(atw))/(2 pi)              (F19)
            eps_a2 = abs(z2)*(tan(alpha_a2) - tan(atw))/(2 pi)         (I09)
            eps_alpha = eps_a1 - eps_a2                                (I09)
            eps_beta = b*tan(betab)/pbt, b = min(b1, b2)               (F20)
            eps_gamma = eps_alpha + eps_beta                           (F21)
            galpha = 1/2*(sqrt(da1**2 - db1**2)
                     + s2*sqrt(da2**2 - db2**2) - 2*al*sin(atw))       (I10)
            ReqI = 1/(1/(rl1 sin(atw)) + 1/(rl2 sin(atw)))             (F23)
            T1T2 = s2*al*sin(atw);  T1E = sqrt(ra1**2 - rb1**2)
            T2A = s2*sqrt(ra2**2 - rb2**2);  T2C = s2*rb2*tan(atw)     (I10)
            T1A, T1B, T1C, T1D, T2E, T2B, T2D, AE, AB, AC, AD, rA1 ... rD2
            as in SpurHelicalGearMeshing (F26, F27);  HV (F28); Ram, Rrms, RzS
            (F29) as there, with u = z2/z1 < 0 in HV.

        a, b, at, atw are alpha, beta, alphat, alphatw of gear1; pbt, betab
        are those of gear1; b is the face width [mm].

        Warnings
        --------
        - With the signed z2 < 0, (z2 - z1) in k is -(abs(z2) + z1).
        - gear1_w.galpha and gear2_w.galpha are per-gear terms rb*(tan(alpha_a)
          - tan(atw)); the pair value ``galpha`` is the separate expression (I10).
        - rA2, rB2, rD2: same mirrored naming as in ``SpurHelicalGearMeshing``
          (radii of gear2 at E, D, B, since the code uses AE, AD, AB).
        - For z2 < 0 (s2 = -1) the pair value ``galpha`` (I10) is not the
          length of the path of contact AE = T1E - T1A: with T1T2 =
          s2*al*sin(atw), AE contains +al*sin(atw) where ``galpha`` contains
          -al*sin(atw), so they differ by 2*al*sin(atw). The contact ratio
          epslon_alpha is computed from the addendum angles, not from
          ``galpha``.
        - As in the external pair: da_w is used where ISO_6336-1_2019, 10.5.1
          uses the active tip diameter d_Na, and pbt where the standard writes
          p_et; ReqI is the transverse reduced radius at the working pitch point
          (Formulae (96)-(97) with d_CP = d_w) without the 1/cos(beta_b) of
          Formula (95), with the plus-sign sum for the internal pair too
          (ISO_6336-1_2019, 10.1, Formula (96), prints one form and does not
          say on that page how it applies to internal gearing; by Table 2,
          footnote a, the diameters of internal gearing are negative in the
          standard). No radius of curvature along the
          line of action (G2), no load sharing (G3), no sliding or rolling
          speeds (G4).

        References
        ----------
        .. [1] ISO_6336-1_2019, Table 2 (eps_gamma), 10.1 Formulae (95)-(97),
           10.5.1 Formulae (104)-(112).
        .. [2] KHK Gear Technical Reference section 4.2 ("minus" for the
           internal contact ratio) and "MAAG Gear Book" (k, Ohlendorf loss
           factor), as named in the code; not in the library; not checked
           against the source (source to find).
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
            """Involute function inv(ang) = tan(ang) - ang [rad], ang in [rad]."""
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

        # ring gear tip diameter: no addendum reduction k; s2 = sgn(z2) = -1 for the ring
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
    # Forces: same definitions as the external pair, using the working
    # pitch radius of the DRIVING gear (see forces()). The attribution to
    # ISO 6336-1 has no support in the library (only F_t is printed there).
    # ------------------------------------------------------------------

    def forces(self, T_in: float, phi_deg: float = 0.0,
            rotation_dir_in: int = 1) -> dict[str, float]:
        """
        Tooth force components and output torque of the pair for a torque T_in.

        Same definitions as ``SpurHelicalGearMeshing.forces``, with the input
        torque taken on the gear named by ``driver``.

        Parameters
        ----------
        T_in : float
            Torque on the driving gear [N m].
        phi_deg : float, optional
            Angle of the line of centres gear1 -> gear2 in the shaft cross-
            section, from +Y toward +Z [deg].
        rotation_dir_in : int, optional
            +1 or -1, sense of rotation of the driving gear about +X [-]. (The
            keyword of the external pair is ``rotation_dir``.)

        Returns
        -------
        dict
            Ft, Fr, Fa, Fn : tangential, radial, axial and normal force [N];
            T_out : output torque on the driven gear [N m]; theta_Fr_driver,
            theta_Ft_driver, theta_Fr_driven, theta_Ft_driven : angles
            [deg, 0-360) from +Y toward +Z; rotation_dir_out : sense of
            rotation of the driven gear [-].

        Notes
        -----
        As coded (I11). For ``driver`` = "gear1": rl_in = gear1_w.rl [mm],
        beta_in = gear1.beta, u_eff = u; for "gear2": rl_in = gear2_w.rl,
        beta_in = gear2.beta, u_eff = 1/u::

            Ft = T_in / (rl_in/1000)
            Fr = Ft * tan(alphatw)
            Fa = Ft * tan(beta_in)         (reference helix angle)
            Fn = Ft / cos(alphatw)
            T_out = T_in * u_eff
            rotation_dir_out = -u_eff/abs(u_eff) * rotation_dir_in
            theta_* as in SpurHelicalGearMeshing.forces (F37; see its Warnings
            on the directions). Here rotation_dir_out = +rotation_dir_in, so
            theta_Ft_driven = theta_Ft_driver + 180.

        With z2 < 0, u < 0: T_out has the sign opposite to T_in times
        abs(u_eff), and rotation_dir_out = +rotation_dir_in (ring and pinion
        turn the same way). With z2 > 0 it would be -rotation_dir_in. Fa has no
        angle and no hand of helix exists (the direction is fixed by convention
        in the system layer). ``Fn`` is returned and read by no caller.

        Warnings
        --------
        - Differs from ISO_6336-1_2019, 4.2.2, Formula (1): F_t = 2000 T / d is
          written on the reference diameter; here Ft uses the working pitch
          radius.
        - Fn = Ft / cos(alphatw) is the coded form (as in the external pair);
          the form Ft / (cos(alphatw) cos(beta)) differs for helical gears;
          neither is printed in a library source.

        References
        ----------
        .. [1] ISO_6336-1_2019, 4.2.2, Formula (1)-(2).
        .. [2] F_r, F_a, F_n: the code names "ISO 6336-1", which prints none of
           them in the pages read (source to find).
        """
        if self.driver == "gear1":
            rl_in, beta_in, u_eff = self.gear1_w.rl, self.gear1.beta, self.u
        else:
            rl_in, beta_in, u_eff = self.gear2_w.rl, self.gear2.beta, 1.0 / self.u

        # F_t on the WORKING pitch radius of the driving gear; T [N m] / r [m] gives [N]
        Ft = T_in / (rl_in / 1000)
        Fr = Ft * np.tan(self.alphatw)
        Fa = Ft * np.tan(beta_in)
        Fn = Ft / np.cos(self.alphatw)
        T_out = T_in * u_eff

        theta_Fr_driver = phi_deg % 360.0
        theta_Ft_driver = (phi_deg + 90.0 * rotation_dir_in) % 360.0

        rotation_dir_out = - u_eff/abs(u_eff) * rotation_dir_in     # = +rotation_dir_in for z2 < 0 (u < 0)

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
            (2) |x2-x1 implied by al - declared| <= 1e-4, only when al was
            imposed, with (inv(alphatw) - inv(alphat))*(z2 - z1)/(2*tan(alpha))
            as the implied value (I05); (3) epsilon_alpha >= 1.0 (a value
            below 1.2 adds a message starting "Warning:" to the list, so
            ``validate_or_raise`` raises); (4) ``InternalGear.validate_mesh``
            (involute, trochoid and trimming interference) for gear1.z, gear1.x.

        Warnings
        --------
        The trimming test of ``InternalGear`` is (z - z1) < 2 with the signed z,
        which is True for every ring with z < 0 (and z1 > 0). As coded, the list is
        therefore never empty for a signed ring: it always contains the
        trimming message, and ``validate_or_raise`` raises for these pairs.
        The trochoid and involute messages depend on the geometry and are not
        covered by this statement.
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
        """Raise ``ValueError`` with all messages of ``validate()`` if any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Multi-line text with the main data of the pair (lengths [mm], angles [deg])."""
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
        """One-line representation: tooth numbers, module, helix angle, al, alphatw, eps_gamma."""
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