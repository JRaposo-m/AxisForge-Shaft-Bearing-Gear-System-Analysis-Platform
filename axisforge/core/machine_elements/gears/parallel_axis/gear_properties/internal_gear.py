"""
Geometry of one internal (ring) cylindrical involute gear, spur or helical.

Computes the reference, base, tip and root circles and the pitches of a gear
with internal teeth, evaluates the involute at an arbitrary diameter, and
provides the interference checks against a mating external gear. The tooth
number z is signed (z < 0 for internal teeth); see the class docstring.
"""

from __future__ import annotations
import numpy as np


class InternalGear:
    """
    Internal (ring) cylindrical involute gear, spur or helical.

    Used as gear2 of ``InternalGearMeshing`` and as the ring of
    ``PlanetaryGearTrainMeshing``. The tooth number carries a sign: z < 0 means
    internal teeth (the planetary class requires it). All diameters, radii and
    pitches stored here are positive magnitudes (abs(z) is used in d); the
    sign of z only selects the direction of the addendum and dedendum offsets.

    Parameters
    ----------
    mn : float
        Normal module [mm].
    z : int
        Signed number of teeth [-]; negative for internal teeth. Must not be
        zero.
    x : float, optional
        Profile shift coefficient [-]. Default 0.0.
    b : float, optional
        Face width [mm]. Default 0.0.
    alpha_n_deg : float, optional
        Normal pressure angle [deg]; stored in radians as ``alpha``.
        Default 20.0.
    beta_n_deg : float, optional
        Helix angle at the reference cylinder [deg]; stored in radians as
        ``beta``. Default 0.0. ``validate`` does not check its range.
    haP : float, optional
        Addendum coefficient of the basic rack [-]. Default 1.0.
    cP : float, optional
        Tip clearance coefficient of the basic rack [-]. Default 0.25.
    rfP : float, optional
        Root fillet radius coefficient of the basic rack [-]; stored only.
        Default 0.38.
    Ra, Rq, Rz : float, optional
        Arithmetic mean, root mean square and mean peak-to-valley roughness
        [um]. Defaults 0.8, 1.0, 4.0.
    label : str, optional
        Name used in validation messages.
    material_id : str, optional
        Material key; stored only.
    position : float, optional
        Position of the gear [mm]; stored only.

    Attributes
    ----------
    alpha, beta : float
        Normal pressure angle and reference helix angle [rad].
    hfP : float
        Dedendum coefficient, haP + cP [-].
    mt : float
        Transverse module [mm].
    mx : float
        Axial module, mt / tan(beta) [mm]; infinite for beta = 0.
    pn, pt, px : float
        Normal circular pitch, transverse pitch, axial pitch [mm]
        (px, as pi*mx, is infinite for beta = 0).
    pz : float
        Lead, abs(z) * mt * pi / tan(beta) [mm]; infinite for beta = 0.
    d, r : float
        Reference diameter and radius, magnitudes [mm].
    beta_b : float
        Base helix angle [rad].
    db, rb : float
        Base diameter and radius [mm].
    alphat : float
        Transverse pressure angle [rad].
    tau : float
        Angular pitch, 2 pi / abs(z) [rad].
    pbt, pbn : float
        Transverse base pitch and normal base pitch [mm].
    dv, rv : float
        Diameter and radius of the profile-shifted reference circle [mm].
    da, ra : float
        Tip diameter and radius [mm].
    df, rf : float
        Root diameter and radius [mm].

    Notes
    -----
    Equations as coded (s = z / abs(z) = +1 or -1; angles in radians)::

        mt = mn / cos(beta);  pn = pi*mn;  pt = pi*mt          (F01, F04)
        d  = mt * abs(z)                                       (F05i)
        beta_b = arcsin(sin(beta) / cos(alpha))                (F03i)
        db = abs(z) * mn * cos(alpha) / cos(beta_b)            (F05i)
        alphat = arccos(db / d)                                (F02i)
        pbt = pt*cos(alphat);  pbn = pn*cos(alpha)             (F04)
        dv = d + 2*s*x*mn                                      (F06i)
        da = d + 2*mn*s*(x + haP + cP)                         (F06i)
        df = d - 2*mn*s*(hfP - x)                              (F06i)

    For z < 0: da = d - 2 mn (x + haP + cP) < d and df > d, so the tip circle
    lies inside the reference circle, as for a ring gear. ``validate`` accepts
    only da < d.

    Warnings
    --------
    - Sign convention: ISO_6336-1_2019, Table 2, footnote a, gives d, d_a, a
      and z2 a negative sign for internal gearing (z1 positive, "all
      calculated diameters have a negative sign"). Here z is signed but d, db,
      da, df, dv are positive magnitudes.
    - Base helix angle and base diameter differ from ``SpurHelicalGear``: there
      betab = arcsin(sin(beta) cos(alpha)) and db = d cos(alphat) with
      alphat = arctan(tan(alpha)/cos(beta)); here beta_b = arcsin(sin(beta) /
      cos(alpha)) and db = abs(z) mn cos(alpha) / cos(beta_b). The two agree
      only at beta = 0. ISO_6336-1_2019 prints neither form (source to find).
      For beta != 0 the lead on the base cylinder, pi*db/tan(beta_b), then
      differs from the lead on the reference cylinder, pz = pi*d/tan(beta);
      in ``SpurHelicalGear`` the two are equal.
    - Tip diameter: da = d + 2 mn s (x + haP + cP) contains the clearance
      coefficient cP (see ``SpurHelicalGear``, Warnings); da - df = 2 mn
      (2 haP + 2 cP) in magnitude. ``trochoid_interference`` uses haP + x1
      only for the mating external gear. No library source prints the tip
      diameter (ISO 21771: source to find).
    - There are no ``zn``, ``betab`` or ``k`` attributes (the base helix angle
      is ``beta_b``); ``InternalGearMeshing(..., equalise_gs=True)`` reads
      ``zn`` of the internal gear and raises ``AttributeError``.

    References
    ----------
    .. [1] ISO_6336-1_2019, Table 2, footnote a (sign convention for internal
       gearing).
    .. [2] ISO 21771:2007, named in the code; not in the library; not checked
       against the source (source to find). Needed for F01-F06i.
    .. [3] KHK Gear Technical Reference, section 4.2 (internal gear
       dimensions), named in the code; not in the library; not checked against
       the source (source to find).
    .. [4] ISO 53, named in the code for haP = 1, cP = 0.25, rfP = 0.38; not in
       the library (source to find).
    """

    def __init__(self,
                 mn: float,
                 z: int,
                 x: float = 0.0,
                 b: float = 0.0,
                 alpha_n_deg: float = 20.0,
                 beta_n_deg: float = 0.0,
                 haP: float = 1.0,
                 cP: float = 0.25,
                 rfP: float = 0.38,
                 Ra: float = 0.8,
                 Rq: float = 1.0,
                 Rz: float = 4.0,
                 label: str = "",
                 material_id: str = "",
                 position: float = 0.0):
        """
        Store the inputs and compute the reference geometry.

        The arguments and the derived attributes are described in the class
        docstring.
        """
        # --- metadata ---
        self.label       = label
        self.material_id = material_id
        self.position    = position
        
        # --- input parameters ---
        self.mn          = mn
        self.z           = z
        self.x           = x
        self.b           = b
        self.alpha_n_deg = alpha_n_deg
        self.alpha       = np.radians(alpha_n_deg)
        self.beta_n_deg  = beta_n_deg
        self.beta        = np.radians(beta_n_deg)
        self.haP         = haP
        self.cP          = cP
        self.hfP         = haP + cP          # = 1.25 for ISO 53 standard
        self.rfP         = rfP
        self.Ra          = Ra
        self.Rq          = Rq
        self.Rz          = Rz

        # --- reference geometry (ISO 21771) ---
        self.mt   = mn / np.cos(self.beta)
        self.mx   = self.mt / np.tan(self.beta) if self.beta != 0 else np.inf
        self.pn   = np.pi * self.mn                         # circular pitch, normal plane [mm]
        self.pt   = np.pi * self.mt                          # transverse pitch [mm]
        self.pz   = abs(z) * self.mt * np.pi / np.tan(self.beta) if self.beta != 0 else np.inf
        self.px   = np.pi * self.mx

        self.d    = self.mt * abs(z)                         # reference (pitch) diameter [mm]
        self.r    = self.d / 2.0                              # reference radius [mm]

        self.beta_b = np.arcsin(np.sin(self.beta) / np.cos(self.alpha))

        self.db     = abs(z) * self.mn * np.cos(self.alpha) / np.cos(self.beta_b)
        self.rb     = self.db / 2
        self.alphat = np.arccos(self.db / self.d)
        self.tau    = 2 * np.pi / abs(z)                     # angular pitch [rad]

        self.pbt  = self.pt * np.cos(self.alphat)             # transverse base pitch [mm]
        self.pbn  = self.pn * np.cos(self.alpha)               # base pitch, normal plane [mm]

        self.dv   = self.d + 2 * z / abs(z) * self.x * self.mn
        self.rv   = self.dv / 2
        self.da   = self.d + 2 * self.mn * z / abs(z) * (self.x + self.haP + self.cP)
        self.ra   = self.da / 2
        self.df   = self.d - 2 * self.mn * z / abs(z) * (self.hfP - self.x)
        self.rf   = self.df / 2

    def values_at(self, dy: float) -> dict:
        """
        Involute quantities at an arbitrary diameter dy of the flank.

        Parameters
        ----------
        dy : float
            Diameter at which the involute is evaluated [mm]; dy >= db is
            needed (otherwise arccos gives nan).

        Returns
        -------
        dict
            beta_y : helix angle at dy [rad]; alpha_yt : transverse pressure
            angle at dy [rad]; alpha_yn : normal pressure angle at dy [rad];
            L_y : radius of curvature of the profile at dy, signed with
            z / abs(z) [mm]; inv_alpha_yt : tan(alpha_yt) - alpha_yt [rad];
            p_yt : transverse pitch at dy [mm]; p_yn : normal pitch at dy [mm].

        Notes
        -----
        ::

            beta_y   = arctan(tan(beta) * dy / d)
            alpha_yt = arccos(db / dy)
            alpha_yn = arctan(tan(alpha_yt) * cos(beta_y))
            L_y      = sgn(z) * db * tan(alpha_yt) / 2               (I12)
            p_yt     = dy * pt / d;   p_yn = p_yt * cos(beta_y)

        The magnitude of L_y equals the transverse radius of curvature of
        ISO_6336-1_2019, 10.1, Formula (97), rho_t = sqrt((d_CP^2 - d_b^2)/4),
        with d_CP = dy. The other relations are not printed in the library
        (source to find).
        """
        beta_y       = np.arctan(np.tan(self.beta) * dy / self.d)               # helix angle at dy
        alpha_yt     = np.arccos(self.db / dy)                                   # transverse pressure angle
        alpha_yn     = np.arctan(np.tan(alpha_yt) * np.cos(beta_y))              # normal pressure angle
        L_y          = (self.z / abs(self.z)) * self.db * np.tan(alpha_yt) / 2   # radius of curvature
        inv_alpha_yt = np.tan(alpha_yt) - alpha_yt
        p_yt         = dy * self.pt / self.d
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

    # ------------------------------------------------------------------ #
    # Interference checks                                                  #
    # ------------------------------------------------------------------ #

    def tip_below_base_circle(self) -> bool:
        """
        Check that the tip circle does not fall on or inside the base circle.

        Returns
        -------
        bool
            True if da <= db, i.e. there is no involute profile at the tip
            (invalid geometry). The valid condition is da > db.
        """
        return self.da <= self.db

    def involute_interference(self, z1: int, x1: float = 0.0) -> bool:
        """
        Involute interference check against a mating external gear.

        Parameters
        ----------
        z1 : int
            Number of teeth of the mating external gear [-].
        x1 : float, optional
            Profile shift coefficient of the mating external gear [-].

        Returns
        -------
        bool
            True when involute interference is present (alphaa2 > alphaw).

        Notes
        -----
        ::

            inv(alphaw) = inv(alpha) + 2*tan(alpha)*(x1 - x)/(z - z1)  (I13)
            alphaa2     = arccos(db / da)

        alphaw is obtained from inv(alphaw) by ``_solve_inv`` (Newton). The
        normal pressure angle alpha is used in inv(alpha) (not alphat), and
        z is the signed tooth number, so for z < 0 the denominator is
        -(abs(z) + z1).

        References
        ----------
        .. [1] KHK Gear Technical Reference, section 4.2, and the article
           "Internal ring gears - design and considerations" (Gear Solutions),
           both named in the code; not in the library; not checked against the
           source (source to find).
        """
        # working pressure angle from involute function
        inv_alpha = np.tan(self.alpha) - self.alpha
        inv_alphaw = inv_alpha + 2 * np.tan(self.alpha) * (x1 - self.x) / (self.z - z1)
        # solve inv(alphaw) numerically via Newton-Raphson
        alphaw = self._solve_inv(inv_alphaw)
        alphaa2 = np.arccos(self.db / self.da)          # pressure angle at tip of internal gear
        return alphaa2 > alphaw

    def trochoid_interference(self, z1: int, x1: float = 0.0) -> bool:
        """
        Trochoid (flank) interference check against a mating external gear.

        Parameters
        ----------
        z1 : int
            Number of teeth of the mating external gear [-].
        x1 : float, optional
            Profile shift coefficient of the mating external gear [-].

        Returns
        -------
        bool
            True when trochoid interference is present (alphaa1 > alphaw).

        Notes
        -----
        ::

            d1  = mn * z1;  db1 = d1 * cos(alpha)
            da1 = d1 + 2*mn*(haP + x1)
            alphaa1 = arccos(db1 / da1)
            inv(alphaw) = inv(alpha) + 2*tan(alpha)*(x1 - x)/(z - z1)  (I13)

        The mating gear is taken as a spur gear: d1 uses mn (not mt) and the
        tip diameter da1 has no tip clearance cP, unlike ``SpurHelicalGear.da``.
        z is the signed tooth number of this gear (see
        ``involute_interference``).

        References
        ----------
        .. [1] KHK Gear Technical Reference, section 4.2, named in the code; not
           in the library; not checked against the source (source to find).
        """
        d1  = self.mn * z1
        db1 = d1 * np.cos(self.alpha)

        # working pressure angle
        inv_alpha  = np.tan(self.alpha) - self.alpha
        inv_alphaw = inv_alpha + 2 * np.tan(self.alpha) * (x1 - self.x) / (self.z - z1)
        alphaw     = self._solve_inv(inv_alphaw)

        da1    = d1 + 2 * self.mn * (self.haP + x1)    # tip diameter of external gear
        alphaa1 = np.arccos(db1 / da1)
        return alphaa1 > alphaw

    def trimming_interference(self, z1: int) -> bool:
        """
        Trimming interference check: tooth-number difference below a limit.

        Parameters
        ----------
        z1 : int
            Number of teeth of the mating external gear [-].

        Returns
        -------
        bool
            True when (z - z1) < 2, with z the signed tooth number of this gear.

        Warnings
        --------
        The exact limit depends on the pinion-type cutter geometry (KHK Table 4.7,
        named in the code; not in the library; not checked against the source
        (source to find)); 2 is the quick conservative value of the code. For
        z < 0, z - z1 is negative and the test is always True.
        """
        return (self.z - z1) < 2

    def validate_mesh(self, z1: int, x1: float = 0.0) -> list[str]:
        """
        Interference checks of this gear against a given mating external gear.

        Separate from ``validate()`` because interference is a property of the
        pair, not of one gear.

        Parameters
        ----------
        z1 : int
            Number of teeth of the mating external gear [-].
        x1 : float, optional
            Profile shift coefficient of the mating external gear [-].

        Returns
        -------
        list of str
            One message per interference found (``involute_interference``,
            ``trochoid_interference``, ``trimming_interference``); empty if none.
        """
        errors: list[str] = []
        tag = self.label or self.__class__.__name__

        if self.involute_interference(z1, x1):
            errors.append(
                f"{tag}: involute interference with mating gear (z1={z1}, x1={x1})."
            )
        if self.trochoid_interference(z1, x1):
            errors.append(
                f"{tag}: trochoid (flank) interference with mating gear (z1={z1}, x1={x1})."
            )
        if self.trimming_interference(z1):
            errors.append(
                f"{tag}: trimming interference — tooth difference z2-z1={self.z - z1} < 2."
            )

        return errors

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate(self) -> list[str]:
        """
        Check the inputs and the tip circle of the gear alone.

        Returns
        -------
        list of str
            One message per violation; empty if valid. Checks: mn > 0, z != 0,
            b >= 0, 0 < alpha_n_deg < 45, Ra, Rq, Rz >= 0, da < d (for the default
            rack, x + haP + cP > 0, only true for z < 0) and ``tip_below_base_circle()`` is False. The helix
            angle is not checked.
        """
        errors: list[str] = []
        tag = self.label or self.__class__.__name__

        if self.mn <= 0:
            errors.append(f"{tag}: mn must be > 0, got {self.mn}")
        if self.z == 0:
            errors.append(f"{tag}: z must be != 0, got {self.z}")
        if self.b < 0:
            errors.append(f"{tag}: b must be >= 0, got {self.b}")
        if not (0.0 < self.alpha_n_deg < 45.0):
            errors.append(f"{tag}: alpha_n_deg out of range (0, 45), got {self.alpha_n_deg}")
        if self.Ra < 0:
            errors.append(f"{tag}: Ra must be >= 0, got {self.Ra}")
        if self.Rq < 0:
            errors.append(f"{tag}: Rq must be >= 0, got {self.Rq}")
        if self.Rz < 0:
            errors.append(f"{tag}: Rz must be >= 0, got {self.Rz}")
        if self.da >= self.d:
            errors.append(
                f"{tag}: tip diameter da={self.da:.4f} ≥ pitch diameter d={self.d:.4f}. "
                "Internal gear addendum must reduce da below d."
            )
        if self.tip_below_base_circle():
            errors.append(
                f"{tag}: tip circle (da={self.da:.4f}) ≤ base circle (db={self.db:.4f}). "
                "No valid involute at tooth tip — increase z or x, or reduce haP."
            )

        return errors

    def validate_or_raise(self) -> None:
        """Raise ``ValueError`` with all messages of ``validate()`` if any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))

    # ------------------------------------------------------------------ #
    # Helper                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _solve_inv(inv_target: float, alpha0: float = 0.3491) -> float:
        """
        Inverse involute: the angle alpha with tan(alpha) - alpha = inv_target.

        Parameters
        ----------
        inv_target : float
            Value of the involute function [rad].
        alpha0 : float, optional
            Starting angle [rad]; 0.3491 rad = 20 deg.

        Returns
        -------
        float
            Angle alpha [rad].

        Notes
        -----
        Newton-Raphson on f = tan(alpha) - alpha - inv_target with
        f' = tan(alpha)**2, at most 50 iterations, stopped when abs(f) < 1e-12.
        There is no convergence test after the loop. ``SpurHelicalGearMeshing``
        and ``InternalGearMeshing`` solve the same equation with
        ``scipy.optimize.brentq`` on [1e-6, 0.7] rad (F10 against F10i).
        """
        alpha = alpha0
        for _ in range(50):
            f  = np.tan(alpha) - alpha - inv_target
            df = np.tan(alpha) ** 2          # d/dα [tan(α) - α] = tan²(α)
            alpha -= f / df
            if abs(f) < 1e-12:
                break
        return alpha