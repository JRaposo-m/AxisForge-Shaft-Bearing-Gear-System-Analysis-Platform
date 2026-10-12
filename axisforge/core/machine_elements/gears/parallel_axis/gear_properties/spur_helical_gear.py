"""
Geometry of one external cylindrical involute gear, spur or helical.

Computes the reference, base, tip and root circles, the pitches and the helix
angles of a gear defined by its basic rack, module, tooth number and profile
shift, and checks it for undercut. Used by the pair and planetary classes.
"""

from __future__ import annotations
import numpy as np


class SpurHelicalGear:
    """
    External cylindrical involute gear (spur when beta = 0, helical otherwise).

    One gear of a pair: it holds the basic-rack data, the tooth number and the
    profile shift, and derives the diameters, pitches and angles listed under
    Attributes. It is gear1/gear2 of ``SpurHelicalGearMeshing`` and the sun and
    planet of ``PlanetaryGearTrainMeshing``.

    Parameters
    ----------
    mn : float
        Normal module [mm].
    z : int
        Number of teeth [-]; positive (external gear).
    Ca : float, optional
        Tip relief. Stored only; no formula of the gear classes uses it. The
        unit is not stated in the code (ISO_6336-1_2019, Table 2, gives [um]
        for C_a and C_f). Default 0.0.
    Cf : float, optional
        Root relief. Same remarks as ``Ca``. Default 0.0.
    x : float, optional
        Profile shift coefficient [-]. Default 0.0.
    b : float, optional
        Face width [mm]. Default 0.0.
    alpha_n_deg : float, optional
        Normal pressure angle [deg]; stored in radians as ``alpha``.
        Default 20.0.
    beta_n_deg : float, optional
        Helix angle at the reference cylinder [deg]; stored in radians as
        ``beta``. The formulas use it as beta (mt = mn / cos(beta)); the
        "_n" in the name is not used in them. Default 0.0.
    haP : float, optional
        Addendum coefficient of the basic rack [-] (addendum = haP * mn).
        Default 1.0.
    cP : float, optional
        Tip clearance coefficient of the basic rack [-]. Default 0.25.
    rfP : float, optional
        Root fillet radius coefficient of the basic rack [-]
        (rhoF = rfP * mn). Default 0.38.
    Ra : float, optional
        Arithmetic mean roughness [um]. Default 0.8.
    Rq : float, optional
        Root mean square roughness [um]. Default 1.0.
    Rz : float, optional
        Mean peak-to-valley roughness [um]. Default 4.0.
    label : str, optional
        Name used in validation messages.
    material_id : str, optional
        Material key; stored only.
    position : float, optional
        Position of the gear [mm]; stored only, not used here.

    Attributes
    ----------
    alpha, beta : float
        Normal pressure angle and reference helix angle [rad].
    hfP : float
        Dedendum coefficient of the basic rack, haP + cP [-].
    mt : float
        Transverse module [mm].
    alphat : float
        Transverse pressure angle [rad].
    betab : float
        Base helix angle [rad].
    rhoF : float
        Root fillet radius of the basic rack [mm].
    pb, pt, pbt : float
        Normal base pitch, transverse pitch, transverse base pitch [mm].
    d, r : float
        Reference (pitch) diameter and radius [mm].
    db, rb : float
        Base diameter and radius [mm].
    da, ra : float
        Tip diameter and radius [mm].
    df, rf : float
        Root diameter and radius [mm].
    zn : float
        Virtual number of teeth of the equivalent spur gear [-].
    de : float
        mn * zn [mm] (see the Warnings on the symbol d_e).

    Notes
    -----
    Equations as coded (angles in radians; the ids refer to the review map)::

        mt     = mn / cos(beta)                                  (F01)
        alphat = arctan(tan(alpha) / cos(beta))                  (F02)
        betab  = arcsin(sin(beta) * cos(alpha))                  (F03)
        pb = pi*mn*cos(alpha);  pt = pi*mt;  pbt = pt*cos(alphat)  (F04)
        d  = mt*z;  db = d*cos(alphat)                           (F05)
        da = d + 2*mn*(haP + cP + x)                             (F06)
        df = d + 2*mn*(x - hfP),  hfP = haP + cP                 (F06)
        rhoF = rfP*mn;  zn = z / cos(beta)**3;  de = mn*zn       (F07)

    ``da`` carries no addendum reduction: the working tip diameter with the
    reduction k is built in ``SpurHelicalGearMeshing.gear_geometry``.

    Assumptions and conventions: external gear, z > 0; the helix angle is a
    magnitude (negative values are rejected by ``validate``), so there is no
    hand of helix; lengths in mm.

    Warnings
    --------
    - ISO_6336-1_2019, Table 2, lists h_aP and h_fP as lengths [mm]; here
      haP, cP, hfP are dimensionless coefficients multiplied by mn.
    - The symbol d_e of ISO_6336-1_2019, Table F.1, is the diameter of the
      circle through the outer point of single pair tooth contact; ``de`` here
      is mn * zn, a different quantity.
    - ``Ca`` and ``Cf`` are stored and not used by any gear formula.
    - ``da`` contains the clearance coefficient: da = d + 2 mn (haP + cP + x),
      so da - df = 2 mn (2 haP + 2 cP) and, for a pair at al = a, the distance
      al - ra1 - rf2 is zero (no bottom clearance). The contact ratios and the path of contact use this
      ``da``. ``InternalGear.trochoid_interference`` uses haP + x1 only. No
      library source prints the tip diameter (ISO 21771: source to find).

    References
    ----------
    .. [1] ISO_6336-1_2019, 9.3.1 NOTE, Formula (81): z_n ~ z / cos^3(beta),
       "approximately" (used for ``zn``); Table 2 and Table F.1 for the symbols
       and units.
    .. [2] ISO 21771:2007, named in the code; not in the library; not checked
       against the source (source to find). Needed for F01-F07.
    .. [3] ISO 53:2013, named in the code as the origin of haP = 1, cP = 0.25,
       rfP = 0.38; not in the library; ISO_6336-1_2019, clause 2, lists
       ISO 53:1998 instead; not checked against the source (source to find).
    """

    def __init__(self,
                 mn: float,
                 z: int,
                 Ca: float = 0.0,
                 Cf: float = 0.0,
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
        self.Ca          = Ca
        self.Cf          = Cf
        self.x           = x
        self.b           = b
        self.alpha_n_deg = alpha_n_deg
        self.alpha       = np.radians(alpha_n_deg)
        self.beta_n_deg  = beta_n_deg
        self.beta        = np.radians(beta_n_deg)
        self.haP         = haP
        self.cP          = cP
        self.hfP         = haP + cP
        self.rfP         = rfP
        self.Ra          = Ra
        self.Rq          = Rq
        self.Rz          = Rz

        # --- reference geometry (ISO 21771) ---
        self.mt     = self.mn / np.cos(self.beta)                     # transverse module [mm]
        self.alphat = np.arctan(np.tan(self.alpha) / np.cos(self.beta))  # transverse pressure angle
        self.betab  = np.arcsin(np.sin(self.beta) * np.cos(self.alpha))  # base helix angle
        self.rhoF   = rfP * mn                                        # fillet radius [mm]
        self.pb     = np.pi * mn * np.cos(self.alpha)                 # base pitch [mm]
        self.pt     = np.pi * self.mt                                 # transverse pitch [mm]
        self.pbt    = self.pt * np.cos(self.alphat)                   # transverse base pitch [mm]
        self.d      = self.mt * z                                     # pitch diameter [mm]
        self.r      = self.d / 2.0                                    # pitch radius [mm]
        self.db     = self.d * np.cos(self.alphat)                    # base diameter [mm]
        self.rb     = self.db / 2.0                                   # base radius [mm]
        self.da     = self.d + 2 * mn * (haP + cP + x)                     # addendum diameter [mm]
        self.ra     = self.da / 2.0                                   # addendum radius [mm]
        self.df     = self.d + 2 * mn * (x - self.hfP)                # dedendum diameter [mm]
        self.rf     = self.df / 2.0                                   # dedendum radius [mm]
        self.zn     = z / (np.cos(self.beta) ** 3)                    # equivalent number of teeth
        self.de     = mn * self.zn                                    # equivalent diameter [mm]

    # ------------------------------------------------------------------
    # Geometry checks
    # ------------------------------------------------------------------

    def undercutting(self) -> bool:
        """
        Undercut check of the tooth root against a minimum tooth number.

        Returns
        -------
        bool
            True if z < z_min, i.e. the gear is considered undercut.

        Notes
        -----
        ::

            z_min = 2 * cos(beta) / sin(alphat)**2 * (haP - x)       (F08)

        z_min is a real number evaluated with the transverse pressure angle
        alphat and the profile shift x, and compared with z.

        Warnings
        --------
        The formula has no named source in the code and none in the library
        (ISO_6336-1_2019, clause 1, only states that its methods do not apply
        when tip/root-fillet interference occurs) (source to find).
        """
        z_min = 2.0 * (np.cos(self.beta) / (np.sin(self.alphat) ** 2)) * (self.haP - self.x)
        return bool(self.z < z_min)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Check the inputs and the undercut condition.

        Returns
        -------
        list of str
            One message per violation; empty if the gear is valid. Checks:
            mn > 0, z >= 1, b >= 0, 0 < alpha_n_deg < 45, 0 <= beta_n_deg < 45
            (negative helix angles, i.e. a hand of helix, are rejected),
            Ra, Rq, Rz >= 0, and ``undercutting()`` is False.
        """
        errors: list[str] = []
        tag = self.label or self.__class__.__name__

        if self.mn <= 0:
            errors.append(f"{tag}: mn must be > 0, got {self.mn}")
        if self.z < 1:
            errors.append(f"{tag}: z must be >= 1, got {self.z}")
        if self.b < 0:
            errors.append(f"{tag}: b must be >= 0, got {self.b}")
        if not (0.0 < self.alpha_n_deg < 45.0):
            errors.append(
                f"{tag}: alpha_n_deg out of range (0, 45), got {self.alpha_n_deg}"
            )
        if not (0.0 <= self.beta_n_deg < 45.0):
            errors.append(
                f"{tag}: beta_n_deg out of range [0, 45), got {self.beta_n_deg}"
            )
        if self.Ra < 0:
            errors.append(f"{tag}: Ra must be >= 0, got {self.Ra}")
        if self.Rq < 0:
            errors.append(f"{tag}: Rq must be >= 0, got {self.Rq}")
        if self.Rz < 0:
            errors.append(f"{tag}: Rz must be >= 0, got {self.Rz}")
        if self.undercutting():
            errors.append(
                f"{tag}: gear is undercut — z={self.z}, mn={self.mn}, "
                f"alpha={self.alpha_n_deg}°, beta={self.beta_n_deg}°, x={self.x}"
            )

        return errors

    def validate_or_raise(self) -> None:
        """Raise ``ValueError`` with all messages of ``validate()`` if any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))