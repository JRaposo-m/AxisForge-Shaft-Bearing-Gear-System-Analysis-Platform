"""
core/machine_elements/Bearings/types/roller_bearing/subtype/cylindrical_roller.py

Cylindrical Roller Bearing — ISO/TS 16281 line contact (NU/N-type, zero
nominal contact angle, no axial capacity).

All contact-mechanics math is delegated to RollerBearingGeometry
(roller_bearing/roller_bearing.py), then mirrored onto self so solvers
always access geometry via bearing.Dwe, bearing.x_k, bearing.cL, etc.,
uniformly across every subtype.


References:
  - ISO/TS 16281:2008 §5.2, eq.(34)-(46) — internal load distribution, line contact
  - ISO/TS 16281:2008 §6.2 — reference roller profile, eq.(42)-(44) (not yet available)
"""

from __future__ import annotations
import numpy as np

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.types.roller_bearing.roller_bearing import RollerBearingGeometry

_LOG_ARG_EPS = 1e-12  # floor for the log() argument in the profile function

_GEOMETRY_ATTRS = (
    "Dwe", "Lwe", "Dpw", "Z", "s", "n_s", "alpha_0",
    "x_k", "phi_j", "gamma",
)


class CylindricalRollerBearing(Bearing):
    """
    NU/N-type cylindrical roller bearing — line contact, no axial capacity.

    Always constructed with arrangement="floating": a cylindrical roller
    bearing (without a locating flange) cannot react an axial load, so it
    can never be the "locating" bearing on a shaft — that role stays with
    a ball (or tapered/spherical roller, once implemented) bearing.

    Usage
    -----
    b = CylindricalRollerBearing(d=20, D=47, b=14, C=28_500.0, C0=22_000.0,
                                 designation="NU204", position=100.0)
    b.setup_internal_geometry(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=40)
    cL = b.compute_line_contact_spring_constant()
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("bearing_type", BearingType.CYLINDRICAL_ROLLER)
        arrangement = kwargs.setdefault("arrangement", "floating")
        if arrangement not in ("floating", "non-locating"):
            raise ValueError(
                f"CylindricalRollerBearing(label={kwargs.get('label', '')!r}): "
                f"arrangement={arrangement!r} is not valid — a cylindrical roller "
                f"bearing (NU/N-type) has no flange to react axial load, so it "
                f"cannot be 'locating'. Use 'floating' or 'non-locating' for this "
                f"bearing, or a ball (or tapered/spherical roller) bearing for the "
                f"locating position on this shaft."
            )
        super().__init__(**kwargs)
        self._geometry = RollerBearingGeometry()

        self.Dwe     = None
        self.Lwe     = None
        self.Dpw     = None
        self.Z       = None
        self.s       = None
        self.n_s     = None
        self.alpha_0 = None
        self.x_k     = None
        self.phi_j   = None
        self.gamma   = None   # Dwe*cos(alpha_0)/Dpw 
        self.cL      = None
        self.cs      = None
        self._P_xk   = None   # cache for the P_xk property


    # ------------------------------------------------------------------
    # Geometry sentinel
    # ------------------------------------------------------------------

    def has_internal_geometry(self) -> bool:
        """True if setup_internal_geometry() has been called."""
        return self.Dwe is not None

    # ------------------------------------------------------------------
    # Reference roller profile — ISO/TS 16281 §6.2, eq.(42)-(44)
    # ------------------------------------------------------------------

    @property
    def P_xk(self) -> np.ndarray:
        """
        Roller profile P(x_k) [mm] -- ISO/TS 16281 Sec 6.2, eq.(42)-(44).
        Crowning depth subtracted (as 2*P(x_k)) from the raw lamina
        deflection in eq.(41) by the solver, so a purely cylindrical
        roller's theoretical edge-stress singularity doesn't appear.

        Lazily computed and cached on first access -- purely geometric
        (depends only on x_k, Dwe, Lwe, none of which change during the
        iterative solve), so it's computed once per setup_internal_geometry()
        call rather than every root-finding iteration.
        """
        if not self.has_internal_geometry():
            raise RuntimeError(
                f"CylindricalRollerBearing(label={self.label!r}): "
                f"P_xk requires setup_internal_geometry() to be called first."
            )
        if self._P_xk is None:
            self._P_xk = self._compute_reference_roller_profile()
        return self._P_xk

    def _compute_reference_roller_profile(self) -> np.ndarray:
        x_k, Dwe, Lwe = self.x_k, self.Dwe, self.Lwe
        P = np.zeros_like(x_k)

        if Lwe <= 2.5 * Dwe:
            arg = 1.0 - (2.0 * x_k / Lwe) ** 2
            arg = np.maximum(arg, _LOG_ARG_EPS)
            P = 0.000350 * Dwe * np.log(1.0 / arg)
        else:
            half_flat = (Lwe - 2.5 * Dwe) / 2.0
            edge = np.abs(x_k) > half_flat
            if np.any(edge):
                xe = x_k[edge]
                arg = 1.0 - ((2.0 * np.abs(xe) - (Lwe - 2.5 * Dwe)) / (2.5 * Dwe)) ** 2
                arg = np.maximum(arg, _LOG_ARG_EPS)
                P[edge] = 0.000500 * Dwe * np.log(1.0 / arg)
            # flat centre region (|x_k| <= half_flat) stays 0.0 -- eq.(43)

        return P

    def setup_internal_geometry(self, Dwe, Lwe, Dpw, Z, s, n_s=30, alpha_0_deg=0.0) -> None:
        self._geometry.setup(Dwe=Dwe, Lwe=Lwe, Dpw=Dpw, Z=Z, s=s,
                             n_s=n_s, alpha_0_deg=alpha_0_deg)
        for attr in _GEOMETRY_ATTRS:
            setattr(self, attr, getattr(self._geometry, attr))
        self._P_xk = None   # geometry changed -- invalidate cache
    # ------------------------------------------------------------------
    # Internal geometry setup
    # ------------------------------------------------------------------

    def setup_internal_geometry(self,
                                Dwe: float,
                                Lwe: float,
                                Dpw: float,
                                Z: int,
                                s: float,
                                n_s: int = 30,
                                alpha_0_deg: float = 0.0) -> None:
        """
        Parameters
        ----------
        Dwe         : roller diameter [mm]
        Lwe         : effective roller length [mm]
        Dpw         : pitch circle diameter [mm]
        Z           : number of rollers
        s           : diametral operating clearance [mm]  (idiomatic input
                      for this family — no §6 default exists for it, unlike
                      ri/re for the ball families)
        n_s         : number of laminae (>= 30 per ISO/TS 16281 §5.2.2)
        alpha_0_deg : nominal contact angle [deg] — 0.0 for a standard
                      radial NU/N-type bearing (default)
        """
        self._geometry.setup(Dwe=Dwe, Lwe=Lwe, Dpw=Dpw, Z=Z, s=s,
                             n_s=n_s, alpha_0_deg=alpha_0_deg)

        for attr in _GEOMETRY_ATTRS:
            setattr(self, attr, getattr(self._geometry, attr))

    # ------------------------------------------------------------------
    # Line-contact spring constant
    # ------------------------------------------------------------------

    def compute_line_contact_spring_constant(self) -> float:
        """
        Line-contact spring constant c_L [N/mm^(10/9)] — ISO/TS 16281
        eq.(35), and the per-lamina c_s — eq.(37).

        Requires setup_internal_geometry() to have been called first.
        """
        cL = self._geometry.hertz_spring_constant()
        self.cL = self._geometry.cL
        self.cs = self._geometry.cs
        return cL