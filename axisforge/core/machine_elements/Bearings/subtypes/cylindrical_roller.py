"""
Bearings/subtypes/cylindrical_roller.py

Cylindrical Roller Bearing — ISO/TS 16281 line contact (NU/N-type, zero
nominal contact angle, no axial capacity).

Extends Bearing base with:
  - internal geometry slots declared in __init__
  - setup_internal_geometry()             -> populates geometry slots on self
  - compute_line_contact_spring_constant() -> returns cL [N/mm^(10/9)]
  - has_internal_geometry()               -> sentinel: Dwe is not None

All internal geometry is delegated to RollerBearingGeometry, then mirrored
onto self for uniform solver access — mirrors DeepGrooveBallBearing's split
with BallBearingGeometry exactly (see subtypes/deep_groove_ball.py).

Consumed by ISO16281RollerSolver (bearings/ISO_16281/Roller_Bearing/
roller_bearing.py) via BearingType.CYLINDRICAL_ROLLER, dispatched through
RollingBearingSolver alongside any DEEP_GROOVE_BALL / ANGULAR_CONTACT
bearings on the same shaft.

References:
  - ISO/TS 16281:2008 §5.2, eq.(34)-(46) — internal load distribution, line contact
"""

from __future__ import annotations
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.geometry.roller_geometry import RollerBearingGeometry

# Geometry attributes mirrored from RollerBearingGeometry onto self —
# exactly the set ISO16281RollerSolver._REQUIRED_ATTRS checks for, plus x_k/phi_j.
_GEOMETRY_ATTRS = (
    "Dwe", "Lwe", "Dpw", "Z", "s", "n_s", "alpha_0",
    "x_k", "phi_j",
)


class CylindricalRollerBearing(Bearing):
    """
    NU/N-type cylindrical roller bearing — line contact, no axial capacity.

    Always constructed with arrangement="floating": a cylindrical roller
    bearing (without a locating flange) cannot react an axial load, so it
    can never be the "locating" bearing on a shaft — that role stays with
    a ball (or tapered/spherical roller, once implemented) bearing. Passing
    arrangement="locating" explicitly raises, rather than silently building
    a bearing that will trip warn_if_floating_loaded() the moment the shaft
    has any axial reaction at this node.

    Usage
    -----
    b = CylindricalRollerBearing(d=20, D=47, b=14, C=28_500.0, C0=22_000.0,
                                 designation="NU204", position=100.0)
    b.setup_internal_geometry(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=40)
    cL = b.compute_line_contact_spring_constant()
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("bearing_type", BearingType.CYLINDRICAL_ROLLER)
        kwargs.setdefault("contact_angle_deg", 0.0)
        arrangement = kwargs.setdefault("arrangement", "floating")
        if arrangement != "floating":
            raise ValueError(
                f"CylindricalRollerBearing(label={kwargs.get('label', '')!r}): "
                f"arrangement={arrangement!r} is not valid — a cylindrical roller "
                f"bearing (NU/N-type) has no flange to react axial load, so it "
                f"cannot be 'locating'. Use a ball (or tapered/spherical roller) "
                f"bearing for the locating position on this shaft."
            )
        super().__init__(**kwargs)
        self._geometry = RollerBearingGeometry()

        # --- internal geometry slots — line contact ---
        # Populated by setup_internal_geometry(); None until then.
        self.Dwe     = None   # roller diameter [mm]
        self.Lwe     = None   # effective roller length [mm]
        self.Dpw     = None   # pitch circle diameter [mm]
        self.Z       = None   # number of rollers
        self.s       = None   # diametral operating clearance [mm]
        self.n_s     = None   # number of laminae (>= 30)
        self.alpha_0 = None   # nominal contact angle [rad] — 0.0 once set
        self.x_k     = None   # lamina midpoint positions [mm]
        self.phi_j   = None   # roller angular positions [rad]
        self.cL      = None   # line-contact spring constant [N/mm^(10/9)]
        self.cs      = None   # per-lamina spring constant [N/mm^(10/9)]

    # ------------------------------------------------------------------
    # Geometry sentinel
    # ------------------------------------------------------------------

    def has_internal_geometry(self) -> bool:
        """True if setup_internal_geometry() has been called."""
        return self.Dwe is not None

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
        Compute and cache internal geometry.

        Parameters
        ----------
        Dwe         : roller diameter [mm]
        Lwe         : effective roller length [mm]
        Dpw         : pitch circle diameter [mm]
        Z           : number of rollers
        s           : diametral operating clearance [mm]
        n_s         : number of laminae (>= 30 per ISO/TS 16281 §5.2.2)
        alpha_0_deg : nominal contact angle [deg] — 0.0 for a standard
                      radial NU/N-type bearing (default)

        Populates on self:
            Dwe, Lwe, Dpw, Z, s, n_s, alpha_0, x_k, phi_j
        """
        self._geometry.setup(Dwe=Dwe, Lwe=Lwe, Dpw=Dpw, Z=Z, s=s,
                             n_s=n_s, alpha_0_deg=alpha_0_deg)

        # Mirror onto self — solvers access bearing.Dwe, bearing.x_k, etc.
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
        Populates self.cL, self.cs; returns cL.
        """
        cL = self._geometry.hertz_spring_constant()
        self.cL = self._geometry.cL
        self.cs = self._geometry.cs
        return cL