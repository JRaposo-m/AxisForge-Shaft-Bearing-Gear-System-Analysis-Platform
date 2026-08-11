"""
Bearings/subtypes/deep_groove_ball.py

Deep Groove Ball Bearing — ISO/TS 16281 point contact.

Extends Bearing base with:
  - internal geometry slots declared in __init__
  - setup_internal_geometry()       → populates geometry slots on self
  - compute_hertz_point_contact()   → returns cp [N/mm^(3/2)]
  - has_internal_geometry()         → sentinel: Dw is not None

All internal geometry is delegated to BallBearingGeometry,
then mirrored onto self for uniform solver access.

References:
  - ISO/TS 16281:2008 — internal load distribution, point contact
"""

from __future__ import annotations
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.geometry.ball_geometry import BallBearingGeometry

# Geometry attributes mirrored from BallBearingGeometry onto self
_GEOMETRY_ATTRS = (
    "ri", "re", "Dw", "Dpw", "Z", "s", "E", "nu",
    "A", "alpha_0", "Ri", "phi_j",
)


class DeepGrooveBallBearing(Bearing):
    """
    DGBB — point contact, no thrust capacity in pure radial service.

    Usage
    -----
    b = DeepGrooveBallBearing(d=40, D=80, b=18, C=29600, C0=17800,
                               designation="6208", position=50.0)

    # via diametral clearance
    b.setup_internal_geometry(ri=6.6, re=6.85, Dw=12.0, Dpw=60.0,
                               Z=9, E=206000, s=0.015)

    # or via free contact angle
    b.setup_internal_geometry(ri=6.6, re=6.85, Dw=12.0, Dpw=60.0,
                               Z=9, E=206000, alpha_0_deg=0.5)

    cp = b.compute_hertz_point_contact()
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("bearing_type", BearingType.DEEP_GROOVE_BALL)
        kwargs.setdefault("contact_angle_deg", 0.0)
        super().__init__(**kwargs)
        self._geometry = BallBearingGeometry(contact_angle=self.contact_angle)

        # --- internal geometry slots — ball / point contact ---
        # Populated by setup_internal_geometry(); None until then.
        self.ri      = None   # inner groove radius [mm]
        self.re      = None   # outer groove radius [mm]
        self.Dw      = None   # ball diameter [mm]
        self.Dpw     = None   # pitch circle diameter [mm]
        self.Z       = None   # number of balls
        self.s       = None   # diametral operating clearance [mm]
        self.E       = None   # Young's modulus [MPa]
        self.nu      = None   # Poisson's ratio
        self.A       = None   # radial clearance auxiliary: ri + re - Dw [mm]
        self.alpha_0 = None   # free contact angle [rad]
        self.Ri      = None   # inner raceway radius to contact [mm]
        self.phi_j   = None   # rolling element angular positions [rad]
        self.cp      = None   # Hertzian spring constant [N/mm^(3/2)]

    # ------------------------------------------------------------------
    # Geometry sentinel
    # ------------------------------------------------------------------

    def has_internal_geometry(self) -> bool:
        """True if setup_internal_geometry() has been called."""
        return self.Dw is not None

    # ------------------------------------------------------------------
    # Internal geometry setup
    # ------------------------------------------------------------------

    def setup_internal_geometry(self,
                                ri: float,
                                re: float,
                                Dw: float,
                                Dpw: float,
                                Z: int,
                                E: float,
                                nu: float = 0.3,
                                **kwargs) -> None:
        """
        Compute and cache internal geometry.

        Parameters
        ----------
        ri, re   : inner/outer groove radii [mm]
        Dw       : ball diameter [mm]
        Dpw      : pitch circle diameter [mm]
        Z        : number of balls
        E        : Young's modulus [MPa]
        nu       : Poisson's ratio
        **kwargs : clearance — exactly one of:
                     s           : diametral operating clearance [mm]
                     alpha_0_deg : free contact angle [°]

        Populates on self:
            ri, re, Dw, Dpw, Z, s, E, nu, A, alpha_0, Ri, phi_j
        """
        self._geometry.setup(ri, re, Dw, Dpw, Z, E, nu, **kwargs)

        # Mirror onto self — solvers access bearing.Dw, bearing.ri, etc.
        for attr in _GEOMETRY_ATTRS:
            setattr(self, attr, getattr(self._geometry, attr))

    # ------------------------------------------------------------------
    # Hertzian spring constant
    # ------------------------------------------------------------------

    def compute_hertz_point_contact(self) -> float:
        """
        Hertzian spring constant c_p [N/mm^(3/2)].
        ISO/TS 16281 eq.(5)–(11).

        Requires setup_internal_geometry() to have been called first.

        Returns
        -------
        cp : float
        """
        if not self.has_internal_geometry():
            raise RuntimeError(
                f"Bearing '{self.label or self.designation}': "
                "call setup_internal_geometry() before compute_hertz_point_contact()."
            )
        self.cp = self._geometry.hertz_spring_constant()
        return self.cp

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors = super().validate()
        tag = self.label or self.designation or self.__class__.__name__

        if self.has_internal_geometry():
            if self.ri <= self.Dw / 2.0:
                errors.append(f"{tag}: ri must be > Dw/2 (conformity < 0.5)")
            if self.re <= self.Dw / 2.0:
                errors.append(f"{tag}: re must be > Dw/2 (conformity < 0.5)")
            if self.s < 0.0:
                errors.append(f"{tag}: diametral clearance s must be >= 0")

        return errors