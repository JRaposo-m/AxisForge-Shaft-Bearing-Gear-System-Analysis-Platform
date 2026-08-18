"""
core/machine_elements/Bearings/types/ball_bearing/subtype/deep_groove_ball.py

Deep Groove Ball Bearing — ISO/TS 16281 point contact.

Extends Bearing base with:
  - internal geometry slots declared in __init__ (Bearing declares none)
  - reference_raceway_radii()       -> ri, re from Dw alone (ISO/TS 16281 §6.3)
  - setup_internal_geometry()       -> populates geometry slots on self
  - compute_hertz_point_contact()   -> returns cp [N/mm^(3/2)]
  - has_internal_geometry()         -> sentinel: Dw is not None

All contact-mechanics math is delegated to BallBearingGeometry
(ball_bearing/ball_bearing.py — pure math, no family knowledge), then
mirrored onto self so solvers always access geometry via bearing.Dw,
bearing.ri, bearing.cp, etc., uniformly across every subtype.

What THIS file owns: the ISO/TS 16281 §6.3 reference relation (what ri/re
default to when catalog data isn't known) and the idiomatic input for this
family (s — diametral operating clearance).

References:
  - ISO/TS 16281:2008 §5 eq.(2)-(11)    — point contact load distribution
  - ISO/TS 16281:2008 §6.3 eq.(72),(73) — reference raceway groove radii
"""

from __future__ import annotations
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.types.ball_bearing.ball_bearing_core import BallBearingGeometry

# Geometry attributes mirrored from BallBearingGeometry onto self.
_GEOMETRY_ATTRS = (
    "ri", "re", "Dw", "Dpw", "Z", "s", "E", "nu",
    "A", "alpha_0", "Ri", "phi_j", "gamma",
)

# ISO/TS 16281 §6.3 eq.(72)-(73) — reference raceway groove radii, as a
# fraction of ball diameter D_w. Same two numbers used verbatim by
# angular_contact.py (§6.3 groups both families under them) — duplicated
# there rather than imported, so neither subtype depends on the other.
_RI_OVER_DW = 0.52   # eq.(73) — inner ring
_RE_OVER_DW = 0.53   # eq.(72) — outer ring


class DeepGrooveBallBearing(Bearing):
    """
    DGBB — point contact, no thrust capacity in pure radial service.

    Usage
    -----
    # Manufacturer catalog data known (preferred — overrides the §6.3
    # reference geometry below):
    b = DeepGrooveBallBearing(d=40, D=80, b=18, C=29600, C0=17800,
                               designation="6208", position=50.0)
    b.setup_internal_geometry(ri=6.6, re=6.85, Dw=12.0, Dpw=60.0,
                               Z=9, E=206000, s=0.015)

    # Manufacturer data NOT known — ri/re fall back to ISO/TS 16281 §6.3
    # (r_e = 0.53*Dw, r_i = 0.52*Dw):
    b.setup_internal_geometry(Dw=12.0, Dpw=60.0, Z=9, E=206000, s=0.015)

    cp = b.compute_hertz_point_contact()
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("bearing_type", BearingType.DEEP_GROOVE_BALL)
        super().__init__(**kwargs)
        self._geometry = BallBearingGeometry()

        # --- internal geometry slots — None until setup_internal_geometry() ---
        self.ri      = None   # inner groove radius [mm]
        self.re      = None   # outer groove radius [mm]
        self.Dw      = None   # ball diameter [mm]
        self.Dpw     = None   # pitch circle diameter [mm]
        self.Z       = None   # number of balls
        self.s       = None   # diametral operating clearance [mm]
        self.E       = None   # Young's modulus [MPa]
        self.nu      = None   # Poisson's ratio
        self.A       = None   # radial clearance auxiliary: ri + re - Dw [mm]
        self.alpha_0 = None   # contact angle [rad]
        self.Ri      = None   # inner raceway radius to contact [mm]
        self.phi_j   = None   # rolling element angular positions [rad]
        self.gamma   = None   # Dw*cos(alpha_0)/Dpw 
        self.cp      = None   # Hertzian spring constant [N/mm^(3/2)]

        # Whether ri/re were supplied by the caller (catalog data) or fell
        # back to the ISO/TS 16281 §6.3 reference geometry.
        self.raceway_radii_from_reference: bool | None = None

    # ------------------------------------------------------------------
    # Geometry sentinel
    # ------------------------------------------------------------------

    def has_internal_geometry(self) -> bool:
        """True if setup_internal_geometry() has been called."""
        return self.Dw is not None

    # ------------------------------------------------------------------
    # Reference geometry — ISO/TS 16281 §6.3 eq.(72)-(73)
    # ------------------------------------------------------------------

    @staticmethod
    def reference_raceway_radii(Dw: float) -> tuple[float, float]:
        """
        Approximate cross-sectional raceway groove radii for a standard
        deep groove ball bearing, from ball diameter alone.

            r_i = 0.52 * D_w   eq.(73)   inner ring groove radius
            r_e = 0.53 * D_w   eq.(72)   outer ring groove radius

        Approximate reference values only (§6.1) — pass explicit ri/re to
        setup_internal_geometry() whenever catalog/drawing data is available.
        """
        return _RI_OVER_DW * Dw, _RE_OVER_DW * Dw

    # ------------------------------------------------------------------
    # Internal geometry setup
    # ------------------------------------------------------------------

    def setup_internal_geometry(self,
                                Dw: float,
                                Dpw: float,
                                Z: int,
                                E: float,
                                s: float,
                                nu: float = 0.3) -> None:
        """
        Parameters
        ----------
        Dw   : ball diameter [mm]
        Dpw  : pitch circle diameter [mm]
        Z    : number of balls
        E    : Young's modulus [MPa]
        s    : diametral operating clearance [mm] -- the only clearance input
            accepted for this family (alpha_0_deg is the angular-contact
            idiom, not this one).
        nu   : Poisson's ratio

        ri, re (inner/outer groove radii) are not inputs here -- always
        derived from reference_raceway_radii(Dw). Accepting arbitrary ri/re
        would let this constructor build something that isn't actually a
        DeepGrooveBallBearing in the catalog-conformity sense.
        """
        ri, re = self.reference_raceway_radii(Dw)

        self._geometry.setup(ri, re, Dw, Dpw, Z, E, nu, s=s)

        # Mirror onto self — solvers access bearing.Dw, bearing.ri, etc.
        for attr in _GEOMETRY_ATTRS:
            setattr(self, attr, getattr(self._geometry, attr))

    # ------------------------------------------------------------------
    # Hertzian spring constant
    # ------------------------------------------------------------------

    def compute_hertz_point_contact(self) -> float:
        """
        Hertzian spring constant c_p [N/mm^(3/2)]. ISO/TS 16281 eq.(5)-(11).

        Requires setup_internal_geometry() to have been called first.
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