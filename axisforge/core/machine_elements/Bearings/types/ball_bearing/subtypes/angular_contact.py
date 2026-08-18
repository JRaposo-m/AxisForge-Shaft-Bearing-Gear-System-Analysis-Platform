"""
core/machine_elements/Bearings/types/ball_bearing/subtype/angular_contact.py

Angular Contact Ball Bearing — ISO/TS 16281 point contact, same family of
formulas as DeepGrooveBallBearing (see deep_groove_ball.py), with one
difference that matters enough to warrant its own subtype: how the bearing
is normally specified.

A DGBB is normally specified by diametral operating clearance (s) — the
contact angle is a secondary, load-dependent quantity. An ACB is normally
specified by its nominal (free) contact angle (alpha_0_deg) — clearance is
the secondary quantity. Both feed the exact same BallBearingGeometry.setup()
— only the idiomatic input direction flips.

References:
  - ISO/TS 16281:2008 §5 eq.(2)-(11)    — point contact load distribution
  - ISO/TS 16281:2008 §6.3 eq.(72),(73) — reference raceway groove radii
"""

from __future__ import annotations
import numpy as np   

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.types.ball_bearing.ball_bearing import BallBearingGeometry

_GEOMETRY_ATTRS = (
    "ri", "re", "Dw", "Dpw", "Z", "s", "E", "nu",
    "A", "alpha_0", "Ri", "phi_j", "gamma",
)

# ISO/TS 16281 §6.3 eq.(72)-(73) — same formula as DeepGrooveBallBearing;
# see that file's module docstring for why it's duplicated, not imported.
_RI_OVER_DW = 0.52   # eq.(73) — inner ring
_RE_OVER_DW = 0.53   # eq.(72) — outer ring


class AngularContactBallBearing(Bearing):
    """
    ACB — point contact, carries axial load (unlike DGBB in pure radial
    service). Commonly the locating bearing on a shaft, alone or paired
    (DB/DF/DT); pairing arrangements are out of scope here — this class
    models a single ring set.

    Usage
    -----
    b = AngularContactBallBearing(d=40, D=80, b=18, C=35000, C0=26000,
                                   designation="7208B", position=50.0,
                                   contact_angle_deg=40.0)
    b.setup_internal_geometry(Dw=11.5, Dpw=60.0, Z=13, E=206000,
                              alpha_0_deg=40.0)

    # Clearance-based spec also accepted (same as DGBB):
    b.setup_internal_geometry(Dw=11.5, Dpw=60.0, Z=13, E=206000, s=0.01)

    cp = b.compute_hertz_point_contact()
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("bearing_type", BearingType.ANGULAR_CONTACT)
        super().__init__(**kwargs)
        self._geometry = BallBearingGeometry()

        self.ri      = None
        self.re      = None
        self.Dw      = None
        self.Dpw     = None
        self.Z       = None
        self.s       = None
        self.E       = None
        self.nu      = None
        self.A       = None
        self.alpha_0 = None
        self.Ri      = None
        self.phi_j   = None
        self.gamma   = None  
        self.cp      = None

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
        """(ri, re) from ball diameter alone — ISO/TS 16281 §6.3 eq.(72)-(73)."""
        return _RI_OVER_DW * Dw, _RE_OVER_DW * Dw

    # ------------------------------------------------------------------
    # Internal geometry setup
    # ------------------------------------------------------------------

    def setup_internal_geometry(self,
                                Dw: float,
                                Dpw: float,
                                Z: int,
                                E: float,
                                alpha_0_deg: float,
                                nu: float = 0.3) -> None:
        """
        Parameters
        ----------
        Dw          : ball diameter [mm]
        Dpw         : pitch circle diameter [mm]
        Z           : number of balls
        E           : Young's modulus [MPa]
        alpha_0_deg : nominal (free) contact angle [deg] -- the only contact
                    input accepted for this family. An angular contact ball
                    bearing is specified by its nominal contact angle (its
                    catalog suffix, e.g. 7208B -> 40 deg), not by a clearance
                    value -- s is the idiomatic input for deep groove ball
                    bearings, not this one, so it's deliberately not accepted
                    here.
        nu          : Poisson's ratio

        ri, re (inner/outer groove radii) are not inputs here -- always
        derived from reference_raceway_radii(Dw), same reasoning as
        DeepGrooveBallBearing.
        """
        ri, re = self.reference_raceway_radii(Dw)

        self._geometry.setup(ri, re, Dw, Dpw, Z, E, nu, alpha_0_deg=alpha_0_deg)

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
            if self.alpha_0 <= 0.0:
                errors.append(
                    f"{tag}: contact angle should be > 0 for an angular "
                    f"contact ball bearing (got {np.degrees(self.alpha_0):.2f} deg)"
                )

        return errors