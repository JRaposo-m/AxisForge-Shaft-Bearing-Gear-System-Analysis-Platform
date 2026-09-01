"""
fixtures/construction/bearings/ball_radial_fixture.py

Construction-stage factories for the three point-contact, radial-duty
ball bearing families (core/machine_elements/bearings/families/ball_bearing/radial/):
DeepGrooveBallFamily, AngularContactFamily, SelfAligningBallFamily.

Each factory is a thin, direct wrapper around Bearing.assemble() -- no
Spec dataclass (same choice as gears). It builds the BearingCatalog from
plain kwargs, builds the family instance, forwards the geometry kwargs
verbatim to family.assemble_geometry() via Bearing.assemble(geometry=...),
and always enables that family's own single capability
(analyses={"point_contact": True}) -- Construction does not decide
analysis dispatch, that's Resolution/ElementAnalysis's concern; every
core usage example already assembles with its own capability enabled.

Geometry kwargs mirror each family's assemble_geometry() signature
exactly (verified against the real core source, not inferred):
  DeepGrooveBallFamily  : Dw, Dpw, Z, E, s,            nu=0.3, i=1
  AngularContactFamily  : Dw, Dpw, Z, E, alpha_0_deg,  nu=0.3, i=1
  SelfAligningBallFamily: Dw, Dpw, Z, E, alpha_0_deg,  nu=0.3, i=1

Do not modify these three factories' validation behaviour -- ri/re
bounds, alpha_0_deg range, i in {1,2} are all enforced inside each
family's own assemble_geometry(), left untouched here on purpose (same
principle as the gear fixtures leaving .validate() to the caller).
"""
from __future__ import annotations

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.bearings.families.ball_bearing.radial.subtypes.deep_groove import (
    DeepGrooveBallFamily,
)
from axisforge.core.machine_elements.bearings.families.ball_bearing.radial.subtypes.angular_contact import (
    AngularContactFamily,
)
from axisforge.core.machine_elements.bearings.families.ball_bearing.radial.subtypes.self_aligning import (
    SelfAligningBallFamily,
)


def make_deep_groove_ball_bearing(
    *,
    d: float,
    D: float,
    Dw: float,
    Dpw: float,
    Z: int,
    E: float,
    s: float,
    b: float = 0.0,
    C: float = 0.0,
    C0: float = 0.0,
    designation: str = "",
    label: str = "",
    position: float = 0.0,
    arrangement: str = "locating",
    nu: float = 0.3,
    i: int = 1,
) -> Bearing:
    """DGBB -- clearance-specified (s), point contact, radial duty."""
    catalog = BearingCatalog(
        d=d, D=D, b=b, C=C, C0=C0, designation=designation,
        label=label, position=position, arrangement=arrangement,
    )
    return Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=catalog,
        geometry=dict(Dw=Dw, Dpw=Dpw, Z=Z, E=E, s=s, nu=nu, i=i),
        analyses={"point_contact": True},
    )


def make_angular_contact_bearing(
    *,
    d: float,
    D: float,
    Dw: float,
    Dpw: float,
    Z: int,
    E: float,
    alpha_0_deg: float,
    b: float = 0.0,
    C: float = 0.0,
    C0: float = 0.0,
    designation: str = "",
    label: str = "",
    position: float = 0.0,
    arrangement: str = "locating",
    nu: float = 0.3,
    i: int = 1,
) -> Bearing:
    """ACB -- contact-angle-specified (alpha_0_deg), point contact, radial
    duty. alpha_0_deg must be in (0, 45] -- enforced by the family itself,
    not re-checked here."""
    catalog = BearingCatalog(
        d=d, D=D, b=b, C=C, C0=C0, designation=designation,
        label=label, position=position, arrangement=arrangement,
    )
    return Bearing.assemble(
        family=AngularContactFamily(),
        catalog=catalog,
        geometry=dict(Dw=Dw, Dpw=Dpw, Z=Z, E=E, alpha_0_deg=alpha_0_deg, nu=nu, i=i),
        analyses={"point_contact": True},
    )


def make_self_aligning_ball_bearing(
    *,
    d: float,
    D: float,
    Dw: float,
    Dpw: float,
    Z: int,
    E: float,
    alpha_0_deg: float,
    b: float = 0.0,
    C: float = 0.0,
    C0: float = 0.0,
    designation: str = "",
    label: str = "",
    position: float = 0.0,
    arrangement: str = "locating",
    nu: float = 0.3,
    i: int = 1,
) -> Bearing:
    """Self-aligning ball bearing -- contact-angle-specified (alpha_0_deg),
    point contact, radial duty. re is derived from gamma(alpha_0) inside
    the family (NOT a fixed ratio of Dw, unlike DGBB/ACB) -- see
    SelfAligningBallFamily.re_from_gamma()."""
    catalog = BearingCatalog(
        d=d, D=D, b=b, C=C, C0=C0, designation=designation,
        label=label, position=position, arrangement=arrangement,
    )
    return Bearing.assemble(
        family=SelfAligningBallFamily(),
        catalog=catalog,
        geometry=dict(Dw=Dw, Dpw=Dpw, Z=Z, E=E, alpha_0_deg=alpha_0_deg, nu=nu, i=i),
        analyses={"point_contact": True},
    )