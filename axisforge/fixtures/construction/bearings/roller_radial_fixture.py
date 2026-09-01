"""
fixtures/construction/bearings/roller_radial_fixture.py

Construction-stage factory for the single radial-duty roller bearing
family (core/machine_elements/bearings/families/roller_bearing/radial/):
CylindricalRollerFamily (NU/N-type, line contact).

`arrangement` defaults to "floating" here, NOT BearingCatalog's own
dataclass default of "locating" -- CylindricalRollerFamily.assemble_geometry()
raises ValueError if catalog.arrangement == "locating" (an NU/N-type
bearing has no flange to react axial load, confirmed in that family's own
source). Overriding the default here means a bare
make_cylindrical_roller_bearing(...) call with no `arrangement=` doesn't
immediately crash on the one thing every caller would otherwise have to
remember to set. The actual guard still lives in the family, not here --
this default only avoids the surprise; passing arrangement="locating"
explicitly still raises, as it should.
"""
from __future__ import annotations

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.bearings.families.roller_bearing.radial.subtypes.cylindrical_roller import (
    CylindricalRollerFamily,
)


def make_cylindrical_roller_bearing(
    *,
    d: float,
    D: float,
    Dwe: float,
    Lwe: float,
    Dpw: float,
    Z: int,
    s: float,
    b: float = 0.0,
    C: float = 0.0,
    C0: float = 0.0,
    designation: str = "",
    label: str = "",
    position: float = 0.0,
    arrangement: str = "floating",
    n_s: int = 30,
    alpha_0_deg: float = 0.0,
    i: int = 1,
) -> Bearing:
    """NU/N-type cylindrical roller bearing -- line contact, radial duty.

    arrangement must be "floating" or "non-locating" -- see this module's
    own docstring for why the default here differs from BearingCatalog's.
    """
    catalog = BearingCatalog(
        d=d, D=D, b=b, C=C, C0=C0, designation=designation,
        label=label, position=position, arrangement=arrangement,
    )
    return Bearing.assemble(
        family=CylindricalRollerFamily(),
        catalog=catalog,
        geometry=dict(Dwe=Dwe, Lwe=Lwe, Dpw=Dpw, Z=Z, s=s, n_s=n_s, alpha_0_deg=alpha_0_deg, i=i),
        analyses={"line_contact": True},
    )