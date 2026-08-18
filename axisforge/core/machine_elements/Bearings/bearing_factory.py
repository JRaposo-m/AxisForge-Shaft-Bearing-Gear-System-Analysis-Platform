"""
Bearings/bearing_factory.py

Factory for rolling bearing instantiation.
Returns the correct class for a given BearingType.

Usage
-----
from axisforge.core.machine_elements.Bearings.bearing_factory import make_bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType

b = make_bearing(BearingType.DEEP_GROOVE_BALL,
                 d=40, D=80, b=18,
                 C=29600, C0=17800,
                 designation="6208",
                 position=50.0)

Use make_bearing() when bearing type comes from a database, config file, or GUI.
For manual instantiation in tests, import the class directly.
"""

from __future__ import annotations
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.types.ball_bearing.subtypes.deep_groove_ball import DeepGrooveBallBearing
from axisforge.core.machine_elements.Bearings.types.ball_bearing.subtypes.angular_contact import AngularContactBallBearing

from axisforge.core.machine_elements.Bearings.types.roller_bearing.subtypes.cylindrical_roller import CylindricalRollerBearing

# Populated incrementally as families are implemented
_TYPE_MAP: dict[BearingType, type[Bearing]] = {
    BearingType.DEEP_GROOVE_BALL: DeepGrooveBallBearing,
    BearingType.ANGULAR_CONTACT: AngularContactBallBearing,
    BearingType.CYLINDRICAL_ROLLER: CylindricalRollerBearing,
    # BearingType.TAPERED_ROLLER:     TaperedRollerBearing,        # Phase 2
    # BearingType.SPHERICAL_ROLLER:   SphericalRollerBearing,      # Phase 2
}


def make_bearing(bearing_type: BearingType, **kwargs) -> Bearing:
    """
    Instantiate the correct bearing class for bearing_type.

    Parameters
    ----------
    bearing_type : BearingType
    **kwargs     : passed directly to the class __init__

    Raises
    ------
    NotImplementedError : if bearing_type has no registered class yet
    """
    cls = _TYPE_MAP.get(bearing_type)
    if cls is None:
        raise NotImplementedError(
            f"BearingType.{bearing_type.name} is not yet implemented. "
            f"Available: {[t.name for t in _TYPE_MAP]}"
        )
    return cls(**kwargs)