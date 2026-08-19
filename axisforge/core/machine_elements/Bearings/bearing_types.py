"""
core/machine_elements/Bearings/bearing_types.py

BearingType -- informational label only.

Historically this enum drove type-dispatch (bearing_factory.make_bearing(),
and still does on the solver side via the _SOLVER_MAP dispatch table in
solvers/machine_elements/bearings/ISO_16281/rolling_bearing_solver.py).

Under the new architecture (see family.py, bearing.py) dispatch on the
core/ side is no longer by enum -- a BearingFamily instance is passed
directly to Bearing.assemble(), so a new family/subtype never requires
editing this file. BearingType survives purely as a label every family
declares via its BEARING_TYPE class attribute, mirrored onto
bearing.bearing_type at assembly time, so that:

  - reports/GUI can group/filter bearings by family without importing
    every family class
  - existing solver-side dispatch tables keyed by BearingType keep
    working unmodified

Extend it when a genuinely new *family* is added (a new life-exponent
group, a new ISO 281 p value) -- not for every new subtype variant within
an existing family.
"""
from enum import Enum, auto


class BearingType(Enum):
    DEEP_GROOVE_BALL   = auto()
    ANGULAR_CONTACT    = auto()
    SELF_ALIGNING_BALL = auto()
    THRUST_BALL        = auto()
    CYLINDRICAL_ROLLER = auto()
    TAPERED_ROLLER     = auto()
    SPHERICAL_ROLLER   = auto()
    THRUST_CYLINDRICAL_ROLLER = auto()
    THRUST_NEEDLE_ROLLER      = auto()
    # extensivel sem tocar no resto do codigo