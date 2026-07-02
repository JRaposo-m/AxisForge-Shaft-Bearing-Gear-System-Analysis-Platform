from enum import Enum, auto

class BearingType(Enum):
    DEEP_GROOVE_BALL   = auto()
    ANGULAR_CONTACT     = auto()
    CYLINDRICAL_ROLLER  = auto()
    TAPERED_ROLLER      = auto()
    SPHERICAL_ROLLER    = auto()
    # extensível sem tocar no resto do código