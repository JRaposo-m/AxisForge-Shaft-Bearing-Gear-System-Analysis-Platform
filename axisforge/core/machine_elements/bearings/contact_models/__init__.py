"""Contact spring constants of rolling bearings after ISO/TS 16281, and the ``ContactAnalysis`` switch.

Point contact: cp from the bearing geometry (Dw, Dpw, gamma, ri, re) and two
materials. Line contact: cL, cs and the reference roller profile from Lwe and
Dwe, steel only. A bearing family selects the model at assembly with
``ContactAnalysis``.
"""
from axisforge.core.machine_elements.bearings.contact_models.analysis import ContactAnalysis
from axisforge.core.machine_elements.bearings.contact_models.iso16281 import (
    PointContactStiffness,
    SelfAligningPointContactStiffness,
    LineContactStiffness,
)

__all__ = [
    "ContactAnalysis",
    "PointContactStiffness",
    "SelfAligningPointContactStiffness",
    "LineContactStiffness",
]