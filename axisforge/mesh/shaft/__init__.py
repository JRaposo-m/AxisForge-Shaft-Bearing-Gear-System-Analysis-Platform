"""
Rollup of the shaft mesh package: beam elements and formulations (element_type), 1D node generation and grading (mesh_generation) and the beam model settings.
"""
from __future__ import annotations

from .element_type import (
    BeamFormulation,
    ShearDeformableBeamFormulation,
    EulerBernoulliBeam,
    TimoshenkoBeam,
    FrameElement,
    ElemBase,
    Elem,
    QuadraticTimoshenkoElem,
    ShearFactor,
    available_formulations,
)
from .mesh_generation import Mesh1D, Grader
from .beam_model_settings import (
    BeamModelSettings,
    VALID_BEAM_THEORIES,
    VALID_SHEAR_THEORIES,
    VALID_INTEGRATION_METHODS,
)

__all__ = [
    # element_type
    "BeamFormulation",
    "ShearDeformableBeamFormulation",
    "EulerBernoulliBeam",
    "TimoshenkoBeam",
    "FrameElement",
    "ElemBase",
    "Elem",
    "QuadraticTimoshenkoElem",
    "ShearFactor",
    "available_formulations",
    # mesh_generation
    "Mesh1D",
    "Grader",
    # beam_model_settings
    "BeamModelSettings",
    "VALID_BEAM_THEORIES",
    "VALID_SHEAR_THEORIES",
    "VALID_INTEGRATION_METHODS",
]