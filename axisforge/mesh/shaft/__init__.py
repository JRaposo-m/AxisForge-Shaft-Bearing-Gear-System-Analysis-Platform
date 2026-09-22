"""
axisforge/mesh/shaft/__init__.py

Eager rollup de tudo dentro de mesh/shaft/: element_type/ (Elem e toda
a hierarquia de formulações de viga), mesh_generation/ (Mesh1D,
Grader), e beam_model_settings.py (BeamModelSettings, a configuração ao
nível da análise que liga beam_theory/shear_theory/integration_method).
Sem lazy loading -- mesma lógica dos __init__ dos subpacotes.
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