"""
axisforge/mesh/__init__.py

Por agora mesh/ só tem shaft/ lá dentro -- este rollup faz
`from axisforge.mesh import Elem` (ou Mesh1D, BeamModelSettings, ...)
funcionar sem quem importa precisar de saber que vive especificamente
em mesh.shaft. Nomes repetidos explicitamente aqui (em vez de
`from .shaft import *`) de propósito -- mantém grep-abilidade (é dado
ver aqui exactamente o que é re-exportado, sem ambiguidade de wildcard
import) e segue o mesmo estilo explícito usado em element_type/ e
shaft/.

Quando aparecer um subpacote irmão (mesh/gear/, mesh/bearing/, ...),
soma-se aqui o rollup dele da mesma forma: import explícito + extensão
de __all__. Cada __init__.py de subpacote continua a ser o único sítio
que sabe o que é seu -- mesh/__init__.py só agrega, nunca redefine.
"""
from __future__ import annotations

from .shaft import (
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
    Mesh1D,
    Grader,
    BeamModelSettings,
    VALID_BEAM_THEORIES,
    VALID_SHEAR_THEORIES,
    VALID_INTEGRATION_METHODS,
)

__all__ = [
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
    "Mesh1D",
    "Grader",
    "BeamModelSettings",
    "VALID_BEAM_THEORIES",
    "VALID_SHEAR_THEORIES",
    "VALID_INTEGRATION_METHODS",
]