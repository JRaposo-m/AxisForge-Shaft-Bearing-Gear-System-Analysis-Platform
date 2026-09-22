"""
axisforge/mesh/shaft/element_type/__init__.py

Eager rollup, sem lazy loading, sem __getattr__ -- no mesmo espírito de
core/machine_elements/bearings/families/__init__.py. elem.py e
shear_factor.py são Python/NumPy puro, sem dependências pesadas, por
isso o import directo aqui não tem custo que justifique lazy loading --
e um import directo é o que torna
`from axisforge.mesh.shaft.element_type import Elem` trivial a partir
de outro repositório, sem precisar de saber a disposição interna dos
módulos.

Diferença deliberada face ao __init__.py dos bearings: lá, __all__ vem
directamente do registry (_FAMILY_REGISTRY é nome -> classe, então
sorted(_FAMILY_REGISTRY) já É a lista de nomes públicos certa). Aqui
_FORMULATION_REGISTRY é (beam_theory, element_order) -> instância, não
nome -> classe -- não há forma de derivar __all__ dele sem reescrever o
próprio registry, por isso __all__ fica escrito à mão. available_formulations()
expõe o registry de forma legível para quem quiser inspeccionar/validar
sem importar o nome privado _FORMULATION_REGISTRY directamente (ex.: um
teste que confirma que toda a teoria válida tem formulação registada).
"""
from __future__ import annotations

from .elem import (
    BeamFormulation,
    ShearDeformableBeamFormulation,
    EulerBernoulliBeam,
    TimoshenkoBeam,
    FrameElement,
    ElemBase,
    Elem,
    QuadraticTimoshenkoElem,
    _FORMULATION_REGISTRY,
)
from .shear_factor import ShearFactor

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
]


def available_formulations() -> dict[tuple[str, str], type]:
    """(beam_theory, element_order) -> concrete BeamFormulation class.
    Para listar/validar sem importar _FORMULATION_REGISTRY directamente
    (ex.: uma UI de definições, ou um teste que confirma que toda a
    combinação válida em VALID_BEAM_THEORIES tem formulação registada)."""
    return {key: type(formulation) for key, formulation in _FORMULATION_REGISTRY.items()}