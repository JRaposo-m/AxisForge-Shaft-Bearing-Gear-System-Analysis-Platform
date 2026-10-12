"""
Rollup of the beam element package: Elem, the beam formulations (Euler-Bernoulli, Timoshenko), the shear correction factor and available_formulations().
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
    """Registered beam formulations, without importing the private registry.

    Returns
    -------
    dict of (str, str) to type
        Maps ``(beam_theory, element_order)`` to the concrete
        BeamFormulation class registered for it, e.g.
        ``("timoshenko", "linear")``.

    Notes
    -----
    Not listed in ``__all__`` of this module; it is imported by name in
    mesh.shaft and mesh.
    """
    return {key: type(formulation) for key, formulation in _FORMULATION_REGISTRY.items()}