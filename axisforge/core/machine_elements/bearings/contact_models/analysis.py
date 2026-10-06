"""
axisforge/core/machine_elements/bearings/contact_models/analysis.py

ContactAnalysis: which contact-stiffness analysis (if any) a family's
assemble_geometry should run. Drives both the required-input check and
what ends up on the assembled Bearing -- NONE means nothing Hertz-shaped
is stored at all, not even as None.
"""
from __future__ import annotations
from enum import Enum


class ContactAnalysis(str, Enum):
    NONE = "none"                               # NONE gives every data related to hertz analysis
    ISO16281 = "iso16281"                       # cp -- geometry auto-derived from ri/re/alpha_0/Dpw
    # this way I can then update this with new analysis 
    
MATERIAL_FIELDS = frozenset({"e1", "e2", "v1", "v2"})
HERTZ_GEOMETRY_FIELDS = frozenset({"r_rolling_el", "r_inner", "r_outer"})

_CONTACT_ANALYSIS_REQUIRES = {
    ContactAnalysis.NONE: MATERIAL_FIELDS | HERTZ_GEOMETRY_FIELDS,
    ContactAnalysis.ISO16281: MATERIAL_FIELDS,
}

def _resolve_pair(a: float | None, b: float | None,
                   name_a: str, name_b: str) -> tuple[float, float]:
    """If exactly one of a/b is given, broadcast it to the other. If
    neither is given, raise. If both given, use as-is."""
    if a is None and b is None:
        raise ValueError(f"At least one of {name_a}/{name_b} must be given")
    if a is None:
        a = b
    if b is None:
        b = a
    return a, b


def _verify_contact_inputs(analysis: ContactAnalysis, **given) -> None:
    """Raises listing exactly what's missing for the requested analysis.
    A param present but not required for this analysis is simply unused,
    not an error."""
    required = _CONTACT_ANALYSIS_REQUIRES[analysis]
    missing = [name for name in required if given.get(name) is None]
    if missing:
        raise ValueError(f"ContactAnalysis.{analysis.name} requires "
                          f"{sorted(required)}, missing: {missing}")