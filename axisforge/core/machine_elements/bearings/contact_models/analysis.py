"""Selection of the contact-stiffness analysis that a bearing family runs at assembly.

``ContactAnalysis`` tells a family's ``assemble_geometry`` whether to store the
materials and the ISO/TS 16281 contact constants on the bearing, with the
helpers that complete a pair of material values and check the inputs of an
analysis.
"""
from __future__ import annotations
from enum import Enum


class ContactAnalysis(str, Enum):
    """Contact-stiffness analysis that a family runs in ``assemble_geometry``.

    Attributes
    ----------
    NONE
        Geometry only (radii, groove ratios, ...); the materials and the
        contact constants are not stored.
    ISO16281
        Also stores ``e1``, ``e2``, ``nu1``, ``nu2`` and the contact constants
        of ISO_TS_16281_2008: ``cp``, ``Ri`` for a ball, ``cL``, ``cs``,
        ``P_xk`` for a roller (see ``contact_models.iso16281``).
    """
    NONE = "none"                               # geometry only: no materials, no contact constants stored
    ISO16281 = "iso16281"                       # also materials and the ISO/TS 16281 constants (cp, Ri or cL, cs, P_xk)
    # further analyses can be added as new members

# names of the material inputs and of the geometry stored for the Hertz contact;
# the families name the Poisson ratios nu1, nu2 (not v1, v2) and these sets are not read by the families
MATERIAL_FIELDS = frozenset({"e1", "e2", "v1", "v2"})
HERTZ_GEOMETRY_FIELDS = frozenset({"r_rolling_el", "r_inner", "r_outer"})

_CONTACT_ANALYSIS_REQUIRES = {
    ContactAnalysis.NONE: MATERIAL_FIELDS | HERTZ_GEOMETRY_FIELDS,
    ContactAnalysis.ISO16281: MATERIAL_FIELDS,
}

def _resolve_pair(a: float | None, b: float | None,
                   name_a: str, name_b: str) -> tuple[float, float]:
    """Complete a pair of material values: one given is used for both.

    Parameters
    ----------
    a, b : float or None
        The two values (e.g. e1, e2 or nu1, nu2); None if not given.
    name_a, name_b : str
        Names of the two values, for the error message.

    Returns
    -------
    tuple[float, float]
        (a, b); the given one is copied to the missing one, both given are
        returned unchanged.

    Raises
    ------
    ValueError
        If neither is given.
    """
    if a is None and b is None:
        raise ValueError(f"At least one of {name_a}/{name_b} must be given")
    if a is None:
        a = b
    if b is None:
        b = a
    return a, b


def _verify_contact_inputs(analysis: ContactAnalysis, **given) -> None:
    """Check that the inputs required by an analysis are given.

    Not called by the families. Its required-field table uses the names
    ``v1``, ``v2`` for the Poisson ratios while the families use ``nu1``,
    ``nu2``, and it requires the Hertz geometry fields for ``NONE``.

    Parameters
    ----------
    analysis : ContactAnalysis
        Requested analysis.
    **given
        Input values by name; a value of None counts as missing. A parameter
        present but not required is ignored.

    Raises
    ------
    ValueError
        Listing the missing required fields.
    """
    required = _CONTACT_ANALYSIS_REQUIRES[analysis]
    missing = [name for name in required if given.get(name) is None]
    if missing:
        raise ValueError(f"ContactAnalysis.{analysis.name} requires "
                          f"{sorted(required)}, missing: {missing}")