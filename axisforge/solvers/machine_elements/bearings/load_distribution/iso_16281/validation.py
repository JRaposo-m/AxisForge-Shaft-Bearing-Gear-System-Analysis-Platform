"""Readiness check of a bearing for the ISO/TS 16281 contact solvers."""

from axisforge.core.machine_elements.bearings.families.family import (
    is_point_contact_family, is_line_contact_family,
)

_CAPABILITY_CHECK = {
    "point_contact": is_point_contact_family,
    "line_contact": is_line_contact_family,
}

def check_bearing_ready(bearing, label, capability):
    """Check that a bearing can be solved by a contact solver of a capability.

    Parameters
    ----------
    bearing : Bearing
        Assembled bearing.
    label : str
        Bearing label for the error messages.
    capability : str
        "point_contact" or "line_contact". An unknown value skips the family
        check.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the bearing family does not have the contact type of
        ``capability`` (checked first), or if ``bearing.iso16281_analysis`` is
        missing or false, i.e. the family did not store the ISO/TS 16281
        stiffness at assembly.

    Notes
    -----
    The capability table of this module repeats the one in ``dispatch``;
    ``resolve_solver_cls`` raises ``SolverDispatchError`` for the same two
    conditions.
    """
    check = _CAPABILITY_CHECK.get(capability)
    if check is not None and not check(bearing.family):
        raise RuntimeError(
            f"Bearing '{label}': family {bearing.family.name!r} is not a "
            f"{capability} family."
        )
    if not getattr(bearing, "iso16281_analysis", False):
        raise RuntimeError(
            f"Bearing '{label}': iso16281_analysis is not True -- the family "
            f"never completed the ISO/TS 16281 stiffness assembly."
        )