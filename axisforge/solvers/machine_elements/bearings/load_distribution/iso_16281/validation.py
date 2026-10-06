from axisforge.core.machine_elements.bearings.families.family import (
    is_point_contact_family, is_line_contact_family,
)

_CAPABILITY_CHECK = {
    "point_contact": is_point_contact_family,
    "line_contact": is_line_contact_family,
}

def check_bearing_ready(bearing, label, capability):
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