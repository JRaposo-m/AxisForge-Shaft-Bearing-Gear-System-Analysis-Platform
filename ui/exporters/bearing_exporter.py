# ui/exporters/bearing_exporter.py
"""
Exporter for BearingLifeResult → .txt
"""
from __future__ import annotations
import math

from .base import SEP_THICK, SEP_THIN, check, field, flt, make_header, write_txt


def export_bearing(
    result,
    path: str,
    *,
    project: str = "",
    element: str = "",
    script: str = "",
    C: float = 0.0,
    C0: float = 0.0,
) -> None:
    """Write BearingLifeResult to a .txt file."""
    r = result
    lines = [
        make_header(
            "Bearing Life Analysis",
            project, element, script,
            solver="BearingLifeSolver v1.0",
            standard="ISO 281:2007",
        ),
        "",
        "INPUTS",
        SEP_THIN,
        field("Bearing",        r.bearing_label),
        field("Position",       f"{flt(r.bearing_position, '.2f')}",  "mm"),
        field("Speed",          f"{flt(r.speed_rpm, '.1f')}",         "rpm"),
        field("C (dynamic)",    f"{flt(C, '.0f')}",                   "N"),
        field("C0 (static)",    f"{flt(C0, '.0f')}",                  "N"),
        field("Life target",    f"{flt(r.life_target_hours, '.0f')}", "h"),
        field("Life exponent p",f"{flt(r.life_exponent, '.4g')}"),
        "",
        "LOADS",
        SEP_THIN,
        field("Fr (radial)",    f"{flt(r.Fr, '.2f')}",  "N"),
        field("Fa (axial)",     f"{flt(r.Fa, '.2f')}",  "N"),
        field("X",              f"{flt(r.X, '.4f')}"),
        field("Y",              f"{flt(r.Y, '.4f')}"),
        field("X0",             f"{flt(r.X0, '.4f')}"),
        field("Y0",             f"{flt(r.Y0, '.4f')}"),
        field("P (dynamic)",    f"{flt(r.P, '.2f')}",   "N"),
        field("P0 (static)",    f"{flt(r.P0, '.2f')}",  "N"),
        "",
        "RESULTS",
        SEP_THIN,
        field("C/P",            f"{flt(r.C_over_P, '.4f')}"),
        field("L10",            f"{flt(r.L10, '.4f')}",   "10⁶ rev"),
        field("L10h",           f"{flt(r.L10h, '.0f')}",  "h"),
        field("S0",             f"{flt(r.S0, '.3f')}"),
        field("Life ratio",     f"{flt(r.life_ratio, '.3f')}"),
        "",
        "CHECKS",
        SEP_THIN,
        field("Meets life target", f"{check(r.meets_life_target)}  (L10h = {flt(r.L10h,'.0f')} h, target = {flt(r.life_target_hours,'.0f')} h)"),
        field("Static safety",     f"{check(r.meets_static_safety)}  (S0 = {flt(r.S0,'.3f')} ≥ 1.0)"),
        field("Overall safe",      check(r.is_safe)),
    ]

    status = "OK" if r.is_safe else "WARNING: bearing does not meet design requirements"
    lines += [
        "",
        SEP_THICK,
        f"STATUS : {status}",
        SEP_THICK,
    ]

    write_txt(path, "\n".join(lines))
