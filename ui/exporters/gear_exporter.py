# ui/exporters/gear_exporter.py
"""
Exporters for GearGeometryResult and GearForceResult → .txt
"""
from __future__ import annotations
import math

from .base import SEP_THICK, SEP_THIN, check, field, flt, make_header, write_txt


def export_gear_geometry(
    result,
    path: str,
    *,
    project: str = "",
    element: str = "",
    script: str = "",
) -> None:
    """Write GearGeometryResult to a .txt file."""
    g = result
    lines = [
        make_header(
            "Gear Pair — Geometry",
            project, element, script,
            solver="GearSolver v1.0",
            standard="ISO 21771:2007",
        ),
        "",
        "INPUTS",
        SEP_THIN,
        field("mn",             f"{flt(g.mn, '.4g')}",          "mm"),
        field("z1 (pinion)",    f"{g.z1}"),
        field("z2 (wheel)",     f"{g.z2}"),
        field("alpha_n",        f"{flt(g.alpha_n_deg, '.4g')}",  "°"),
        field("beta",           f"{flt(g.beta_deg, '.4g')}",     "°"),
        field("x1",             f"{flt(g.x1, '.4f')}"),
        field("x2",             f"{flt(g.x2, '.4f')}"),
        field("b (face width)", f"{flt(g.b, '.2f')}",           "mm"),
        "",
        "DERIVED",
        SEP_THIN,
        field("mt",             f"{flt(g.mt, '.6f')}",           "mm"),
        field("alpha_t",        f"{flt(g.alpha_t_deg, '.6f')}",  "°"),
        field("beta_b",         f"{flt(g.beta_b_deg, '.6f')}",   "°"),
        field("alpha_tw",       f"{flt(g.alpha_tw_deg, '.6f')}", "°"),
        field("u (gear ratio)", f"{flt(g.u, '.6f')}"),
        "",
        "CENTRE DISTANCES",
        SEP_THIN,
        field("a (standard)",   f"{flt(g.a, '.4f')}",            "mm"),
        field("al (working)",   f"{flt(g.al, '.4f')}",           "mm"),
        field("Standard centre",check(g.is_standard_centre)),
        "",
        "DIAMETERS",
        SEP_THIN,
        field("d1",             f"{flt(g.d1, '.4f')}",           "mm"),
        field("d2",             f"{flt(g.d2, '.4f')}",           "mm"),
        field("db1",            f"{flt(g.db1, '.4f')}",          "mm"),
        field("db2",            f"{flt(g.db2, '.4f')}",          "mm"),
        field("da1",            f"{flt(g.da1, '.4f')}",          "mm"),
        field("da2",            f"{flt(g.da2, '.4f')}",          "mm"),
        field("df1",            f"{flt(g.df1, '.4f')}",          "mm"),
        field("df2",            f"{flt(g.df2, '.4f')}",          "mm"),
        field("dl1 (working)",  f"{flt(g.dl1, '.4f')}",          "mm"),
        field("dl2 (working)",  f"{flt(g.dl2, '.4f')}",          "mm"),
        "",
        "CONTACT RATIOS",
        SEP_THIN,
        field("p_bt",           f"{flt(g.p_bt, '.6f')}",         "mm"),
        field("eps_alpha",      f"{flt(g.eps_alpha, '.6f')}"),
        field("eps_beta",       f"{flt(g.eps_beta, '.6f')}"),
        field("eps_gamma",      f"{flt(g.eps_gamma, '.6f')}"),
        field("Spur gear",      check(g.is_spur)),
    ]

    lines += [
        "",
        SEP_THICK,
        "STATUS : OK",
        SEP_THICK,
    ]
    write_txt(path, "\n".join(lines))


def export_gear_forces(
    result,
    path: str,
    *,
    project: str = "",
    element: str = "",
    script: str = "",
    geometry=None,
) -> None:
    """Write GearForceResult to a .txt file."""
    f = result
    lines = [
        make_header(
            "Gear Pair — Forces",
            project, element, script,
            solver="GearSolver v1.0",
            standard="ISO 21771:2007 / Shigley §13-7",
        ),
        "",
        "TORQUES",
        SEP_THIN,
        field("T1 (pinion)",    f"{flt(f.T1_Nmm, '.2f')}",      "N·mm"),
        field("T2 (wheel)",     f"{flt(f.T2_Nmm, '.2f')}",      "N·mm"),
        field("Gear ratio u",   f"{flt(f.gear_ratio, '.6f')}"),
        "",
        "FORCES",
        SEP_THIN,
        field("Ft (tangential)",f"{flt(f.Ft, '.2f')}",           "N"),
        field("Fr (radial)",    f"{flt(f.Fr, '.2f')}",           "N"),
        field("Fa (axial)",     f"{flt(f.Fa, '.2f')}",           "N"),
        field("Fn (normal)",    f"{flt(f.Fn, '.2f')}",           "N"),
        field("F_transverse",   f"{flt(f.F_transverse_resultant, '.2f')}", "N"),
    ]

    if geometry is not None:
        lines += [
            "",
            "GEOMETRY REFERENCE",
            SEP_THIN,
            field("rl1 (working r1)", f"{flt(geometry.rl1, '.4f')}", "mm"),
            field("rl2 (working r2)", f"{flt(geometry.rl2, '.4f')}", "mm"),
        ]

    lines += [
        "",
        "INTERMEDIATE",
        SEP_THIN,
        field("Fbt",            f"{flt(f.Fbt, '.2f')}",          "N"),
        field("Fbn",            f"{flt(f.Fbn, '.2f')}",          "N"),
        field("Consistency dev",f"{flt(f.consistency_deviation_pct, '.4f')}", "%"),
    ]

    status = "OK" if f.consistency_deviation_pct < 2.0 else f"WARNING: consistency deviation {flt(f.consistency_deviation_pct,'.2f')} % > 2 %"
    lines += [
        "",
        SEP_THICK,
        f"STATUS : {status}",
        SEP_THICK,
    ]
    write_txt(path, "\n".join(lines))
