# ui/exporters/statics_exporter.py
"""
Exporter for StaticsResult → .txt

Sections: Inputs → Reactions → Extremes → Status
"""
from __future__ import annotations

from .base import SEP_THICK, SEP_THIN, check, field, flt, make_header, write_txt


def export_statics(
    result,
    path: str,
    *,
    project: str = "",
    element: str = "",
    script: str = "",
) -> None:
    """
    Write StaticsResult to a .txt file.

    Parameters
    ----------
    result  : StaticsResult
    path    : destination file path (created if needed)
    project : project name for header
    element : element name for header
    script  : script filename for header
    """
    r = result
    rx = r.reactions

    lines = [
        make_header(
            "Static Analysis — Shaft",
            project, element, script,
            solver="StaticsSolver v1.0",
            standard="Shigley 10th ed. §3",
        ),
        "",
        "REACTIONS",
        SEP_THIN,
        field("A_xz",  f"{flt(rx.get('A_xz', 0.0), '.2f')}",  "N"),
        field("A_xy",  f"{flt(rx.get('A_xy', 0.0), '.2f')}",  "N"),
        field("B_xz",  f"{flt(rx.get('B_xz', 0.0), '.2f')}",  "N"),
        field("B_xy",  f"{flt(rx.get('B_xy', 0.0), '.2f')}",  "N"),
        field("Axial", f"{flt(rx.get('axial', 0.0), '.2f')}", "N"),
        "",
        "EXTREMES",
        SEP_THIN,
        field("V_xz_max",   f"{flt(r.V_xz_max, '.2f')}",   "N"),
        field("V_xy_max",   f"{flt(r.V_xy_max, '.2f')}",   "N"),
        field("M_xz_max",   f"{flt(r.M_xz_max, '.2f')}",   "N·mm"),
        field("M_xy_max",   f"{flt(r.M_xy_max, '.2f')}",   "N·mm"),
        field("M_res_max",  f"{flt(r.M_res_max, '.2f')}",  "N·mm"),
        field("x @ M_res_max", f"{flt(r.x_at_M_res_max, '.2f')}", "mm"),
        field("T_max",      f"{flt(r.T_max, '.2f')}",      "N·mm"),
        "",
        SEP_THICK,
        "STATUS : OK",
        SEP_THICK,
    ]

    write_txt(path, "\n".join(lines))
