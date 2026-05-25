# ui/exporters/static_failure_exporter.py
"""
Exporter for StaticFailureResult → .txt
"""
from __future__ import annotations
import math

from .base import SEP_THICK, SEP_THIN, check, field, flt, make_header, write_txt


def export_static_failure(
    result,
    path: str,
    *,
    project: str = "",
    element: str = "",
    script: str = "",
) -> None:
    """Write StaticFailureResult to a .txt file."""
    r = result
    lines = [
        make_header(
            "Static Failure Analysis — Shaft",
            project, element, script,
            solver="StaticFailureSolver v1.0",
            standard="Shigley 10th ed. §5-4/5-5",
        ),
        "",
        f"  Material : {r.material_id}",
        f"  Sy       : {flt(r.Sy, '.2f')} MPa",
        "",
        SEP_THIN,
    ]

    if not r.sections:
        lines += ["  No sections evaluated.", ""]
    else:
        lines += [
            f"  {'x [mm]':>8}  {'d [mm]':>8}  {'σ_x [MPa]':>10}  {'τ_xy [MPa]':>10}  "
            f"{'n_DE':>7}  {'n_MSS':>7}  {'Gov.':>5}  {'Status':>10}",
            "  " + "-" * 72,
        ]
        for s in r.sections:
            n_de  = flt(s.n_DE,  ".3f") if math.isfinite(s.n_DE)  else "∞"
            n_mss = flt(s.n_MSS, ".3f") if math.isfinite(s.n_MSS) else "∞"
            lines.append(
                f"  {s.x:>8.2f}  {s.diameter:>8.2f}  {s.sigma_x:>10.3f}  {s.tau_xy:>10.3f}  "
                f"{n_de:>7}  {n_mss:>7}  {s.governing_theory:>5}  {s.risk_label:>10}"
            )

        lines.append("")
        lines.append("CRITICAL SECTION")
        lines.append(SEP_THIN)
        c = r.critical_section
        if c is not None:
            lines += [
                field("x",              f"{flt(c.x, '.2f')}",           "mm"),
                field("diameter",       f"{flt(c.diameter, '.2f')}",    "mm"),
                field("sigma_x",        f"{flt(c.sigma_x, '.3f')}",     "MPa"),
                field("tau_xy",         f"{flt(c.tau_xy, '.3f')}",      "MPa"),
                field("sigma_prime",    f"{flt(c.sigma_prime, '.3f')}", "MPa"),
                field("sigma_eff_mss",  f"{flt(c.sigma_eff_mss, '.3f')}", "MPa"),
                field("n_DE",           flt(c.n_DE,  ".3f") if math.isfinite(c.n_DE)  else "∞"),
                field("n_MSS",          flt(c.n_MSS, ".3f") if math.isfinite(c.n_MSS) else "∞"),
                field("governing_n",    flt(c.governing_n, ".3f") if math.isfinite(c.governing_n) else "∞"),
                field("governing_theory", c.governing_theory),
                field("yielded",        check(not c.yielded)),
                field("risk",           c.risk_label),
            ]

    status = "OK" if not r.any_yielded else f"WARNING: yielding detected — min n = {flt(r.min_governing_n, '.3f')}"
    lines += [
        "",
        SEP_THICK,
        f"STATUS : {status}",
        SEP_THICK,
    ]

    write_txt(path, "\n".join(lines))
