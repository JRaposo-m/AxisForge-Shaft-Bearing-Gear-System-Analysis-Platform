# ui/exporters/stress_exporter.py
"""
Exporter for StressResult → .txt

One block per CriticalSection, sorted by nf_goodman ascending (already sorted in result).
"""
from __future__ import annotations
import math

from .base import SEP_THICK, SEP_THIN, check, field, flt, make_header, write_txt


def export_stress(
    result,
    path: str,
    *,
    project: str = "",
    element: str = "",
    script: str = "",
    mat=None,
) -> None:
    """
    Write StressResult to a .txt file.

    Parameters
    ----------
    result  : StressResult
    path    : destination file path
    mat     : Material object (optional, for Sut/Se display)
    """
    r = result
    lines = [
        make_header(
            "Fatigue Stress Analysis — Shaft",
            project, element, script,
            solver="StressSolver v1.0",
            standard="Shigley 10th ed. §7-4",
        ),
        "",
        f"  Material      : {r.material_id}",
        f"  Surface finish: {r.finish}",
        f"  Reliability   : {r.reliability_percent:.0f} %",
    ]

    if mat is not None:
        try:
            lines += [
                f"  Sut           : {flt(mat.Sut, '.1f')} MPa",
                f"  Sy            : {flt(mat.Sy, '.1f')} MPa",
            ]
        except AttributeError:
            pass

    lines += ["", SEP_THIN]

    if not r.sections:
        lines += ["  No critical sections evaluated.", ""]
    else:
        lines += [
            f"  {'x [mm]':>8}  {'d [mm]':>8}  {'Kf':>6}  {'Kfs':>6}  "
            f"{'σ_a [MPa]':>10}  {'σ_m [MPa]':>10}  "
            f"{'nf_GM':>7}  {'nf_ASME':>8}  {'ny':>7}  {'Safe':>5}",
            "  " + "-" * 76,
        ]
        for s in r.sections:
            nfg  = flt(s.nf_goodman, ".3f") if math.isfinite(s.nf_goodman) else "∞"
            nfa  = flt(s.nf_asme,    ".3f") if math.isfinite(s.nf_asme)    else "∞"
            ny   = flt(s.ny,         ".3f") if math.isfinite(s.ny)         else "∞"
            safe = check(s.is_safe)
            lines.append(
                f"  {s.x:>8.2f}  {s.diameter:>8.2f}  {s.Kf:>6.3f}  {s.Kfs:>6.3f}  "
                f"{s.sigma_a:>10.3f}  {s.sigma_m:>10.3f}  "
                f"{nfg:>7}  {nfa:>8}  {ny:>7}  {safe:>5}"
            )

        lines.append("")
        lines.append("MOST CRITICAL SECTION")
        lines.append(SEP_THIN)
        c = r.most_critical
        if c is not None:
            lines += [
                field("x",          f"{flt(c.x, '.2f')}",          "mm"),
                field("diameter",   f"{flt(c.diameter, '.2f')}",   "mm"),
                field("Kt",         f"{flt(c.Kt, '.4f')}"),
                field("Kf",         f"{flt(c.Kf, '.4f')}"),
                field("Kts",        f"{flt(c.Kts, '.4f')}"),
                field("Kfs",        f"{flt(c.Kfs, '.4f')}"),
                field("q (bending)",f"{flt(c.q, '.4f')}"),
                field("qs (torsion)",f"{flt(c.qs, '.4f')}"),
                field("Ma",         f"{flt(c.Ma, '.2f')}",         "N·mm"),
                field("Tm",         f"{flt(c.Tm, '.2f')}",         "N·mm"),
                field("Se'",        f"{flt(c.Se_prime, '.2f')}",   "MPa"),
                field("ka",         f"{flt(c.ka, '.4f')}"),
                field("kb",         f"{flt(c.kb, '.4f')}"),
                field("ke",         f"{flt(c.ke, '.4f')}"),
                field("sigma_a",    f"{flt(c.sigma_a, '.3f')}",    "MPa"),
                field("sigma_m",    f"{flt(c.sigma_m, '.3f')}",    "MPa"),
                field("nf_goodman", flt(c.nf_goodman, ".3f") if math.isfinite(c.nf_goodman) else "∞"),
                field("nf_asme",    flt(c.nf_asme,    ".3f") if math.isfinite(c.nf_asme)    else "∞"),
                field("ny",         flt(c.ny,          ".3f") if math.isfinite(c.ny)          else "∞"),
                field("Safe",       check(c.is_safe)),
                field("Langer OK",  check(c.langer_ok)),
            ]

    # Status
    all_safe = all(s.is_safe for s in r.sections) if r.sections else True
    status = "OK" if all_safe else f"WARNING: {sum(1 for s in r.sections if not s.is_safe)} section(s) unsafe"
    lines += [
        "",
        SEP_THICK,
        f"STATUS : {status}",
        SEP_THICK,
    ]

    write_txt(path, "\n".join(lines))
