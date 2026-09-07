"""
fixtures/studies/shafts/fem_studies/outputs/resolution_csv.py

CONTENT ONLY, single-solve CSV export for one shaft_fem run's
per-node data (section 3 of resolution_report.py's own docstring: x,
M_xz/xy/M, V_xz/xy/V, v_xz/xy/v, T, d, W, Wt, sigma_b, tau) -- sibling
of resolution_report.py, same "builds text, does not write any file"
contract. Exists specifically for numerical comparison against an
external solver (Abaqus) row-by-row: resolution_report.py's fixed-width
tables are tuned for human legibility (1-4 decimals, unit-labelled
headers) and lose precision the comparison needs; this module keeps
full float precision and a machine-parseable format instead, at the
cost of being unreadable as a text report -- the two modules are not
alternatives to each other, they serve different readers (a person
skimming vs. a diff/plot script).

Does not reuse resolution_report.py's _table() (fixed-width, string
cells, no precision control) -- csv.writer over io.StringIO instead,
same as resolution_report.py's own docstring notes comparison_report.py
duplicating _table() rather than importing it: formatting helpers here
are not shared across report shapes.

No file I/O -- returns the CSV as a str. fixtures/studies/text_report.py
(or a comparison-specific caller) opens the file and writes it, same
division of responsibility as resolution_report.py's own blocks.

Dependency: axisforge.results.fem_results.shaft_results.ShaftResults
(TYPE_CHECKING only) -- reads a ShaftResults handed to it, never builds
one.
"""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.results.fem_results.shaft_results import ShaftResults


# Column order matches resolution_report.py's three per-node tables,
# concatenated (BENDING & SHEAR, DEFLECTION & TORSION, SECTION &
# STRESS), x kept once instead of once per table.
_COLUMNS = (
    "x_mm", "M_xz_Nmm", "M_xy_Nmm", "M_Nmm", "V_xz_N", "V_xy_N", "V_N",
    "v_xz_mm", "v_xy_mm", "v_mm", "T_Nm",
    "d_mm", "W_mm3", "Wt_mm3", "sigma_b_MPa", "tau_MPa",
)


def resolution_csv(result: "ShaftResults", decimals: int = 10) -> str:
    """
    Full-precision CSV of every section-3 per-node array, one row per
    node, columns in `_COLUMNS` order. `decimals` controls the number
    of digits after the decimal point for every numeric column
    uniformly (Abaqus report/field-output values are typically
    exported at similar or higher precision, so 10 is a safe default
    for a diff against them; pass fewer if the comparison side has
    less precision and trailing digits would just be noise).

    Uses repr-free fixed-point formatting (f"{value:.{decimals}f}"),
    not general/scientific notation, so every column lines up at the
    same decimal precision when opened in a spreadsheet -- easier to
    subtract column-by-column against an Abaqus export than mixed
    notation would be.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_COLUMNS)

    for i, x in enumerate(result.x_nodes):
        row = [
            x,
            result.M_xz[i], result.M_xy[i], result.M[i],
            result.V_xz[i], result.V_xy[i], result.V[i],
            result.v_xz[i], result.v_xy[i], result.v[i], result.T[i],
            result.d[i], result.W[i], result.Wt[i],
            result.sigma_b[i], result.tau[i],
        ]
        writer.writerow(f"{value:.{decimals}f}" for value in row)

    return buf.getvalue()