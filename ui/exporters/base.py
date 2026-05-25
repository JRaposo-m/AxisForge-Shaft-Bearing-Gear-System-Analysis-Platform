# ui/exporters/base.py
"""
Shared utilities for all AxisForge result exporters.

Rules:
  - Never import from axisforge/ here — receives objects, doesn't construct them.
  - Always UTF-8.
  - 80-char width.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

SEP_THICK = "=" * 80
SEP_THIN  = "-" * 80


def make_header(
    analysis_type: str,
    project: str,
    element: str,
    script: str,
    solver: str,
    standard: str,
) -> str:
    return "\n".join([
        SEP_THICK,
        f"AxisForge — {analysis_type}",
        SEP_THICK,
        f"Project  : {project  or '—'}",
        f"Element  : {element  or '—'}",
        f"Script   : {script   or '—'}",
        f"Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Solver   : {solver}",
        f"Standard : {standard}",
        SEP_THICK,
    ])


def write_txt(path: str | Path, content: str) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")


def field(label: str, value, unit: str = "", width: int = 22) -> str:
    """Format a single input/output field line."""
    val_str = f"{value}" if not isinstance(value, float) else f"{value:.4g}"
    unit_str = f"  [{unit}]" if unit else ""
    return f"  {label:<{width}}: {val_str}{unit_str}"


def flt(value: float, fmt: str = ".4g") -> str:
    """Format float — handles inf gracefully."""
    if value == float("inf"):
        return "∞"
    if value == float("-inf"):
        return "-∞"
    return format(value, fmt)


def check(ok: bool) -> str:
    return "✓" if ok else "✗"
