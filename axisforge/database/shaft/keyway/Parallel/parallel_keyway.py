"""
database/shaft/keyway/Parallel/parallel_keyway.py

Lookup for DIN 6885 parallel keyway dimensions by shaft diameter.
Tolerance bands are pre-resolved into _min/_max columns so callers can
pick the critical value (nominal, min or max) for simulation.

Units: mm.
"""

from __future__ import annotations

import csv
from pathlib import Path

_TABLE_PATH = Path(__file__).parent / "parallel_keyway.csv"

_FLOAT_COLUMNS = {
    "d_min", "d_max", "b", "h", "b_fine", "h_fine", "L_min", "L_max",
    "t_shaft", "t_shaft_min", "t_shaft_max", "t_fine_shaft",
    "T_hub", "T_hub_min", "T_hub_max", "T_fine_hub",
    "c_min", "c_max", "r_min",
}


def _load_table() -> list[dict]:
    rows: list[dict] = []
    with open(_TABLE_PATH, newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict = {}
            for key, value in row.items():
                if key in _FLOAT_COLUMNS:
                    parsed[key] = float(value) if value else None
                else:
                    parsed[key] = value or None
            rows.append(parsed)
    return rows


_TABLE = _load_table()


def lookup_parallel_keyway(shaft_diameter: float) -> dict:
    """
    Return the DIN 6885 row matching shaft_diameter.

    Row selection: d_min < shaft_diameter <= d_max.
    Includes nominal, min and max for t_shaft / T_hub, and L_min / L_max.

    Raises
    ------
    ValueError
        If shaft_diameter is outside the table range.
    """
    for row in _TABLE:
        if row["d_min"] < shaft_diameter <= row["d_max"]:
            return row
    raise ValueError(
        f"shaft_diameter={shaft_diameter} mm is outside DIN 6885 table range "
        f"({_TABLE[0]['d_min']}-{_TABLE[-1]['d_max']} mm)"
    )