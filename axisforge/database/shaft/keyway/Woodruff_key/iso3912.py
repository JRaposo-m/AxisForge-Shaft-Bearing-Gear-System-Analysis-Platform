"""
database/shaft/keyway/Woodruff/iso3912.py

Lookup for ISO 3912 woodruff (disc) keyway dimensions by shaft diameter.
Two series (different diameter bands, same geometry columns).

Units: mm.
"""

from __future__ import annotations

import csv
from pathlib import Path

_DIR = Path(__file__).parent

_FLOAT_COLUMNS = {
    "d_min", "d_max", "b", "h", "D", "ht", "c_min", "c_max",
    "t_shaft", "t_shaft_min", "t_shaft_max",
    "T_hub", "T_hub_min", "T_hub_max",
    "r_max", "r_min",
}


def _load_table(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if k in _FLOAT_COLUMNS})
    return rows


_TABLES = {
    1: _load_table(_DIR / "iso3912_series1.csv"),
    2: _load_table(_DIR / "iso3912_series2.csv"),
}


def lookup_woodruff_keyway(shaft_diameter: float, series: int = 1) -> dict:
    """
    Return the ISO 3912 row matching shaft_diameter for the given series.

    Row selection: d_min < shaft_diameter <= d_max.

    Parameters
    ----------
    shaft_diameter : float
        Shaft diameter [mm].
    series : int
        1 or 2 — selects the diameter band table.

    Raises
    ------
    ValueError
        If series is not 1 or 2, or shaft_diameter is outside the table range.
    """
    if series not in _TABLES:
        raise ValueError(f"series must be 1 or 2, got {series}")

    table = _TABLES[series]
    for row in table:
        if row["d_min"] < shaft_diameter <= row["d_max"]:
            return row
    raise ValueError(
        f"shaft_diameter={shaft_diameter} mm is outside ISO 3912 series {series} "
        f"table range ({table[0]['d_min']}-{table[-1]['d_max']} mm)"
    )
