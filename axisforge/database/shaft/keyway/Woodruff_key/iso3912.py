"""
Lookup of Woodruff (disc) key keyway dimensions by shaft diameter.

Reads ``iso3912_series1.csv`` and ``iso3912_series2.csv`` (named "ISO 3912" in the
code; the files carry no edition or source). The two series differ in their
diameter bands and share the same columns. All lengths in mm.
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
    """Read one CSV, keeping only the float columns (an empty cell would fail in ``float``)."""
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
    Woodruff keyway dimensions of the table row that contains a shaft diameter.

    Used by ``Keyway.from_standard`` (core/machine_elements/shaft/shaft.py), its
    only caller, which reads ``b``, ``t_shaft``, ``D``, ``t_shaft_min`` and
    ``t_shaft_max`` (``D``, the key diameter, becomes the keyway length there).

    Parameters
    ----------
    shaft_diameter : float
        Shaft diameter [mm].
    series : int
        1 or 2. Both tables hold the same 14 key sizes (identical dimension columns
        row by row); only the shaft diameter bands differ. What each series is
        for is not stated in the code.

    Returns
    -------
    row : dict
        The whole table row (the same dict object held by the module). Keys,
        all [mm]: ``d_min``, ``d_max`` (shaft diameter band), ``b`` (width),
        ``h`` (key height), ``D`` (key diameter), ``ht`` (meaning not stated in
        the file), ``c_min``, ``c_max`` (chamfer), ``t_shaft``, ``t_shaft_min``,
        ``t_shaft_max`` (keyseat depth in the shaft), ``T_hub``, ``T_hub_min``,
        ``T_hub_max`` (keyseat depth in the hub), ``r_max``, ``r_min`` (radius).

    Raises
    ------
    ValueError
        If ``series`` is not 1 or 2, or ``shaft_diameter`` is not in (3, 28] mm
        (series 1) / (3, 40] mm (series 2).

    Notes
    -----
    Row selection: ``d_min < shaft_diameter <= d_max``.

    Tolerance convention of the data: the nominal ``t_shaft`` equals
    ``t_shaft_min`` and the nominal ``T_hub`` equals ``T_hub_min``. The parallel
    key table uses the opposite convention for ``t_shaft`` (nominal = max).

    References
    ----------
    ISO 3912, named in the comments and docstrings of the code; not in the
    library; not checked against the source ((source to find)).
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
