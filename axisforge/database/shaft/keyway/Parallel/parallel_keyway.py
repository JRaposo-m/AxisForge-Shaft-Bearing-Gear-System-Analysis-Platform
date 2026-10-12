"""
Lookup of parallel-key keyway dimensions by shaft diameter.

Reads ``parallel_keyway.csv`` (named "DIN 6885" in the code; the file carries no
part number, edition or source). Tolerance bands are pre-resolved into _min/_max
columns so callers can pick the critical value. All lengths in mm.
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
    """Read the CSV: float columns to float, empty cell to ``None``, others kept as str."""
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
    Keyway dimensions of the table row that contains a shaft diameter.

    Used by ``Keyway.from_standard`` (core/machine_elements/shaft/shaft.py), its
    only caller, which reads ``b``, ``t_shaft``, ``t_shaft_min``, ``t_shaft_max``,
    ``L_min`` and ``L_max``.

    Parameters
    ----------
    shaft_diameter : float
        Shaft diameter [mm].

    Returns
    -------
    row : dict
        The whole table row (the same dict object held by the module, not a
        copy). Keys, all lengths [mm]; an empty cell is ``None``:

        - ``d_min``, ``d_max``: shaft diameter band.
        - ``b``, ``h``: key width and height.
        - ``b_fine``, ``h_fine``, ``t_fine_shaft``, ``T_fine_hub``: key section
          and keyseat depths of a second variant (``None`` for d <= 12 mm; not
          labelled in the file).
        - ``L_min``, ``L_max``: key length range.
        - ``t_shaft``, ``t_shaft_min``, ``t_shaft_max``: depth of the keyseat in
          the shaft.
        - ``T_hub``, ``T_hub_min``, ``T_hub_max``: depth of the keyseat in the
          hub.
        - ``c_min``, ``c_max``: chamfer; ``r_min``: keyseat radius.
        - ``screw``: string (not a length, e.g. ``"M2.5x6"``), or ``None``.

    Raises
    ------
    ValueError
        If ``shaft_diameter`` is not in (6, 75] mm (the lower limit of the first
        row is excluded).

    Notes
    -----
    Row selection: ``d_min < shaft_diameter <= d_max``.

    Tolerance convention of the data: the nominal ``t_shaft`` equals
    ``t_shaft_max`` and the nominal ``T_hub`` equals ``T_hub_min``. The Woodruff
    table uses the opposite convention for ``t_shaft`` (nominal = min).

    References
    ----------
    DIN 6885, named in the comments and docstrings of the code; not in the
    library; not checked against the source ((source to find)).
    """
    for row in _TABLE:
        if row["d_min"] < shaft_diameter <= row["d_max"]:
            return row
    raise ValueError(
        f"shaft_diameter={shaft_diameter} mm is outside DIN 6885 table range "
        f"({_TABLE[0]['d_min']}-{_TABLE[-1]['d_max']} mm)"
    )