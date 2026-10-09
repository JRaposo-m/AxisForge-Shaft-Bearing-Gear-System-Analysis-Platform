"""
axisforge/outputs/_format.py

Formatting and CSV helpers shared by every module of ``axisforge.outputs``.

Nothing here knows about bearings, shafts or gears: it turns lists of plain dicts ("records")
into aligned text and CSV files, and back. Only the standard library is used.

Record conventions
------------------
* A record is a ``dict`` of scalars (``float``, ``int``, ``bool``, ``str``).
* The unit of a number is part of its key (``Fr_N``, ``delta_r_mm``, ``psi_rad``). A key without
  a unit suffix is dimensionless or text.
* NaN means "not available"; it is written as ``nan`` in CSV and as ``-`` in text. ``inf`` is
  kept as ``inf``.
* Descriptions of a built system use the four-key schema
  ``{"section", "parameter", "value", "unit"}`` (see ``kv_record``).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

NAN = math.nan

# Column of a text table: (key, header, format_spec) or (key, header, format_spec, scale).
Column = tuple


def format_value(value, spec: str = ".6g") -> str:
    """Format one scalar for a text table.

    Parameters
    ----------
    value: bool, int, float, str or None
        The value. ``bool`` gives ``yes`` / ``no``; ``None`` and NaN give ``-``.
    spec: str
        ``format`` specification applied to ``float`` values (and to ``int`` values when it is
        not empty).

    Returns
    -------
    text: str
        The formatted value.
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "-" if math.isnan(value) else format(value, spec)
    if isinstance(value, int) and spec:
        try:
            return format(value, spec)
        except ValueError:
            return str(value)
    return str(value)


def format_table(columns: Sequence[Column], records: Sequence[dict], indent: int = 0) -> str:
    """Fixed-width text table.

    Parameters
    ----------
    columns: sequence of tuple
        ``(key, header, format_spec)`` or ``(key, header, format_spec, scale)``. When ``scale``
        is given, a float value is multiplied by it before formatting; this is how one record
        key is shown in two units (for example ``psi_rad`` as ``psi [rad]`` and, with
        ``scale=1e3``, as ``psi [mrad]``). A key missing from a record counts as NaN.
    records: sequence of dict
        One table row per record.
    indent: int
        Number of leading spaces on every line.

    Returns
    -------
    text: str
        Header, a rule and one line per record, right-aligned. Empty ``records`` give the header
        and the rule only.
    """
    cells = []
    for record in records:
        row = []
        for column in columns:
            key, _, spec = column[:3]
            value = record.get(key, NAN)
            if len(column) > 3 and isinstance(value, float):
                value = value * column[3]
            row.append(format_value(value, spec))
        cells.append(row)
    headers = [column[1] for column in columns]
    widths = [max([len(h)] + [len(row[i]) for row in cells]) for i, h in enumerate(headers)]
    pad = " " * indent
    lines = [pad + "  ".join(h.rjust(w) for h, w in zip(headers, widths)),
             pad + "  ".join("-" * w for w in widths)]
    lines += [pad + "  ".join(c.rjust(w) for c, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)


def kv_record(section: str, parameter: str, value, unit: str = "") -> dict:
    """One row of a system description.

    Parameters
    ----------
    section: str
        Group the row belongs to (for example ``"shaft shaft_1"``).
    parameter: str
        Name of the parameter.
    value: float, int, bool or str
        The value.
    unit: str
        Unit text, empty when dimensionless.

    Returns
    -------
    record: dict
        ``{"section", "parameter", "value", "unit"}``.
    """
    return dict(section=section, parameter=parameter, value=value, unit=unit)


def format_key_values(records: Iterable[dict], name_width: int = 34, value_width: int = 18,
                      spec: str = ".6g") -> str:
    """Aligned ``parameter value unit`` blocks, one block per section.

    Parameters
    ----------
    records: iterable of dict
        Records with the keys of ``kv_record``, in display order. Consecutive records with the
        same ``section`` form one block.
    name_width: int
        Width of the parameter column.
    value_width: int
        Width of the value column.
    spec: str
        Format specification for float values.

    Returns
    -------
    text: str
        The blocks separated by a blank line; each starts with ``<section>:``.
    """
    lines: list[str] = []
    section = None
    for r in records:
        if r["section"] != section:
            section = r["section"]
            if lines:
                lines.append("")
            lines.append(f"{section}:")
        lines.append(f"    {r['parameter']:<{name_width}} "
                     f"{format_value(r['value'], spec):>{value_width}}  {r.get('unit', '')}".rstrip())
    return "\n".join(lines)


def union_fields(records: Iterable[dict]) -> list[str]:
    """Every key of a list of dicts, in order of first appearance.

    Parameters
    ----------
    records: iterable of dict
        The records.

    Returns
    -------
    fields: list of str
        Keys without repetition.
    """
    fields: dict[str, None] = {}
    for r in records:
        fields.update(dict.fromkeys(r))
    return list(fields)


def to_python(value):
    """Convert a NumPy scalar to the matching Python scalar; anything else is returned as is.

    Parameters
    ----------
    value: object
        Any value.

    Returns
    -------
    value: object
        ``value.item()`` for a 0-d NumPy scalar or array, otherwise ``value`` itself.
    """
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return value.item()
    return value


def attribute_rows(obj, section: str, attributes, prefix: str = "") -> list[dict]:
    """``kv_record`` rows for the attributes an object actually has.

    Parameters
    ----------
    obj: object
        Any object.
    section: str
        Section of every row.
    attributes: iterable of tuple
        ``(attribute name, unit)`` or ``(attribute name, unit, parameter label)``. The label
        defaults to the attribute name.
    prefix: str
        Text put before every parameter label.

    Returns
    -------
    rows: list of dict
        One row per attribute that exists and holds a scalar; the others are skipped, so the
        listing does not break when a class gains or loses an attribute.
    """
    rows = []
    for item in attributes:
        name, unit = item[0], item[1]
        label = item[2] if len(item) > 2 else name
        if not hasattr(obj, name):
            continue
        value = to_python(getattr(obj, name))
        if isinstance(value, (bool, int, float, str)):
            rows.append(kv_record(section, f"{prefix}{label}", value, unit))
    return rows


def scalar_attributes(obj) -> dict:
    """Public scalar attributes of an object.

    Parameters
    ----------
    obj: object
        A dataclass, an object with ``__dict__`` or an object with ``__slots__``.

    Returns
    -------
    attributes: dict
        ``{name: value}`` for every public attribute that is a ``bool``, ``int``, ``float`` or
        ``str`` (NumPy scalars are converted to Python scalars); arrays, objects and names that
        start with an underscore are left out.
    """
    if hasattr(obj, "__dataclass_fields__"):
        names = list(obj.__dataclass_fields__)
    elif hasattr(obj, "__dict__"):
        names = list(vars(obj))
    else:
        names = [n for cls in type(obj).__mro__ for n in getattr(cls, "__slots__", ())]
    out = {}
    for name in names:
        if name.startswith("_"):
            continue
        value = getattr(obj, name, None)
        if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
            value = value.item()
        if isinstance(value, (bool, int, float, str)):
            out[name] = value
    return out


def write_records_csv(path, records: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    """Write records to a CSV file.

    Parameters
    ----------
    path: str or Path
        Output file; the parent folder must exist.
    records: sequence of dict
        The records.
    fields: sequence of str, optional (None)
        Column order. None uses the union of the keys in order of first appearance. A key that
        is in a record but not in ``fields`` is ignored; a missing key is written empty.

    Notes
    -----
    UTF-8, ``newline=""``. NaN is written as ``nan``, ``inf`` as ``inf``, booleans as ``True`` /
    ``False``.
    """
    columns = list(fields) if fields is not None else union_fields(records)
    with open(Path(path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def read_records_csv(path, text_fields: Iterable[str] = (),
                     bool_fields: Iterable[str] = ()) -> list[dict]:
    """Read a CSV written by ``write_records_csv``.

    Parameters
    ----------
    path: str or Path
        The file.
    text_fields: iterable of str
        Columns kept as text.
    bool_fields: iterable of str
        Columns read as ``bool`` (``"True"`` is True, anything else False).

    Returns
    -------
    records: list of dict
        Numbers as ``float`` (``nan`` and ``inf`` included), an empty cell as NaN, and any other
        non-numeric cell as text.
    """
    text_fields, bool_fields = set(text_fields), set(bool_fields)

    def parse(name: str, cell: str):
        if name in text_fields:
            return cell
        if name in bool_fields:
            return cell == "True"
        if cell == "":
            return NAN
        try:
            return float(cell)
        except ValueError:
            return cell

    with open(Path(path), newline="", encoding="utf-8") as f:
        return [{k: parse(k, v) for k, v in row.items()} for row in csv.DictReader(f)]
