"""
axisforge/outputs/construction/loads.py

Records and text of the loads of a ``ShaftSystem`` (``axisforge.core.loads``): the user loads
and the loads injected by ``SpurHelicalGearSystem.resolve``.

The text is the ``__repr__`` of each load, so it always matches what AxisForge prints for them.
"""
from __future__ import annotations

from axisforge.outputs._format import attribute_rows, kv_record

# Unit of ``magnitude`` by load class. The classes document N for forces; TorqueLoad is in N m
# (see SpurHelicalGearSystem.resolve). ExternalMoment is left empty: its unit is not stated in
# axisforge.core.loads.
MAGNITUDE_UNIT = {
    "RadialLoad": "N",
    "AxialLoad": "N",
    "DistributedRadialLoad": "N",
    "TorqueLoad": "N m",
    "ExternalMoment": "",
}


def load_records(shaft_system) -> list[dict]:
    """One group of rows per load on a shaft.

    Parameters
    ----------
    shaft_system: ShaftSystem
        The shaft; its ``loads`` property is read (sorted by AxisForge as it stores them).

    Returns
    -------
    records: list of dict
        Rows ``{section, parameter, value, unit}`` with section ``loads <shaft name>``. Every
        load gives its type, source (``user`` or ``gear_mesh``), axial position (or ``x_lo`` /
        ``x_hi`` for a distributed load), magnitude and, where the load has a constant one,
        the angular position ``theta_deg`` [deg] from +Y toward +Z.
    """
    section = f"loads {shaft_system.name}"
    rows: list[dict] = []
    for i, load in enumerate(shaft_system.loads, start=1):
        kind = type(load).__name__
        prefix = f"{i} {load.label or kind} "
        rows.append(kv_record(section, f"{prefix}type", kind))
        rows += attribute_rows(load, section, [
            ("source", ""), ("position", "mm"), ("x_lo", "mm"), ("x_hi", "mm"),
            ("magnitude", MAGNITUDE_UNIT.get(kind, "")), ("theta_deg", "deg"),
        ], prefix=prefix)
    return rows


def loads_text(shaft_system) -> str:
    """The loads of a shaft as AxisForge prints them.

    Parameters
    ----------
    shaft_system: ShaftSystem
        The shaft.

    Returns
    -------
    text: str
        A title and one ``repr`` line per load; ``(none)`` if the shaft has no loads.
    """
    lines = [f"Loads on {shaft_system.name}:"]
    lines += [f"    {load!r}" for load in shaft_system.loads] or ["    (none)"]
    return "\n".join(lines)


def print_loads(shaft_system) -> None:
    """Print ``loads_text`` to the console.

    Parameters
    ----------
    shaft_system: ShaftSystem
        The shaft.
    """
    print(loads_text(shaft_system))
