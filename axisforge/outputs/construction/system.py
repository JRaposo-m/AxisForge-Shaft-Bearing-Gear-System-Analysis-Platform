"""
axisforge/outputs/construction/system.py

Records and text of a ``SpurHelicalGearSystem`` (``axisforge.core.mechanical_system.
parallel_axis.spur_helical.gear_system``): the gear stage, the mesh links and every shaft.

Read from the BUILT system, so the report always matches the model that was solved and never a
set of constants kept somewhere else.
"""
from __future__ import annotations

import math

from axisforge.outputs._format import attribute_rows, format_key_values, format_table, kv_record
from axisforge.outputs.construction.bearings import bearings_table_text
from axisforge.outputs.construction.loads import loads_text
from axisforge.outputs.construction.shaft import (
    SECTION_COLUMNS, SHOULDER_COLUMNS, section_records, shaft_system_records, shaft_system_text,
    shoulder_records)

# SpurHelicalGearMeshing attributes (the contact ratios exist only after gear_geometry() ran).
MESH_ATTRIBUTES = [
    ("a", "mm", "reference centre distance a"),
    ("al", "mm", "working centre distance"),
    ("u", "", "gear ratio u"),
    ("alphatw", "rad", "transverse working pressure angle"),
    ("epslon_alpha", "", "transverse contact ratio"),
    ("epslon_beta", "", "overlap ratio"),
    ("epslon_gamma", "", "total contact ratio"),
]


def source_shaft(system):
    """The shaft that is not driven by any link.

    Parameters
    ----------
    system: SpurHelicalGearSystem
        The gear system.

    Returns
    -------
    shaft: ShaftSystem or None
        The unique source shaft, or None if there is none or more than one.
    """
    driven = {id(link.shaft_b) for link in system.links}
    sources = [s for s in system.shafts if id(s) not in driven]
    return sources[0] if len(sources) == 1 else None


def system_records(system, power_W: float | None = None) -> list[dict]:
    """Description of the whole gear system.

    Parameters
    ----------
    system: SpurHelicalGearSystem
        The built (and, for the loads, resolved) system.
    power_W: float, optional (None)
        Transmitted power [W] that was given to ``system.resolve``. The system does not keep
        it, so it is passed in; when given, the power and the torque of the source shaft are
        added.

    Returns
    -------
    records: list of dict
        Rows ``{section, parameter, value, unit}``: the system (label, number of shafts and
        links, source shaft, resolved or not), the power and torque, one section per mesh link
        (line-of-centres angle ``phi_deg`` [deg], torque split, and the attributes of
        ``MESH_ATTRIBUTES`` that the meshing object has), the resolved torque of every shaft,
        and ``shaft_system_records`` of every shaft.

    Notes
    -----
    The resolved torques are read from the gear system's internal record of ``resolve``
    (``_resolved``), because there is no public accessor yet; the key is skipped if it is
    absent. The mesh forces Ft, Fr, Fa appear as the gear-mesh loads of each shaft.
    """
    s = "gear system"
    source = source_shaft(system)
    resolved = getattr(system, "_resolved", None)
    rows = [kv_record(s, "label", system.label),
            kv_record(s, "number of shafts", len(system.shafts)),
            kv_record(s, "number of mesh links", len(system.links)),
            kv_record(s, "source shaft", source.name if source is not None else "none or several"),
            kv_record(s, "resolved", resolved is not None)]
    if power_W is not None:
        rows.append(kv_record(s, "power", power_W, "W"))
        if source is not None and source.speed_rpm > 0.0:
            omega = source.speed_rpm * 2.0 * math.pi / 60.0
            rows.append(kv_record(s, "torque of the source shaft", power_W / omega, "N m"))
    for i, link in enumerate(system.links, start=1):
        sec = f"mesh {link.label or i} ({link.shaft_a.name} -> {link.shaft_b.name})"
        rows.append(kv_record(sec, "line of centres angle phi", link.phi_deg, "deg"))
        rows.append(kv_record(sec, "torque split",
                              "mode B (100 %)" if link.torque_split is None else link.torque_split))
        rows.append(kv_record(sec, "distribute loads", bool(link.distribute_loads)))
        rows.append(kv_record(sec, "driver gear", link.gear_a.label or link.gear_a.role))
        rows.append(kv_record(sec, "driven gear", link.gear_b.label or link.gear_b.role))
        rows += attribute_rows(link.meshing, sec, MESH_ATTRIBUTES)
    if resolved is not None:
        for entry in resolved.values():
            sec = f"resolved shaft {entry['name']}"
            if entry.get("T_out_Nm") is not None:
                rows.append(kv_record(sec, "torque T_out", entry["T_out_Nm"], "N m"))
            rows.append(kv_record(sec, "rotation direction", entry.get("rotation_dir")))
    for shaft in system.shafts:
        rows += shaft_system_records(shaft)
    return rows


# Parameters shown in the short text (the records and the CSV keep everything).
SUMMARY_PARAMETERS = {
    "torque of the source shaft", "line of centres angle phi", "gear ratio u",
    "reference centre distance a", "transverse contact ratio", "overlap ratio",
    "normal module mn", "number of teeth z", "face width b", "helix angle beta",
    "normal pressure angle alpha_n", "reference diameter d", "material",
}


def _without_shaft(records):
    return [{k: v for k, v in r.items() if k != "shaft"} for r in records]


def system_text(system, power_W: float | None = None) -> str:
    """Short description of the whole gear system, for reading.

    Parameters
    ----------
    system: SpurHelicalGearSystem
        The built system.
    power_W: float, optional (None)
        Transmitted power [W]; see ``system_records``.

    Returns
    -------
    text: str
        The gear stage (power, speeds, torque, gears and mesh, main parameters only), the
        sections and shoulders of the shafts (once, when they are identical), a table of the
        bearings of the first shaft and the loads on every shaft as AxisForge prints them. The
        complete description is ``system_records`` (and ``shaft_system_text`` for one shaft).
    """
    shafts = list(system.shafts)
    first = shafts[0]
    stage = []
    if power_W is not None:
        stage.append(kv_record("gear stage", "power", power_W, "W"))
    stage += [kv_record("gear stage", f"speed {ss.name}", ss.speed_rpm, "rpm") for ss in shafts]
    stage += [r for r in system_records(system, power_W)
              if r["parameter"] in SUMMARY_PARAMETERS
              and (r["section"].startswith(("mesh ", "gear ")) or r["section"] == "gear system")]
    parts = [format_key_values(stage)]

    sections = [section_records(ss) for ss in shafts]
    shoulders = [shoulder_records(ss) for ss in shafts]
    identical = all(_without_shaft(sections[0]) == _without_shaft(x) for x in sections[1:]) and \
        all(_without_shaft(shoulders[0]) == _without_shaft(x) for x in shoulders[1:])
    for ss, sec, sho in zip(shafts, sections, shoulders):
        if identical and ss is not first:
            break
        name = " = ".join(s.name for s in shafts) if identical and len(shafts) > 1 else ss.name
        text = (f"Shaft {name} (length {ss.shaft.total_length:g} mm):\n"
                + format_table(SECTION_COLUMNS, sec, indent=2))
        if sho:
            text += "\n\n" + format_table(SHOULDER_COLUMNS, sho, indent=2)
        parts.append(text)

    note = (f"\n  (same bearings on {', '.join(s.name for s in shafts[1:])})"
            if len(shafts) > 1 else "")
    parts.append(f"Bearings of {first.name}:\n" + bearings_table_text(first.bearings, 2) + note)
    parts += [loads_text(ss) for ss in shafts]
    return "\n\n".join(parts)


def system_full_text(system, power_W: float | None = None) -> str:
    """Complete description of the whole gear system.

    Parameters
    ----------
    system: SpurHelicalGearSystem
        The built system.
    power_W: float, optional (None)
        Transmitted power [W]; see ``system_records``.

    Returns
    -------
    text: str
        ``SpurHelicalGearSystem.summary()``, the repr of every link, the power, the torque and
        the mesh blocks, and the text of every shaft (``shaft_system_text``).
    """
    stage = [r for r in system_records(system, power_W)
             if r["section"].startswith("mesh ")
             or (r["section"] == "gear system"
                 and r["parameter"] in ("power", "torque of the source shaft"))]
    parts = [system.summary(), "\n".join(repr(link) for link in system.links),
             format_key_values(stage)]
    parts += [shaft_system_text(shaft) for shaft in system.shafts]
    return "\n\n".join(p for p in parts if p)


def print_system(system, power_W: float | None = None) -> None:
    """Print ``system_text`` to the console.

    Parameters
    ----------
    system: SpurHelicalGearSystem
        The built system.
    power_W: float, optional (None)
        Transmitted power [W].
    """
    print(system_text(system, power_W))
