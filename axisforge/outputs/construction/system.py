"""Records and text of a ``Transmission`` (``axisforge.core.mechanical_system.transmission``):
the gear stages of its written power path and every shaft.

Read from the BUILT system, so the report always matches the model that was solved and never a
set of constants kept somewhere else. Every function takes either the ``Transmission`` or the
``ResolvedTransmission`` that its ``resolve`` returns. The mesh loads, the torques and the senses
of rotation are in the ``ResolvedTransmission`` only: a ``Transmission`` is described with its
shafts as given, without mesh loads.
"""
from __future__ import annotations

import math

from axisforge.outputs._format import attribute_rows, format_key_values, format_table, kv_record
from axisforge.outputs.construction.bearings import bearings_table_text
from axisforge.outputs.construction.loads import loads_text
from axisforge.outputs.construction.shaft import (
    SECTION_COLUMNS, SHOULDER_COLUMNS, section_records, shaft_system_records, shaft_system_text,
    shoulder_records)

# Attributes of a cylindrical ``GearPair``.
MESH_ATTRIBUTES = [
    ("a", "mm", "reference centre distance a"),
    ("al", "mm", "working centre distance"),
    ("u", "", "gear ratio u"),
    ("alphatw", "rad", "transverse working pressure angle"),
    ("epslon_alpha", "", "transverse contact ratio"),
    ("epslon_beta", "", "overlap ratio"),
    ("epslon_gamma", "", "total contact ratio"),
]


def system_records(system, power_W: float | None = None) -> list[dict]:
    """Description of the whole gear system.

    Parameters
    ----------
    system : Transmission or ResolvedTransmission
        The built system, or what its ``resolve`` returned. A ``ResolvedTransmission`` is
        described with its loaded copies (loads and position), torques and senses of rotation; a
        ``Transmission`` with its shafts as given, which carry no mesh load (``resolve`` does
        not modify them), and with no resolved section.
    power_W : float, optional (None)
        Transmitted power [W] that was given to ``resolve``. The system does not keep it, so it
        is passed in; when given, the power and the torque of the source shaft are added.

    Returns
    -------
    records : list of dict
        Rows ``{section, parameter, value, unit}`` (``kv_record`` schema). Section
        ``gear system``: label, number of shafts, number of mesh links (the stages of the
        path), source shaft, resolved (bool); with ``power_W`` also power [W] and, if the source
        shaft has ``speed_rpm > 0``, its torque [N m] = ``power_W / omega`` with
        ``omega = speed_rpm * 2 pi / 60`` [rad/s]. One section ``mesh ...`` per stage: torque
        split (the ``share`` of the stage [-]; the text "mode B (100 %)" when it is 1.0) and,
        for a ``PairStage``, line-of-centres angle ``phi_deg`` [deg],
        ``distribute_loads``, driver and driven gear, and the attributes of ``MESH_ATTRIBUTES``
        of the pair (distances [mm], angle ``alphatw`` [rad], ratios without unit). For a
        ``ResolvedTransmission``, one section ``resolved shaft <name>`` per shaft: torque
        ``T_out`` [N m] entering it and rotation direction. Then ``shaft_system_records`` of
        every shaft.

    Notes
    -----
    The mesh forces Ft, Fr, Fa appear as the gear-mesh loads of each shaft only for a
    ``ResolvedTransmission``.

    Warnings
    --------
    The ``speed`` rows of the shafts and the torque of the source shaft come from
    ``ShaftSystem.speed_rpm`` as set by the user: ``resolve`` returns the speed of every shaft
    in ``ResolvedTransmission.speed_rpm`` and does not write it on the shafts. The torque of
    the source shaft is shown in N m, as returned by the formula above.
    """
    s = "gear system"
    # a ResolvedTransmission holds the Transmission it was resolved from; a Transmission is itself
    built = getattr(system, "transmission", system)
    resolved = system if built is not system else None
    system = built
    source = system.source
    rows = [kv_record(s, "label", system.label),
            kv_record(s, "number of shafts", len(system.shafts)),
            kv_record(s, "number of mesh links", len(system.path)),
            kv_record(s, "source shaft", source.name),
            kv_record(s, "resolved", resolved is not None)]
    if power_W is not None:
        rows.append(kv_record(s, "power", power_W, "W"))
        if source.speed_rpm > 0.0:
            omega = source.speed_rpm * 2.0 * math.pi / 60.0
            rows.append(kv_record(s, "torque of the source shaft", power_W / omega, "N m"))
    for i, stage in enumerate(system.path, start=1):
        sec = f"mesh {stage.label or i} ({' -> '.join(sh.name for sh in stage.shafts)})"
        pair_stage = hasattr(stage, "pair")       # a PairStage; another stage type has only the common rows
        if pair_stage:
            rows.append(kv_record(sec, "line of centres angle phi", stage.phi_deg, "deg"))
        # QUESTION(Q46): row name and the text "mode B (100 %)" are those written for the replaced link when it took the full torque (torque_split=None); now share = 1.0. The alternative is the number.
        rows.append(kv_record(sec, "torque split", "mode B (100 %)" if stage.share == 1.0 else stage.share))
        if pair_stage:
            rows.append(kv_record(sec, "distribute loads", bool(stage.distribute_loads)))
            # Q17 (José, 2026-10-15): the gear element has no role; the driver of a stage is its gear_a
            rows.append(kv_record(sec, "driver gear", stage.gear_a.label or "gear_a"))
            rows.append(kv_record(sec, "driven gear", stage.gear_b.label or "gear_b"))
            rows += attribute_rows(stage.pair, sec, MESH_ATTRIBUTES)
    if resolved is not None:
        for shaft, torque, rotation in zip(resolved.shafts, resolved.torque_Nm, resolved.rotation):
            sec = f"resolved shaft {shaft.name}"
            rows.append(kv_record(sec, "torque T_out", torque, "N m"))
            rows.append(kv_record(sec, "rotation direction", rotation))
    for shaft in (system.shafts if resolved is None else resolved.shafts):
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
    """Copy of flat records without the ``shaft`` key, to compare shafts by geometry only.

    Parameters
    ----------
    records : iterable of dict
        Records such as those of ``section_records`` or ``shoulder_records``.

    Returns
    -------
    records : list of dict
        The same records without the ``shaft`` (name) key.
    """
    return [{k: v for k, v in r.items() if k != "shaft"} for r in records]


def system_text(system, power_W: float | None = None) -> str:
    """Short description of the whole gear system, for reading.

    Parameters
    ----------
    system : Transmission or ResolvedTransmission
        The built system (at least one shaft), or what its ``resolve`` returned; see ``system_records``.
    power_W : float, optional (None)
        Transmitted power [W]; see ``system_records``.

    Returns
    -------
    text : str
        The gear stage (power [W], speed of every shaft [rpm], torque [N m], gears and mesh,
        the parameters in ``SUMMARY_PARAMETERS`` only), the sections and shoulders of the
        shafts (once, when they are identical), a table of the bearings of the first shaft and
        the loads on every shaft as AxisForge prints them. The complete description is
        ``system_records`` (and ``shaft_system_text`` for one shaft).

    Warnings
    --------
    The ``speed <shaft>`` lines show ``ShaftSystem.speed_rpm`` as set by the user; ``resolve``
    does not write the speed on the shafts, so a driven shaft shows whatever was set on it.
    """
    shafts = list(system.shafts)       # of the Transmission, or the loaded copies of the ResolvedTransmission
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
    system : Transmission or ResolvedTransmission
        The built system, or what its ``resolve`` returned; see ``system_records``.
    power_W : float, optional (None)
        Transmitted power [W]; see ``system_records``.

    Returns
    -------
    text : str
        ``ResolvedTransmission.summary()`` (for a ``ResolvedTransmission``), the repr of every
        stage, the power [W], the torque [N m] and the mesh blocks, and the text of every shaft
        (``shaft_system_text``), joined by blank lines; empty parts are left out.
    """
    stage = [r for r in system_records(system, power_W)
             if r["section"].startswith("mesh ")
             or (r["section"] == "gear system"
                 and r["parameter"] in ("power", "torque of the source shaft"))]
    built = getattr(system, "transmission", system)
    parts = [system.summary() if built is not system else "",
             "\n".join(repr(st) for st in built.path), format_key_values(stage)]
    parts += [shaft_system_text(shaft) for shaft in system.shafts]
    return "\n\n".join(p for p in parts if p)


def print_system(system, power_W: float | None = None) -> None:
    """Print ``system_text`` to the console.

    Parameters
    ----------
    system : Transmission or ResolvedTransmission
        The built system, or what its ``resolve`` returned; see ``system_records``.
    power_W : float, optional (None)
        Transmitted power [W].
    """
    print(system_text(system, power_W))
