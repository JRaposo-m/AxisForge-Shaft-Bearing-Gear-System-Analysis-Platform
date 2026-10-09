"""
axisforge/outputs/construction/shaft.py

Records and text of a ``ShaftSystem`` (``axisforge.core.mechanical_system.parallel_axis.
spur_helical.shaft_system``): the shaft, its gears, its bearings and its loads.

A ``ShaftSystem`` already reaches everything that is built on one shaft (``.shaft``, ``.gears``,
``.bearings``, ``.loads``), so this module only needs the ``ShaftSystem``. The text is made of
AxisForge's own ``summary()`` and ``__repr__`` output (``ShaftSystem.summary``,
``Bearing.summary``, the load reprs) plus aligned tables of the geometry.
"""
from __future__ import annotations

from axisforge.outputs._format import (
    attribute_rows, format_key_values, format_table, kv_record)
from axisforge.outputs.construction.bearings import bearing_records, bearing_text
from axisforge.outputs.construction.loads import load_records, loads_text

# SpurHelicalGear attributes. The gear stores its pressure and helix angles as inputs in degrees
# (alpha_n_deg, beta_n_deg).
GEAR_ATTRIBUTES = [
    ("mn", "mm", "normal module mn"),
    ("z", "", "number of teeth z"),
    ("x", "", "profile shift coefficient x"),
    ("b", "mm", "face width b"),
    ("alpha_n_deg", "deg", "normal pressure angle alpha_n"),
    ("beta_n_deg", "deg", "helix angle beta"),
    ("d", "mm", "reference diameter d"),
    ("da", "mm", "tip diameter da"),
    ("df", "mm", "root diameter df"),
    ("material_id", "", "material"),
]


def shaft_geometry_records(shaft_system) -> list[dict]:
    """Operating data and geometry of the shaft itself.

    Parameters
    ----------
    shaft_system: ShaftSystem
        The shaft container.

    Returns
    -------
    records: list of dict
        Rows ``{section, parameter, value, unit}`` for section ``shaft <name>``: label, speed
        [rpm], design life [h], axis offset (y, z) and origin_x [mm], total length [mm]; then,
        for every section of the shaft, length, diameter, inner diameter [mm], material and
        surface finish Ra, and for every shoulder its axial position, fillet radius and the two
        diameters [mm].

    Notes
    -----
    ``speed_rpm`` is the value stored on the ``ShaftSystem``; the gear system does not set it
    when it resolves the loads, so it is whatever the user (or the study) gave.
    """
    shaft = shaft_system.shaft
    s = f"shaft {shaft_system.name}"
    y, z = shaft_system.shaft_position
    rows = [kv_record(s, "label", shaft_system.label or getattr(shaft, "label", "")),
            kv_record(s, "speed", shaft_system.speed_rpm, "rpm"),
            kv_record(s, "design life", shaft_system.design_life_hours, "h"),
            kv_record(s, "axis offset y", y, "mm"),
            kv_record(s, "axis offset z", z, "mm"),
            kv_record(s, "origin_x", shaft_system.shaft_origin_x, "mm"),
            kv_record(s, "total length", shaft.total_length, "mm"),
            kv_record(s, "number of sections", len(shaft.sections))]
    sec = f"{s} - sections"
    for i, section in enumerate(shaft.sections, start=1):
        rows += attribute_rows(section, sec, [
            ("length", "mm"), ("diameter", "mm"), ("inner_diameter", "mm"),
            ("material_id", ""), ("surface_finish_ra", "um", "surface finish Ra")],
            prefix=f"{i} {section.label or 'section'} ".rstrip() + " ")
        rows.append(kv_record(sec, f"{i} {section.label or 'section'} keyways",
                              len(getattr(section, "keyways", []))))
    shoulders = shaft.shoulders() if callable(getattr(shaft, "shoulders", None)) else []
    sho = f"{s} - shoulders"
    for z_pos, shoulder in shoulders:
        prefix = f"at z = {z_pos:g} mm "
        rows.append(kv_record(sho, f"{prefix}fillet radius", shoulder.fillet_radius, "mm"))
        rows.append(kv_record(sho, f"{prefix}large diameter", shoulder.diameter_large, "mm"))
        rows.append(kv_record(sho, f"{prefix}small diameter", shoulder.diameter_small, "mm"))
    return rows


SECTION_COLUMNS = [
    ("index", "#", ".0f"), ("label", "section", ""), ("length_mm", "L [mm]", ".1f"),
    ("diameter_mm", "d [mm]", ".1f"), ("inner_diameter_mm", "d_in [mm]", ".1f"),
    ("material_id", "material", ""), ("surface_finish_ra_um", "Ra [um]", ".2f"),
    ("n_keyways", "keyways", ".0f"),
]
SHOULDER_COLUMNS = [
    ("z_mm", "z [mm]", ".1f"), ("diameter_small_mm", "d small [mm]", ".1f"),
    ("diameter_large_mm", "d large [mm]", ".1f"), ("fillet_radius_mm", "fillet r [mm]", ".2f"),
]


def section_records(shaft_system) -> list[dict]:
    """One record per shaft section, from left to right.

    Parameters
    ----------
    shaft_system: ShaftSystem
        The shaft container.

    Returns
    -------
    records: list of dict
        ``shaft``, ``index`` (from 1), ``label``, ``length_mm``, ``diameter_mm``,
        ``inner_diameter_mm``, ``material_id``, ``surface_finish_ra_um`` and ``n_keyways``.
    """
    records = []
    for i, sec in enumerate(shaft_system.shaft.sections, start=1):
        records.append(dict(shaft=shaft_system.name, index=i, label=sec.label,
                            length_mm=float(sec.length), diameter_mm=float(sec.diameter),
                            inner_diameter_mm=float(sec.inner_diameter),
                            material_id=sec.material_id,
                            surface_finish_ra_um=float(sec.surface_finish_ra),
                            n_keyways=len(getattr(sec, "keyways", []))))
    return records


def shoulder_records(shaft_system) -> list[dict]:
    """One record per shoulder (change of diameter) of the shaft.

    Parameters
    ----------
    shaft_system: ShaftSystem
        The shaft container.

    Returns
    -------
    records: list of dict
        ``shaft``, ``z_mm`` (axial position), ``diameter_small_mm``, ``diameter_large_mm`` and
        ``fillet_radius_mm``. Empty if the shaft has no shoulder.
    """
    shaft = shaft_system.shaft
    shoulders = shaft.shoulders() if callable(getattr(shaft, "shoulders", None)) else []
    return [dict(shaft=shaft_system.name, z_mm=float(z), diameter_small_mm=float(sh.diameter_small),
                 diameter_large_mm=float(sh.diameter_large),
                 fillet_radius_mm=float(sh.fillet_radius)) for z, sh in shoulders]


def gear_element_records(shaft_system) -> list[dict]:
    """The gears mounted on a shaft.

    Parameters
    ----------
    shaft_system: ShaftSystem
        The shaft container.

    Returns
    -------
    records: list of dict
        Per ``GearElement``: role, rotation direction, axial position [mm] and every attribute
        of ``GEAR_ATTRIBUTES`` that the gear has. Empty if the shaft has no gear.
    """
    rows: list[dict] = []
    for i, ge in enumerate(shaft_system.gears, start=1):
        section = f"gear {ge.label or i} ({shaft_system.name})"
        rows.append(kv_record(section, "role", ge.role))
        rows.append(kv_record(section, "rotation direction",
                              "propagated" if ge.rotation_dir is None else ge.rotation_dir))
        rows.append(kv_record(section, "axial position", ge.position, "mm"))
        rows += attribute_rows(ge.gear, section, GEAR_ATTRIBUTES)
    return rows


def shaft_system_records(shaft_system) -> list[dict]:
    """Everything built on one shaft.

    Parameters
    ----------
    shaft_system: ShaftSystem
        The shaft container.

    Returns
    -------
    records: list of dict
        ``shaft_geometry_records``, ``gear_element_records``, then ``bearing_records`` of every
        bearing and ``load_records``.
    """
    rows = shaft_geometry_records(shaft_system) + gear_element_records(shaft_system)
    for bearing in shaft_system.bearings:
        rows += bearing_records(bearing, shaft_system.name)
    return rows + load_records(shaft_system)


def shaft_system_text(shaft_system) -> str:
    """Readable description of one shaft.

    Parameters
    ----------
    shaft_system: ShaftSystem
        The shaft container.

    Returns
    -------
    text: str
        ``ShaftSystem.summary()`` (speed, life, position, counts), the table of sections and
        the table of shoulders, the gears, the ``summary()`` of every bearing with its geometry,
        and the ``repr`` of every load.
    """
    parts = [shaft_system.summary(),
             "Sections:\n" + format_table(SECTION_COLUMNS, section_records(shaft_system), indent=2)]
    shoulders = shoulder_records(shaft_system)
    if shoulders:
        parts.append("Shoulders:\n" + format_table(SHOULDER_COLUMNS, shoulders, indent=2))
    gears = gear_element_records(shaft_system)
    if gears:
        parts.append(format_key_values(gears))
    parts += [bearing_text(b) for b in shaft_system.bearings]
    parts.append(loads_text(shaft_system))
    return "\n\n".join(parts)


def print_shaft_system(shaft_system) -> None:
    """Print ``shaft_system_text`` to the console.

    Parameters
    ----------
    shaft_system: ShaftSystem
        The shaft container.
    """
    print(shaft_system_text(shaft_system))
