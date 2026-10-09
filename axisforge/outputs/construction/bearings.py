"""
axisforge/outputs/construction/bearings.py

Records and text of an assembled ``Bearing`` (``axisforge.core.machine_elements.bearings``).

The record lists the attributes the bearing actually carries (a deep groove ball bearing has no
``Lwe``, a roller bearing has no ``alpha_0`` set by hand, ...), so it follows the family without
a per-family branch. The text is ``Bearing.summary()``, i.e. what AxisForge prints itself.
"""
from __future__ import annotations

import math

from axisforge.outputs._format import (
    NAN, attribute_rows, format_key_values, format_table, kv_record, to_python)

# (attribute, unit, label). Angles stored by the bearing are in rad (alpha_0).
GEOMETRY_ATTRIBUTES = [
    ("Dw", "mm", "ball diameter Dw"),
    ("Dwe", "mm", "roller diameter Dwe"),
    ("Lwe", "mm", "roller effective length Lwe"),
    ("Dpw", "mm", "pitch diameter Dpw"),
    ("Z", "", "rolling elements per row Z"),
    ("i", "", "number of rows i"),
    ("n_s", "", "laminae per roller n_s"),
    ("s", "mm", "radial internal clearance s"),
    ("A", "mm", "groove curvature-centre distance A"),
    ("ri", "mm", "inner groove radius ri"),
    ("re", "mm", "outer groove radius re"),
    ("alpha_0", "rad", "free contact angle alpha_0"),
    ("gamma", "", "gamma"),
    ("reduction_factor", "", "reduction factor"),
]

# Material and contact constants used by the ISO/TS 16281 analysis. e1/nu1 belong to the rolling
# element and e2/nu2 to the raceways (see the family classes). Units of e1, e2 are not stated
# by the bearing classes and are left empty.
CONTACT_ATTRIBUTES = [
    ("e1", "", "elastic modulus, rolling element e1"),
    ("e2", "", "elastic modulus, raceway e2"),
    ("nu1", "", "Poisson ratio, rolling element nu1"),
    ("nu2", "", "Poisson ratio, raceway nu2"),
    ("cp", "N/mm^1.5", "contact constant cp (Q = cp delta^1.5)"),
    ("Ri", "mm", "raceway contact radius Ri"),
    ("cL", "", "line-contact constant cL"),
    ("cs", "", "line-contact constant cs"),
]


def bearing_records(bearing, shaft_name: str = "") -> list[dict]:
    """Description of one assembled bearing.

    Parameters
    ----------
    bearing: Bearing
        An assembled bearing.
    shaft_name: str
        Shaft that carries it; only used in the section title.

    Returns
    -------
    records: list of dict
        Rows ``{section, parameter, value, unit}``: family, designation, arrangement, axial
        position [mm], d / D / b [mm], the dynamic rating C [N] (when it can be computed), the
        ISO/TS 16281 flag, and every geometry and contact attribute of ``GEOMETRY_ATTRIBUTES``
        and ``CONTACT_ATTRIBUTES`` that the bearing has.

    Notes
    -----
    Only ``C`` is reported as a rating: AxisForge computes no static rating C0 for the bearing.
    """
    tag = bearing.label or bearing.designation or type(bearing).__name__
    section = f"bearing {tag}" + (f" ({shaft_name})" if shaft_name else "")
    rows = [kv_record(section, "family", bearing.family.name),
            kv_record(section, "designation", bearing.designation),
            kv_record(section, "arrangement", bearing.arrangement)]
    rows += attribute_rows(bearing, section, [
        ("position", "mm", "axial position"), ("d", "mm", "bore d"), ("D", "mm", "outer D"),
        ("b", "mm", "width b")])
    try:
        rows.append(kv_record(section, "basic dynamic load rating C", float(bearing.C), "N"))
    except Exception:                                              # noqa: BLE001
        pass                                    # not assembled, or the family cannot compute it
    rows.append(kv_record(section, "ISO/TS 16281 analysis", bool(bearing.has_iso16281_analysis())))
    rows += attribute_rows(bearing, section, GEOMETRY_ATTRIBUTES)
    if bearing.has_iso16281_analysis():
        rows += attribute_rows(bearing, section, CONTACT_ATTRIBUTES)
    return rows


BEARING_TABLE_COLUMNS = [
    ("label", "bearing", ""), ("family", "family", ""), ("arrangement", "arrangement", ""),
    ("position_mm", "x [mm]", ".1f"), ("d_mm", "d [mm]", ".1f"), ("D_mm", "D [mm]", ".1f"),
    ("b_mm", "b [mm]", ".1f"), ("Z", "Z", ".0f"),
    ("element_diameter_mm", "Dw/Dwe [mm]", ".3f"), ("element_length_mm", "Lwe [mm]", ".3f"),
    ("Dpw_mm", "Dpw [mm]", ".2f"), ("s_mm", "s [mm]", ".4f"),
    ("alpha_0_rad", "alpha0 [deg]", ".2f", 180.0 / math.pi), ("C_N", "C [N]", ".0f"),
]


def bearing_summary_record(bearing) -> dict:
    """One flat record per bearing, for a comparison table.

    Parameters
    ----------
    bearing: Bearing
        An assembled bearing.

    Returns
    -------
    record: dict
        ``label``, ``family``, ``arrangement``, ``position_mm``, ``d_mm``, ``D_mm``, ``b_mm``,
        ``Z``, ``element_diameter_mm`` (Dw for a ball, Dwe for a roller), ``element_length_mm``
        (Lwe, NaN for a ball), ``Dpw_mm``, ``s_mm``, ``alpha_0_rad`` and ``C_N``. A quantity the
        bearing does not have is NaN.
    """
    def get(name):
        value = to_python(getattr(bearing, name, NAN))
        return float(value) if isinstance(value, (int, float)) else NAN

    try:
        rating = float(bearing.C)
    except Exception:                                              # noqa: BLE001
        rating = NAN
    ball = hasattr(bearing, "Dw")
    return dict(label=bearing.label, family=bearing.family.name, arrangement=bearing.arrangement,
                position_mm=get("position"), d_mm=get("d"), D_mm=get("D"), b_mm=get("b"),
                Z=int(get("Z")) if not math.isnan(get("Z")) else NAN,
                element_diameter_mm=get("Dw") if ball else get("Dwe"),
                element_length_mm=get("Lwe"), Dpw_mm=get("Dpw"), s_mm=get("s"),
                alpha_0_rad=get("alpha_0"), C_N=rating)


def bearings_table_text(bearings, indent: int = 0) -> str:
    """One line per bearing.

    Parameters
    ----------
    bearings: iterable of Bearing
        Assembled bearings.
    indent: int
        Number of leading spaces on every line.

    Returns
    -------
    text: str
        Fixed-width table with the family, arrangement, position, envelope, rolling-element
        size, pitch diameter, clearance, free contact angle (deg) and rating C.
    """
    return format_table(BEARING_TABLE_COLUMNS, [bearing_summary_record(b) for b in bearings],
                        indent)


def bearing_geometry_records(bearing) -> list[dict]:
    """Geometry of one bearing: the part of ``bearing_records`` that ``summary()`` does not print.

    Parameters
    ----------
    bearing: Bearing
        An assembled bearing.

    Returns
    -------
    records: list of dict
        Rows ``{section, parameter, value, unit}`` for every attribute of ``GEOMETRY_ATTRIBUTES``
        that the bearing has (rolling-element size, pitch diameter, number of elements,
        clearance, free contact angle, ...). Empty if it has none.
    """
    tag = bearing.label or bearing.designation or type(bearing).__name__
    return attribute_rows(bearing, f"{tag} geometry", GEOMETRY_ATTRIBUTES)


def bearing_text(bearing) -> str:
    """The bearing as AxisForge prints it, plus its geometry.

    Parameters
    ----------
    bearing: Bearing
        An assembled bearing.

    Returns
    -------
    text: str
        ``bearing.summary()`` followed by the geometry block (``bearing_geometry_records``),
        because ``summary()`` lists the envelope and the contact constants but not the rolling
        element geometry.
    """
    text = str(bearing.summary())
    geometry = bearing_geometry_records(bearing)
    return text + "\n\n" + format_key_values(geometry) if geometry else text


def print_bearing(bearing) -> None:
    """Print ``bearing_text`` to the console.

    Parameters
    ----------
    bearing: Bearing
        An assembled bearing.
    """
    print(bearing_text(bearing))
