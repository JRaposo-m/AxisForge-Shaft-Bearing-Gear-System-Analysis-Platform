"""
axisforge/outputs/solvers/fem_results/shaft_results.py

Records and text of ``axisforge.results.fem_results.shaft_results`` (``ShaftResults`` and its
``BearingNodeData``). Mirrors the module it reads.

Units are the ones AxisForge uses everywhere: N, mm, N mm, rad, MPa. Records carry them in the
key; the text shows the slopes both in rad and in mrad.
"""
from __future__ import annotations

import math

import numpy as np

from axisforge.outputs._format import NAN, format_key_values, format_table, kv_record

# BearingNodeData field -> record key (the unit is in the key). The test suite checks that every
# field of BearingNodeData is listed here, so a new field cannot be dropped silently.
NODE_FIELDS: dict[str, str] = {
    "label": "label",
    "position": "x_mm",
    "u": "u_mm",
    "v_xz": "v_xz_mm",
    "v_xy": "v_xy_mm",
    "theta_xz": "theta_xz_rad",
    "theta_xy": "theta_xy_rad",
    "Fr_xz": "Fr_xz_N",
    "Fr_xy": "Fr_xy_N",
    "Fr": "Fr_N",
    "Fa": "Fa_N",
    "M_xz": "M_xz_Nmm",
    "M_xy": "M_xy_Nmm",
    "psi_xz": "psi_xz_rad",
    "psi_xy": "psi_xy_rad",
}

# ShaftResults scalar -> record key.
SUMMARY_FIELDS: dict[str, str] = {
    "M_max": "M_max_Nmm",
    "x_M_max": "x_M_max_mm",
    "v_max": "v_max_mm",
    "x_v_max": "x_v_max_mm",
    "sigma_b_max": "sigma_b_max_MPa",
    "x_sigma_b_max": "x_sigma_b_max_mm",
    "tau_max": "tau_max_MPa",
    "x_tau_max": "x_tau_max_mm",
    "phi_max": "phi_max_rad",
    "x_phi_max": "x_phi_max_mm",
}


# ShaftResults array (one value per mesh node) -> record key. Units are the AxisForge ones
# (N, mm, N mm, rad, MPa); ShaftResults does not state them for M, V, T, so confirm before quoting.
MESH_ARRAYS: dict[str, str] = {
    "x": "x_mm",
    "d": "d_mm",
    "M": "M_Nmm",
    "V": "V_N",
    "T": "T_Nmm",
    "v": "v_mm",
    "u": "u_mm",
    "theta_xz": "theta_xz_rad",
    "theta_xy": "theta_xy_rad",
    "phi": "phi_rad",
    "sigma_b": "sigma_b_MPa",
    "tau": "tau_MPa",
}


def bearing_node_records(shaft_results, shaft_name: str | None = None) -> list[dict]:
    """One record per bearing node of a shaft FEM result.

    Parameters
    ----------
    shaft_results: ShaftResults
        Output of the shaft FEM for one shaft.
    shaft_name: str, optional (None)
        Name written in the ``shaft`` column; None uses ``shaft_results.name``.

    Returns
    -------
    records: list of dict
        ``shaft`` plus every field of ``BearingNodeData`` under the key given by ``NODE_FIELDS``
        (for example ``Fr_N``, ``M_xz_Nmm``, ``psi_xz_rad``). Empty if the result has no bearing
        node.

    Notes
    -----
    ``psi_xz`` and ``psi_xy`` are the beam rotation at the bearing node, in rad (the contact
    solver uses them as radians). ``position`` is the axial coordinate along the shaft.
    """
    shaft = shaft_name if shaft_name is not None else shaft_results.name
    records = []
    for node in getattr(shaft_results, "bearing_nodes", None) or []:
        record = {"shaft": shaft}
        for field, key in NODE_FIELDS.items():
            record[key] = getattr(node, field)
        records.append(record)
    return records


def mesh_node_records(shaft_results, shaft_name: str | None = None) -> list[dict]:
    """One record per mesh node along the shaft.

    Parameters
    ----------
    shaft_results: ShaftResults
        Output of the shaft FEM for one shaft.
    shaft_name: str, optional (None)
        Name written in the ``shaft`` column; None uses ``shaft_results.name``.

    Returns
    -------
    records: list of dict
        ``shaft``, ``index`` and one key per array of ``MESH_ARRAYS`` (``x_mm``, ``d_mm``,
        ``M_Nmm``, ``V_N``, ``T_Nmm``, ``v_mm``, ``u_mm``, ``theta_xz_rad``, ``theta_xy_rad``,
        ``phi_rad``, ``sigma_b_MPa``, ``tau_MPa``). An array that is empty or does not have one
        value per node gives NaN in that key. Empty if the result has no node positions.
    """
    shaft = shaft_name if shaft_name is not None else shaft_results.name
    x = np.asarray(getattr(shaft_results, "x", []), dtype=float)
    if x.size == 0:
        x = np.asarray(getattr(shaft_results, "x_nodes", []), dtype=float)
    columns = {}
    for attribute, key in MESH_ARRAYS.items():
        values = x if attribute == "x" else np.asarray(getattr(shaft_results, attribute, []),
                                                       dtype=float)
        columns[key] = values if values.shape == x.shape else np.full(x.shape, NAN)
    return [dict(shaft=shaft, index=i, **{key: float(values[i]) for key, values in columns.items()})
            for i in range(x.size)]


def shaft_summary_record(shaft_results, shaft_name: str | None = None) -> dict:
    """Maxima and sizes of one shaft FEM result.

    Parameters
    ----------
    shaft_results: ShaftResults
        Output of the shaft FEM for one shaft.
    shaft_name: str, optional (None)
        Name written in the ``shaft`` column; None uses ``shaft_results.name``.

    Returns
    -------
    record: dict
        ``shaft``, ``n_nodes`` (mesh nodes), ``n_bearing_nodes`` and the maxima of
        ``SUMMARY_FIELDS`` with their axial positions.

    Notes
    -----
    ``ShaftResults`` does not state the units of its maxima. They are taken from the N, mm, MPa
    convention of AxisForge (bending moment in N mm, stress in MPa); phi_max is documented in
    rad. Confirm before quoting them.
    """
    record = dict(
        shaft=shaft_name if shaft_name is not None else shaft_results.name,
        n_nodes=len(getattr(shaft_results, "x_nodes", [])),
        n_bearing_nodes=len(getattr(shaft_results, "bearing_nodes", [])),
    )
    for field, key in SUMMARY_FIELDS.items():
        record[key] = float(getattr(shaft_results, field, NAN))
    return record


def bearing_nodes_text(shaft_results, shaft_name: str | None = None) -> str:
    """Readable block of every bearing node of a shaft FEM result.

    Parameters
    ----------
    shaft_results: ShaftResults
        Output of the shaft FEM for one shaft.
    shaft_name: str, optional (None)
        Name written in the block titles; None uses ``shaft_results.name``.

    Returns
    -------
    text: str
        One block per bearing node. The slopes psi and the rotations theta are given in rad and
        in mrad.
    """
    rows = []
    for rec in bearing_node_records(shaft_results, shaft_name):
        section = f"{rec['shaft']} | bearing node {rec['label']}"
        rows.append(kv_record(section, "axial position", rec["x_mm"], "mm"))
        for key, name, unit in (("Fr_N", "radial reaction Fr", "N"),
                                ("Fr_xz_N", "radial reaction Fr (XZ)", "N"),
                                ("Fr_xy_N", "radial reaction Fr (XY)", "N"),
                                ("Fa_N", "axial reaction Fa", "N"),
                                ("M_xz_Nmm", "moment reaction M (XZ)", "N mm"),
                                ("M_xy_Nmm", "moment reaction M (XY)", "N mm"),
                                ("u_mm", "axial displacement u", "mm"),
                                ("v_xz_mm", "transverse displacement v (XZ)", "mm"),
                                ("v_xy_mm", "transverse displacement v (XY)", "mm")):
            rows.append(kv_record(section, name, rec[key], unit))
        for plane in ("xz", "xy"):
            for kind, name in (("psi", "slope psi"), ("theta", "rotation theta")):
                value = rec[f"{kind}_{plane}_rad"]
                rows.append(kv_record(section, f"{name} ({plane.upper()})", value, "rad"))
                rows.append(kv_record(section, f"{name} ({plane.upper()})",
                                      value * 1e3 if isinstance(value, float) and
                                      not math.isnan(value) else value, "mrad"))
    return format_key_values(rows, spec=".8g") if rows else "(no bearing-node data)"


MESH_TABLE_COLUMNS = [
    ("x_mm", "x [mm]", ".1f"), ("d_mm", "d [mm]", ".1f"),
    ("M_Nmm", "M [N m]", ".2f", 1e-3), ("V_N", "V [N]", ".1f"), ("T_Nmm", "T [N m]", ".2f", 1e-3),
    ("v_mm", "v [mm]", ".5f"), ("u_mm", "u [mm]", ".5f"),
    ("theta_xz_rad", "theta_xz [mrad]", ".4f", 1e3),
    ("theta_xy_rad", "theta_xy [mrad]", ".4f", 1e3),
    ("phi_rad", "twist [mrad]", ".4f", 1e3),
    ("sigma_b_MPa", "sigma_b [MPa]", ".2f"), ("tau_MPa", "tau [MPa]", ".2f"),
]


def mesh_nodes_table_text(records, indent: int = 0) -> str:
    """One line per mesh node of one shaft.

    Parameters
    ----------
    records: sequence of dict
        Records of ``mesh_node_records`` of ONE shaft (and one power).
    indent: int
        Number of leading spaces on every line.

    Returns
    -------
    text: str
        Fixed-width table: position, diameter, bending moment and shear force (resultants),
        torque, deflection, axial displacement, bending rotations, twist, bending and shear
        stress at every node. The moment and the torque are shown in N m.
    """
    return format_table(MESH_TABLE_COLUMNS, records, indent)


NODE_TABLE_COLUMNS = [
    ("shaft", "shaft", ""), ("label", "bearing node", ""), ("x_mm", "x [mm]", ".1f"),
    ("Fr_N", "Fr [N]", ".2f"), ("Fr_xz_N", "Fr_xz [N]", ".2f"), ("Fr_xy_N", "Fr_xy [N]", ".2f"),
    ("Fa_N", "Fa [N]", ".2f"),
    ("psi_xz_rad", "psi_xz [rad]", ".4e"), ("psi_xz_rad", "psi_xz [mrad]", ".4f", 1e3),
    ("psi_xy_rad", "psi_xy [rad]", ".4e"), ("psi_xy_rad", "psi_xy [mrad]", ".4f", 1e3),
]


def bearing_nodes_table_text(records, indent: int = 0) -> str:
    """One line per bearing node.

    Parameters
    ----------
    records: sequence of dict
        Records of ``bearing_node_records``; a record may carry an extra ``power_W`` key (the
        power of the solve), which then becomes the first column.
    indent: int
        Number of leading spaces on every line.

    Returns
    -------
    text: str
        Fixed-width table with the position, the reactions Fr, Fr_xz, Fr_xy, Fa and the slopes
        psi of both planes in rad and in mrad. The moments, displacements and rotations stay in
        the records (and the CSV).
    """
    columns = list(NODE_TABLE_COLUMNS)
    if any("power_W" in r for r in records):
        columns.insert(0, ("power_W", "P [W]", ".1f"))
    return format_table(columns, records, indent)


SUMMARY_TABLE_COLUMNS = [
    ("shaft", "shaft", ""), ("n_nodes", "mesh nodes", ".0f"),
    ("M_max_Nmm", "M_max [N m]", ".2f", 1e-3), ("x_M_max_mm", "at x [mm]", ".1f"),
    ("v_max_mm", "v_max [mm]", ".5f"), ("x_v_max_mm", "at x [mm]", ".1f"),
    ("sigma_b_max_MPa", "sigma_b [MPa]", ".2f"), ("x_sigma_b_max_mm", "at x [mm]", ".1f"),
    ("tau_max_MPa", "tau [MPa]", ".2f"), ("x_tau_max_mm", "at x [mm]", ".1f"),
    ("phi_max_rad", "phi_max [mrad]", ".4f", 1e3), ("x_phi_max_mm", "at x [mm]", ".1f"),
]


def shaft_summary_table_text(records, indent: int = 0) -> str:
    """One line per shaft with the maxima of the FEM result.

    Parameters
    ----------
    records: sequence of dict
        Records of ``shaft_summary_record``; a record may carry an extra ``power_W`` key, which
        then becomes the first column.
    indent: int
        Number of leading spaces on every line.

    Returns
    -------
    text: str
        Fixed-width table with the maximum bending moment (N m), deflection, bending stress,
        shear stress and torsion angle, each with its axial position.

    Notes
    -----
    ``ShaftResults`` does not state the units of its maxima; see ``shaft_summary_record``.
    """
    columns = list(SUMMARY_TABLE_COLUMNS)
    if any("power_W" in r for r in records):
        columns.insert(0, ("power_W", "P [W]", ".1f"))
    return format_table(columns, records, indent)


def print_bearing_nodes(shaft_results, shaft_name: str | None = None) -> None:
    """Print ``bearing_nodes_text`` to the console.

    Parameters
    ----------
    shaft_results: ShaftResults
        Output of the shaft FEM for one shaft.
    shaft_name: str, optional (None)
        Name written in the block titles.
    """
    print(bearing_nodes_text(shaft_results, shaft_name))
