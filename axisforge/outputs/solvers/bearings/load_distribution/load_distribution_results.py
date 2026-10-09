"""
axisforge/outputs/solvers/bearings/load_distribution/load_distribution_results.py

Records and text of ``axisforge.results.bearings.load_distribution.load_distribution_results``
(``BearingResult`` and its rows, ISO/TS 16281 internal load distribution). Mirrors the module it
reads.

The functions only read what the result holds. They do not derive engineering quantities
(Q_max/(Fr/Z), loaded-zone angle, ...) and they do not decide pass or fail: that is the job of
the studies that use AxisForge, which must stay independent of AxisForge's own post-processing.

Units: N, mm, N mm, rad. Records carry the unit in the key; the text shows psi in rad and mrad
and the element angles in degrees.
"""
from __future__ import annotations

import math

import numpy as np

from axisforge.outputs._format import (NAN, format_key_values, format_table, kv_record)

CONTACT_POINT = "point"
CONTACT_LINE = "line"

ELEMENT_COLUMNS = [
    ("j", "j", ".0f"),
    ("phi_rad", "phi [deg]", ".1f", 180.0 / math.pi),
    ("Q_N", "Q [N]", ".3f"),
    ("delta_mm", "delta [mm]", ".6f"),
    ("alpha_rad", "alpha [deg]", ".2f", 180.0 / math.pi),
]


def contact_type(result) -> str:
    """``"line"`` for a line-contact bearing result, ``"point"`` otherwise.

    Parameters
    ----------
    result: BearingResult
        Bearing-level load distribution result.

    Returns
    -------
    contact: str
        ``"line"`` or ``"point"``.
    """
    return CONTACT_LINE if result.is_line_contact else CONTACT_POINT


def bearing_result_record(result) -> dict:
    """Bearing-level summary of one load distribution result.

    Parameters
    ----------
    result: BearingResult
        Bearing-level result (``BallBearingResult`` or ``RollerBearingResult``).

    Returns
    -------
    record: dict
        ``label``, ``contact``, ``n_rows``, ``n_loaded`` (single-row bearings only, else NaN),
        the applied load (``Fr_xz_N``, ``Fr_xy_N``, ``Fr_N``, ``Fa_N``), ``Mz_Nmm``, the shared
        state of the solve (``delta_r_mm``, ``delta_a_mm``, ``psi_rad``, ``phi_Fr_rad``), the
        equilibrium errors (``equilibrium_error_Fr_N``, ``equilibrium_error_Fa_N``), the
        convergence (``ok``, ``residual``, ``n_iter``) and the stiffness when present
        (``stiffness_available``, ``delta_r_xz_mm``, ``delta_r_xy_mm``, ``Kr_xz_N_per_mm``,
        ``Kr_xy_N_per_mm``, ``Ka_N_per_mm``, ``Ka_regime``); those are NaN (``Ka_regime`` empty)
        when the result was not post-processed.

    Notes
    -----
    ``phi_Fr_rad`` is the direction of Fr in the global frame; the element angles ``phi_rad``
    of ``element_records`` are measured from it. ``n_iter`` holds the number of function
    evaluations of the root find (see ``BearingResult``).

    The stiffness is the secant stiffness of the stationary analysis, projected on each plane
    (Kr_xz = Fr_xz / delta_r_xz, Kr_xy = Fr_xy / delta_r_xy). Where the load has no component
    in a plane (for example Kr_xy when phi_Fr = 0) the corresponding projected displacement is
    ~0 and the secant stiffness does not exist in a stationary analysis: AxisForge returns
    ``inf`` and it is kept as ``inf`` here.

    Multi-row bearings (``n_rows > 1``) are reported structurally, one record per row (see
    ``row_records``). ISO/TS 16281 does not cover them, so those values are not validated
    against the standard; the single-row case is the one in use.
    """
    eq_r, eq_a = result.equilibrium_error
    record = dict(
        label=result.label,
        contact=contact_type(result),
        n_rows=result.n_rows,
        n_loaded=result.row.n_loaded if result.is_single else NAN,
        Fr_xz_N=result.Fr_xz, Fr_xy_N=result.Fr_xy, Fr_N=result.Fr, Fa_N=result.Fa,
        Mz_Nmm=result.Mz,
        delta_r_mm=result.delta_r, delta_a_mm=result.delta_a,
        psi_rad=result.psi, phi_Fr_rad=result.phi_Fr,
        equilibrium_error_Fr_N=eq_r, equilibrium_error_Fa_N=eq_a,
        ok=bool(result.ok), residual=result.residual, n_iter=result.n_iter,
    )
    st = result.stiffness
    record.update(
        stiffness_available=st is not None,
        delta_r_xz_mm=st.delta_r_xz if st is not None else NAN,
        delta_r_xy_mm=st.delta_r_xy if st is not None else NAN,
        Kr_xz_N_per_mm=st.Kr_xz if st is not None else NAN,
        Kr_xy_N_per_mm=st.Kr_xy if st is not None else NAN,
        Ka_N_per_mm=(st.Ka if st is not None and st.Ka is not None else NAN),
        Ka_regime=(st.Ka_regime or "") if st is not None else "",
    )
    return record


def row_records(result) -> list[dict]:
    """One record per bearing row (single-row bearings give one).

    Parameters
    ----------
    result: BearingResult
        Bearing-level result.

    Returns
    -------
    records: list of dict
        ``label``, ``row``, ``Z``, ``n_loaded``, ``Fr_row_N``, ``Fa_row_N``, ``Mz_Nmm``.
    """
    return [dict(label=result.label, row=i, Z=r.Z, n_loaded=r.n_loaded, Fr_row_N=r.Fr_row,
                 Fa_row_N=r.Fa_row, Mz_Nmm=r.Mz) for i, r in enumerate(result.rows)]


def wrap_angle(phi) -> np.ndarray:
    """Wrap angles to [-pi, pi).

    Parameters
    ----------
    phi: array_like
        Angles [rad].

    Returns
    -------
    wrapped: ndarray
        Angles in [-pi, pi) [rad]; 0 stays 0.

    Notes
    -----
    This is the expression used by the design studies (``(phi + pi) % (2 pi) - pi``), kept
    identical so that the stored values do not change.
    """
    return (np.asarray(phi, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


def element_records(result, row: int = 0) -> list[dict]:
    """One record per rolling element of one row.

    Parameters
    ----------
    result: BearingResult
        Bearing-level result.
    row: int
        Row index (0 for single-row bearings).

    Returns
    -------
    records: list of dict
        ``label``, ``row``, ``j``, ``phi_rad`` (local frame wrapped to [-pi, pi); 0 is the
        element on the load line), ``Q_N``, ``delta_mm`` (element deflection; for rollers the
        raw approach, which may be negative), ``alpha_rad`` (operating contact angle) and, for
        line contact, ``psi_j_rad`` (roller tilt).

    Raises
    ------
    IndexError
        If ``row`` is out of range.
    """
    if not 0 <= row < result.n_rows:
        raise IndexError(f"{result.label}: row {row} out of range (n_rows = {result.n_rows}).")
    r = result.rows[row]
    phi = wrap_angle(r.phi_j)
    line = result.is_line_contact
    records = []
    for j in range(r.Z):
        rec = dict(label=result.label, row=row, j=j, phi_rad=float(phi[j]), Q_N=float(r.Q_j[j]),
                   delta_mm=float(r.delta_j[j]), alpha_rad=float(r.alpha_j[j]))
        if line:
            rec["psi_j_rad"] = float(r.psi_j[j])
        records.append(rec)
    return records


def lamina_records(result, row: int = 0, element: int | None = None) -> list[dict]:
    """Lamina loads of a line-contact row.

    Parameters
    ----------
    result: BearingResult
        Bearing-level result.
    row: int
        Row index (0 for single-row bearings).
    element: int, optional (None)
        Only this roller; None gives every roller.

    Returns
    -------
    records: list of dict
        ``label``, ``row``, ``j``, ``k``, ``x_mm`` (lamina mid-position), ``delta_mm``,
        ``q_N``. Empty for point contact.
    """
    if not result.is_line_contact:
        return []
    r = result.rows[row]
    rollers = range(r.Z) if element is None else [element]
    return [dict(label=result.label, row=row, j=j, k=k, x_mm=float(r.x_k[k]),
                 delta_mm=float(r.delta_jk[j, k]), q_N=float(r.q_jk[j, k]))
            for j in rollers for k in range(r.n_s)]


def bearing_result_text(result, row: int = 0) -> str:
    """Readable summary of one load distribution result.

    Parameters
    ----------
    result: BearingResult
        Bearing-level result.
    row: int
        Row whose element table is printed (0 for single-row bearings).

    Returns
    -------
    text: str
        A key-value block (psi in rad and in mrad) followed by the element table of ``row``
        (angles in degrees).
    """
    rec = bearing_result_record(result)
    s = f"bearing {rec['label']} ({rec['contact']} contact, {rec['n_rows']} row(s))"
    rows = [
        kv_record(s, "radial load Fr", rec["Fr_N"], "N"),
        kv_record(s, "axial load Fa", rec["Fa_N"], "N"),
        kv_record(s, "moment Mz", rec["Mz_Nmm"], "N mm"),
        kv_record(s, "radial approach delta_r", rec["delta_r_mm"], "mm"),
        kv_record(s, "axial approach delta_a", rec["delta_a_mm"], "mm"),
        kv_record(s, "misalignment psi", rec["psi_rad"], "rad"),
        kv_record(s, "misalignment psi", 1e3 * rec["psi_rad"], "mrad"),
        kv_record(s, "load direction phi_Fr", rec["phi_Fr_rad"], "rad"),
        kv_record(s, "loaded elements", rec["n_loaded"], ""),
        kv_record(s, "equilibrium error Fr", rec["equilibrium_error_Fr_N"], "N"),
        kv_record(s, "equilibrium error Fa", rec["equilibrium_error_Fa_N"], "N"),
        kv_record(s, "converged", rec["ok"], ""),
        kv_record(s, "residual", rec["residual"], ""),
        kv_record(s, "function evaluations", rec["n_iter"], ""),
    ]
    if rec["stiffness_available"]:
        rows += [kv_record(s, "radial stiffness Kr (XZ)", rec["Kr_xz_N_per_mm"], "N/mm"),
                 kv_record(s, "radial stiffness Kr (XY)", rec["Kr_xy_N_per_mm"], "N/mm"),
                 kv_record(s, "axial stiffness Ka", rec["Ka_N_per_mm"], "N/mm"),
                 kv_record(s, "axial regime", rec["Ka_regime"], "")]
    else:
        rows.append(kv_record(s, "stiffness", "not post-processed", ""))
    text = format_key_values(rows, spec=".8g")
    table = format_table(ELEMENT_COLUMNS, element_records(result, row), indent=2)
    return f"{text}\n\n  Element loads, row {row} (phi = 0 on the load line):\n{table}"


def print_bearing_result(result, row: int = 0) -> None:
    """Print ``bearing_result_text`` to the console.

    Parameters
    ----------
    result: BearingResult
        Bearing-level result.
    row: int
        Row whose element table is printed.
    """
    print(bearing_result_text(result, row))
