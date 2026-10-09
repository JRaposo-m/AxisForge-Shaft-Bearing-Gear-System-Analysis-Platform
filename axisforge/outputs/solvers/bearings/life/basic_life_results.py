"""
axisforge/outputs/solvers/bearings/life/basic_life_results.py

Records and text of ``axisforge.results.bearings.life.basic_life_results``
(``BasicReferenceRatingLifeResult``, ISO/TS 16281:2008 basic reference rating life).
Mirrors the module it reads.

The life is the ISO/TS 16281 basic REFERENCE rating life L10r in millions of revolutions. It is
not an ISO 281 catalogue life, and no modification factor is applied.
"""
from __future__ import annotations

from axisforge.outputs._format import NAN, format_key_values, kv_record, scalar_attributes


def _row_key(name: str) -> str:
    """Record key of a scalar attribute of a per-row life object (the unit goes in the key)."""
    if name == "L10r":
        return "L10r_Mrev"
    if name.startswith("Q_"):
        return f"{name}_N"
    return name


def basic_life_record(life) -> dict:
    """Bearing-level life record.

    Parameters
    ----------
    life: BasicReferenceRatingLifeResult or None
        Life result of one bearing; None means "not computed".

    Returns
    -------
    record: dict
        ``label``, ``method``, ``n_rows``, ``L10r_Mrev``, ``Pref_r_N``, ``Pref_a_N``,
        ``available``. Numbers are NaN (and label/method empty) when ``life`` is None;
        ``Pref_r_N`` / ``Pref_a_N`` are NaN when the reference load does not apply.
    """
    if life is None:
        return dict(label="", method="", n_rows=0, L10r_Mrev=NAN, Pref_r_N=NAN, Pref_a_N=NAN,
                    available=False)
    pref_r, pref_a = life.Pref_r, life.Pref_a
    return dict(label=life.label, method=life.METHOD, n_rows=life.n_rows,
                L10r_Mrev=float(life.L10r),
                Pref_r_N=NAN if pref_r is None else float(pref_r),
                Pref_a_N=NAN if pref_a is None else float(pref_a),
                available=True)


def row_life_records(life) -> list[dict]:
    """One record per bearing row.

    Parameters
    ----------
    life: BasicReferenceRatingLifeResult or None
        Life result of one bearing; None gives an empty list.

    Returns
    -------
    records: list of dict
        ``label``, ``row`` and every scalar attribute of the row life object (``L10r`` becomes
        ``L10r_Mrev``; the equivalent loads ``Q_*`` become ``Q_*_N``). Array attributes (the
        lamina loads of rollers) are not exported.
    """
    if life is None:
        return []
    records = []
    for i, row in enumerate(life.rows):
        rec = dict(label=life.label, row=i)
        rec.update({_row_key(k): v for k, v in scalar_attributes(row).items() if k != "label"})
        records.append(rec)
    return records


def basic_life_text(life) -> str:
    """Readable summary of one life result.

    Parameters
    ----------
    life: BasicReferenceRatingLifeResult or None
        Life result of one bearing.

    Returns
    -------
    text: str
        Key-value block; a single line saying that the life was not computed when ``life`` is
        None.
    """
    rec = basic_life_record(life)
    if not rec["available"]:
        return "basic reference rating life: not computed"
    s = f"basic reference rating life {rec['label']} ({rec['method']})"
    rows = [kv_record(s, "L10r (bearing)", rec["L10r_Mrev"], "10^6 rev"),
            kv_record(s, "reference load Pref_r", rec["Pref_r_N"], "N"),
            kv_record(s, "reference load Pref_a", rec["Pref_a_N"], "N")]
    for r in row_life_records(life):
        rows.append(kv_record(s, f"row {r['row']} L10r", r["L10r_Mrev"], "10^6 rev"))
    return format_key_values(rows, spec=".8g")


def print_basic_life(life) -> None:
    """Print ``basic_life_text`` to the console.

    Parameters
    ----------
    life: BasicReferenceRatingLifeResult or None
        Life result of one bearing.
    """
    print(basic_life_text(life))
