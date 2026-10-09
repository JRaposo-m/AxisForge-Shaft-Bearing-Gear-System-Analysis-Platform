"""
axisforge/outputs/solvers/bearings/bearing_analysis_result.py

Records and text of ``axisforge.results.bearings.bearing_analysis_result``
(``BearingAnalysisResult``: load distribution plus, optionally, life). Mirrors the module it
reads, and like it is the only output module that joins ``load_distribution/`` and ``life/``.
"""
from __future__ import annotations

from axisforge.outputs._format import format_table
from axisforge.outputs.solvers.bearings.life.basic_life_results import (
    basic_life_record, basic_life_text)
from axisforge.outputs.solvers.bearings.load_distribution.load_distribution_results import (
    bearing_result_record, bearing_result_text)

SUMMARY_COLUMNS = [
    ("label", "bearing", ""),
    ("contact", "contact", ""),
    ("Fr_N", "Fr [N]", ".2f"),
    ("Fa_N", "Fa [N]", ".2f"),
    ("psi_rad", "psi [rad]", ".3e"),
    ("psi_rad", "psi [mrad]", ".4f", 1e3),
    ("delta_r_mm", "delta_r [mm]", ".5f"),
    ("n_loaded", "n_loaded", ".0f"),
    ("equilibrium_error_Fr_N", "dFr [N]", ".2e"),
    ("equilibrium_error_Fa_N", "dFa [N]", ".2e"),
    ("L10r_Mrev", "L10r [1e6 rev]", ".4g"),
    ("ok", "ok", ""),
]


def analysis_record(analysis) -> dict:
    """Summary record of one bearing: load distribution and life together.

    Parameters
    ----------
    analysis: BearingAnalysisResult
        Result of one bearing.

    Returns
    -------
    record: dict
        Every key of ``bearing_result_record`` plus ``L10r_Mrev``, ``Pref_r_N``, ``Pref_a_N``
        (NaN when there is no life) and ``postprocessed`` (stiffness and life both present).
    """
    record = bearing_result_record(analysis.load_distribution)
    life = basic_life_record(analysis.basic_life)
    record.update(L10r_Mrev=life["L10r_Mrev"], Pref_r_N=life["Pref_r_N"],
                  Pref_a_N=life["Pref_a_N"],
                  postprocessed=record["stiffness_available"] and life["available"])
    return record


def analysis_records(analyses: dict) -> list[dict]:
    """Summary records of several bearings.

    Parameters
    ----------
    analyses: dict
        ``{label: BearingAnalysisResult}``, as returned by the bearing solvers.

    Returns
    -------
    records: list of dict
        One ``analysis_record`` per bearing, in the order of the dict.
    """
    return [analysis_record(a) for a in analyses.values()]


def analysis_text(analysis) -> str:
    """Readable report of one bearing: load distribution, element table and life.

    Parameters
    ----------
    analysis: BearingAnalysisResult
        Result of one bearing.

    Returns
    -------
    text: str
        ``bearing_result_text`` followed by ``basic_life_text``.
    """
    return (bearing_result_text(analysis.load_distribution) + "\n\n"
            + basic_life_text(analysis.basic_life))


def analyses_table_text(analyses: dict) -> str:
    """One line per bearing.

    Parameters
    ----------
    analyses: dict
        ``{label: BearingAnalysisResult}``.

    Returns
    -------
    text: str
        Fixed-width table with the load, the misalignment psi (rad and mrad), the loaded-element
        count, the equilibrium errors, the life and the convergence flag.
    """
    return format_table(SUMMARY_COLUMNS, analysis_records(analyses))


def print_analysis(analysis) -> None:
    """Print ``analysis_text`` to the console.

    Parameters
    ----------
    analysis: BearingAnalysisResult
        Result of one bearing.
    """
    print(analysis_text(analysis))


def print_analyses(analyses: dict) -> None:
    """Print ``analyses_table_text`` to the console.

    Parameters
    ----------
    analyses: dict
        ``{label: BearingAnalysisResult}``.
    """
    print(analyses_table_text(analyses))
