"""Result containers of the mesh-convergence study: ``ConvergenceRecord`` (the history of one
interval) and ``MeshRefinementResult`` (all intervals of one shaft). Data only: the computation
is in ``RichardsonGCI`` and ``MeshConvergenceStudy`` (``solvers/mesh``), referenced here by type
only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.solvers.mesh.convergence_solver import (
        RichardsonGCI,
        _DummyGCI,
    )

__all__ = ["ConvergenceRecord", "MeshRefinementResult"]


@dataclass
class ConvergenceRecord:
    """Convergence history of one interval (a gear face width, a bearing extent, a load span, ...).

    Holds every mesh grade tried, the metric values at each grade, the GCI computed once three or more
    levels exist, and whether the interval converged.

    Parameters
    ----------
    label : str
        Name of the interval.
    x_lo, x_hi : float
        Interval bounds [mm].
    levels : list of str
        Grade names in the order tried.
    point_metrics_history : list of dict
        Per level, ``{metric_name: value}`` with the key built by ``_request_label`` (for example
        ``"M_xz:max"``); values in the units of the field.
    gci_history : list of dict
        Per level transition (from the third level on), ``{metric_name: RichardsonGCI or _DummyGCI}``.
        A ``_DummyGCI`` (NaN, not converged) replaces ``RichardsonGCI`` when its computation raised
        ``ValueError``, ``ZeroDivisionError`` or ``ArithmeticError``.
    peak_shifted : dict of str to list of bool
        Declared for peak-tracking metrics but never written by ``MeshConvergenceStudy``; always
        empty in records it builds.
    converged : bool
        ``True`` if every metric converged at some level triple.
    x_final : list of float
        Node positions [mm] of the last level; set on convergence, and also by
        ``finalize_unconverged`` when the maximum number of levels was reached.
    """

    label: str
    x_lo: float
    x_hi: float

    levels: list[str] = field(default_factory=list)

    point_metrics_history: list[dict[str, float]] = field(default_factory=list)
    """Per level: {metric_name: aggregated_value}, e.g.
    {"v_xz": ..., "v_xy": ..., "v_res": ...} for the default displacement
    metric set. Was: tuple[float, float, float] fixed to
    (f_xz, f_xy, f_res)."""

    gci_history: list[dict[str, "RichardsonGCI | _DummyGCI"]] = field(default_factory=list)
    """Per level transition (once len(levels) >= 3): {metric_name: ...},
    each value a RichardsonGCI (successful fit) or _DummyGCI (non-uniform
    refinement ratio, a degenerate observed order, or a metric landing
    exactly on 0.0 -- see convergence_solver.py's own RichardsonGCI/
    _compute_gci()). Type hint is unchanged from before this pass -- it
    was already an open dict[str, ...]; only the key set it's populated
    with widened."""

    peak_shifted: dict[str, list[bool]] = field(default_factory=dict)
    """Per metric name that uses a peak-tracking evaluation strategy: one
    bool per transition in gci_history, True if that metric's sampled
    extremum moved more than its strategy's stable_tol_mm between the
    two grades feeding that transition. Absent, or not meaningfully
    populated, for metrics sampled at a fixed geometric point (today's
    displacement metrics) -- FixedPointStrategy never repositions, so
    drift is not a concept that applies to them.

    Exists because RichardsonGCI's order-of-convergence math assumes the
    same physical location is being compared across grades; a metric
    whose peak is still migrating can report GCI_f_m < threshold (i.e.
    "converged") while actually only meaning "the value AT A DRIFTING
    LOCATION stopped changing much," which is a materially weaker claim.
    A future convergence_report.py should render this (e.g. flag the row)
    rather than let that distinction disappear into a bare True/False
    converged column."""

    converged: bool = False
    x_final: list[float] = field(default_factory=list)

    def metric_names(self) -> list[str]:
        """Metric names present in the record, in first-seen order.

        Returns
        -------
        list of str
            Keys of the first entry of ``point_metrics_history`` (empty if there is none).
        """
        if not self.point_metrics_history:
            return []
        return list(self.point_metrics_history[0].keys())


@dataclass
class MeshRefinementResult:
    """Aggregated convergence result of one shaft: one ``ConvergenceRecord`` per interval.

    Parameters
    ----------
    shaft_name : str
        Name of the shaft system.
    per_load : dict of str to ConvergenceRecord
        Record of each interval, keyed by its label.
    """

    shaft_name: str = ""
    per_load: dict[str, ConvergenceRecord] = field(default_factory=dict)

    @property
    def all_extra_nodes(self) -> list[float]:
        """Sorted union of the ``x_final`` node positions of all intervals [mm].

        Ready to pass as ``extra_mandatory`` to ``Mesh1D``.

        Returns
        -------
        list of float

        Warnings
        --------
        Intervals with ``converged == False`` are included too, because ``x_final`` is also set when the
        maximum number of levels was reached without convergence.
        """
        nodes: set[float] = set()
        for rec in self.per_load.values():
            nodes.update(rec.x_final)
        return sorted(nodes)