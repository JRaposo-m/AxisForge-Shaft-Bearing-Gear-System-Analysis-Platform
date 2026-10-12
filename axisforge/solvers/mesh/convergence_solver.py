"""Mesh-convergence bookkeeping for the shaft sub-model: reduces the fields of a ``SubmodelResult``
to scalar metrics, and computes the observed order of convergence, the Richardson extrapolation
and the Grid Convergence Index (GCI) over three consecutive refinement levels. No solve is run
here: one ``SubmodelResult`` per level is supplied by the caller. Richardson (1911) and Roache
(1998) are named in the code and are not in the library (source to find).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np

from axisforge.results.convergence_results.convergence_results import ConvergenceRecord, MeshRefinementResult

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.results.fem_results.submodel_results import SubmodelResult


# ===========================================================================
# Registry of "evaluation forms" (reducers of a field array to one number),
# independent of any variable. Same pattern as register_contact_solver().
# ===========================================================================

# Aggregate form: (values) -> float, reduces the whole array and does not need node
# positions (e.g. max, mean, max_minus_mean).
EvalFormAggregate = Callable[[np.ndarray], float]
"""Assinatura para uma forma que agrega sobre o array inteiro -- não
precisa de saber onde é cada nó, só dos valores. Ex: max, mean,
max_minus_mean."""

# Point form: (values, x_nodes, x) -> float, reads one position (e.g. at_point).
EvalFormPoint = Callable[[np.ndarray, list, float], float]
"""Assinatura para uma forma que lê um único ponto -- recebe
(values, x_nodes, x). Ex: at_point."""

_EVAL_FORM_REGISTRY: dict[str, "EvalFormAggregate | EvalFormPoint"] = {}
_POINT_FORMS: set[str] = set()


def register_eval_form(name: str, *, needs_point: bool = False):
    """Decorator that registers an evaluation form (a reducer of a field array to one number) by name.

    The registered function never sees the variable (M, v, theta, ...): it receives only the array
    of values that the caller has already fetched with ``getattr``.

    Parameters
    ----------
    name : str
        Registry name of the form.
    needs_point : bool, optional
        ``True`` if the function signature is ``(values, x_nodes, x)``, ``False`` (default) if it is
        ``(values)``. ``MeshConvergenceStudy.add_level`` uses it to know whether a named point must be
        resolved before calling the form.

    Raises
    ------
    ValueError
        If ``name`` is already registered.
    """
    def decorator(fn):
        """Register ``fn`` under ``name`` (ValueError if taken) and return it unchanged."""
        if name in _EVAL_FORM_REGISTRY:
            raise ValueError(f"register_eval_form: '{name}' already registered")
        _EVAL_FORM_REGISTRY[name] = fn
        if needs_point:
            _POINT_FORMS.add(name)
        return fn
    return decorator


def resolve_eval_form(name: str):
    """Return the evaluation form registered under ``name``.

    Parameters
    ----------
    name : str
        Registry name.

    Returns
    -------
    callable
        The registered function.

    Raises
    ------
    ValueError
        If the name is not registered (the message lists the registered names).
    """
    try:
        return _EVAL_FORM_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"resolve_eval_form: unknown eval form '{name}' -- "
            f"registered: {sorted(_EVAL_FORM_REGISTRY)}"
        ) from None


# --- concrete forms -- generic, none knows which variable the array came from ---

@register_eval_form("max")
def _eval_max(values: np.ndarray) -> float:
    """Largest absolute value of the field.

    Parameters
    ----------
    values : ndarray
        Nodal values of the field (units of the field).

    Returns
    -------
    float
        ``max(abs(values))`` (units of the field).
    """
    return float(np.max(np.abs(values)))


@register_eval_form("mean")
def _eval_mean(values: np.ndarray) -> float:
    """Mean value of the field over the nodes.

    Parameters
    ----------
    values : ndarray
        Nodal values of the field (units of the field).

    Returns
    -------
    float
        ``mean(values)`` (units of the field); signed, not an absolute value.
    """
    return float(np.mean(values))


@register_eval_form("max_minus_mean")
def _eval_max_minus_mean(values: np.ndarray) -> float:
    """Difference between the largest absolute value and the mean of the field.

    The peak of M(x) moves slightly between mesh grades, so comparing max-minus-mean is more robust
    than evaluating the field at a fixed x.

    Parameters
    ----------
    values : ndarray
        Nodal values of the field (units of the field).

    Returns
    -------
    float
        ``max(abs(values)) - mean(values)`` (units of the field).
    """
    return float(np.max(np.abs(values)) - np.mean(values))


@register_eval_form("max_minus_mean_relative")
def _eval_max_minus_mean_relative(values: np.ndarray) -> float:
    """Relative form of ``max_minus_mean``: (max(abs(values)) - mean) / mean.

    Parameters
    ----------
    values : ndarray
        Nodal values of the field.

    Returns
    -------
    float
        Dimensionless ratio [-].

    Raises
    ------
    ValueError
        If the mean is exactly 0.0 (degenerate case, for example an antisymmetric field over the
        interval). It is raised inside ``MeshConvergenceStudy.add_level`` when the metric is
        evaluated and is not caught there (``_compute_gci`` only guards ``RichardsonGCI``).
    """
    mean = float(np.mean(values))
    if mean == 0.0:
        raise ValueError("max_minus_mean_relative: mean is 0.0 -- cannot normalize")
    return float((np.max(np.abs(values)) - mean) / mean)


@register_eval_form("at_point", needs_point=True)
def _eval_at_point(values: np.ndarray, x_nodes: list, x: float) -> float:
    """Value of the field at the node whose position equals ``x``.

    Parameters
    ----------
    values : ndarray
        Nodal values of the field (units of the field).
    x_nodes : list of float
        Node positions [mm].
    x : float
        Requested position [mm]; it must coincide with a node (the caller makes it a mandatory node).

    Returns
    -------
    float
        The nodal value at ``x`` (units of the field).

    Raises
    ------
    ValueError
        If no node lies within 1e-6 mm of ``x``.
    """
    for i, xi in enumerate(x_nodes):
        if abs(xi - x) < 1e-6:
            return float(values[i])
    raise ValueError(f"at_point: x={x} not found in x_nodes (tol=1e-6)")


# ===========================================================================
# GCI placeholder for the non-uniform-refinement-ratio case
# ===========================================================================

@dataclass
class _DummyGCI:
    """Placeholder returned when the GCI cannot be computed: NaN GCI values and ``converged = False``.
    """
    GCI_f_m:   float = float("nan")
    GCI_m_c:   float = float("nan")
    converged: bool  = False


# ===========================================================================
# Richardson GCI -- metric-agnostic (only floats and lists of x_nodes).
# ===========================================================================

class RichardsonGCI:
    """Grid Convergence Index (GCI) from Richardson extrapolation over three refinement levels.

    Needs three consecutive levels to compute the observed order of convergence p and the GCI of each
    interval. Metric-agnostic: it takes the metric values of the coarse, medium and fine levels as
    plain floats.

    Parameters
    ----------
    f_coarse, f_medium, f_fine : float
        Value of the evaluation form at each level (units of the metric).
    x_nodes_coarse, x_nodes_medium, x_nodes_fine : list of float
        Node positions of each level [mm]; the refinement ratio uses the number of elements
        ``len(x_nodes) - 1``.
    gci_threshold : float, optional
        GCI below which an interval counts as converged [-]. Default 0.01 (1 %).
    safety_factor : float, optional
        Safety factor Fs of the GCI [-]. Default 1.25; the code uses this value as given and never
        switches to another one.
    min_p, max_p : float or None, optional
        Lower and upper clamp of the observed order p [-].

    Raises
    ------
    ValueError
        If a level has fewer than 2 nodes; if the refinement ratios are not above 1 or are not equal;
        if ``f_medium == f_fine``; if the convergence ratio is not positive (non-monotonic);
        if p is not positive or ``r**p - 1`` is ~0; if ``f_coarse`` or ``f_medium`` is 0.0.

    Warnings
    --------
    The relative differences divide by the coarser value of each pair (``f_c`` and ``f_m``), not the
    finer one. The refinement ratios are compared for equality as floats (``r_f_m != r_m_c``).
    ``safety_factor`` is used as given; no other value is selected by the code. ``min_p`` / ``max_p``
    clamp p before the extrapolation, so they change ``f_h0`` and both GCI. Convergence needs both
    ``GCI_m_c`` and ``GCI_f_m`` below the threshold.

    Notes
    -----
    With ``r = n_medium / n_coarse = n_fine / n_medium`` (n = number of elements) and levels
    ``f_c, f_m, f_f``:

    - observed order: ``p = ln((f_c - f_m) / (f_m - f_f)) / ln(r)``
    - Richardson extrapolation: ``f_h0 = f_f + (f_f - f_m) / (r**p - 1)``
    - relative differences: ``e_m_c = (f_m - f_c) / f_c``, ``e_f_m = (f_f - f_m) / f_m``
    - ``GCI_m_c = Fs |e_m_c| / (r**p - 1)``, ``GCI_f_m = Fs |e_f_m| / (r**p - 1)``
    - converged if both GCI are below ``gci_threshold``.

    Flat-field shortcut: if both level differences are at most ``max(1e-9, 1e-8 max(|f|, 1))`` then
    ``p = nan``, the GCI are 0 and the interval is converged.

    References
    ----------
    Roache (1998) and Richardson (1911): named in the code, not in the library (source to find).
    """

    def __init__(self,
                 f_coarse: float,
                 f_medium: float,
                 f_fine: float,
                 x_nodes_coarse: list,
                 x_nodes_medium: list,
                 x_nodes_fine: list,
                 gci_threshold: float = 0.01,
                 safety_factor: float = 1.25,
                 min_p: float | None = None,
                 max_p: float | None = None):
        """Store the arguments; see the class docstring."""

        self.x_nodes_coarse = x_nodes_coarse
        self.x_nodes_medium = x_nodes_medium
        self.x_nodes_fine   = x_nodes_fine
        self.gci_threshold  = gci_threshold
        self.safety_factor  = safety_factor
        self.f_coarse       = f_coarse
        self.f_medium       = f_medium
        self.f_fine         = f_fine

        n_c = len(x_nodes_coarse) - 1
        n_m = len(x_nodes_medium) - 1
        n_f = len(x_nodes_fine) - 1
        if n_c <= 0 or n_m <= 0 or n_f <= 0:
            raise ValueError(
                f"Cannot compute refinement ratio: level(s) with fewer "
                f"than 2 nodes (n_c={n_c}, n_m={n_m}, n_f={n_f})."
            )
        self.r_m_c = n_m / n_c
        self.r_f_m = n_f / n_m

        if self.r_m_c <= 1.0:
            raise ValueError(
                f"Refinement ratio r_m_c must be > 1. Got {self.r_m_c:.4f}. "
                f"Medium mesh must be finer than coarse mesh."
            )
        if self.r_f_m <= 1.0:
            raise ValueError(
                f"Refinement ratio r_f_m must be > 1. Got {self.r_f_m:.4f}. "
                f"Fine mesh must be finer than medium mesh."
            )
        if self.r_f_m != self.r_m_c:
            raise ValueError(
                f"Non-uniform refinement ratio: r_f_m={self.r_f_m:.4f} != "
                f"r_m_c={self.r_m_c:.4f}. "
                f"Grades must be generated by uniform bisection."
            )

        self.r = self.r_f_m

        scale = max(abs(self.f_coarse), abs(self.f_medium), abs(self.f_fine), 1.0)
        RELATIVE_FLAT_TOL = 1e-8
        ABS_FLAT_TOL = 1e-9
        d_m_c = abs(self.f_medium - self.f_coarse)
        d_f_m = abs(self.f_fine - self.f_medium)
        flat_tol = max(ABS_FLAT_TOL, RELATIVE_FLAT_TOL * scale)
        if d_m_c <= flat_tol and d_f_m <= flat_tol:
            self.p = float("nan")
            self.e_m_c = 0.0
            self.e_f_m = 0.0
            self.f_h0 = self.f_fine
            self.GCI_m_c = 0.0
            self.GCI_f_m = 0.0
            self.converged_m_c = True
            self.converged_f_m = True
            self.converged = True
            return

        if self.f_medium == self.f_fine:
            raise ValueError(
                "f_medium == f_fine -- cannot compute observed order p "
                "(division by zero in the convergence ratio) for this "
                "metric/interval."
            )
        ratio = (self.f_coarse - self.f_medium) / (self.f_medium - self.f_fine)
        if ratio <= 0.0:
            raise ValueError(
                f"Non-monotonic convergence (f_coarse-f_medium and "
                f"f_medium-f_fine have opposite sign, ratio={ratio:.6g} <= 0) "
                f"-- cannot compute a real observed order p for this "
                f"metric/interval."
            )
        self.p = float(np.log(ratio) / np.log(self.r))
        
        if self.p <= 0.0:
            raise ValueError(
                f"Non-positive observed order p={self.p:.6g} "
                f"(ratio={ratio:.6g} <= 1: differences between levels are not "
                f"decreasing) -- not in the asymptotic range, GCI undefined "
                f"for this metric/interval."
            )

        if min_p is not None:
            self.p = max(self.p, min_p)
        if max_p is not None:
            self.p = min(self.p, max_p)

        if abs(self.r**self.p - 1.0) < 1e-12:
            raise ValueError(
                f"Degenerate observed order p={self.p:.6g} (r**p - 1 ~= 0) -- "
                f"cannot extrapolate or compute GCI for this metric/interval."
            )

        if self.f_coarse == 0.0:
            raise ValueError(
                "f_coarse == 0.0 -- cannot compute relative error e_m_c "
                "(division by zero) for this metric/interval."
            )
        self.e_m_c = (self.f_medium - self.f_coarse) / self.f_coarse

        if self.f_medium == 0.0:
            raise ValueError(
                "f_medium == 0.0 -- cannot compute relative error e_f_m "
                "(division by zero) for this metric/interval."
            )
        self.e_f_m = (self.f_fine - self.f_medium) / self.f_medium

        self.f_h0 = self.f_fine + (self.f_fine - self.f_medium) / (self.r**self.p - 1)

        self.GCI_m_c = (self.safety_factor * abs(self.e_m_c) / (self.r_m_c**self.p - 1))
        self.GCI_f_m = (self.safety_factor * abs(self.e_f_m) / (self.r_f_m**self.p - 1))

        self.converged_m_c = self.GCI_m_c < self.gci_threshold
        self.converged_f_m = self.GCI_f_m < self.gci_threshold
        self.converged     = self.converged_f_m and self.converged_m_c


def _compute_gci(f_c, f_m, f_f, x_c, x_m, x_f, *,
                  gci_threshold: float, safety_factor: float,
                  min_p: float | None, max_p: float | None) -> "RichardsonGCI | _DummyGCI":
    """Compute the GCI of one metric over three levels, with a fallback.

    The thresholds are the same for every request of a ``MeshConvergenceStudy``.

    Parameters
    ----------
    f_c, f_m, f_f : float
        Metric values at the coarse, medium and fine levels.
    x_c, x_m, x_f : list of float
        Node positions of the three levels [mm].
    gci_threshold, safety_factor, min_p, max_p
        As in ``RichardsonGCI``.

    Returns
    -------
    RichardsonGCI or _DummyGCI
        ``_DummyGCI`` (NaN GCI, not converged) if ``RichardsonGCI`` raises ``ValueError``,
        ``ZeroDivisionError`` or ``ArithmeticError``.
    """
    try:
        return RichardsonGCI(
            f_coarse=f_c, f_medium=f_m, f_fine=f_f,
            x_nodes_coarse=x_c, x_nodes_medium=x_m, x_nodes_fine=x_f,
            gci_threshold=gci_threshold, safety_factor=safety_factor,
            min_p=min_p, max_p=max_p,
        )
    except (ValueError, ZeroDivisionError, ArithmeticError):
        return _DummyGCI()


# ===========================================================================
# Request label -- dictionary key used in
# ConvergenceRecord.point_metrics_history/gci_history, derived from the
# tuple (field, form[, point]).
# ===========================================================================

EvalRequest = tuple  # (field_name, form_name) or (field_name, form_name, point_name)


def _request_label(request: EvalRequest) -> str:
    """Dictionary key of an evaluation request: the tuple elements joined with ``":"``.

    Used as key of ``ConvergenceRecord.point_metrics_history`` and ``gci_history``.

    Parameters
    ----------
    request : tuple of str
        ``(field_name, form_name)`` or ``(field_name, form_name, point_name)``.

    Returns
    -------
    str
    """
    return ":".join(request)


# ===========================================================================
# Orchestrator -- no solves. Receives one SubmodelResult per level, already
# solved by a dispatcher; it only does the convergence and GCI bookkeeping.
# ===========================================================================

class MeshConvergenceStudy:
    """Convergence bookkeeping per interval, given one ``SubmodelResult`` per refinement level.

    The results come from outside; this class runs no solve. For every level it evaluates the
    requested metrics, and from the third level on it computes the GCI of each metric over the last
    three levels. The study of an interval stops (``add_level`` returns ``True``) at the first level
    triple where the GCI of every request is below the threshold.

    Parameters
    ----------
    requests : list of tuple
        Each item is ``(field_name, form_name)`` or ``(field_name, form_name, point_name)``.
        ``field_name`` is read from the ``SubmodelResult`` with ``getattr``; ``form_name`` must be
        registered with ``register_eval_form``; ``point_name`` is needed only for forms registered with
        ``needs_point=True`` and its position [mm] comes from the ``points`` argument of ``add_level``.
    gci_threshold : float, optional
        GCI below which a metric counts as converged [-]. Default 0.01.
    safety_factor : float, optional
        Safety factor Fs of the GCI [-]. Default 1.25 (see ``RichardsonGCI``).
    min_p, max_p : float or None, optional
        Clamp of the observed order p [-].

    Raises
    ------
    ValueError
        If a form is unknown, or a request has the wrong number of items for its form.

    Notes
    -----
    The thresholds apply to all requests alike. At least 3 levels are needed for a GCI.
    """

    _MIN_LEVELS_FOR_GCI = 3

    def __init__(
        self,
        requests: list[EvalRequest],
        gci_threshold: float = 0.01,
        safety_factor: float = 1.25,
        min_p: float | None = None,
        max_p: float | None = None,
    ):
        """Store the arguments; see the class docstring."""
        self._requests = list(requests)
        for request in self._requests:
            form_name = request[1]
            resolve_eval_form(form_name)  # validate early: fail at construction, not mid-study
            needs_point = form_name in _POINT_FORMS
            if needs_point and len(request) != 3:
                raise ValueError(
                    f"MeshConvergenceStudy: request {request} uses form "
                    f"'{form_name}', which needs a point -- expected "
                    f"(field_name, form_name, point_name)."
                )
            if not needs_point and len(request) != 2:
                raise ValueError(
                    f"MeshConvergenceStudy: request {request} uses form "
                    f"'{form_name}', which takes no point -- expected "
                    f"(field_name, form_name)."
                )

        self._gci_threshold = gci_threshold
        self._safety_factor = safety_factor
        self._min_p = min_p
        self._max_p = max_p

        # Private bookkeeping: x_nodes per level, per record. Not stored in
        # ConvergenceRecord (it is not a field of that dataclass); it lives
        # as long as this MeshConvergenceStudy instance, keyed by id(rec).
        # Assumes each ConvergenceRecord is processed to the end (converged
        # or max levels exhausted) before the object is discarded; not meant
        # to survive a record rebuilt with the same id() after the original
        # was garbage-collected.
        self._x_nodes_history: dict[int, list[list[float]]] = {}

    def new_record(self, label: str, x_lo: float, x_hi: float) -> ConvergenceRecord:
        """Create an empty ``ConvergenceRecord`` for the interval ``[x_lo, x_hi]``.

        Parameters
        ----------
        label : str
            Name of the load or interval.
        x_lo, x_hi : float
            Interval bounds [mm].

        Returns
        -------
        ConvergenceRecord
        """
        return ConvergenceRecord(label=label, x_lo=x_lo, x_hi=x_hi)

    def add_level(
        self,
        rec: ConvergenceRecord,
        grade: str,
        result: "SubmodelResult",
        points: dict[str, float] | None = None,
    ) -> bool:
        """Register one more refinement level in ``rec`` and test convergence.

        Parameters
        ----------
        rec : ConvergenceRecord
            Record of the interval; ``levels``, ``point_metrics_history`` and (from the third level)
            ``gci_history`` are appended to.
        grade : str
            Name of the mesh grade of this level.
        result : SubmodelResult
            Sub-model result at this level (fields in their own units).
        points : dict of str to float or None, optional
            ``{point_name: x [mm]}`` for the requests that need a point, already resolved by the caller;
            ``None`` if no request needs one.

        Returns
        -------
        bool
            ``True`` if this level reached convergence (``rec.converged`` and ``rec.x_final`` are then
            set); the caller uses it to stop the grade loop.

        Raises
        ------
        ValueError
            If a request needs a point that is not in ``points``; also from the evaluation form
            itself (``at_point``: no node at the point; ``max_minus_mean_relative``: zero mean).
        """
        metrics: dict[str, float] = {}
        for request in self._requests:
            field_name, form_name = request[0], request[1]
            values = getattr(result, field_name)
            form = resolve_eval_form(form_name)

            if form_name in _POINT_FORMS:
                point_name = request[2]
                if points is None or point_name not in points:
                    raise ValueError(
                        f"add_level: request {request} needs point "
                        f"'{point_name}', not provided in `points`."
                    )
                value = form(values, result.x_nodes, points[point_name])
            else:
                value = form(values)

            metrics[_request_label(request)] = value

        rec.levels.append(grade)
        rec.point_metrics_history.append(metrics)

        x_history = self._x_nodes_history.setdefault(id(rec), [])
        x_history.append(list(result.x_nodes))

        if len(rec.levels) >= self._MIN_LEVELS_FOR_GCI:
            gci_this_transition: dict[str, "RichardsonGCI | _DummyGCI"] = {}
            for label in metrics:
                f_c = rec.point_metrics_history[-3][label]
                f_m = rec.point_metrics_history[-2][label]
                f_f = rec.point_metrics_history[-1][label]
                x_c, x_m, x_f = x_history[-3], x_history[-2], x_history[-1]
                gci_this_transition[label] = _compute_gci(
                    f_c, f_m, f_f, x_c, x_m, x_f,
                    gci_threshold=self._gci_threshold,
                    safety_factor=self._safety_factor,
                    min_p=self._min_p, max_p=self._max_p,
                )
            rec.gci_history.append(gci_this_transition)

            if all(g.converged for g in gci_this_transition.values()):
                rec.converged = True
                rec.x_final = list(result.x_nodes)
                return True

        return False

    def finalize_unconverged(self, rec: ConvergenceRecord, result: "SubmodelResult") -> None:
        """Publish ``rec.x_final`` when the maximum number of levels was reached without convergence.

        Parameters
        ----------
        rec : ConvergenceRecord
            Record of the interval.
        result : SubmodelResult
            Sub-model result of the last level; its nodes become ``rec.x_final`` [mm].

        Warnings
        --------
        ``x_final`` is set although ``rec.converged`` is ``False``; the caller decides whether to use it
        (see ``MeshRefinementResult.all_extra_nodes``).
        """
        rec.x_final = list(result.x_nodes)


def build_mesh_refinement_result(shaft_name: str, per_load: dict[str, ConvergenceRecord]) -> MeshRefinementResult:
    """Assemble the final ``MeshRefinementResult`` from the filled convergence records.

    Parameters
    ----------
    shaft_name : str
        Name of the shaft.
    per_load : dict of str to ConvergenceRecord
        One record per interval.

    Returns
    -------
    MeshRefinementResult
    """
    return MeshRefinementResult(shaft_name=shaft_name, per_load=per_load)