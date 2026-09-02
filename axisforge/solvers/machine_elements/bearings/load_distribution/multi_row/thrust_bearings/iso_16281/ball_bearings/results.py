"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_results.py

The LOCAL library for point-contact (ball) bearings: the result *shapes* a
solve produces (BallLoadDistributionResult, BallBearingResult) together
with the registry that holds one BallBearingResult per bearing label
(BallLoadDistributionLibrary). "results" IS the local library -- not a
separate concept from it, mirroring Roller_Bearing/roller_bearing_results.py
on the line-contact side.

RECONSTRUCTED, this turn -- unifying single-row and multi-row
-----------------------------------------------------------------
Previously, a single-row solve produced a bare BallLoadDistributionResult
and a multi-row solve produced a completely different class,
MultiRowBallLoadDistributionResult (defined inside
ball_bearing_multirow_solver.py, not here). That asymmetry is what forced
every downstream consumer -- postprocess_and_record() in
rolling_bearing_solver.py, the multi-row postprocessing module -- to branch
on `_is_multirow(bearing)` and handle the two shapes separately.

This file now defines TWO result types instead:

  BallLoadDistributionResult -- UNCHANGED. Still the raw output of ONE
      point-contact solve (ISO16281BallSolver.solve_contact(), formerly
      _solve_bearing()) -- delta_r, delta_a, psi, phi_Fr, delta_j, alpha_j,
      Mz, n_iter, residual, ok. This is a PER-ROW result: for a multi-row
      bearing, row j's entry in BallBearingResult.rows is exactly what this
      class would hold had row j been solved alone with its converged share
      of the total load. Nothing about its fields changes with row count --
      it never needed to know about rows in the first place.

  BallBearingResult -- NEW. The unified per-BEARING result, single-row or
      multi-row alike. `rows: list[BallLoadDistributionResult]` is the only
      required field -- length 1 for an ordinary single-row bearing, length
      i for a multi-row bearing's i rows (index-aligned with bearing.rows).
      `is_multirow` is derived from `len(rows) >= 2`, not stored as a second
      flag that could drift out of sync with the list it describes. The
      outer-iteration diagnostics that only exist for a multi-row solve
      (f_r, f_a, outer_n_iter, outer_residual, outer_ok -- see
      ball_bearing_multirow_solver.py's module docstring for what they mean)
      are optional fields, None for a single-row result, where no outer
      iteration ever ran.

      This replaces MultiRowBallLoadDistributionResult, which used to live
      in ball_bearing_multirow_solver.py -- that class is retired;
      ball_bearing_multirow_solver.py now builds a BallBearingResult via
      the .multirow() classmethod instead of defining its own container.
      ISO16281BallSolver.solve_contact() output is wrapped the same way via
      .single() wherever it is recorded per-label (see
      BallLoadDistributionLibrary below) -- so BOTH solvers now hand back
      the exact same result TYPE, differing only in how many rows are
      inside it and whether the outer-iteration fields are populated.

Neither class imports anything from library.py, real or TYPE_CHECKING --
they only happen to structurally match library.LoadDistributionResult (a
Protocol); they do not depend on it. This keeps the module's only runtime
dependency chain going the other way: ball_bearing_postprocessing.py and
ball_bearing_multirow_solver.py import from here, ball_bearing_solver.py
imports from here, and this file imports nothing from either of them.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# BallLoadDistributionResult -- one point-contact solve, per row. UNCHANGED.
# ---------------------------------------------------------------------------

class BallLoadDistributionResult:
    """
    Result of a single point-contact (ball) bearing internal load
    distribution solve -- every field defined directly here. Fully
    self-contained: no import, subclassing, or other runtime dependency on
    library.py or on Roller_Bearing -- it only happens to structurally match
    library.LoadDistributionResult (a Protocol), it does not depend on it.

    This is a PER-ROW result -- see BallBearingResult below for the
    container that holds one or more of these for a whole bearing.

    Attributes
    ----------
    delta_r   float [mm]        radial ring displacement in the resultant-force plane
    delta_a   float [mm]        axial ring displacement
    psi       float [rad]       prescribed ring misalignment in the resultant-force plane
    phi_Fr    float [rad]       angle of resultant Fr in the global frame (output/plots only)
    delta_j   ndarray(Z,) [mm]  elastic deflection per rolling element
    alpha_j   ndarray(Z,) [rad] effective contact angle per element
    Mz        float [N*mm]      moment reaction
    n_iter    int               solver function evaluations
    residual  float [N]         final ||R||
    ok        bool              solver convergence flag

    Contact force per element (Q_j = cp * delta_j^n) is computed by
    ball_bearing_postprocessing.Q_j(), not stored here.
    """
    __slots__ = (
        "delta_r", "delta_a", "psi", "phi_Fr",
        "delta_j", "alpha_j",
        "Mz", "n_iter", "residual", "ok",
    )

    def __init__(self, *, delta_r, delta_a, psi, phi_Fr,
                 delta_j, alpha_j, Mz, n_iter, residual, ok):
        self.delta_r  = delta_r
        self.delta_a  = delta_a
        self.psi      = psi
        self.phi_Fr   = phi_Fr
        self.delta_j  = delta_j
        self.alpha_j  = alpha_j
        self.Mz       = Mz
        self.n_iter   = n_iter
        self.residual = residual
        self.ok       = ok


# ---------------------------------------------------------------------------
# BallBearingResult -- unified per-bearing container, single-row or
# multi-row. NEW this turn -- replaces MultiRowBallLoadDistributionResult.
# ---------------------------------------------------------------------------

class BallBearingResult:
    """
    Result of solving a ball bearing's internal load distribution -- single-
    row or multi-row, through the SAME type. `rows` is always a list:
    length 1 for an ordinary single-row bearing, length i for a multi-row
    bearing's i rows (index-aligned with bearing.rows). `is_multirow` is a
    property derived from `len(rows) >= 2` -- there is no separate flag
    that could disagree with the list it describes.

    Build via the two classmethods below rather than the constructor
    directly -- .single() and .multirow() name the two cases explicitly at
    the call site instead of leaving a reader to infer them from which
    optional arguments were passed.

    Outer-iteration fields (f_r, f_a, outer_n_iter, outer_residual,
    outer_ok) are the converged load-split diagnostics from
    ISO16281MultiRowBallSolver.solve_bearing()'s outer fixed-point
    iteration -- see that module's docstring for the physical meaning
    (co-located rows sharing one rigid-ring displacement). They are None
    for a single-row result, where no outer iteration ever ran -- callers
    that need to branch on "did an outer split actually happen" should
    check `is_multirow`, not `outer_ok is not None` (the two happen to
    agree today, but is_multirow is the one defined to always be correct).

    delta_r / delta_a / psi read rows[0] -- for a multi-row result this is
    the shared rigid-ring displacement every row agrees on at convergence
    (see ball_bearing_multirow_solver.py); for a single-row result rows[0]
    is simply the only row there is. Either way, "the bearing's own
    (delta_r, delta_a, psi)" is a well-defined question with a single
    answer, so exposing it here (rather than forcing every caller to write
    `result.rows[0].delta_r`) is correct for both cases, not a multi-row-only
    convenience.

    Attributes
    ----------
    rows            list[BallLoadDistributionResult]   length 1 or i
    f_r             ndarray(i,) | None   converged radial-load fractions
    f_a             ndarray(i,) | None   converged axial-load fractions
    outer_n_iter    int   | None   outer solver function evaluations
    outer_residual  float | None   outer solver final ||R||
    outer_ok        bool  | None   outer solver convergence flag
    """
    __slots__ = ("rows", "f_r", "f_a", "outer_n_iter", "outer_residual", "outer_ok")

    def __init__(self,
                 rows: Sequence[BallLoadDistributionResult],
                 f_r: np.ndarray | None = None,
                 f_a: np.ndarray | None = None,
                 outer_n_iter: int | None = None,
                 outer_residual: float | None = None,
                 outer_ok: bool | None = None):
        rows = list(rows)
        if not rows:
            raise ValueError("BallBearingResult needs at least 1 row, got 0")
        self.rows           = rows
        self.f_r             = f_r
        self.f_a             = f_a
        self.outer_n_iter    = outer_n_iter
        self.outer_residual  = outer_residual
        self.outer_ok         = outer_ok

    @classmethod
    def single(cls, row: BallLoadDistributionResult) -> "BallBearingResult":
        """Wrap an ordinary single-row solve -- the common case."""
        return cls(rows=[row])

    @classmethod
    def multirow(cls,
                 rows: Sequence[BallLoadDistributionResult],
                 f_r: np.ndarray, f_a: np.ndarray,
                 n_iter: int, residual: float, ok: bool) -> "BallBearingResult":
        """
        Wrap a converged multi-row solve -- see
        ISO16281MultiRowBallSolver.solve_bearing() in
        ball_bearing_multirow_solver.py, the only caller of this
        classmethod. `rows` must have len >= 2 (enforced by
        solve_bearing() itself before this is ever called).
        """
        if len(rows) < 2:
            raise ValueError(
                f"BallBearingResult.multirow() needs >=2 rows, got {len(rows)} "
                f"-- use .single() for an ordinary single-row result."
            )
        return cls(rows=rows, f_r=f_r, f_a=f_a,
                   outer_n_iter=n_iter, outer_residual=residual, outer_ok=ok)

    @property
    def is_multirow(self) -> bool:
        return len(self.rows) >= 2

    @property
    def delta_r(self) -> float:
        return self.rows[0].delta_r

    @property
    def delta_a(self) -> float:
        return self.rows[0].delta_a

    @property
    def psi(self) -> float:
        return self.rows[0].psi


# ---------------------------------------------------------------------------
# BallLoadDistributionLibrary -- LOCAL registry, point-contact bearings only
# ---------------------------------------------------------------------------

class BallLoadDistributionLibrary:
    """
    Local registry of BallBearingResult, keyed by bearing label -- scoped to
    point-contact bearings (DEEP_GROOVE_BALL, ANGULAR_CONTACT today; a
    multi-row THRUST_BALL bearing solved directly via
    ISO16281MultiRowBallSolver is NOT recorded here -- that solver returns
    its BallBearingResult straight to its caller, it does not populate this
    library. See ball_bearing_multirow_solver.py's own docstring).

    CHANGED, this turn: entries are now BallBearingResult, not a bare
    BallLoadDistributionResult -- ISO16281BallSolver.solve() wraps every
    per-label solve_contact() output via BallBearingResult.single() before
    calling set(). `.get(label).rows[0]` recovers the old bare per-row
    result if a caller specifically needs it.

    Populated by ISO16281BallSolver.solve() (in ball_bearing_solver.py) and
    read back by rolling_bearing_solver.RollingBearingSolver.solve(), which
    merges this together with whatever other per-type local libraries
    (e.g. RollerLoadDistributionLibrary) were produced for the rest of a
    shaft's bearing set. This class is NOT the final, cross-type registry --
    that is BearingResultsLibrary in the global library.py one level up.
    This one only ever holds the load distribution result for point-contact
    bearings.

    Same shape as SimpleFEMResultsLibrary / BearingResultsLibrary /
    RollerLoadDistributionLibrary (set/get/labels/__contains__/__iter__/
    __len__) -- satisfies library.LoadDistributionLibrary structurally,
    without importing it.
    """

    def __init__(self):
        self._results: dict[str, BallBearingResult] = {}

    def set(self, label: str, result: BallBearingResult) -> None:
        self._results[label] = result

    def get(self, label: str) -> BallBearingResult:
        """Raises KeyError if nothing has been recorded for this label yet."""
        if label not in self._results:
            raise KeyError(
                f"No load distribution recorded for bearing '{label}' -- "
                f"nothing has been set on it yet."
            )
        return self._results[label]

    def labels(self) -> list[str]:
        """All bearing labels with a recorded result."""
        return list(self._results)

    def __contains__(self, label: str) -> bool:
        return label in self._results

    def __iter__(self):
        return iter(self._results.values())

    def __len__(self) -> int:
        return len(self._results)