"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/roller_bearing_results.py

The LOCAL library for CYLINDRICAL_ROLLER bearings: the result *shapes* a
solve produces (RollerLoadDistributionResult, RollerBearingResult) together
with the registry that holds one RollerBearingResult per bearing label
(RollerLoadDistributionLibrary). "results" IS the local library -- not a
separate concept from it. Mirrors Ball_Bearing/ball_bearing_results.py.

RECONSTRUCTED, this turn -- unified shape, container ready for multi-row,
prepared ahead of need for a FUTURE thrust roller family
-----------------------------------------------------------------------------
This file defines TWO result types, mirroring Ball_Bearing/
ball_bearing_results.py's CONTAINER SHAPE:

  RollerLoadDistributionResult -- UNCHANGED across every reconstruction
      pass. Still the raw output of ONE line-contact solve
      (ISO16281RollerSolver.solve_contact(), formerly _solve_bearing()) --
      the base fields plus the per-lamina extras (x_k, psi_j, delta_jk,
      q_jk).

  RollerBearingResult -- the unified per-BEARING result. `rows:
      list[RollerLoadDistributionResult]` is the only required field --
      length 1 for CylindricalRollerFamily (radial NU/N-type, the only
      family that exists in core/ today) and, once a thrust roller family
      exists, length i for that family's rows. Built via .single()
      (ISO16281RollerSolver.solve()) or .multirow() (a future
      ISO16281MultiRowRollerSolver, not yet written -- see below).

  NOT the same "multi-row" as CylindricalRollerFamily's own `i`
  ----------------------------------------------------------------
  CylindricalRollerFamily (radial, NU/N-type) is RADIAL duty, same as
  DeepGrooveBallFamily -- and DeepGrooveBallFamily has NO multi-row family
  counterpart: its multi-row case (i=2) is a plain capacity-rating
  multiplier on ONE raceway (REDUCTION_FACTOR_BY_ROWS, ISO 281:2007
  Table 1), solved as a single delta_r, no rows list, no outer load-split.
  CylindricalRollerFamily's own `i` (assemble_geometry(), see that file)
  does the exact same thing for radial rollers -- a genuinely-separate
  MultiRowCylindricalRollerFamily / cylindrical_roller_multirow.py was
  tried and retired; do not reintroduce it, radial multi-row stays the
  plain `i` multiplier, mirroring DGBB.

  The `.multirow()` classmethod below exists for a DIFFERENT, not-yet-built
  case instead: a future thrust roller family (needle or cylindrical
  thrust roller -- not yet in core/), mirroring MultiRowThrustBallFamily/
  ISO16281MultiRowBallSolver on the ball side, where THRUST duty genuinely
  needs axially-stacked rows sharing a load-split compatibility solve. Kept
  here, unexercised, so this container's shape does not need to change
  again once that family exists -- exactly the same "prepared ahead of
  need" reasoning roller_bearing_postprocessing.py's _row_views() already
  uses. Building the actual ISO16281MultiRowRollerSolver orchestration
  (mirroring roller_bearing_multirow_solver.py's retired first attempt) is
  deliberately NOT done yet: that first attempt was framed around the
  WRONG physical idealization (radial -- delta_r shared, Fa hardcoded to
  0), which does not carry over to thrust roller (Fa is the primary load
  there, mirroring the ball thrust solver's delta_a-sharing, not delta_r).
  Write that solver fresh, correctly framed around thrust duty, once the
  thrust roller core family actually exists -- guessing at its physics now
  would not be sourced.

Neither class imports anything from library.py, real or TYPE_CHECKING --
they only happen to structurally match library.LoadDistributionResult (a
Protocol); they do not depend on it. roller_bearing_postprocessing.py
imports both from here (not from roller_bearing_solver.py);
roller_bearing_solver.py never imports roller_bearing_postprocessing.py,
and this module never imports either of them.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# RollerLoadDistributionResult -- one line-contact solve, per row. UNCHANGED.
# ---------------------------------------------------------------------------

class RollerLoadDistributionResult:
    """
    Result of a single CYLINDRICAL_ROLLER bearing internal load distribution
    solve -- every field defined directly here, with the per-lamina data the
    §5.2 lamina model produces added alongside. Fully self-contained: no
    import, subclassing, or other runtime dependency on library.py or on
    Ball_Bearing -- it only happens to structurally match
    library.LoadDistributionResult (a Protocol), it does not depend on it.

    This is a PER-ROW result -- see RollerBearingResult below for the
    container that holds one or more of these for a whole bearing.

    Base fields keep the same meaning documented in library.py, adapted for
    line contact:
      delta_r  : radial ring displacement [mm] -- the one unknown solved
      delta_a  : always 0.0 -- radial roller bearings (NU/N-type) carry no
                 axial load
      psi      : prescribed ring misalignment [rad]
      phi_Fr   : angle of resultant Fr in the global frame (output/plots only)
      delta_j  : roller-centreline deflection per roller (Z,) [mm],
                 eq.(38), BEFORE the lamina/profile correction -- the
                 per-lamina deflection is delta_jk, not this
      alpha_j  : bearing.alpha_0 broadcast to (Z,) -- for a cylindrical
                 roller the contact normal stays radial regardless of tilt,
                 unlike a ball's alpha_j, which genuinely varies with load
      Mz       : diagnostic reaction moment, eq.(46), evaluated at the
                 converged (delta_r, psi) -- NOT a solve constraint here,
                 see roller_bearing_solver.py's module docstring
      n_iter   : solver function evaluations
      residual : final ||R|| [N]
      ok       : solver convergence flag

    Extra fields (line contact only)
    ----------------------------------
    x_k       ndarray(n_s,) [mm]     lamina positions, eq.(38)-figure 3
    psi_j     ndarray(Z,)   [rad]    per-roller local misalignment, eq.(39)
    delta_jk  ndarray(Z,n_s)[mm]     per-lamina elastic deflection, eq.(41)
    q_jk      ndarray(Z,n_s)[N]      per-lamina contact force, eq.(36)
    """
    __slots__ = (
        "delta_r", "delta_a", "psi", "phi_Fr",
        "delta_j", "alpha_j",
        "Mz", "n_iter", "residual", "ok",
        "x_k", "psi_j", "delta_jk", "q_jk",
    )

    def __init__(self, *, delta_r, delta_a, psi, phi_Fr,
                 delta_j, alpha_j, Mz, n_iter, residual, ok,
                 x_k, psi_j, delta_jk, q_jk):
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
        self.x_k      = x_k
        self.psi_j    = psi_j
        self.delta_jk = delta_jk
        self.q_jk     = q_jk


# ---------------------------------------------------------------------------
# RollerBearingResult -- unified per-bearing container, single-row today
# (CylindricalRollerFamily) or multi-row once a thrust roller family exists.
# Built via .single() (ISO16281RollerSolver) or .multirow() (a future
# ISO16281MultiRowRollerSolver, not yet written -- see module docstring).
# ---------------------------------------------------------------------------

class RollerBearingResult:
    """
    Result of solving a roller bearing's internal load distribution, through
    the same container shape as Ball_Bearing's BallBearingResult. `rows` is
    always a list -- length 1 for CylindricalRollerFamily (the only family
    in core/ today), length i once a thrust roller family's i rows exist.
    `is_multirow` (== `len(rows) >= 2`) is derived from that, not a second
    flag that could drift out of sync.

    Build via .single() for an ordinary single-row solve. .multirow() is
    kept ready for a future thrust-roller ISO16281MultiRowRollerSolver
    (not yet written -- see module docstring for why building it now would
    mean guessing at unsourced physics); nothing calls it today.

    delta_r / delta_a / psi read rows[0] -- for a future multi-row result
    this would be the shared rigid-ring displacement every row agrees on at
    convergence; for a single-row result rows[0] is simply the only row
    there is.

    f_a is ALWAYS None here, unlike BallBearingResult -- see the class's
    own docstring reasoning on the ball side: CylindricalRollerFamily
    (radial) carries no axial load at all (delta_a is always 0.0 on every
    row's RollerLoadDistributionResult), so there is no axial split to
    record for it. A future thrust roller family would need this
    reconsidered (Fa becomes the primary load there) -- not done here.

    Attributes
    ----------
    rows            list[RollerLoadDistributionResult]   length 1 or i
    f_r             ndarray | None   converged radial-load fractions (multi-row only)
    f_a             ndarray | None   ALWAYS None today -- see above
    outer_n_iter    int   | None     outer solver function evaluations (multi-row only)
    outer_residual  float | None     outer solver final ||R|| (multi-row only)
    outer_ok        bool  | None     outer solver convergence flag (multi-row only)
    """
    __slots__ = ("rows", "f_r", "f_a", "outer_n_iter", "outer_residual", "outer_ok")

    def __init__(self,
                 rows: Sequence[RollerLoadDistributionResult],
                 f_r: np.ndarray | None = None,
                 f_a: np.ndarray | None = None,
                 outer_n_iter: int | None = None,
                 outer_residual: float | None = None,
                 outer_ok: bool | None = None):
        rows = list(rows)
        if not rows:
            raise ValueError("RollerBearingResult needs at least 1 row, got 0")
        self.rows           = rows
        self.f_r             = f_r
        self.f_a             = f_a
        self.outer_n_iter    = outer_n_iter
        self.outer_residual  = outer_residual
        self.outer_ok         = outer_ok

    @classmethod
    def single(cls, row: RollerLoadDistributionResult) -> "RollerBearingResult":
        """Wrap an ordinary single-row solve."""
        return cls(rows=[row])

    @classmethod
    def multirow(cls,
                 rows: Sequence[RollerLoadDistributionResult],
                 f_r: np.ndarray,
                 n_iter: int, residual: float, ok: bool) -> "RollerBearingResult":
        """
        Wrap a converged multi-row solve -- for a future thrust roller
        family's ISO16281MultiRowRollerSolver.solve_bearing(), not yet
        written (see module docstring). `rows` must have len >= 2. No
        `f_a` parameter here yet either -- whether a thrust roller
        multi-row solve splits Fa, Fr, or both is exactly the physics that
        solver needs to get right when it is actually written; not
        guessed at here.
        """
        if len(rows) < 2:
            raise ValueError(
                f"RollerBearingResult.multirow() needs >=2 rows, got {len(rows)} "
                f"-- use .single() for an ordinary single-row result."
            )
        return cls(rows=rows, f_r=f_r, f_a=None,
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
# RollerLoadDistributionLibrary -- LOCAL registry, CYLINDRICAL_ROLLER only
# ---------------------------------------------------------------------------

class RollerLoadDistributionLibrary:
    """
    Local registry of RollerBearingResult, keyed by bearing label -- scoped
    to CYLINDRICAL_ROLLER bearings only.

    CHANGED, this turn: entries are now RollerBearingResult, not a bare
    RollerLoadDistributionResult -- ISO16281RollerSolver.solve() wraps
    every per-label solve_contact() output via RollerBearingResult.single()
    before calling set(). `.get(label).rows[0]` recovers the old bare
    per-row result if a caller specifically needs it.

    This is NOT the final, cross-type registry -- that is BearingResultsLibrary
    in the global library.py one level up, which also holds capacity/
    dynamic_equivalent_load/stiffness/extra for every bearing regardless of
    type. This one only ever holds the load distribution result for
    CYLINDRICAL_ROLLER bearings.

    Same shape as SimpleFEMResultsLibrary / BearingResultsLibrary /
    BallLoadDistributionLibrary (set/get/labels/__contains__/__iter__/
    __len__) -- satisfies library.LoadDistributionLibrary structurally,
    without importing it.
    """

    def __init__(self):
        self._results: dict[str, RollerBearingResult] = {}

    def set(self, label: str, result: RollerBearingResult) -> None:
        self._results[label] = result

    def get(self, label: str) -> RollerBearingResult:
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