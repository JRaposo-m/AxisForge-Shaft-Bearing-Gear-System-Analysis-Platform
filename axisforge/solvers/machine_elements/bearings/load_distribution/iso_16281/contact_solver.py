"""
axisforge/solvers/machine_elements/bearings/load_distribution/single_row/iso_16281/contact_solver.py

Single-row ISO/TS 16281 contact solvers -- abstract contract on top,
concrete point-contact (ball) and line-contact (roller) solvers below.
Everything that RUNS the iterative solve lives here; result shapes live
in results/bearings/load_distribution/single_row/*; postprocessing lives
in contact_postprocessing.py (never imported from here -- one-directional).

Multi-row (shared-displacement) solvers now live here too, immediately
after their single-row sibling, following the same abstract-on-top /
concrete-below pattern (MultiRowSolverBase after LineContactSolverBase,
then one concrete multi-row class per contact type next to its
single-row counterpart). They are NOT reviewed for bearing-to-bearing
coupling -- the shared-displacement formulation assumes co-located rows
(zero axial offset) and superposes each row's own reaction; it is a
step beyond what ISO/TS 16281 itself states (the standard gives
per-raceway equations, not row combination), valid as-is only for
thrust bearings. That physics is explicitly NOT touched by this pass --
only paths, decorator style and code shape were aligned with the
single-row classes.

References
----------
ISO/TS 16281:2008 Sec 4.2, eq.(12)-(15); Sec 4.3.1, eq.(19)-(28) -- point contact
ISO/TS 16281:2008 Sec 5.2, eq.(34)-(46) -- line contact, lamina model
Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch. 6
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem
from axisforge.results.fem_results.shaft_results import ShaftResults
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.numerics import run_root
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.validation import (
    check_bearing_ready, warn_if_floating_loaded,
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.dispatch import register_contact_solver
from axisforge.results.bearings.load_distribution.single_row.ball_bearing_results import (
    BallLoadDistributionResult, BallBearingResult,
)
from axisforge.results.bearings.load_distribution.single_row.roller_bearing_results import (
    RollerLoadDistributionResult, RollerBearingResult,
)
from axisforge.config import SOLVER_TOLERANCE


# =====================================================================
# ---- SolverBase / LineContactSolverBase -- abstract contract --------
# =====================================================================

class SolverBase(ABC):
    """
    Contract for a single-row ISO/TS 16281 contact solver. The batch loop
    over a shaft's bearing set (solve()) is identical between point and
    line contact -- grouping by label, projecting psi, checking readiness,
    warning on a floating-but-loaded bearing, wrapping the result -- so it
    is concrete here, once. What genuinely differs (solve_contact's own
    signature and equilibrium equations) stays on each concrete solver,
    reached through _solve_one_bearing() rather than forced into a
    uniform signature here (point contact needs Fa/delta_a_init; line
    contact has no axial capacity at all).

    CAPABILITY/REQUIRED_ATTRS are written onto concrete subclasses by
    register_contact_solver(), not declared in the class body -- see
    dispatch.py. _result_cls (the *BearingResult wrapper) and
    _solve_one_bearing() are set/implemented directly by each concrete
    subclass. _wrap_result() is a hook so that a solver whose
    _solve_one_bearing() doesn't return a single *LoadDistributionResult
    (e.g. a multi-row solver, which returns a whole row set + fractions)
    can still go through this same loop without solve() knowing about it.
    """

    CAPABILITY: str = ""
    REQUIRED_ATTRS: tuple[str, ...] = ()
    MULTIROW_SOLVER: type | None = None
    _result_cls: type | None = None

    def __init__(self, tol: float = SOLVER_TOLERANCE, psi_input: bool = False):
        self.tol       = tol
        self.psi_input = psi_input

    def solve(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              shaft_results: ShaftResults,
              psi_override: dict[str, float] | None = None,
              ) -> dict[str, object]:
        """
        Solve the internal load distribution for every bearing in
        `bearings` this solver applies to.

        For a single-row solver, returns one *BearingResult per label,
        each wrapping exactly 1 row. A multi-row solver (MultiRowSolverBase
        subclass) returns one *BearingResult per label wrapping N rows --
        the batch loop itself (grouping, psi, readiness, warnings) is
        identical either way, only _wrap_result() differs.
        """
        if self._result_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set _result_cls to its "
                f"*BearingResult wrapper class."
            )

        if self.psi_input and psi_override:
            unknown = set(psi_override) - set(bearings)
            if unknown:
                warnings.warn(
                    f"psi_override has labels not in `bearings`: {sorted(unknown)} -- "
                    f"check for typos; these entries are ignored."
                )

        node_by_label = {n.label: n for n in shaft_results.bearing_nodes}

        results: dict[str, object] = {}
        for label, b in bearings.items():
            check_bearing_ready(b, label, self.REQUIRED_ATTRS)
            self._extra_ready_checks(b, label)

            node   = node_by_label[label]
            Fr_xz  = node.Fr_xz
            Fr_xy  = node.Fr_xy
            phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

            warn_if_floating_loaded(b, label, node.Fa)

            if (self.psi_input
                    and psi_override is not None
                    and label in psi_override):
                psi = float(psi_override[label])
            else:
                psi = node.psi_xz * np.cos(phi_Fr) + node.psi_xy * np.sin(phi_Fr)

            row = self._solve_one_bearing(b, node, phi_Fr, psi)
            results[label] = self._wrap_result(row)

        return results

    def _extra_ready_checks(self, bearing: Bearing, label: str) -> None:
        """Hook for readiness checks beyond REQUIRED_ATTRS. No-op by
        default; overridden by LineContactSolverBase and MultiRowSolverBase."""
        return None

    def _wrap_result(self, row):
        """Hook: wraps whatever _solve_one_bearing() returns into this
        solver's *BearingResult. Default = single-row (row is ONE
        *LoadDistributionResult). MultiRowSolverBase overrides this --
        its _solve_one_bearing() returns a row set + f_r/f_a instead."""
        return self._result_cls.single(row)

    @abstractmethod
    def _solve_one_bearing(self, bearing: Bearing, node, phi_Fr: float, psi: float):
        """Solve ONE bearing given its FEM node data, phi_Fr and psi.
        Must call self.solve_contact(...) with whatever arguments this
        solver's own physics needs and return the resulting
        *LoadDistributionResult row (or, for a multi-row solver, whatever
        _wrap_result() on that solver expects)."""
        ...


class LineContactSolverBase(SolverBase):
    """Specialization for line-contact (lamina-model) solvers -- adds the
    Sec 5.2.2 minimum-lamina-count check on top of REQUIRED_ATTRS."""

    MIN_LAMINAE = 30  # Sec 5.2.2 -- "the number of laminae shall be at least n_s = 30"

    def _extra_ready_checks(self, bearing: Bearing, label: str) -> None:
        n_s = bearing.n_s
        if n_s < self.MIN_LAMINAE:
            raise ValueError(
                f"Bearing '{label}': n_s = {n_s} < {self.MIN_LAMINAE} -- "
                f"ISO/TS 16281 Sec 5.2.2 requires at least {self.MIN_LAMINAE} laminae."
            )
        if len(bearing.x_k) != n_s:
            raise ValueError(
                f"Bearing '{label}': len(x_k) = {len(bearing.x_k)} != n_s = {n_s} -- "
                f"x_k must have exactly one lamina midpoint per n_s."
            )


# =====================================================================
# ---- MultiRowSolverBase -- shared-displacement multi-row ------------
# ---- (NOT reviewed for row-to-row coupling -- see module docstring) --
# =====================================================================

class MultiRowSolverBase(SolverBase, ABC):
    """
    Base for a multi-row solver built by superposing its single-row
    sibling's static elements() kinematics once per row, under ONE shared
    root-find over the resultant displacement (delta_r[, delta_a]).

    This is deliberately NOT a generic n-row FE-style solve: rows are
    assumed co-located (zero axial offset), each row keeps its own
    geometry/stiffness (cp or cs/cL, phi_j, ...), and there is no
    bearing-to-bearing coupling term. Valid as-is only for thrust
    bearings -- see module docstring. Reviewing/extending this to real
    radial multi-row behavior is a separate, later task.

    SINGLE_ROW_SOLVER: the sibling single-row solver class (e.g.
    ISO16281BallSolver) this reuses for elements() (its static kinematics
    primitive) and for its REQUIRED_ATTRS / MIN_LAMINAE where relevant --
    set by each concrete subclass.
    """

    #: Sibling single-row solver class, reused for elements() + per-row
    #: attribute validation. Set by concrete subclasses.
    SINGLE_ROW_SOLVER: type[SolverBase] | None = None

    @staticmethod
    def _row_views(bearing: Bearing) -> list:
        """bearing.rows, each wrapped as a SimpleNamespace when it's a
        plain dict, so the sibling solver's elements() (which does
        duck-typing on attributes like cp, phi_j, ...) works unmodified
        per row."""
        rows = getattr(bearing, "rows", None) or []
        if len(rows) < 2:
            raise ValueError(
                f"Bearing '{getattr(bearing, 'label', '?')}': multi-row solver "
                f"needs >= 2 rows in bearing.rows; got {len(rows)}."
            )
        return [SimpleNamespace(**row) if isinstance(row, dict) else row for row in rows]

    def _extra_ready_checks(self, bearing: Bearing, label: str) -> None:
        for j, row in enumerate(self._row_views(bearing)):
            row_label = f"{label}[row{j}]"
            check_bearing_ready(row, row_label, self.SINGLE_ROW_SOLVER.REQUIRED_ATTRS)
            self._check_row_ready(row, row_label)

    @abstractmethod
    def _check_row_ready(self, row, row_label: str) -> None:
        """Row-specific readiness check beyond REQUIRED_ATTRS (e.g. cp > 0
        for ball rows, lamina count for roller rows)."""
        ...

    def _wrap_result(self, row):
        row_results, f_r, f_a, n_iter, residual_norm, ok = row
        # NOTE: assumes *BearingResult.multirow(rows, f_r, f_a, n_iter,
        # residual, ok) exists with this signature -- confirm/adjust
        # against the actual classmethod on BallBearingResult /
        # RollerBearingResult (I don't have that file verbatim here).
        return self._result_cls.multirow(
            rows=row_results, f_r=f_r, f_a=f_a,
            n_iter=n_iter, residual=residual_norm, ok=ok,
        )

    @staticmethod
    def _fractions(row_values: list[float], total: float) -> list[float]:
        """Per-row f_r/f_a: fraction of the bearing's total Fr (or Fa)
        absorbed by each row -- equal-split fallback when the total is
        ~0 (e.g. all-radial case for f_a)."""
        n = len(row_values)
        if abs(total) < 1e-9:
            return [1.0 / n] * n
        return [v / total for v in row_values]

    def _solve_one_bearing(self, bearing: Bearing, node, phi_Fr: float, psi: float):
        row_views = self._row_views(bearing)
        row_results, row_reactions, n_iter, residual_norm, ok = self._solve_rows(
            row_views,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy, Fa=getattr(node, "Fa", 0.0),
            psi=psi, phi_Fr=phi_Fr,
        )
        Fr_total = float(np.hypot(node.Fr_xz, node.Fr_xy))
        Fa_total = float(getattr(node, "Fa", 0.0))
        f_r = self._fractions([r for r, _ in row_reactions], Fr_total)
        f_a = self._fractions([a for _, a in row_reactions], Fa_total)
        return row_results, f_r, f_a, n_iter, residual_norm, ok

    @abstractmethod
    def _solve_rows(self, row_views, Fr_xz: float, Fr_xy: float, Fa: float,
                     psi: float, phi_Fr: float):
        """Resolve the shared-displacement system over all rows.

        Returns (row_results, row_reactions, n_iter, residual_norm, ok):
        row_results is a list of *LoadDistributionResult, one per row;
        row_reactions is a list of (Fr_row, Fa_row) floats, one per row
        (used only to back out f_r/f_a -- Fa_row is 0.0 for roller rows).
        Not forced into a common equation count on purpose -- ball solves
        2 unknowns (delta_r, delta_a), roller solves 1 (delta_r only)."""
        ...


# =====================================================================
# ---- ISO16281BallSolver -- point contact -----------------------------
# =====================================================================

@register_contact_solver(capability="point_contact",
                          required_attrs=("A", "alpha_0", "phi_j", "Ri", "cp", "Dpw", "Z"))
class ISO16281BallSolver(SolverBase):
    """Point contact (deep groove / angular contact ball bearings).
    2-equation root (delta_r, delta_a) in the resultant-force plane."""

    _result_cls = BallBearingResult

    def _solve_one_bearing(self, bearing: Bearing, node, phi_Fr: float, psi: float
                            ) -> BallLoadDistributionResult:
        return self.solve_contact(
            bearing,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy, Fa=node.Fa,
            delta_r_init=float(np.hypot(node.v_xz, node.v_xy)),
            delta_a_init=node.u,
            psi=psi, phi_Fr=phi_Fr,
        )

    @staticmethod
    def elements(bearing: Bearing, delta_r: float, delta_a: float, Vpsi: np.ndarray):
        """Per-element elastic deflection and effective contact angle --
        ISO/TS 16281 eq.(12)/(15), Sec 4.2.2.1."""
        A, alpha_0, phi_j = bearing.A, bearing.alpha_0, bearing.phi_j
        cp_j = np.cos(phi_j)

        U_j     = A * np.cos(alpha_0) + delta_r * cp_j
        V_j     = A * np.sin(alpha_0) + delta_a + Vpsi
        delta_j = np.maximum(np.sqrt(U_j**2 + V_j**2) - A, 0.0)

        alpha_j = np.arctan2(V_j, U_j)
        d32     = delta_j ** 1.5

        return delta_j, alpha_j, np.cos(alpha_j), np.sin(alpha_j), cp_j, d32

    def _initial_delta_r(self, bearing: Bearing, Fr: float, hint: float) -> float:
        if hint > 1e-6:
            return hint
        gap   = bearing.A * (1.0 - np.cos(bearing.alpha_0))
        hertz = (abs(Fr) / (bearing.cp * bearing.Z)) ** (2.0/3.0) if Fr != 0.0 else 1e-4
        return gap + hertz

    def _initial_delta_a(self, bearing: Bearing, Fa: float, hint: float) -> float:
        if abs(hint) > 1e-6:
            return hint
        if Fa == 0.0:
            return 0.0
        hertz = (abs(Fa) / (bearing.cp * bearing.Z)) ** (2.0/3.0)
        return np.copysign(hertz, Fa)

    def solve_contact(self,
                      bearing: Bearing,
                      Fr_xz: float, Fr_xy: float, Fa: float,
                      delta_r_init: float, delta_a_init: float,
                      psi: float, phi_Fr: float) -> BallLoadDistributionResult:
        """Equilibrium, Sec 4.2.2.1:
            Fr = cp * sum(delta_j^1.5 * cos(alpha_j) * cos(phi_j))
            Fa = cp * sum(delta_j^1.5 * sin(alpha_j))
        """
        Fr   = float(np.hypot(Fr_xz, Fr_xy))
        Vpsi = bearing.Ri * np.sin(psi) * np.cos(bearing.phi_j)
        cp   = bearing.cp

        def residual(u):
            dr, da = u
            _, _, ca, sa, cp_j, d32 = self.elements(bearing, dr, da, Vpsi)
            return np.array([
                Fr - cp * np.sum(d32 * ca * cp_j),
                Fa - cp * np.sum(d32 * sa),
            ])

        dr0               = self._initial_delta_r(bearing, Fr, delta_r_init)
        da0               = self._initial_delta_a(bearing, Fa, delta_a_init)
        x, nfev, res, ok  = run_root(residual, [dr0, da0], self.tol)
        delta_r, delta_a  = float(x[0]), float(x[1])

        delta_j, alpha_j, _, sa, cp_j, d32 = self.elements(bearing, delta_r, delta_a, Vpsi)
        Mz = (bearing.Dpw / 2.0) * cp * float(np.sum(d32 * sa * cp_j))

        return BallLoadDistributionResult(
            delta_r=delta_r, delta_a=delta_a, psi=psi, phi_Fr=phi_Fr,
            delta_j=delta_j, alpha_j=alpha_j,
            Mz=Mz, n_iter=nfev, residual=res, ok=ok,
        )

    def minimum_axial_load(self,
                            bearing: Bearing,
                            Fr_xz: float, Fr_xy: float,
                            psi: float,
                            delta_r_init: float = 0.0,
                            delta_a_init: float = 0.0,
                            Fa_bracket: tuple[float, float] = (0.0, 5.0e4),
                            xtol: float = 1e-6) -> tuple[float, BallLoadDistributionResult]:
        """Minimum axial preload [N] such that delta_a >= 0 (contact closure)."""
        check_bearing_ready(bearing, bearing.label, self.REQUIRED_ATTRS)
        phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

        res0 = self.solve_contact(bearing, Fr_xz, Fr_xy, 0.0,
                                  delta_r_init, delta_a_init, psi, phi_Fr)
        if res0.delta_a >= 0.0:
            return 0.0, res0

        def g(Fa: float) -> float:
            return self.solve_contact(bearing, Fr_xz, Fr_xy, Fa,
                                      delta_r_init, delta_a_init, psi, phi_Fr).delta_a

        lo, hi = Fa_bracket
        g_hi = g(hi)
        if g_hi < 0.0:
            raise ValueError(
                f"minimum_axial_load: delta_a = {g_hi:.4e} mm at Fa = {hi:.1f} N -- "
                f"widen Fa_bracket."
            )

        Fa_min = brentq(g, lo, hi, xtol=xtol)
        result = self.solve_contact(bearing, Fr_xz, Fr_xy, Fa_min,
                                    delta_r_init, delta_a_init, psi, phi_Fr)
        return Fa_min, result


@register_contact_solver(capability="point_contact", required_attrs=("rows",))
class ISO16281MultiRowBallSolverSharedDisplacement(MultiRowSolverBase):
    """Multi-row ball, shared delta_r/delta_a across all rows -- see
    MultiRowSolverBase and module docstring for the co-located-rows /
    no-coupling caveat (unchanged physics, only structure aligned here)."""

    _result_cls        = BallBearingResult
    SINGLE_ROW_SOLVER  = ISO16281BallSolver

    def _check_row_ready(self, row, row_label: str) -> None:
        if row.cp <= 0:
            raise ValueError(f"Bearing '{row_label}': cp must be positive; got {row.cp}.")

    def _initial_delta_r(self, row_views, Fr: float) -> float:
        # Pooled heuristic only (root-finder converges regardless of x0
        # quality) -- not an ISO/TS 16281 formula.
        A0, alpha0 = row_views[0].A, row_views[0].alpha_0
        cpZ_total  = sum(rv.cp * rv.Z for rv in row_views)
        gap   = A0 * (1.0 - np.cos(alpha0))
        hertz = (abs(Fr) / cpZ_total) ** (2.0/3.0) if Fr != 0.0 else 1e-4
        return gap + hertz

    def _initial_delta_a(self, row_views, Fa: float) -> float:
        if Fa == 0.0:
            return 0.0
        cpZ_total = sum(rv.cp * rv.Z for rv in row_views)
        hertz     = (abs(Fa) / cpZ_total) ** (2.0/3.0)
        return np.copysign(hertz, Fa)

    def _solve_rows(self, row_views, Fr_xz, Fr_xy, Fa, psi, phi_Fr):
        Fr       = float(np.hypot(Fr_xz, Fr_xy))
        elements = self.SINGLE_ROW_SOLVER.elements

        def residual(u):
            dr, da = u
            res_r, res_a = Fr, Fa
            for rv in row_views:
                Vpsi_rv = rv.Ri * np.sin(psi) * np.cos(rv.phi_j)
                _, _, ca, sa, cp_j, d32 = elements(rv, dr, da, Vpsi_rv)
                res_r -= rv.cp * np.sum(d32 * ca * cp_j)
                res_a -= rv.cp * np.sum(d32 * sa)
            return np.array([res_r, res_a])

        dr0 = self._initial_delta_r(row_views, Fr)
        da0 = self._initial_delta_a(row_views, Fa)
        x, nfev, res, ok = run_root(residual, [dr0, da0], self.tol)
        delta_r, delta_a = float(x[0]), float(x[1])

        row_results, row_reactions = [], []
        for rv in row_views:
            Vpsi_rv = rv.Ri * np.sin(psi) * np.cos(rv.phi_j)
            delta_j, alpha_j, ca, sa, cp_j, d32 = elements(rv, delta_r, delta_a, Vpsi_rv)
            Fr_row = float(rv.cp * np.sum(d32 * ca * cp_j))
            Fa_row = float(rv.cp * np.sum(d32 * sa))
            Mz_row = (rv.Dpw / 2.0) * rv.cp * float(np.sum(d32 * sa * cp_j))
            row_results.append(BallLoadDistributionResult(
                delta_r=delta_r, delta_a=delta_a, psi=psi, phi_Fr=phi_Fr,
                delta_j=delta_j, alpha_j=alpha_j,
                Mz=Mz_row, n_iter=nfev, residual=res, ok=ok,
            ))
            row_reactions.append((Fr_row, Fa_row))
        return row_results, row_reactions, nfev, res, ok


# =====================================================================
# ---- ISO16281RollerSolver -- line contact -----------------------------
# =====================================================================

@register_contact_solver(capability="line_contact",
                          required_attrs=("Z", "Dwe", "Lwe", "Dpw", "phi_j", "s", "n_s",
                                          "x_k", "cL", "cs", "alpha_0", "P_xk"))
class ISO16281RollerSolver(LineContactSolverBase):
    """Line contact (radial cylindrical roller bearings, NU/N-type, zero
    nominal contact angle). Sec 5.2 lamina model. 1-equation root (delta_r)
    in the resultant-force plane, psi prescribed -- eq.(46) evaluated
    afterwards as a diagnostic, not a solve constraint."""

    _result_cls = RollerBearingResult

    def _solve_one_bearing(self, bearing: Bearing, node, phi_Fr: float, psi: float
                            ) -> RollerLoadDistributionResult:
        return self.solve_contact(
            bearing,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy,
            delta_r_init=float(np.hypot(node.v_xz, node.v_xy)),
            psi=psi, phi_Fr=phi_Fr,
        )

    @staticmethod
    def elements(bearing: Bearing, delta_r: float, psi: float):
        """Per-roller and per-lamina deflection/force -- eq.(36)-(41)."""
        phi_j = bearing.phi_j
        x_k   = bearing.x_k
        s     = bearing.s

        cp_j    = np.cos(phi_j)
        delta_j = delta_r * cp_j - s / 2.0                          # eq.(38)
        psi_j   = np.arctan(np.tan(psi) * cp_j)                     # eq.(39)

        P_xk = bearing.P_xk   # eq.(42)-(44)

        delta_jk = (delta_j[:, None]
                    - x_k[None, :] * np.tan(psi_j)[:, None]
                    - 2.0 * P_xk[None, :])                          # eq.(41)
        delta_jk = np.maximum(delta_jk, 0.0)

        q_jk = bearing.cs * delta_jk ** (10.0 / 9.0)                # eq.(36)

        return delta_j, psi_j, delta_jk, q_jk, cp_j

    def _initial_delta_r(self, bearing: Bearing, Fr: float, hint: float) -> float:
        if hint > 1e-6:
            return hint
        gap = bearing.s / 2.0
        line_hertz = (abs(Fr) / (bearing.cL * bearing.Z)) ** (9.0 / 10.0) if Fr != 0.0 else 1e-4
        return gap + line_hertz

    def solve_contact(self,
                      bearing: Bearing,
                      Fr_xz: float, Fr_xy: float,
                      delta_r_init: float,
                      psi: float, phi_Fr: float) -> RollerLoadDistributionResult:
        """Equilibrium, Sec 5.2.4.1, eq.(45): Fr = sum_j cos(phi_j) * sum_k q_j,k"""
        Fr = float(np.hypot(Fr_xz, Fr_xy))

        def residual(u):
            dr = u[0]
            _, _, _, q_jk, cp_j = self.elements(bearing, dr, psi)
            Fr_internal = float(np.sum(cp_j * np.sum(q_jk, axis=1)))
            return np.array([Fr - Fr_internal])

        dr0              = self._initial_delta_r(bearing, Fr, delta_r_init)
        x, nfev, res, ok = run_root(residual, [dr0], self.tol)
        delta_r          = float(x[0])

        delta_j, psi_j, delta_jk, q_jk, cp_j = self.elements(bearing, delta_r, psi)
        x_k = bearing.x_k

        Mz = float(np.sum(cp_j * np.sum(x_k[None, :] * q_jk, axis=1)))  # eq.(46), diagnostic
        alpha_j = np.full(bearing.Z, bearing.alpha_0, dtype=float)

        return RollerLoadDistributionResult(
            delta_r=delta_r, delta_a=0.0, psi=psi, phi_Fr=phi_Fr,
            delta_j=delta_j, alpha_j=alpha_j,
            Mz=Mz, n_iter=nfev, residual=res, ok=ok,
            x_k=x_k, psi_j=psi_j, delta_jk=delta_jk, q_jk=q_jk,
        )


@register_contact_solver(capability="line_contact", required_attrs=("rows",))
class ISO16281MultiRowRollerSolverSharedDisplacement(MultiRowSolverBase):
    """Multi-row roller, shared delta_r across all rows -- radial roller
    carries no axial load, so unlike the ball case there is only ONE
    shared unknown (no delta_a/Fa at all).

    UNVALIDATED: no real family produces bearing.rows for a roller
    bearing today (a MultiRowCylindricalRollerFamily was tried and
    retired). Kept for structural parity with the ball side, at your
    request -- do not treat this as production-ready without a real
    multi-row roller family to validate it against, same as before this
    alignment pass."""

    _result_cls        = RollerBearingResult
    SINGLE_ROW_SOLVER  = ISO16281RollerSolver

    def _check_row_ready(self, row, row_label: str) -> None:
        min_laminae = self.SINGLE_ROW_SOLVER.MIN_LAMINAE
        if row.n_s < min_laminae:
            raise ValueError(
                f"Bearing '{row_label}': n_s = {row.n_s} < {min_laminae} -- "
                f"ISO/TS 16281 Sec 5.2.2 requires at least {min_laminae} laminae."
            )
        if len(row.x_k) != row.n_s:
            raise ValueError(
                f"Bearing '{row_label}': len(x_k) = {len(row.x_k)} != n_s = {row.n_s}."
            )

    def _initial_delta_r(self, row_views, Fr: float) -> float:
        s0         = row_views[0].s
        cLZ_total  = sum(rv.cL * rv.Z for rv in row_views)
        gap        = s0 / 2.0
        line_hertz = (abs(Fr) / cLZ_total) ** (9.0/10.0) if Fr != 0.0 else 1e-4
        return gap + line_hertz

    def _solve_rows(self, row_views, Fr_xz, Fr_xy, Fa, psi, phi_Fr):
        Fr       = float(np.hypot(Fr_xz, Fr_xy))
        elements = self.SINGLE_ROW_SOLVER.elements

        def residual(u):
            dr = u[0]
            res_r = Fr
            for rv in row_views:
                _, _, _, q_jk, cp_j = elements(rv, dr, psi)
                res_r -= float(np.sum(cp_j * np.sum(q_jk, axis=1)))
            return np.array([res_r])

        dr0 = self._initial_delta_r(row_views, Fr)
        x, nfev, res, ok = run_root(residual, [dr0], self.tol)
        delta_r = float(x[0])

        row_results, row_reactions = [], []
        for rv in row_views:
            delta_j, psi_j, delta_jk, q_jk, cp_j = elements(rv, delta_r, psi)
            Fr_row  = float(np.sum(cp_j * np.sum(q_jk, axis=1)))
            Mz_row  = float(np.sum(cp_j * np.sum(rv.x_k[None, :] * q_jk, axis=1)))
            alpha_j = np.full(rv.Z, rv.alpha_0, dtype=float)
            row_results.append(RollerLoadDistributionResult(
                delta_r=delta_r, delta_a=0.0, psi=psi, phi_Fr=phi_Fr,
                delta_j=delta_j, alpha_j=alpha_j,
                Mz=Mz_row, n_iter=nfev, residual=res, ok=ok,
                x_k=rv.x_k, psi_j=psi_j, delta_jk=delta_jk, q_jk=q_jk,
            ))
            row_reactions.append((Fr_row, 0.0))
        return row_results, row_reactions, nfev, res, ok


# Ligação single-row -> multi-row (mesma convenção que já tinhas).
ISO16281BallSolver.MULTIROW_SOLVER   = ISO16281MultiRowBallSolverSharedDisplacement
ISO16281RollerSolver.MULTIROW_SOLVER = ISO16281MultiRowRollerSolverSharedDisplacement