"""
axisforge/solvers/machine_elements/bearings/load_distribution/single_row/iso_16281/contact_solver.py

Single-row ISO/TS 16281 contact solvers -- abstract contract on top,
concrete point-contact (ball) and line-contact (roller) solvers below.
Everything that RUNS the iterative solve lives here; result shapes live
in results/bearings/load_distribution/load_distribution_results.py.

Result shape: every solver produces a *BearingResult (frozen) holding the
shared-displacement state ONCE (delta_r, delta_a, psi, phi_Fr, applied
Fr_xz/Fr_xy/Fa, n_iter/residual/ok) plus one *LoadDistributionResult per
row with only the per-row data (phi_j, delta_j, alpha_j, Q_j, Fr_row,
Fa_row, Mz [+ lamina data for roller]). Single-row = 1 row.

solve() returns dict[label, BearingAnalysisResult]:
    .load_distribution  the *BearingResult (always)
    .basic_life         BasicReferenceRatingLifeResult (postprocess=True only)

Postprocessing (contact_postprocessing.py) is imported here on purpose --
this was a one-directional dependency (contact_solver.py never imported
contact_postprocessing.py) until the `postprocess` flag below was added.
The rule was deliberately relaxed: a solver constructed with
postprocess=True now runs contact_postprocessing.py's functions itself --
stiffness is attached to the *BearingResult (with_stiffness()), L10r/Pref
go into BearingAnalysisResult.basic_life -- so an external orchestrator
only ever has to call solve() once to get everything.
contact_postprocessing.py itself still never imports this module -- only
the *Result shapes it produces -- so that half of the one-directional
rule stands.

Multi-row (shared-displacement) solvers now live here too, immediately
after their single-row sibling, following the same abstract-on-top /
concrete-below pattern (MultiRowSolverBase after LineContactSolverBase,
then one concrete multi-row class per contact type next to its
single-row counterpart). They are NOT reviewed for bearing-to-bearing
coupling -- the shared-displacement formulation assumes co-located rows
(zero axial offset) and superposes each row's own reaction; it is a
step beyond what ISO/TS 16281 itself states (the standard gives
per-raceway equations, not row combination). That physics is explicitly
NOT touched by this pass. Postprocessing is likewise NOT wired up for
multi-row solvers -- see MultiRowSolverBase._extra_ready_checks().
NOTE (to review): the earlier claim "valid as-is only for thrust
bearings" is questionable -- thrust bearings carry no radial load, yet
the ball multi-row solve includes Fr.

NOTE (to review): the roller solvers are RADIAL only (Fa, delta_a, Fa_row
= 0.0 by definition). Thrust roller is the mirror case (Fa only, no Fr)
and is not covered here.

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
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281 import (
    contact_postprocessing as pp,
)
from axisforge.results.bearings.load_distribution.load_distribution_results import (
    BearingResult,
    BallLoadDistributionResult, BallBearingResult,
    RollerLoadDistributionResult, RollerBearingResult,
)
from axisforge.results.bearings.life.basic_life_results import BasicReferenceRatingLifeResult
from axisforge.results.bearings.bearing_analysis_result import BearingAnalysisResult
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
    dispatch.py. _result_cls (the *BearingResult class) and
    _solve_one_bearing() are set/implemented directly by each concrete
    subclass. _wrap_result() is a hook so that a solver whose
    _solve_one_bearing() doesn't return a complete *BearingResult (e.g. a
    multi-row solver, which returns a row set + shared state) can still go
    through this same loop without solve() knowing about it.

    postprocess : when True, each concrete single-row solver's
    _attach_postprocessing() attaches bearing-level stiffness to the
    *BearingResult and computes L10r/Pref into BearingAnalysisResult.basic_life.
    False by default -- basic_life is then None.
    """

    CAPABILITY: str = ""
    REQUIRED_ATTRS: tuple[str, ...] = ()
    MULTIROW_SOLVER: type | None = None
    _result_cls: type | None = None

    def __init__(self, tol: float = SOLVER_TOLERANCE, psi_input: bool = False,
                 postprocess: bool = False):
        self.tol         = tol
        self.psi_input   = psi_input
        self.postprocess = postprocess

    def solve(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              shaft_results: ShaftResults,
              psi_override: dict[str, float] | None = None,
              ) -> dict[str, BearingAnalysisResult]:
        """
        Solve the internal load distribution for every bearing in
        `bearings` this solver applies to.

        Returns one BearingAnalysisResult per label. Its .load_distribution
        is a *BearingResult wrapping 1 row (single-row solver) or N rows
        (MultiRowSolverBase subclass) -- the batch loop itself (grouping,
        psi, readiness, warnings) is identical either way, only
        _wrap_result() differs. .basic_life is set only with postprocess=True.
        """
        if self._result_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set _result_cls to its "
                f"*BearingResult class."
            )

        if self.psi_input and psi_override:
            unknown = set(psi_override) - set(bearings)
            if unknown:
                warnings.warn(
                    f"psi_override has labels not in `bearings`: {sorted(unknown)} -- "
                    f"check for typos; these entries are ignored."
                )

        node_by_label = {n.label: n for n in shaft_results.bearing_nodes}

        results: dict[str, BearingAnalysisResult] = {}
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

            result = self._wrap_result(self._solve_one_bearing(b, node, phi_Fr, psi))

            if self.postprocess:
                results[label] = self._attach_postprocessing(b, result)
            else:
                results[label] = BearingAnalysisResult(label=result.label,
                                                       load_distribution=result)

        return results

    def _extra_ready_checks(self, bearing: Bearing, label: str) -> None:
        """Hook for readiness checks beyond REQUIRED_ATTRS. No-op by
        default; overridden by LineContactSolverBase and MultiRowSolverBase."""
        return None

    def _wrap_result(self, row):
        """Hook: turns whatever _solve_one_bearing() returns into this
        solver's *BearingResult. Default = single-row: solve_contact()
        already returns a complete *BearingResult, so nothing to do.
        MultiRowSolverBase overrides this -- its _solve_one_bearing()
        returns a row set + shared state instead."""
        return row

    def _attach_postprocessing(self, bearing: Bearing, result: BearingResult
                               ) -> BearingAnalysisResult:
        """Overridden by each concrete single-row solver. Multi-row refuses
        postprocess=True before reaching here (see MultiRowSolverBase)."""
        raise NotImplementedError(
            f"{type(self).__name__}: postprocessing not implemented."
        )

    @abstractmethod
    def _solve_one_bearing(self, bearing: Bearing, node, phi_Fr: float, psi: float):
        """Solve ONE bearing given its FEM node data, phi_Fr and psi.
        Must call self.solve_contact(...) with whatever arguments this
        solver's own physics needs and return the resulting *BearingResult
        (or, for a multi-row solver, whatever _wrap_result() on that
        solver expects)."""
        ...

    @staticmethod
    def _Cr_Ca(bearing: Bearing) -> tuple[float | None, float | None]:
        """(Cr, Ca) for Pref, dispatched off bearing.duty -- bearing.C is
        the single catalog dynamic rating field, serving as Cr for a
        radial-duty family or Ca for a thrust-duty family (see
        BearingCatalog/Bearing -- there is no separate Cr/Ca field)."""
        if bearing.duty == "radial":
            return bearing.C, None
        if bearing.duty == "thrust":
            return None, bearing.C
        raise NotImplementedError(
            f"Bearing '{bearing.label}': duty={bearing.duty!r} not radial/thrust -- "
            f"Cr/Ca dispatch for Pref is undefined."
        )


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
    bearing-to-bearing coupling term. Reviewing/extending this to real
    radial multi-row behavior is a separate, later task.

    Output: ONE *BearingResult with N rows. Shared state (delta_r,
    delta_a, psi, phi_Fr, applied load, n_iter/residual/ok) is stored once
    on the bearing; each row carries its own reactions Fr_row/Fa_row, from
    which the result derives f_r/f_a.

    postprocess=True is refused here (see _extra_ready_checks()) -- the
    Cr/Ca-per-row and L10r-row-combination semantics for postprocess were
    never designed for the shared-displacement multi-row case, unlike
    single-row where they are unambiguous.

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
        if self.postprocess:
            raise NotImplementedError(
                f"Bearing '{label}': postprocess=True is not supported on "
                f"multi-row solvers ({type(self).__name__}) -- row-level "
                f"postprocessing linkage (Cr/Ca per row, L10r row combination) "
                f"has not been designed/reviewed for shared-displacement "
                f"multi-row bearings. Use postprocess=False and call "
                f"contact_postprocessing.py's functions directly if needed."
            )
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
        row_results, shared = row
        return self._result_cls.multirow(row_results, **shared)

    def _solve_one_bearing(self, bearing: Bearing, node, phi_Fr: float, psi: float):
        row_views = self._row_views(bearing)
        row_results, delta_r, delta_a, Fa, n_iter, residual_norm, ok = self._solve_rows(
            row_views,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy, Fa=getattr(node, "Fa", 0.0),
            psi=psi, phi_Fr=phi_Fr,
        )
        shared = dict(
            label=bearing.label,
            delta_r=delta_r, delta_a=delta_a, psi=psi, phi_Fr=phi_Fr,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy, Fa=Fa,
            n_iter=n_iter, residual=residual_norm, ok=ok,
        )
        return row_results, shared

    @abstractmethod
    def _solve_rows(self, row_views, Fr_xz: float, Fr_xy: float, Fa: float,
                     psi: float, phi_Fr: float):
        """Resolve the shared-displacement system over all rows.

        Returns (row_results, delta_r, delta_a, Fa, n_iter, residual_norm, ok):
        row_results is a list of *LoadDistributionResult, one per row (each
        with its own Fr_row/Fa_row reactions); delta_r/delta_a are the shared
        solution; Fa is the axial load the bearing actually carries (0.0 for
        radial roller rows, whatever the node says). Not forced into a
        common equation count on purpose -- ball solves 2 unknowns
        (delta_r, delta_a), roller solves 1 (delta_r only)."""
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
                            ) -> BallBearingResult:
        return self.solve_contact(
            bearing,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy, Fa=node.Fa,
            delta_r_init=float(np.hypot(node.v_xz, node.v_xy)),
            delta_a_init=node.u,
            psi=psi, phi_Fr=phi_Fr,
        )

    @staticmethod
    def _attach_postprocessing(bearing: Bearing, result: BallBearingResult
                               ) -> BearingAnalysisResult:
        """Runs everything contact_postprocessing.py offers for a ball
        bearing -- stiffness (attached to the *BearingResult via
        with_stiffness(), results are frozen) and L10r/Pref (into
        BearingAnalysisResult.basic_life). inner_rotating/outer_rotating
        are left at DynamicEquivalentRollingElementLoad.from_distribution()'s
        own defaults (True/False) -- Bearing has no such attribute to read
        from, this mirrors the ISO/TS 16281 "usual case" (inner ring
        rotating) assumption baked into that default."""
        stiffness = pp.bearing_stiffness(bearing, result,
                                         Fr_xz=result.Fr_xz, Fr_xy=result.Fr_xy, Fa=result.Fa)
        result = result.with_stiffness(stiffness)

        Q_ci, Q_ce = bearing.family.per_element_dynamic_capacity(bearing)
        equiv = pp.DynamicEquivalentRollingElementLoad.from_distribution(
            bearing, result.row, label=bearing.label,
        )
        life = pp.BallBasicReferenceRatingLife.from_loads(
            label=bearing.label, Q_ci=Q_ci, Q_ei=equiv.Q_ei, Q_ce=Q_ce, Q_ee=equiv.Q_ee,
        )

        Cr, Ca = SolverBase._Cr_Ca(bearing)
        pref = pp.BallDynamicEquivalentReferenceLoad.from_L10r(
            label=bearing.label, L10r=life.L10r, Cr=Cr, Ca=Ca,
        )

        basic_life = BasicReferenceRatingLifeResult(
            label=bearing.label, rows=(life,), L10r=life.L10r, pref=pref,
        )
        return BearingAnalysisResult(label=result.label, load_distribution=result,
                                     basic_life=basic_life)

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
                      psi: float, phi_Fr: float) -> BallBearingResult:
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

        delta_j, alpha_j, ca, sa, cp_j, d32 = self.elements(bearing, delta_r, delta_a, Vpsi)
        Mz = (bearing.Dpw / 2.0) * cp * float(np.sum(d32 * sa * cp_j))

        row = BallLoadDistributionResult(
            phi_j=np.asarray(bearing.phi_j, dtype=float),
            delta_j=delta_j, alpha_j=alpha_j,
            Q_j=cp * d32,
            Fr_row=float(cp * np.sum(d32 * ca * cp_j)),
            Fa_row=float(cp * np.sum(d32 * sa)),
            Mz=Mz,
        )
        return BallBearingResult.single(
            row, label=bearing.label,
            delta_r=delta_r, delta_a=delta_a, psi=psi, phi_Fr=phi_Fr,
            Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa,
            n_iter=nfev, residual=res, ok=ok,
        )

    def minimum_axial_load(self,
                            bearing: Bearing,
                            Fr_xz: float, Fr_xy: float,
                            psi: float,
                            delta_r_init: float = 0.0,
                            delta_a_init: float = 0.0,
                            Fa_bracket: tuple[float, float] = (0.0, 5.0e4),
                            xtol: float = 1e-6) -> tuple[float, BallBearingResult]:
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

        row_results = []
        for rv in row_views:
            Vpsi_rv = rv.Ri * np.sin(psi) * np.cos(rv.phi_j)
            delta_j, alpha_j, ca, sa, cp_j, d32 = elements(rv, delta_r, delta_a, Vpsi_rv)
            row_results.append(BallLoadDistributionResult(
                phi_j=np.asarray(rv.phi_j, dtype=float),
                delta_j=delta_j, alpha_j=alpha_j,
                Q_j=rv.cp * d32,
                Fr_row=float(rv.cp * np.sum(d32 * ca * cp_j)),
                Fa_row=float(rv.cp * np.sum(d32 * sa)),
                Mz=(rv.Dpw / 2.0) * rv.cp * float(np.sum(d32 * sa * cp_j)),
            ))
        return row_results, delta_r, delta_a, Fa, nfev, res, ok


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
    afterwards as a diagnostic, not a solve constraint. No axial capacity:
    Fa, delta_a and Fa_row are 0.0 in the result by definition."""

    _result_cls = RollerBearingResult

    def _solve_one_bearing(self, bearing: Bearing, node, phi_Fr: float, psi: float
                            ) -> RollerBearingResult:
        return self.solve_contact(
            bearing,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy,
            delta_r_init=float(np.hypot(node.v_xz, node.v_xy)),
            psi=psi, phi_Fr=phi_Fr,
        )

    @staticmethod
    def _attach_postprocessing(bearing: Bearing, result: RollerBearingResult
                               ) -> BearingAnalysisResult:
        """Runs everything contact_postprocessing.py offers for a roller
        bearing -- stiffness (attached via with_stiffness()) and L10r/Pref
        (into BearingAnalysisResult.basic_life). Fa is always 0.0 -- radial
        roller bearings carry no axial load. inner_rotating/outer_rotating
        left at LaminaDynamicEquivalentLoad.from_distribution()'s own
        defaults (True/False), same reasoning as the ball solver."""
        stiffness = pp.bearing_stiffness(bearing, result,
                                         Fr_xz=result.Fr_xz, Fr_xy=result.Fr_xy, Fa=0.0)
        result = result.with_stiffness(stiffness)

        q_kci, q_kce = bearing.family.per_lamina_dynamic_capacity(bearing)
        equiv = pp.LaminaDynamicEquivalentLoad.from_distribution(
            bearing, result.row, label=bearing.label,
        )
        q_kci_arr = np.full_like(equiv.q_kei, q_kci)
        q_kce_arr = np.full_like(equiv.q_kee, q_kce)
        life = pp.RollerBasicReferenceRatingLife.from_loads(
            label=bearing.label, q_kci=q_kci_arr, q_kei=equiv.q_kei,
            q_kce=q_kce_arr, q_kee=equiv.q_kee,
        )

        Cr, Ca = SolverBase._Cr_Ca(bearing)
        pref = pp.RollerDynamicEquivalentReferenceLoad.from_L10r(
            label=bearing.label, L10r=life.L10r, Cr=Cr, Ca=Ca,
        )

        basic_life = BasicReferenceRatingLifeResult(
            label=bearing.label, rows=(life,), L10r=life.L10r, pref=pref,
        )
        return BearingAnalysisResult(label=result.label, load_distribution=result,
                                     basic_life=basic_life)

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
                      psi: float, phi_Fr: float) -> RollerBearingResult:
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
        x_k = np.asarray(bearing.x_k, dtype=float)
        Q_j = np.sum(q_jk, axis=1)

        Mz = float(np.sum(cp_j * np.sum(x_k[None, :] * q_jk, axis=1)))  # eq.(46), diagnostic
        alpha_j = np.full(bearing.Z, bearing.alpha_0, dtype=float)

        row = RollerLoadDistributionResult(
            phi_j=np.asarray(bearing.phi_j, dtype=float),
            delta_j=delta_j, alpha_j=alpha_j,
            Q_j=Q_j,
            Fr_row=float(np.sum(cp_j * Q_j)),
            Fa_row=0.0,
            Mz=Mz,
            x_k=x_k, psi_j=psi_j, delta_jk=delta_jk, q_jk=q_jk,
        )
        return RollerBearingResult.single(
            row, label=bearing.label,
            delta_r=delta_r, delta_a=0.0, psi=psi, phi_Fr=phi_Fr,
            Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=0.0,
            n_iter=nfev, residual=res, ok=ok,
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
        # Fa ignored on purpose -- radial roller, no axial capacity (returned as 0.0).
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

        row_results = []
        for rv in row_views:
            delta_j, psi_j, delta_jk, q_jk, cp_j = elements(rv, delta_r, psi)
            x_k = np.asarray(rv.x_k, dtype=float)
            Q_j = np.sum(q_jk, axis=1)
            row_results.append(RollerLoadDistributionResult(
                phi_j=np.asarray(rv.phi_j, dtype=float),
                delta_j=delta_j,
                alpha_j=np.full(rv.Z, rv.alpha_0, dtype=float),
                Q_j=Q_j,
                Fr_row=float(np.sum(cp_j * Q_j)),
                Fa_row=0.0,
                Mz=float(np.sum(cp_j * np.sum(x_k[None, :] * q_jk, axis=1))),
                x_k=x_k, psi_j=psi_j, delta_jk=delta_jk, q_jk=q_jk,
            ))
        return row_results, delta_r, 0.0, 0.0, nfev, res, ok


# Ligação single-row -> multi-row (mesma convenção que já tinhas).
ISO16281BallSolver.MULTIROW_SOLVER   = ISO16281MultiRowBallSolverSharedDisplacement
ISO16281RollerSolver.MULTIROW_SOLVER = ISO16281MultiRowRollerSolverSharedDisplacement