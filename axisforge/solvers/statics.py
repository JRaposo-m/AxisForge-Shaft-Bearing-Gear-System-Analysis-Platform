# solvers/statics.py
"""
StaticsSolver — static analysis of a two-support shaft.

Formulation:
  Euler-Bernoulli beam, two simple supports (bearings A and B).
  Analysis performed independently in XZ and XY planes.
  Results combined vectorially: M_res = sqrt(M_xz² + M_xy²).

Sign convention (documented explicitly):
  Applied loads:  positive = downward in XY, forward in XZ (per external load convention).
  Reactions:      sign is determined by equilibrium equations; not forced positive.
  Torsion:        accumulates left-to-right; positive = counter-clockwise from +x.

ExternalMoment sign convention:
  A pure couple M₀ at position pos contributes no net force (ΣF unchanged).
  In ΣM_A: enters as −M₀ → R_B = +M₀/L, R_A = −M₀/L (verified: M(xB)=0 ✓).
  In moment diagram: step of +M₀ for x ≥ pos (no moment arm).
  Derivation: M(x<pos) = R_A×x; M(x≥pos) = R_A×x + M₀; M(L) = −M₀ + M₀ = 0.

Restrictions:
  - Exactly 2 bearings (Phase 1). Statically determinate.
  - No GUI imports. No database calls. No global state.
  - All methods are pure functions of their arguments.
"""
from __future__ import annotations
import warnings
import numpy as np

from core.system import MechanicalSystem
from core.loads import AxialLoad, TorqueLoad, ExternalMoment, LoadPlane
from models.statics_result import StaticsResult
from config import SOLVER_RESOLUTION, BOUNDARY_MOMENT_TOLERANCE


class StaticsSolver:
    """
    Computes bearing reactions and internal force/moment diagrams for a
    two-support shaft.

    Usage:
        solver = StaticsSolver()
        result = solver.solve(system)

    The solver is stateless — all data flows through arguments and return values.
    """

    def solve(self, system: MechanicalSystem) -> StaticsResult:
        """
        Perform full static analysis.

        Args:
            system: validated MechanicalSystem with exactly 2 bearings.

        Returns:
            StaticsResult with discretised diagrams and reactions dict.

        Raises:
            ValueError: if system is invalid or bearing span is zero.
        """
        system.validate_or_raise()

        if len(system.bearings) != 2:
            raise ValueError(
                f"Phase 1 StaticsSolver requires exactly 2 bearings, "
                f"got {len(system.bearings)}"
            )

        xA = system.bearings[0].position
        xB = system.bearings[1].position
        span = xB - xA

        if span <= 0:
            raise ValueError(
                f"Bearing span is {span:.3f} mm — bearings must be at distinct positions. "
                f"xA={xA} mm, xB={xB} mm"
            )

        # ── Collect loads by plane ──────────────────────────────────────────
        # Use system properties directly — avoids duplicating RadialLoad filtering logic.
        radial_xz: list[tuple[float, float]] = [
            (l.position, l.magnitude) for l in system.radial_loads_xz
        ]
        radial_xy: list[tuple[float, float]] = [
            (l.position, l.magnitude) for l in system.radial_loads_xy
        ]
        axial_loads = list(system.axial_loads)
        torque_loads = list(system.torque_loads)
        ext_moments_xz = [m for m in system.external_moments if m.plane == LoadPlane.XZ]
        ext_moments_xy = [m for m in system.external_moments if m.plane == LoadPlane.XY]

        # ── Decompose GearElements into equivalent point loads ──────────────
        for gear in system.gears:
            # Wt → XZ plane (tangential, in-plane horizontal)
            radial_xz.append((gear.position, gear.tangential_force))
            # Wr → XY plane (radial, in-plane vertical)
            radial_xy.append((gear.position, gear.radial_force))
            if gear.axial_force != 0.0:
                axial_loads.append(AxialLoad(position=gear.position, magnitude=gear.axial_force))
            if gear.torque != 0.0:
                torque_loads.append(TorqueLoad(position=gear.position, magnitude=gear.torque))

        # ── Warn for loads applied outside bearing span (overhang) ──────────
        all_positions = (
            [p for p, _ in radial_xz]
            + [p for p, _ in radial_xy]
            + [a.position for a in axial_loads]
            + [t.position for t in torque_loads]
            + [m.position for m in ext_moments_xz]
            + [m.position for m in ext_moments_xy]
        )
        overhang = sorted({p for p in all_positions if p < xA or p > xB})
        if overhang:
            warnings.warn(
                f"Load(s) applied outside bearing span [{xA:.1f}, {xB:.1f}] mm "
                f"at x = {overhang}. Overhang loads cause M ≠ 0 at supports — "
                f"_validate_result will raise RuntimeError. Phase 1 does not support overhang.",
                stacklevel=2,
            )

        # ── Warn if torque is unbalanced ────────────────────────────────────
        total_torque = sum(t.magnitude for t in torque_loads)
        if abs(total_torque) > 1.0:
            warnings.warn(
                f"Total torque in system is {total_torque:.1f} N·mm (non-zero). "
                f"This is acceptable if an external reaction torque exists (e.g. motor/brake), "
                f"but may indicate a missing torque source.",
                stacklevel=2,
            )

        # ── Solve equilibrium in each plane ─────────────────────────────────
        R_A_xz, R_B_xz = self._equilibrium(radial_xz, xA, xB, ext_moments_xz)
        R_A_xy, R_B_xy = self._equilibrium(radial_xy, xA, xB, ext_moments_xy)
        R_axial = -sum(a.magnitude for a in axial_loads)

        # ── Assemble point-force lists (reactions + applied) ────────────────
        all_xz: list[tuple[float, float]] = [(xA, R_A_xz), (xB, R_B_xz)] + radial_xz
        all_xy: list[tuple[float, float]] = [(xA, R_A_xy), (xB, R_B_xy)] + radial_xy

        # ── Discretise ──────────────────────────────────────────────────────
        x = np.linspace(0.0, system.shaft.total_length, SOLVER_RESOLUTION)

        V_xz = self._shear_diagram(x, all_xz)
        V_xy = self._shear_diagram(x, all_xy)
        M_xz = self._moment_diagram(x, all_xz) + self._moment_external_diagram(x, ext_moments_xz)
        M_xy = self._moment_diagram(x, all_xy) + self._moment_external_diagram(x, ext_moments_xy)
        M_res = np.sqrt(M_xz ** 2 + M_xy ** 2)
        T = self._torsion_diagram(x, torque_loads)
        Fa = self._axial_diagram(x, axial_loads, R_axial, xA)

        # ── Internal consistency check ───────────────────────────────────────
        self._validate_result(M_xz, M_xy, xA, xB, x)

        reactions = {
            "A_xz": R_A_xz,
            "A_xy": R_A_xy,
            "B_xz": R_B_xz,
            "B_xy": R_B_xy,
            "axial": R_axial,
        }

        return StaticsResult(
            x=x,
            V_xz=V_xz,
            V_xy=V_xy,
            M_xz=M_xz,
            M_xy=M_xy,
            M_res=M_res,
            T=T,
            axial_force=Fa,
            reactions=reactions,
        )

    # ── Private methods ───────────────────────────────────────────────────────

    def _equilibrium(
        self,
        loads: list[tuple[float, float]],
        xA: float,
        xB: float,
        external_moments: list[ExternalMoment] | None = None,
    ) -> tuple[float, float]:
        """
        Solve static equilibrium for two pin supports at xA and xB.

        Args:
            loads: list of (position, magnitude) — applied radial loads only,
                   NOT including reactions.
            xA, xB: support positions [mm].
            external_moments: list of ExternalMoment loads (pure couples, no net force).

        Returns:
            (R_A, R_B): reactions at A and B. Signs determined by equilibrium.

        Method:
            ΣM_A = 0 → R_B×L + Σ(F_i×(xi−xA)) − Σ(M₀_j) = 0
            Note: external moments enter with NEGATIVE sign in ΣM_A.
            Derivation: for CCW couple M₀, correct equilibrium gives R_B = M₀/L,
            requiring the term (−M₀) in the moment sum.

            ΣF = 0 → R_A = −ΣF_i − R_B   (couples don't contribute to ΣF)
        """
        if external_moments is None:
            external_moments = []
        span = xB - xA
        moment_about_A = sum(F * (x - xA) for x, F in loads)
        moment_about_A -= sum(m.magnitude for m in external_moments)
        R_B = -moment_about_A / span
        R_A = -sum(F for _, F in loads) - R_B
        return R_A, R_B

    def _moment_external_diagram(
        self,
        x: np.ndarray,
        external_moments: list[ExternalMoment],
    ) -> np.ndarray:
        """
        M(x) contribution from external moment couples.

        A pure couple M₀ at position pos causes a step jump of +M₀ in the
        bending moment diagram at x = pos. There is no moment arm — the
        couple acts instantaneously. This is added to the point-force
        moment diagram from _moment_diagram.

        Convention: x >= pos (inclusive), consistent with _shear_diagram.
        """
        M = np.zeros_like(x)
        for m in external_moments:
            M += np.where(x >= m.position, m.magnitude, 0.0)
        return M

    def _shear_diagram(
        self,
        x: np.ndarray,
        point_forces: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        V(x): shear force at each x.

        For each point force at position pos, its contribution is added to
        all x >= pos. This uses the standard step-function superposition.

        Convention: x >= pos (inclusive) ensures loads exactly at a support
        position are captured in the diagram from that point onward.
        """
        V = np.zeros_like(x)
        for pos, F in point_forces:
            V += np.where(x >= pos, F, 0.0)
        return V

    def _moment_diagram(
        self,
        x: np.ndarray,
        point_forces: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        M(x): bending moment at each x.

        For a point force F at position pos:
          Contribution to M(x) = F × (x - pos)  for x >= pos, else 0.

        This is the moment arm formula; superposition gives the full diagram.
        """
        M = np.zeros_like(x)
        for pos, F in point_forces:
            M += np.where(x >= pos, F * (x - pos), 0.0)
        return M

    def _torsion_diagram(
        self,
        x: np.ndarray,
        torques: list[TorqueLoad],
    ) -> np.ndarray:
        """
        T(x): torsion accumulated left to right.

        Each TorqueLoad steps T by its magnitude at its axial position.
        """
        T = np.zeros_like(x)
        for t in torques:
            T += np.where(x >= t.position, t.magnitude, 0.0)
        return T

    def _axial_diagram(
        self,
        x: np.ndarray,
        axial_loads: list[AxialLoad],
        reaction_axial: float,
        xA: float,
    ) -> np.ndarray:
        """
        Fa(x): axial force distribution.

        The axial reaction is applied at xA (fixed support carries all axial load).
        External axial loads step Fa at their positions.
        """
        Fa = np.zeros_like(x)
        Fa += np.where(x >= xA, reaction_axial, 0.0)
        for a in axial_loads:
            Fa += np.where(x >= a.position, a.magnitude, 0.0)
        return Fa

    def _validate_result(
        self,
        M_xz: np.ndarray,
        M_xy: np.ndarray,
        xA: float,
        xB: float,
        x: np.ndarray,
    ) -> None:
        """
        Post-solution sanity checks.

        Verifies simply-supported boundary conditions:
          |M(xA)| < BOUNDARY_MOMENT_TOLERANCE
          |M(xB)| < BOUNDARY_MOMENT_TOLERANCE

        The residual is a discretisation artefact proportional to 1/N.
        With N=1000 and typical loads, residuals should be < 10 N·mm.
        BOUNDARY_MOMENT_TOLERANCE = 500 N·mm is intentionally generous.
        """
        tol = BOUNDARY_MOMENT_TOLERANCE

        idx_A = int(np.argmin(np.abs(x - xA)))
        idx_B = int(np.argmin(np.abs(x - xB)))

        for plane_name, M in [("XZ", M_xz), ("XY", M_xy)]:
            M_at_A = float(M[idx_A])
            M_at_B = float(M[idx_B])
            if abs(M_at_A) > tol:
                raise RuntimeError(
                    f"Boundary condition violated: M_{plane_name}(xA={xA:.1f}) = "
                    f"{M_at_A:.2f} N·mm, expected |M| < {tol} N·mm. "
                    f"Check equilibrium calculation."
                )
            if abs(M_at_B) > tol:
                raise RuntimeError(
                    f"Boundary condition violated: M_{plane_name}(xB={xB:.1f}) = "
                    f"{M_at_B:.2f} N·mm, expected |M| < {tol} N·mm. "
                    f"Check equilibrium calculation."
                )