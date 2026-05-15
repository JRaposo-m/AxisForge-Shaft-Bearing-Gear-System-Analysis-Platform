# solvers/statics.py
"""
StaticsSolver — static analysis of a two-support shaft.

Formulation:
  Euler-Bernoulli beam, two simple supports (bearings A and B).
  Analysis performed independently in XZ and YZ planes.
  Results combined vectorially: M_res = sqrt(M_xz² + M_yz²).

Sign convention (documented explicitly):
  Applied loads:  positive = downward in YZ, forward in XZ (per external load convention).
  Reactions:      sign is determined by equilibrium equations; not forced positive.
  Torsion:        accumulates left-to-right; positive = counter-clockwise from +x.

Restrictions:
  - Exactly 2 bearings (Phase 1). Statically determinate.
  - No GUI imports. No database calls. No global state.
  - All methods are pure functions of their arguments.
"""
from __future__ import annotations
import warnings
import numpy as np

from core.system import MechanicalSystem
from core.loads import RadialLoad, AxialLoad, TorqueLoad, ExternalMoment, LoadPlane
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
        radial_xz = self._collect_radial(system, LoadPlane.XZ)
        radial_yz = self._collect_radial(system, LoadPlane.YZ)
        axial_loads = list(system.axial_loads)
        torque_loads = list(system.torque_loads)

        # ── Decompose GearElements into equivalent point loads ──────────────
        for gear in system.gears:
            # Wt → XZ plane (tangential, in-plane horizontal)
            radial_xz.append((gear.position, gear.tangential_force))
            # Wr → YZ plane (radial, in-plane vertical)
            radial_yz.append((gear.position, gear.radial_force))
            if gear.axial_force != 0.0:
                axial_loads.append(AxialLoad(position=gear.position, magnitude=gear.axial_force))
            if gear.torque != 0.0:
                torque_loads.append(TorqueLoad(position=gear.position, magnitude=gear.torque))

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
        R_A_xz, R_B_xz = self._equilibrium(radial_xz, xA, xB)
        R_A_yz, R_B_yz = self._equilibrium(radial_yz, xA, xB)
        R_axial = -sum(a.magnitude for a in axial_loads)

        # ── Assemble point-force lists (reactions + applied) ────────────────
        all_xz: list[tuple[float, float]] = [(xA, R_A_xz), (xB, R_B_xz)] + radial_xz
        all_yz: list[tuple[float, float]] = [(xA, R_A_yz), (xB, R_B_yz)] + radial_yz

        # ── Discretise ──────────────────────────────────────────────────────
        x = np.linspace(0.0, system.shaft.total_length, SOLVER_RESOLUTION)

        V_xz = self._shear_diagram(x, all_xz)
        V_yz = self._shear_diagram(x, all_yz)
        M_xz = self._moment_diagram(x, all_xz)
        M_yz = self._moment_diagram(x, all_yz)
        M_res = np.sqrt(M_xz ** 2 + M_yz ** 2)
        T = self._torsion_diagram(x, torque_loads)
        Fa = self._axial_diagram(x, axial_loads, R_axial, xA)

        # ── Internal consistency check ───────────────────────────────────────
        self._validate_result(M_xz, M_yz, xA, xB, x)

        reactions = {
            "A_xz": R_A_xz,
            "A_yz": R_A_yz,
            "B_xz": R_B_xz,
            "B_yz": R_B_yz,
            "axial": R_axial,
        }

        return StaticsResult(
            x=x,
            V_xz=V_xz,
            V_yz=V_yz,
            M_xz=M_xz,
            M_yz=M_yz,
            M_res=M_res,
            T=T,
            axial_force=Fa,
            reactions=reactions,
        )

    # ── Private methods ───────────────────────────────────────────────────────

    def _collect_radial(
        self, system: MechanicalSystem, plane: LoadPlane
    ) -> list[tuple[float, float]]:
        """Extract (position, magnitude) pairs for RadialLoad in a given plane."""
        return [
            (load.position, load.magnitude)
            for load in system.loads
            if isinstance(load, RadialLoad) and load.plane == plane
        ]

    def _equilibrium(
        self,
        loads: list[tuple[float, float]],
        xA: float,
        xB: float,
    ) -> tuple[float, float]:
        """
        Solve static equilibrium for two pin supports at xA and xB.

        Args:
            loads: list of (position, magnitude) — applied external loads only,
                   NOT including reactions.
            xA, xB: support positions [mm].

        Returns:
            (R_A, R_B): reactions at A and B. Signs determined by equilibrium.

        Method:
            ΣM_A = 0 → R_B = -Σ(F_i × (x_i - xA)) / (xB - xA)
            ΣF   = 0 → R_A = -ΣF_i - R_B
        """
        span = xB - xA
        moment_about_A = sum(F * (x - xA) for x, F in loads)
        R_B = -moment_about_A / span
        R_A = -sum(F for _, F in loads) - R_B
        return R_A, R_B

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
        M_yz: np.ndarray,
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

        for plane_name, M in [("XZ", M_xz), ("YZ", M_yz)]:
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
