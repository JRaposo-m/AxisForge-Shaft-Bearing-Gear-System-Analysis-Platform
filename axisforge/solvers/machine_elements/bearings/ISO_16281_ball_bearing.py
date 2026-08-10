"""
axisforge/solvers/machine_elements/bearings/ISO_16281_ball_bearing.py

ISO/TS 16281 internal load distribution — coupled shaft-bearing solver
(prescribed-psi formulation, solved with scipy.optimize.root).

Architecture (ISO/TS 16281 §4.2)
--------------------------------
The SimpleFEMResultsLibrary supplies all FEM quantities needed per bearing
via ShaftResults.bearing_nodes (BearingNodeData):
    Fr_xz, Fr_xy, Fa    — bearing reactions [N]
    psi_xz, psi_xy      — shaft centreline slope across seat [rad]
    u, v_xz, v_xy       — initial displacement estimates [mm]

A per-bearing root solve then resolves the internal load distribution
(delta_r, delta_a) that equilibrates Fr = sqrt(Fr_xz² + Fr_xy²).

The solve is performed in the plane of the resultant force:
    phi_Fr = arctan2(Fr_xy, Fr_xz)   [rad, global frame]

phi_j is kept local in the solver (bearing.phi_j, starting at 0) and
rotated to the global frame only for output:
    phi_j_global = bearing.phi_j + phi_Fr

This ensures solver and plot share the same reference frame with no
post-processing conversion.

Element kinematics — ISO/TS 16281 eq. (12)/(15)
-----------------------------------------------
    U_j     = A*cos(alpha_0) + delta_r*cos(phi_j)
    V_j     = A*sin(alpha_0) + delta_a + Ri*sin(psi)*cos(phi_j)
    delta_j = sqrt(U_j^2 + V_j^2) - A        (set to 0 if negative)
    alpha_j = arctan2(V_j, U_j)

Static equilibrium — ISO/TS 16281 §4.2.2.1
------------------------------------------
    R0 = Fr  - cp*sum(delta_j^1.5 * cos(alpha_j) * cos(phi_j))
    R1 = Fa  - cp*sum(delta_j^1.5 * sin(alpha_j))

Solver
------
scipy.optimize.root (MINPACK 'hybr', with 'lm' fallback).
2-eq root (delta_r, delta_a) for Fr_res and Fa.

Why psi is prescribed
---------------------
psi (ring tilt) is PRESCRIBED from the shaft centreline slope across the
seat, projected onto the plane of the resultant force:
    psi = psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr)

psi_xz / psi_xy are pre-computed by ShaftResultsReader and stored in
BearingNodeData — this solver never accesses FEM arrays directly.

References
----------
  ISO/TS 16281:2008 §4.2, eq. (12)-(15)
  Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch.6
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import root, brentq

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    SimpleFEMResultsLibrary,
    BearingNodeData,
)
from axisforge.config import NR_TOL, NR_MAX_ITER


# ===========================================================================
# LoadDistributionResult
# ===========================================================================

class LoadDistributionResult:
    """
    Output of the internal load distribution solve for a single bearing —
    single solve in the plane of the resultant force.

    Attributes
    ----------
    delta_r : float           — radial ring displacement [mm]
    delta_a : float           — axial ring displacement [mm]
    psi     : float           — prescribed misalignment in resultant plane [rad]
    phi_Fr  : float           — angle of resultant radial force, global frame [rad]
                                used only for plot — solve is always in local frame
                                (phi_j=0 aligned with Fr)
    delta_j : np.ndarray(Z,)  — elastic deflection per element [mm]
    alpha_j : np.ndarray(Z,)  — effective contact angle per element [rad]
    Mz      : float           — moment reaction (OUTPUT) [N.mm]
    n_iter  : int             — solver function evaluations
    residual: float           — final ||R|| [N]
    ok      : bool            — solver success flag

    Note
    ----
    Contact-force per element: Q_j = cp * max(delta_j, 0)**1.5  [N]
    Ball positions in global frame (plot only):
        phi_j_global = (bearing.phi_j + phi_Fr) % (2*pi)
    """

    __slots__ = (
        "delta_r", "delta_a", "psi", "phi_Fr",
        "delta_j", "alpha_j",
        "Mz", "n_iter", "residual", "ok",
    )

    def __init__(self,
                 delta_r, delta_a, psi, phi_Fr,
                 delta_j, alpha_j,
                 Mz, n_iter, residual, ok):

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


# ===========================================================================
# BearingStiffnessState
# ===========================================================================

class BearingStiffnessState:
    """
    Secant stiffness of a single bearing, decomposed onto the two orthogonal
    shaft planes (XZ, XY) plus the axial direction.

    Built from a converged ISO/TS 16281 load distribution (LoadDistributionResult),
    which solves in the plane of the resultant force. This class projects that
    resultant solution back onto the global XZ / XY axes, giving a per-plane
    secant stiffness ready to characterise the bearing as an elastic support.

    Unlike LoadDistributionResult (the raw ISO solve — resultant-plane, per
    rolling element), this is a compact per-plane summary: one radial stiffness
    per plane plus the axial stiffness and its engagement regime.

    Stiffness definitions
    ---------------------
    delta_r_xz = delta_r * cos(phi_Fr)
    delta_r_xy = delta_r * sin(phi_Fr)

    Kr_xz = Fr_xz / delta_r_xz
    Kr_xy = Fr_xy / delta_r_xy
    Ka    = Fa    / delta_a       (locating bearing, "engaged" regime only)

    A stiffness is set to float('inf') when the corresponding displacement is
    ~0 (rigid in that direction — no compliance to report).

    Ka_regime
    ---------
    "no_load"           : Fa == 0  → axial direction carries no load
    "engaged"           : delta_a >= 0 under axial load → axial contact active
    "closing_clearance" : delta_a < 0 → axial play still closing, not yet engaged

    Attributes
    ----------
    label : str — bearing label, matches Bearing.label

    -- XZ plane --
    Fr_xz      : float — radial reaction, XZ [N]
    delta_r_xz : float — radial ring displacement, XZ [mm]
    Kr_xz      : float — secant radial stiffness, XZ [N/mm]

    -- XY plane --
    Fr_xy      : float — radial reaction, XY [N]
    delta_r_xy : float — radial ring displacement, XY [mm]
    Kr_xy      : float — secant radial stiffness, XY [N/mm]

    -- axial (locating bearing only; None for floating bearings) --
    Fa         : float | None — axial reaction [N]
    delta_a    : float | None — axial ring displacement [mm]
    Ka         : float | None — secant axial stiffness [N/mm]
    Ka_regime  : str  | None  — "no_load" | "closing_clearance" | "engaged"
    """

    __slots__ = (
        "label",
        "Fr_xz", "delta_r_xz", "Kr_xz",
        "Fr_xy", "delta_r_xy", "Kr_xy",
        "Fa", "delta_a", "Ka", "Ka_regime",
    )

    def __init__(self,
                 label: str,
                 Fr_xz: float, delta_r_xz: float, Kr_xz: float,
                 Fr_xy: float, delta_r_xy: float, Kr_xy: float,
                 Fa: float | None = None,
                 delta_a: float | None = None,
                 Ka: float | None = None,
                 Ka_regime: str | None = None):

        self.label      = label
        self.Fr_xz      = Fr_xz
        self.delta_r_xz = delta_r_xz
        self.Kr_xz      = Kr_xz
        self.Fr_xy      = Fr_xy
        self.delta_r_xy = delta_r_xy
        self.Kr_xy      = Kr_xy
        self.Fa         = Fa
        self.delta_a    = delta_a
        self.Ka         = Ka
        self.Ka_regime  = Ka_regime

    @classmethod
    def from_load_distribution(cls,
                                label: str,
                                result: "LoadDistributionResult",
                                Fr_xz: float,
                                Fr_xy: float,
                                Fa: float,
                                eps: float = 1e-9) -> "BearingStiffnessState":
        """
        Build a BearingStiffnessState from a resultant-plane ISO solve,
        decomposing delta_r / Kr back onto the global XZ / XY axes.

        Parameters
        ----------
        label  : bearing label
        result : LoadDistributionResult — converged ISO/TS 16281 solve
        Fr_xz  : radial reaction, XZ plane [N]
        Fr_xy  : radial reaction, XY plane [N]
        Fa     : axial reaction [N]
        eps    : displacement threshold below which stiffness → inf
        """
        delta_r_xz = result.delta_r * np.cos(result.phi_Fr)
        delta_r_xy = result.delta_r * np.sin(result.phi_Fr)
        delta_a    = result.delta_a

        Kr_xz = (Fr_xz / delta_r_xz) if abs(delta_r_xz) > eps else float("inf")
        Kr_xy = (Fr_xy / delta_r_xy) if abs(delta_r_xy) > eps else float("inf")

        if Fa == 0.0:
            Ka, regime = float("inf"), "no_load"
        elif abs(delta_a) > eps:
            Ka = Fa / delta_a
            regime = "engaged" if delta_a >= 0.0 else "closing_clearance"
        else:
            Ka, regime = float("inf"), "engaged"

        return cls(
            label=label,
            Fr_xz=Fr_xz, delta_r_xz=delta_r_xz, Kr_xz=Kr_xz,
            Fr_xy=Fr_xy, delta_r_xy=delta_r_xy, Kr_xy=Kr_xy,
            Fa=Fa, delta_a=delta_a, Ka=Ka, Ka_regime=regime,
        )


# ===========================================================================
# IterativeBearingFEMSolver
# ===========================================================================

class IterativeBearingFEMSolver:
    """
    Coupled shaft-bearing solver — ISO/TS 16281 (prescribed-psi, root-based).

    Pipeline
    --------
    1. SimpleFEMResultsLibrary (pre-populated by ShaftResultsReader.read())
       provides per-bearing nodal data via ShaftResults.bearing_nodes:
         Fr_xz, Fr_xy, Fa     — reactions [N]
         psi_xz, psi_xy       — misalignment [rad]
         u, v_xz, v_xy        — displacement estimates [mm]

    2. Per bearing: single 2-eq root (delta_r, delta_a) in the plane of Fr
       phi_Fr = arctan2(Fr_xy, Fr_xz)
       Fr     = sqrt(Fr_xz² + Fr_xy²)
       psi    = psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr)
       phi_j_global = phi_j + phi_Fr  (output only)

    This solver never accesses SimpleFEMSolver arrays directly — all FEM
    data is consumed from the library via BearingNodeData.

    Parameters
    ----------
    tol      : residual tolerance passed to root() [-]
    max_iter : ignored by root(); kept for interface compatibility
    """

    def __init__(self,
                 tol: float = NR_TOL,
                 max_iter: int = NR_MAX_ITER):
        self.tol      = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              library: SimpleFEMResultsLibrary,
              Pd: float = 0.0) -> dict[str, LoadDistributionResult]:
        """
        Parameters
        ----------
        shaft_system : ShaftSystem — fully resolved
        bearings     : {label: Bearing}
                       setup_internal_geometry() and compute_hertz_point_contact()
                       must have been called on each Bearing before this call.
        library      : SimpleFEMResultsLibrary — must contain results for
                       shaft_system.name (populated by ShaftResultsReader.read()).
        Pd           : diametral clearance override [mm] (0 = use bearing.s)

        Returns
        -------
        {label: LoadDistributionResult}
        """
        shaft_results = library.get(shaft_system.name)

        # index bearing_nodes by label for O(1) access
        node_by_label: dict[str, BearingNodeData] = {
            n.label: n for n in shaft_results.bearing_nodes
        }

        results: dict[str, LoadDistributionResult] = {}
        for label, b in bearings.items():
            node = node_by_label[label]
            results[label] = self._solve_bearing_internal(
                b,
                Fr_xz        = node.Fr_xz,
                Fr_xy        = node.Fr_xy,
                Fa           = node.Fa,
                delta_r_init = float(np.hypot(node.v_xz, node.v_xy)),
                delta_a_init = node.u,
                psi_xz       = node.psi_xz,
                psi_xy       = node.psi_xy,
            )

        self.bearing_data = {
            label: {
                'Fr_xz': node_by_label[label].Fr_xz,
                'Fr_xy': node_by_label[label].Fr_xy,
                'Fa':    node_by_label[label].Fa,
            }
            for label in bearings
        }

        return results

    # ------------------------------------------------------------------
    # Element kinematics (ISO eq. 12/15, §4.2.2.1)
    # ------------------------------------------------------------------

    @staticmethod
    def _elements(bearing, delta_r, delta_a, Vpsi):
        """
        Per-element deflection and contact angle for given ring displacements.
        phi_j is bearing.phi_j (local, starting at 0 = direction of Fr).

        Returns delta_j, alpha_j, ca, sa, cp_j, d32, d12, mask.
        """
        A       = bearing.A
        alpha_0 = bearing.alpha_0
        phi_j   = bearing.phi_j
        cp_j    = np.cos(phi_j)

        U_j     = A * np.cos(alpha_0) + delta_r * cp_j
        V_j     = A * np.sin(alpha_0) + delta_a + Vpsi
        Sigma_j = np.sqrt(U_j**2 + V_j**2)
        delta_j = np.maximum(Sigma_j - A, 0.0)
        mask    = delta_j > 0.0

        alpha_j = np.where(mask, np.arctan2(V_j, U_j), 0.0)
        ca      = np.cos(alpha_j)
        sa      = np.sin(alpha_j)
        d32     = np.where(mask, delta_j ** 1.5, 0.0)
        d12     = np.where(mask, delta_j ** 0.5, 0.0)
        return delta_j, alpha_j, ca, sa, cp_j, d32, d12, mask

    def _radial_init(self, bearing, Fr, delta_r_init):
        """Initial delta_r estimate — clears clearance gap and adds Hertz estimate."""
        A       = bearing.A
        alpha_0 = bearing.alpha_0
        cp      = bearing.cp
        Z       = bearing.Z

        gap   = A * (1.0 - np.cos(alpha_0))
        hertz = (abs(Fr) / (cp * Z)) ** (2.0 / 3.0) if Fr != 0.0 else 1e-4
        seed  = gap + hertz

        if abs(delta_r_init) < 1e-6:
            return seed
        return delta_r_init if abs(delta_r_init) > seed else seed

    # ------------------------------------------------------------------
    # Internal load distribution — single solve in resultant plane
    # ------------------------------------------------------------------

    def _solve_bearing_internal(self,
                                bearing: Bearing,
                                Fr_xz: float,
                                Fr_xy: float,
                                Fa: float,
                                delta_r_init: float,
                                delta_a_init: float,
                                psi_xz: float,
                                psi_xy: float) -> LoadDistributionResult:
        """
        Single 2-eq root (delta_r, delta_a) in the plane of the resultant Fr.
        """
        Fr     = float(np.sqrt(Fr_xz**2 + Fr_xy**2))
        phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

        psi  = psi_xz * np.cos(phi_Fr) + psi_xy * np.sin(phi_Fr)
        cp   = bearing.cp
        Dpw  = bearing.Dpw
        Ri   = bearing.Ri
        cp_j = np.cos(bearing.phi_j)
        Vpsi = Ri * np.sin(psi) * cp_j

        def residual(u):
            dr, da = u
            _, _, ca, sa, _, d32, _, _ = self._elements(bearing, dr, da, Vpsi)
            return np.array([
                Fr - cp * np.sum(d32 * ca * cp_j),
                Fa - cp * np.sum(d32 * sa),
            ])

        dr0 = self._radial_init(bearing, Fr, delta_r_init)
        x, nfev, res, ok = self._run_root(residual, [dr0, delta_a_init], 2)
        delta_r, delta_a = float(x[0]), float(x[1])

        delta_j, alpha_j, ca, sa, _, d32, _, _ = self._elements(
            bearing, delta_r, delta_a, Vpsi)
        Mz = (Dpw / 2.0) * cp * float(np.sum(d32 * sa * cp_j))

        return LoadDistributionResult(
            delta_r=delta_r, delta_a=delta_a, psi=psi, phi_Fr=phi_Fr,
            delta_j=delta_j, alpha_j=alpha_j,
            Mz=Mz, n_iter=nfev, residual=res, ok=ok,
        )

    # ------------------------------------------------------------------
    # Post-processing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def Q_j(bearing: Bearing, result: LoadDistributionResult) -> np.ndarray:
        """Contact force at each rolling element [N]. Q_j = cp * max(delta_j,0)^1.5"""
        return bearing.cp * np.maximum(result.delta_j, 0.0) ** 1.5

    @staticmethod
    def phi_j_global(bearing: Bearing, result: LoadDistributionResult) -> np.ndarray:
        """Ball positions in global frame [rad]. phi_j_global = (phi_j + phi_Fr) % 2pi"""
        return (bearing.phi_j + result.phi_Fr) % (2 * np.pi)

    @staticmethod
    def contact_distribution(bearing: Bearing,
                              result: LoadDistributionResult,
                              frame: str = "global") -> np.ndarray:
        """
        Per-element contact force paired with angular position.

        Returns np.ndarray shape (Z, 2): column 0 = phi_j [rad], column 1 = Q_j [N].

        Parameters
        ----------
        frame : "global" (phi_j + phi_Fr, [0,2pi)) | "local" (phi_j=0 aligned with Fr)
        """
        if frame == "global":
            phi = (bearing.phi_j + result.phi_Fr) % (2 * np.pi)
        elif frame == "local":
            phi = bearing.phi_j
        else:
            raise ValueError(f"frame must be 'global' or 'local', got '{frame}'")

        Q = bearing.cp * np.maximum(result.delta_j, 0.0) ** 1.5
        return np.column_stack((phi, Q))

    @staticmethod
    def bearing_stiffness(bearing: Bearing,
                           result: LoadDistributionResult,
                           Fr_xz: float,
                           Fr_xy: float,
                           Fa: float) -> tuple[float, float, float]:
        """
        Secant bearing stiffness (Kr_xz, Kr_xy, Ka) from converged result.

        Returns (Kr_xz, Kr_xy, Ka) [N/mm], each possibly float('inf').
        See module docstring for Ka guard rationale.
        """
        delta_r_xz = result.delta_r * np.cos(result.phi_Fr)
        delta_r_xy = result.delta_r * np.sin(result.phi_Fr)
        delta_a    = result.delta_a
        eps        = 1e-9

        Kr_xz = (Fr_xz / delta_r_xz) if abs(delta_r_xz) > eps else float("inf")
        Kr_xy = (Fr_xy / delta_r_xy) if abs(delta_r_xy) > eps else float("inf")
        Ka    = (Fa / delta_a)        if (Fa != 0.0 and abs(delta_a) > eps) else float("inf")

        return Kr_xz, Kr_xy, Ka

    # ------------------------------------------------------------------
    # Helper — minimum axial preload
    # ------------------------------------------------------------------

    def minimum_axial_load(self,
                            bearing: Bearing,
                            Fr_xz: float,
                            Fr_xy: float,
                            psi_xz: float,
                            psi_xy: float,
                            delta_r_init: float = 0.0,
                            delta_a_init: float = 0.0,
                            Fa_bracket: tuple[float, float] = (0.0, 5.0e4),
                            xtol: float = 1e-6) -> tuple[float, LoadDistributionResult]:
        """
        Minimum axial preload Fa_min [N] such that delta_a >= 0.

        Finds the smallest Fa >= 0 that cancels free-contact-angle settling
        (delta_a ~= -A*sin(alpha_0) under Fa=0). Uses brentq on delta_a(Fa).

        Raises ValueError if Fa_bracket[1] is too small to bracket the root.
        """
        res0 = self._solve_bearing_internal(
            bearing, Fr_xz, Fr_xy, 0.0, delta_r_init, delta_a_init, psi_xz, psi_xy)
        if res0.delta_a >= 0.0:
            return 0.0, res0

        def g(Fa: float) -> float:
            return self._solve_bearing_internal(
                bearing, Fr_xz, Fr_xy, Fa, delta_r_init, delta_a_init,
                psi_xz, psi_xy).delta_a

        lo, hi = Fa_bracket
        g_hi = g(hi)
        if g_hi < 0.0:
            raise ValueError(
                f"minimum_axial_load: delta_a still negative ({g_hi:.4e} mm) "
                f"at Fa={hi:.1f} N — widen Fa_bracket."
            )

        Fa_min = brentq(g, lo, hi, xtol=xtol)
        result = self._solve_bearing_internal(
            bearing, Fr_xz, Fr_xy, Fa_min, delta_r_init, delta_a_init, psi_xz, psi_xy)
        return Fa_min, result

    # ------------------------------------------------------------------
    # Root solver
    # ------------------------------------------------------------------

    def _run_root(self, fun, x0, n_out):
        """hybr with lm fallback. Returns (x, nfev, residual_norm, success)."""
        sol = root(fun, x0, method="hybr", tol=self.tol)
        if not sol.success:
            sol_lm = root(fun, x0, method="lm", tol=self.tol)
            if float(np.linalg.norm(np.atleast_1d(sol_lm.fun))) < \
               float(np.linalg.norm(np.atleast_1d(sol.fun))):
                sol = sol_lm

        x    = np.atleast_1d(sol.x)
        res  = float(np.linalg.norm(np.atleast_1d(sol.fun)))
        nfev = int(getattr(sol, "nfev", 0))
        return x, nfev, res, bool(sol.success)