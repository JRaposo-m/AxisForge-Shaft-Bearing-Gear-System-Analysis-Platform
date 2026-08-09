"""
axisforge/solvers/machine_elements/bearings/ISO_16281_ball_bearing.py

ISO/TS 16281 internal load distribution — coupled shaft-bearing solver
(prescribed-psi formulation, solved with scipy.optimize.root).

Architecture (ISO/TS 16281 §4.2)
--------------------------------
The FEM supplies the bearing reactions (Fr_xz, Fr_xy, Fa) and the shaft
centreline slope across the bearing seat; a per-bearing root solve then
resolves the internal load distribution (delta_r, delta_a) that equilibrates
the radial resultant Fr = sqrt(Fr_xz² + Fr_xy²).

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
from axisforge.mesh.oneD.shaft.Elements.elem import Elem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
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
# CoupledBearingState
# ===========================================================================

class CoupledBearingState:
    """
    Per-bearing state carried across outer coupling (Picard / secant-
    stiffness) iterations of a shaft-bearing FEM loop.

    Unlike LoadDistributionResult (a single ISO solve, resultant-plane,
    stateless), this tracks quantities PER PLANE — because the FEM injects
    stiffness plane-by-plane (Kr_xz into the XZ system, Kr_xy into the XY
    system, Ka only at the locating bearing's axial DOF). It also carries
    the regime classification needed to decide how each plane's stiffness
    should be treated as a boundary condition — that decision itself is
    NOT made here; this class only records the facts an orchestrator acts on.

    Attributes
    ----------
    label : str — bearing label, matches Bearing.label / ShaftSystem lookup

    -- XZ plane --
    Fr_xz      : float — radial reaction, XZ [N]
    delta_r_xz : float — radial ring displacement, XZ [mm]
    Kr_xz      : float — secant radial stiffness, XZ [N/mm]
                 (Fr_xz/delta_r_xz; NOT expected to be negative under normal
                 operation — a negative value here is a solver/convergence
                 flag, not a valid physical regime, and should be surfaced
                 rather than silently substituted)

    -- XY plane --
    Fr_xy      : float — radial reaction, XY [N]
    delta_r_xy : float — radial ring displacement, XY [mm]
    Kr_xy      : float — secant radial stiffness, XY [N/mm]

    -- axial (locating bearing only; None for floating bearings) --
    Fa         : float | None — axial reaction [N]
    delta_a    : float | None — axial ring displacement [mm]
    Ka         : float | None — secant axial stiffness [N/mm]
    Ka_regime  : str  | None  — one of:
                 "no_load"          : Fa == 0, Ka undefined -> report as inf
                 "closing_clearance": Fa != 0 but delta_a < 0 (ring still
                                      taking up the free-contact-angle
                                      clearance; Ka is a legitimate negative
                                      secant value here, not an error, but
                                      NOT a stiffness the FEM should inject
                                      as-is — the orchestrator decides the BC)
                 "engaged"          : Fa != 0 and delta_a > 0, normal
                                      load-deflection response

    -- iteration bookkeeping --
    iteration  : int  — outer coupling iteration index this state belongs to
    converged  : bool — whether this bearing's Fr/Fa/Kr/Ka changed by less
                 than the caller's tolerance since the previous iteration
                 (set by the orchestrator, not computed here)

    Note
    ----
    This class holds data only. Any policy decision — e.g. "treat
    closing_clearance as rigid (Kr_xz -> inf) until Fa exceeds Fa_min", or
    "floating bearings have Fa=Ka=None and are skipped for axial injection"
    — belongs in the Newton-Raphson / Picard orchestrator that consumes
    CoupledBearingState, not in this class or in bearing_stiffness().
    """

    __slots__ = (
        "label",
        "Fr_xz", "delta_r_xz", "Kr_xz",
        "Fr_xy", "delta_r_xy", "Kr_xy",
        "Fa", "delta_a", "Ka", "Ka_regime",
        "iteration", "converged",
    )

    def __init__(self,
                 label: str,
                 Fr_xz: float, delta_r_xz: float, Kr_xz: float,
                 Fr_xy: float, delta_r_xy: float, Kr_xy: float,
                 Fa: float | None = None,
                 delta_a: float | None = None,
                 Ka: float | None = None,
                 Ka_regime: str | None = None,
                 iteration: int = 0,
                 converged: bool = False):

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
        self.iteration  = iteration
        self.converged  = converged

    @classmethod
    def from_load_distribution(cls,
                                label: str,
                                result: "LoadDistributionResult",
                                Fr_xz: float,
                                Fr_xy: float,
                                Fa: float,
                                iteration: int = 0,
                                converged: bool = False,
                                eps: float = 1e-9) -> "CoupledBearingState":
        """
        Build a CoupledBearingState from a single-plane (resultant) ISO solve,
        decomposing delta_r/Kr back onto the global XZ/XY axes.

        Ka_regime classification:
            Fa == 0                -> "no_load",  Ka = inf
            Fa != 0 and delta_a<0  -> "closing_clearance", Ka = Fa/delta_a (<0)
            Fa != 0 and delta_a>=0 -> "engaged",  Ka = Fa/delta_a

        Parameters
        ----------
        label      : bearing label
        result     : LoadDistributionResult from the same solve as Fr_xz/Fr_xy/Fa
        Fr_xz, Fr_xy, Fa : bearing reactions [N] from that same solve
        iteration  : outer coupling iteration index (caller-supplied)
        converged  : convergence flag for this bearing at this iteration
                     (caller-supplied — this method does not compare against
                     a previous state)
        eps        : mm, below which a displacement component is "zero"
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
            iteration=iteration, converged=converged,
        )


# ===========================================================================
# IterativeBearingFEMSolver
# ===========================================================================

class IterativeBearingFEMSolver:
    """
    Coupled shaft-bearing solver — ISO/TS 16281 (prescribed-psi, root-based).

    Pipeline
    --------
    1. SimpleFEMSolver with rigid bearings
       -> Fr_xz, Fr_xy, Fa per bearing
       -> shaft centreline slope across each seat -> psi_xz, psi_xy
    2. Per bearing: single 2-eq root (delta_r, delta_a) in the plane of Fr
       phi_Fr = arctan2(Fr_xy, Fr_xz)
       Fr     = sqrt(Fr_xz² + Fr_xy²)
       psi    = psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr)
       phi_j_global = phi_j + phi_Fr  (output only, not used in solve)

    Single FEM pass; no bearing stiffness injected into the global FEM.

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
              Pd: float = 0.0,
              fem: SimpleFEMSolver | None = None) -> dict[str, LoadDistributionResult]:
        """
        Parameters
        ----------
        shaft_system : ShaftSystem — fully resolved
        bearings     : {label: Bearing}
                       setup_internal_geometry() and compute_hertz_point_contact()
                       must have been called on each Bearing before this call.
        Pd           : diametral clearance override [mm] (0 = use bearing.s)
        fem          : pre-built SimpleFEMSolver (optional); built internally if None

        Returns
        -------
        {label: LoadDistributionResult}
        """
        if fem is None:
            fem = self._iter0_rigid(shaft_system)
        bearing_data = self._extract_bearing_forces(fem, shaft_system)

        results: dict[str, LoadDistributionResult] = {}
        for label, b in bearings.items():
            data = bearing_data[label]
            psi_xz, psi_xy = self._psi_from_displacement_gradient(fem, shaft_system, b)
            results[label] = self._solve_bearing_internal(
                b,
                Fr_xz        = data['Fr_xz'],
                Fr_xy        = data['Fr_xy'],
                Fa           = data['Fa'],
                delta_r_init = data['v'],
                delta_a_init = data['u'],
                psi_xz       = psi_xz,
                psi_xy       = psi_xy,
            )

        self.fem_converged = fem
        self.bearing_data  = bearing_data

        return results

    # ------------------------------------------------------------------
    # Rigid bearing FEM
    # ------------------------------------------------------------------

    def _iter0_rigid(self, shaft_system: ShaftSystem) -> SimpleFEMSolver:
        """Rigid bearing FEM. Returns solved SimpleFEMSolver."""
        fem = SimpleFEMSolver()
        fem.solve(shaft_system)
        return fem

    # ------------------------------------------------------------------
    # Misalignment from displacement gradient across bearing seat width
    # ------------------------------------------------------------------

    def _psi_from_displacement_gradient(self,
                                         fem: SimpleFEMSolver,
                                         shaft_system: ShaftSystem,
                                         bearing: Bearing) -> tuple[float, float]:
        """
        Average slope of the shaft centreline across the bearing seat:
            psi = (v(x_hi) - v(x_lo)) / (x_hi - x_lo)
        Falls back to the nodal section rotation theta if seat width is 0.

        Returns
        -------
        (psi_xz, psi_xy) [rad]
        """
        x_lo, x_hi = shaft_system.bearing_extent(bearing)
        span = x_hi - x_lo

        if span <= 0.0:
            i = Elem.find_node_index(fem.x_nodes, bearing.position)
            return (float(fem.d_total_xz[3 * i + 2]),
                    float(fem.d_total_xy[3 * i + 2]))

        i_lo = Elem.find_node_index(fem.x_nodes, x_lo)
        i_hi = Elem.find_node_index(fem.x_nodes, x_hi)

        psi_xz = float((fem.d_total_xz[3 * i_hi + 1] - fem.d_total_xz[3 * i_lo + 1]) / span)
        psi_xy = float((fem.d_total_xy[3 * i_hi + 1] - fem.d_total_xy[3 * i_lo + 1]) / span)
        return psi_xz, psi_xy

    # ------------------------------------------------------------------
    # Extract bearing forces and displacements
    # ------------------------------------------------------------------

    def _extract_bearing_forces(self,
                                 fem: SimpleFEMSolver,
                                 shaft_system: ShaftSystem) -> dict[str, dict]:
        """
        Returns {label: {Fr_xz, Fr_xy, Fr, phi_Fr, Fa,
                          M_xz, M_xy, M, phi_M,
                          v_xz, v_xy, v, phi_v, u,
                          theta_xz, theta_xy, theta, phi_theta}}
        """
        result = {}
        for b in shaft_system.bearings:
            i = Elem.find_node_index(fem.x_nodes, b.position)

            Fr_xz    = float(fem.f_xz_total[3 * i + 1])
            Fr_xy    = float(fem.f_xy_total[3 * i + 1])
            Fa       = float(fem.f_xz_total[3 * i])
            M_xz     = float(fem.f_xz_total[3 * i + 2])
            M_xy     = float(fem.f_xy_total[3 * i + 2])
            v_xz     = float(fem.d_total_xz[3 * i + 1])
            v_xy     = float(fem.d_total_xy[3 * i + 1])
            u        = float(fem.d_total_xz[3 * i])
            theta_xz = float(fem.d_total_xz[3 * i + 2])
            theta_xy = float(fem.d_total_xy[3 * i + 2])

            Fr    = np.sqrt(Fr_xz**2 + Fr_xy**2)
            M     = np.sqrt(M_xz**2  + M_xy**2)
            v     = np.sqrt(v_xz**2  + v_xy**2)
            theta = np.sqrt(theta_xz**2 + theta_xy**2)

            result[b.label] = {
                'Fr_xz'    : Fr_xz,  'Fr_xy'   : Fr_xy,
                'Fr'       : Fr,     'phi_Fr'  : float(np.arctan2(Fr_xy, Fr_xz)),
                'Fa'       : Fa,
                'M_xz'     : M_xz,   'M_xy'    : M_xy,
                'M'        : M,      'phi_M'   : float(np.arctan2(M_xy, M_xz)),
                'v_xz'     : v_xz,   'v_xy'    : v_xy,
                'v'        : v,      'phi_v'   : float(np.arctan2(v_xy, v_xz)),
                'u'        : u,
                'theta_xz' : theta_xz, 'theta_xy' : theta_xy,
                'theta'    : theta,    'phi_theta': float(np.arctan2(theta_xy, theta_xz)),
            }

        return result

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
        """
        Initial delta_r estimate — clears clearance gap and adds Hertz estimate.
        """
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

        phi_Fr = arctan2(Fr_xy, Fr_xz)  — global angle of resultant force
        Fr     = sqrt(Fr_xz² + Fr_xy²)  — resultant radial force magnitude
        psi    = psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr)  — misalignment
                 projected onto the resultant plane
        phi_j_global = phi_j + phi_Fr   — ball positions in global frame (output)
        """
        Fr     = float(np.sqrt(Fr_xz**2 + Fr_xy**2))
        phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

        # misalignment projected onto resultant plane
        psi = psi_xz * np.cos(phi_Fr) + psi_xy * np.sin(phi_Fr)

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
        """
        Contact force at each rolling element [N].
        Q_j = cp * max(delta_j, 0)^1.5

        Parameters
        ----------
        bearing : Bearing — must have cp set (compute_hertz_point_contact called)
        result  : LoadDistributionResult

        Returns
        -------
        np.ndarray(Z,) [N]
        """
        return bearing.cp * np.maximum(result.delta_j, 0.0) ** 1.5

    @staticmethod
    def phi_j_global(bearing: Bearing, result: LoadDistributionResult) -> np.ndarray:
        """
        Ball positions in the global frame [rad], for plotting only.
        phi_j_global = (phi_j + phi_Fr) % (2*pi)

        phi_j   : local positions (phi_j=0 aligned with Fr, from bearing.phi_j)
        phi_Fr  : global angle of resultant force (from result.phi_Fr)

        Parameters
        ----------
        bearing : Bearing
        result  : LoadDistributionResult

        Returns
        -------
        np.ndarray(Z,) [rad], values in [0, 2*pi)
        """
        return (bearing.phi_j + result.phi_Fr) % (2 * np.pi)

    @staticmethod
    def contact_distribution(bearing: Bearing,
                              result: LoadDistributionResult,
                              frame: str = "global") -> np.ndarray:
        """
        Per-element contact force paired with angular position, in one call.

        Q_j   = cp * max(delta_j, 0)^1.5                     [N]
        phi_j = bearing.phi_j (local)  or  phi_j + phi_Fr (global)

        Parameters
        ----------
        bearing : Bearing — must have cp set (compute_hertz_point_contact called)
        result  : LoadDistributionResult
        frame   : "global" (default) — phi_j + phi_Fr, wrapped to [0, 2*pi)
                  "local"            — bearing.phi_j as-is (phi_j=0 aligned with Fr)

        Returns
        -------
        np.ndarray, shape (Z, 2), dtype float — column 0 = phi_j [rad],
        column 1 = Q_j [N]. Row j is rolling element j (same order/index as
        bearing.phi_j and result.delta_j).

        Example
        -------
            dist = IterativeBearingFEMSolver.contact_distribution(bearing, result)
            phi, Q = dist[:, 0], dist[:, 1]
            for phi_i, Q_i in dist:
                print(f"phi={np.degrees(phi_i):6.1f}  Q={Q_i:8.2f} N")
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
        Secant bearing stiffness (Kr_xz, Kr_xy, Ka) recovered from the
        converged internal load distribution result.

            Kr_xz = Fr_xz / delta_r_xz
            Kr_xy = Fr_xy / delta_r_xy
            Ka    = Fa    / delta_a

        delta_r_xz, delta_r_xy are the radial ring displacement decomposed
        from the resultant-plane solve (result.delta_r, result.phi_Fr) back
        onto the global XZ/XY axes:
            delta_r_xz = delta_r * cos(phi_Fr)
            delta_r_xy = delta_r * sin(phi_Fr)
        Fr_xz, Fr_xy must be the SAME reactions that produced Fr/phi_Fr in
        this solve (i.e. bearing_data[label]['Fr_xz'/'Fr_xy'] from the same
        call to solve()) — passing mismatched values silently corrupts Kr.

        Guards
        ------
        Kr_xz / Kr_xy : if the corresponding displacement component is ~0,
            the ring has not moved in that direction (no measurable
            compliance) -> stiffness reported as inf (rigid).

        Ka : delta_a is, in general, NOT the pure axial response to Fa alone
            — under Fa=0 the ring still settles at delta_a ~= -A*sin(alpha_0)
            to balance the free contact angle (see module docstring / ISO
            eq. 12). That settling is an assembly/clearance artefact, not a
            load-deflection response, and computing Fa/delta_a there (with
            Fa=0) would report Ka=0 — implying "no axial stiffness", which
            is physically wrong (the DGBB is not axially free; it simply
            has no axial load applied in this load case).
            Therefore Ka is only evaluated when Fa != 0 (a real axial load
            is present); otherwise Ka -> inf (no axial stiffness deficiency
            can be inferred from a zero-load case).
            A negative delta_a with Fa != 0 would indicate an assembly/sign
            inconsistency (ring displacing opposite to the applied axial
            load) — Ka is still returned (Fa/delta_a, negative), and the
            caller should treat a negative Ka as an error flag rather than
            a physical stiffness.

        Parameters
        ----------
        bearing : Bearing
        result  : LoadDistributionResult — from the same solve() call as
                  Fr_xz, Fr_xy, Fa below
        Fr_xz, Fr_xy, Fa : float — bearing reactions [N], typically
                  bearing_data[label]['Fr_xz'/'Fr_xy'/'Fa']

        Returns
        -------
        (Kr_xz, Kr_xy, Ka) [N/mm], each possibly float('inf')
        """
        delta_r_xz = result.delta_r * np.cos(result.phi_Fr)
        delta_r_xy = result.delta_r * np.sin(result.phi_Fr)
        delta_a    = result.delta_a

        eps = 1e-9  # mm, below which a displacement component is "zero"

        Kr_xz = (Fr_xz / delta_r_xz) if abs(delta_r_xz) > eps else float("inf")
        Kr_xy = (Fr_xy / delta_r_xy) if abs(delta_r_xy) > eps else float("inf")

        if Fa == 0.0:
            Ka = float("inf")
        elif abs(delta_a) > eps:
            Ka = Fa / delta_a
        else:
            Ka = float("inf")

        return Kr_xz, Kr_xy, Ka

    # ------------------------------------------------------------------
    # Helper
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
        Minimum axial preload Fa_min [N] that must be applied to the shaft so
        that this bearing's axial ring displacement delta_a >= 0.

        Context
        -------
        Under Fa=0 a radial DGBB settles at delta_a ~= -A*sin(alpha_0) (ISO
        eq. 12 free-contact-angle balance) — a negative axial displacement
        that is NOT itself an error (see bearing_stiffness docstring: it is
        clearance take-up, not a load response). It only becomes a problem
        for arrangements that require the ring to stay at or above its
        nominal axial position (e.g. this bearing is meant to be preloaded,
        or delta_a < 0 would mean the ring rides on the wrong flank of the
        raceway for the intended locating direction).

        This helper finds the smallest Fa >= 0 that brings delta_a exactly
        to 0, i.e. cancels the free-contact-angle settling. Applying less
        than Fa_min leaves the ring in the negative (clearance-only)
        position; applying Fa_min or more keeps delta_a >= 0.

        Method
        ------
        Root search (brentq) on g(Fa) = delta_a(Fa), bracketed by Fa_bracket.
        delta_a(Fa) is obtained by re-running the full internal-distribution
        solve (_solve_bearing_internal) at each trial Fa — Fr_xz, Fr_xy,
        psi_xz, psi_xy are held fixed throughout (only Fa is varied).

        If delta_a(Fa=0) is already >= 0 (no clearance-settling problem for
        this load case), Fa_min = 0.0 is returned without a root search.

        Parameters
        ----------
        bearing      : Bearing — cp, geometry already set up
        Fr_xz, Fr_xy : radial reactions [N] (held fixed)
        psi_xz, psi_xy : prescribed misalignment [rad] (held fixed)
        delta_r_init, delta_a_init : starting guesses for the inner solve
        Fa_bracket   : (Fa_lo, Fa_hi) [N] search bracket; must bracket the
                       root (delta_a(Fa_lo) < 0 <= delta_a(Fa_hi)). Widen if
                       ValueError is raised.
        xtol         : brentq tolerance on Fa [N]

        Returns
        -------
        (Fa_min, result) : Fa_min [N], and the LoadDistributionResult at
                            Fa_min (delta_a ~= 0 there, up to xtol)

        Raises
        ------
        ValueError if Fa_bracket does not bracket the root (delta_a stays
        negative even at Fa_bracket[1] — widen the bracket).
        """
        res0 = self._solve_bearing_internal(
            bearing, Fr_xz, Fr_xy, 0.0, delta_r_init, delta_a_init, psi_xz, psi_xy)
        if res0.delta_a >= 0.0:
            return 0.0, res0

        def g(Fa: float) -> float:
            res = self._solve_bearing_internal(
                bearing, Fr_xz, Fr_xy, Fa, delta_r_init, delta_a_init, psi_xz, psi_xy)
            return res.delta_a

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

    def _run_root(self, fun, x0, n_out):
        """
        Run scipy.optimize.root with 'hybr', falling back to 'lm' if hybr
        does not report success. Returns (x, nfev, residual_norm, success).
        """
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