"""
axisforge/solvers/machine_elements/bearings/load_distribution_ISO_16281_ball_bearing.py

ISO/TS 16281 internal load distribution — coupled shaft-bearing solver
(prescribed-psi formulation, solved with scipy.optimize.root).

Architecture (ISO/TS 16281 §4.2)
--------------------------------
The FEM supplies the bearing reactions (Fr_xz, Fr_xy, Fa) and the shaft
centreline slope across the bearing seat; a per-bearing root solve then
resolves the internal load distribution (delta_r, delta_a) that equilibrates
those radial/axial reactions. The root solve IS the bearing model — no
bearing stiffness is injected back into the global FEM K. Rolling-element
supports remain rigid displacement BCs in the FEM.

Element kinematics — ISO/TS 16281 eq. (12)/(15)
-----------------------------------------------
    U_j     = A*cos(alpha_0) + delta_r*cos(phi_j)
    V_j     = A*sin(alpha_0) + delta_a + Ri*sin(psi)*cos(phi_j)
    delta_j = sqrt(U_j^2 + V_j^2) - A        (set to 0 if negative)
    alpha_j = arctan2(V_j, U_j)

Static equilibrium — ISO/TS 16281 §4.2.2.1
------------------------------------------
    R0 = Fr - cp*sum(delta_j^1.5 * cos(alpha_j) * cos(phi_j))
    R1 = Fa - cp*sum(delta_j^1.5 * sin(alpha_j))                (XZ only)

Solver
------
scipy.optimize.root (MINPACK 'hybr', with 'lm' fallback) replaces the hand-
rolled Newton. It supplies automatic variable scaling and a trust-region /
line-search globalisation, which the radial DGBB under Fa=0 needs: delta_r is
stiff, delta_a is soft, and a plain Newton either overshoots delta_a or
stalls on the badly-scaled 2x2 Jacobian. root() handles both without manual
damping.

Why psi is prescribed
---------------------
For a radial DGBB the moment reaction is negligible; feeding the beam bending
moment into the solve has no physical root and diverges. psi (ring tilt) is
PRESCRIBED from the shaft centreline slope across the seat:
    psi = (v(x_hi) - v(x_lo)) / (x_hi - x_lo)
Mz then falls out as a pure OUTPUT of the converged distribution.

    XZ : 2-eq root  (delta_r_xz, delta_a)   -- psi prescribed
    XY : 1-eq root  (delta_r_xy)            -- delta_a shared from XZ

References
----------
  ISO/TS 16281:2008 §4.2, eq. (12)-(15)
  Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch.6
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import root

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
    Output of the internal load distribution solve for a single bearing
    position — two independent planar solves with psi prescribed.

    Attributes
    ----------
    -- XZ plane (2-eq root: delta_r_xz, delta_a | psi_xz prescribed) --
    delta_r_xz  : float           — radial ring displacement, XZ [mm]
    delta_a     : float           — axial ring displacement [mm]
    psi_xz      : float           — prescribed misalignment, XZ [rad]
    delta_j_xz  : np.ndarray(Z,)  — elastic deflection per element, XZ [mm]
    alpha_j_xz  : np.ndarray(Z,)  — effective contact angle per element, XZ [rad]
    Mz_xz       : float           — moment reaction (OUTPUT), XZ [N.mm]
    n_iter_xz   : int             — solver function evaluations
    residual_xz : float           — final ||R|| XZ [N]
    ok_xz       : bool            — solver success flag

    -- XY plane (1-eq root: delta_r_xy | delta_a fixed, psi_xy prescribed) --
    delta_r_xy  : float           — radial ring displacement, XY [mm]
    psi_xy      : float           — prescribed misalignment, XY [rad]
    delta_j_xy  : np.ndarray(Z,)  — elastic deflection per element, XY [mm]
    alpha_j_xy  : np.ndarray(Z,)  — effective contact angle per element, XY [rad]
    Mz_xy       : float           — moment reaction (OUTPUT), XY [N.mm]
    n_iter_xy   : int             — solver function evaluations
    residual_xy : float           — final ||R|| XY [N]
    ok_xy       : bool            — solver success flag

    Note
    ----
    Contact-force distribution is recovered downstream as
    Q_j = cp * max(delta_j, 0)**1.5.
    """

    __slots__ = (
        "delta_r_xz", "delta_a", "psi_xz",
        "delta_j_xz", "alpha_j_xz", "Mz_xz",
        "n_iter_xz", "residual_xz", "ok_xz",
        "delta_r_xy", "psi_xy",
        "delta_j_xy", "alpha_j_xy", "Mz_xy",
        "n_iter_xy", "residual_xy", "ok_xy",
    )

    def __init__(self,
                 delta_r_xz, delta_a, psi_xz,
                 delta_j_xz, alpha_j_xz, Mz_xz, n_iter_xz, residual_xz, ok_xz,
                 delta_r_xy, psi_xy,
                 delta_j_xy, alpha_j_xy, Mz_xy, n_iter_xy, residual_xy, ok_xy):

        self.delta_r_xz  = delta_r_xz
        self.delta_a     = delta_a
        self.psi_xz      = psi_xz
        self.delta_j_xz  = delta_j_xz
        self.alpha_j_xz  = alpha_j_xz
        self.Mz_xz       = Mz_xz
        self.n_iter_xz   = n_iter_xz
        self.residual_xz = residual_xz
        self.ok_xz       = ok_xz

        self.delta_r_xy  = delta_r_xy
        self.psi_xy      = psi_xy
        self.delta_j_xy  = delta_j_xy
        self.alpha_j_xy  = alpha_j_xy
        self.Mz_xy       = Mz_xy
        self.n_iter_xy   = n_iter_xy
        self.residual_xy = residual_xy
        self.ok_xy       = ok_xy


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
    2. Per bearing: _solve_bearing_internal
         XZ 2-eq root (delta_r_xz, delta_a)   -- psi_xz prescribed
         XY 1-eq root (delta_r_xy)            -- delta_a shared, psi_xy prescribed

    Single FEM pass; no bearing stiffness injected into the global FEM.

    Requires bearing.setup_internal_geometry() and
    bearing.compute_hertz_point_contact() on every Bearing before solve().

    Parameters
    ----------
    tol      : residual tolerance passed to root() (xtol/ftol) [-]
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
                Fr_xz           = data['Fr_xz'],
                Fr_xy           = data['Fr_xy'],
                Fa              = data['Fa'],
                delta_r_xz_init = data['v_xz'],
                delta_r_xy_init = data['v_xy'],
                delta_a_init    = data['u'],
                psi_xz          = psi_xz,
                psi_xy          = psi_xy,
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
        Falls back to the nodal section rotation theta if the seat width is 0.

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

        Fa is the axial reaction at the locating bearing node (0 at floating).
        M_xz/M_xy are the beam bending moments — reported for inspection only,
        NOT fed into the bearing solve (see module docstring).
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
                'Fr'       : Fr,     'phi_Fr'  : np.arctan2(Fr_xy, Fr_xz),
                'Fa'       : Fa,
                'M_xz'     : M_xz,   'M_xy'    : M_xy,
                'M'        : M,      'phi_M'   : np.arctan2(M_xy, M_xz),
                'v_xz'     : v_xz,   'v_xy'    : v_xy,
                'v'        : v,      'phi_v'   : np.arctan2(v_xy, v_xz),
                'u'        : u,
                'theta_xz' : theta_xz, 'theta_xy' : theta_xy,
                'theta'    : theta,    'phi_theta': np.arctan2(theta_xy, theta_xz),
            }

        return result

    # ------------------------------------------------------------------
    # Element kinematics (ISO eq. 12/15, §4.2.2.1)
    # ------------------------------------------------------------------

    @staticmethod
    def _elements(bearing, delta_r, delta_a, Vpsi):
        """
        Per-element deflection and contact angle for given ring displacements
        (ISO eq. 12/15).

        Vpsi : constant array Ri*sin(psi)*cos(phi_j).

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
        Initial delta_r that clears the clearance gap A*(1-cos(alpha_0)) and
        adds a Hertzian contact estimate, so the residual is well-defined
        (elements in contact) at the starting point.
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
    # Internal load distribution — psi prescribed, solved with root()
    # ------------------------------------------------------------------

    def _solve_bearing_internal(self,
                             bearing: Bearing,
                             Fr_xz: float,
                             Fr_xy: float,
                             Fa: float,
                             delta_r_xz_init: float,
                             delta_r_xy_init: float,
                             delta_a_init: float,
                             psi_xz: float,
                             psi_xy: float) -> LoadDistributionResult:
        """
        XZ: 2-eq root  (delta_r_xz, delta_a)   -- psi_xz prescribed
        XY: 1-eq root  (delta_r_xy)            -- delta_a fixed from XZ

        Requires bearing.cp, bearing.A, bearing.alpha_0,
                 bearing.phi_j, bearing.Dpw, bearing.Ri to exist.
        """
        delta_r_xz, delta_a, delta_j_xz, alpha_j_xz, \
            Mz_xz_res, nfev_xz, res_xz, ok_xz = self._root_xz(
                bearing, Fr_xz, Fa, psi_xz,
                delta_r_init = delta_r_xz_init,
                delta_a_init = delta_a_init,
            )

        delta_r_xy, delta_j_xy, alpha_j_xy, \
            Mz_xy_res, nfev_xy, res_xy, ok_xy = self._root_xy(
                bearing, Fr_xy, delta_a, psi_xy,
                delta_r_init = delta_r_xy_init,
            )

        return LoadDistributionResult(
            delta_r_xz=delta_r_xz, delta_a=delta_a, psi_xz=psi_xz,
            delta_j_xz=delta_j_xz, alpha_j_xz=alpha_j_xz,
            Mz_xz=Mz_xz_res, n_iter_xz=nfev_xz, residual_xz=res_xz, ok_xz=ok_xz,
            delta_r_xy=delta_r_xy, psi_xy=psi_xy,
            delta_j_xy=delta_j_xy, alpha_j_xy=alpha_j_xy,
            Mz_xy=Mz_xy_res, n_iter_xy=nfev_xy, residual_xy=res_xy, ok_xy=ok_xy,
        )

    def _run_root(self, fun, x0, n_out):
        """
        Run scipy.optimize.root with 'hybr', falling back to 'lm' if hybr
        does not report success. Returns (x, nfev, residual_norm, success).
        """
        sol = root(fun, x0, method="hybr", tol=self.tol)
        if not sol.success:
            sol_lm = root(fun, x0, method="lm", tol=self.tol)
            # keep whichever has the smaller residual
            if float(np.linalg.norm(np.atleast_1d(sol_lm.fun))) < \
               float(np.linalg.norm(np.atleast_1d(sol.fun))):
                sol = sol_lm

        x    = np.atleast_1d(sol.x)
        res  = float(np.linalg.norm(np.atleast_1d(sol.fun)))
        nfev = int(getattr(sol, "nfev", 0))
        return x, nfev, res, bool(sol.success)

    # ------------------------------------------------------------------
    # XZ — 2-eq root (delta_r, delta_a), psi prescribed
    # ------------------------------------------------------------------

    def _root_xz(self, bearing, Fr_xz, Fa, psi, delta_r_init, delta_a_init):
        cp    = bearing.cp
        Dpw   = bearing.Dpw
        Ri    = bearing.Ri
        cp_j  = np.cos(bearing.phi_j)
        Vpsi  = Ri * np.sin(psi) * cp_j

        def residual(u):
            dr, da = u
            _, _, ca, sa, _, d32, _, _ = self._elements(bearing, dr, da, Vpsi)
            return np.array([
                Fr_xz - cp * np.sum(d32 * ca * cp_j),
                Fa    - cp * np.sum(d32 * sa),
            ])

        dr0 = self._radial_init(bearing, Fr_xz, delta_r_init)
        x, nfev, res, ok = self._run_root(residual, [dr0, delta_a_init], 2)
        delta_r, delta_a = float(x[0]), float(x[1])

        delta_j, alpha_j, ca, sa, _, d32, _, _ = self._elements(bearing, delta_r, delta_a, Vpsi)
        Mz_res = (Dpw / 2.0) * cp * float(np.sum(d32 * sa * cp_j))
        return delta_r, delta_a, delta_j, alpha_j, Mz_res, nfev, res, ok

    # ------------------------------------------------------------------
    # XY — 1-eq root (delta_r), psi prescribed, delta_a fixed from XZ
    # ------------------------------------------------------------------

    def _root_xy(self, bearing, Fr_xy, delta_a_fixed, psi, delta_r_init):
        cp    = bearing.cp
        Dpw   = bearing.Dpw
        Ri    = bearing.Ri
        cp_j  = np.cos(bearing.phi_j)
        Vpsi  = Ri * np.sin(psi) * cp_j

        def residual(u):
            dr = u[0]
            _, _, ca, _, _, d32, _, _ = self._elements(bearing, dr, delta_a_fixed, Vpsi)
            return np.array([Fr_xy - cp * np.sum(d32 * ca * cp_j)])

        dr0 = self._radial_init(bearing, Fr_xy, delta_r_init)
        x, nfev, res, ok = self._run_root(residual, [dr0], 1)
        delta_r = float(x[0])

        delta_j, alpha_j, ca, sa, _, d32, _, _ = self._elements(bearing, delta_r, delta_a_fixed, Vpsi)
        Mz_res = (Dpw / 2.0) * cp * float(np.sum(d32 * sa * cp_j))
        return delta_r, delta_j, alpha_j, Mz_res, nfev, res, ok