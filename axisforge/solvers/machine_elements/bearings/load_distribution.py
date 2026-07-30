"""
axisforge/solvers/machine_elements/bearings/load_distribution.py

ISO/TS 16281 internal load distribution + coupled shaft-bearing solver.

References:
  ISO/TS 16281:2008 §4.2
  Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch.6
"""
from __future__ import annotations

import numpy as np

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
    Output of the internal load distribution Newton-Raphson solve
    for a single bearing position — two independent planar solves.

    Attributes
    ----------
    -- XZ plane (3x3 Newton: delta_r_xz, delta_a, psi_xz) --
    delta_r_xz  : float           — radial ring displacement, XZ [mm]
    delta_a     : float           — axial ring displacement [mm]
    psi_xz      : float           — misalignment angle, XZ [rad]
    K_xz        : np.ndarray(3,3) — bearing stiffness matrix, XZ
    delta_j_xz  : np.ndarray(Z,)  — elastic deflection per element, XZ [mm]
    alpha_j_xz  : np.ndarray(Z,)  — effective contact angle per element, XZ [rad]
    Mz_xz       : float           — moment reaction, XZ [N.mm]
    n_iter_xz   : int
    residual_xz : float           — final ||R|| XZ [N]

    -- XY plane (2x2 Newton: delta_r_xy, psi_xy | delta_a fixed) --
    delta_r_xy  : float           — radial ring displacement, XY [mm]
    psi_xy      : float           — misalignment angle, XY [rad]
    K_xy        : np.ndarray(2,2) — bearing stiffness matrix, XY
    delta_j_xy  : np.ndarray(Z,)  — elastic deflection per element, XY [mm]
    alpha_j_xy  : np.ndarray(Z,)  — effective contact angle per element, XY [rad]
    Mz_xy       : float           — moment reaction, XY [N.mm]
    n_iter_xy   : int
    residual_xy : float           — final ||R|| XY [N]
    """

    __slots__ = (
        "delta_r_xz", "delta_a", "psi_xz",
        "K_xz", "delta_j_xz", "alpha_j_xz", "Mz_xz",
        "n_iter_xz", "residual_xz",
        "delta_r_xy", "psi_xy",
        "K_xy", "delta_j_xy", "alpha_j_xy", "Mz_xy",
        "n_iter_xy", "residual_xy",
    )

    def __init__(self,
                 delta_r_xz, delta_a, psi_xz,
                 K_xz, delta_j_xz, alpha_j_xz,
                 Mz_xz, n_iter_xz, residual_xz,
                 delta_r_xy, psi_xy,
                 K_xy, delta_j_xy, alpha_j_xy,
                 Mz_xy, n_iter_xy, residual_xy):

        self.delta_r_xz  = delta_r_xz
        self.delta_a     = delta_a
        self.psi_xz      = psi_xz
        self.K_xz        = K_xz
        self.delta_j_xz  = delta_j_xz
        self.alpha_j_xz  = alpha_j_xz
        self.Mz_xz       = Mz_xz
        self.n_iter_xz   = n_iter_xz
        self.residual_xz = residual_xz

        self.delta_r_xy  = delta_r_xy
        self.psi_xy      = psi_xy
        self.K_xy        = K_xy
        self.delta_j_xy  = delta_j_xy
        self.alpha_j_xy  = alpha_j_xy
        self.Mz_xy       = Mz_xy
        self.n_iter_xy   = n_iter_xy
        self.residual_xy = residual_xy


# ===========================================================================
# IterativeBearingFEMSolver
# ===========================================================================

class IterativeBearingFEMSolver:
    """
    Coupled shaft-bearing solver -- ISO/TS 16281.

    Iter 0 : SimpleFEMSolver (rigid bearings)
             -> Fr_xz, Fr_xy, Fa, Mz, theta per bearing
    Iter k : _solve_bearing_internal (Newton 3x3 XZ, 2x2 XY)
             -> K_xz, K_xy per bearing
             inject K_xz, K_xy into global FEM K before condensation
             FEM with bearing stiffness -> updated reactions + theta
    until |DeltaFr| < tol

    Requires bearing.setup_internal_geometry() and
    bearing.compute_hertz_point_contact() to have been called on every
    Bearing instance before solve() is invoked.

    Parameters
    ----------
    tol      : convergence tolerance on bearing reaction force [N]
    max_iter : maximum outer coupling iterations
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
              Pd: float = 0.0) -> dict[str, LoadDistributionResult]:
        """
        Parameters
        ----------
        shaft_system : ShaftSystem -- fully resolved
        bearings     : {label: Bearing}
                       setup_internal_geometry() and compute_hertz_point_contact()
                       must have been called on each Bearing before this call.
        Pd           : diametral clearance override [mm] (0 = use bearing.s)

        Returns
        -------
        {label: LoadDistributionResult}
        """
        # iter 0 -- rigid bearing FEM
        fem          = self._iter0_rigid(shaft_system)
        bearing_data = self._extract_bearing_forces(fem, shaft_system)

        results: dict[str, LoadDistributionResult] = {}
        Fr_old = {lbl: (d['Fr_xz'], d['Fr_xy']) for lbl, d in bearing_data.items()}

        for _ in range(self.max_iter):

            # Newton per bearing
            K_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for label, b in bearings.items():
                data = bearing_data[label]
                res  = self._solve_bearing_internal(
                    b,
                    Fr_xz           = data['Fr_xz'],
                    Fr_xy           = data['Fr_xy'],
                    Fa              = data['Fa'],
                    Mz_xz           = data['M_xz'],
                    Mz_xy           = data['M_xy'],
                    delta_r_xz_init = data['v_xz'],
                    delta_r_xy_init = data['v_xy'],
                    delta_a_init    = data['u'],
                    psi_xz_init     = data['theta_xz'],
                    psi_xy_init     = data['theta_xy'],
                )
                results[label] = res
                K_map[label]   = (res.K_xz, res.K_xy)

            # FEM with bearing stiffness injected
            fem          = self._solve_with_bearing_stiffness(shaft_system, K_map)
            bearing_data = self._extract_bearing_forces(fem, shaft_system)

            Fr_new = {lbl: (d['Fr_xz'], d['Fr_xy']) for lbl, d in bearing_data.items()}

            if self._converged(Fr_old, Fr_new):
                break

            Fr_old = Fr_new

        # --- expose converged FEM for downstream use (convergence study) ---
        self.fem_converged = fem
        self.bearing_data  = bearing_data

        return results

    # ------------------------------------------------------------------
    # Iter 0 -- rigid bearing FEM
    # ------------------------------------------------------------------

    def _iter0_rigid(self, shaft_system: ShaftSystem) -> SimpleFEMSolver:
        """Rigid bearing FEM. Returns solved SimpleFEMSolver."""
        fem = SimpleFEMSolver()
        fem.solve(shaft_system)
        return fem

    # ------------------------------------------------------------------
    # Extract bearing forces and displacements
    # ------------------------------------------------------------------

    def _extract_bearing_forces(self,
                                 fem: SimpleFEMSolver,
                                 shaft_system: ShaftSystem) -> dict[str, dict]:
        """
        Returns {label: {Fr_xz, Fr_xy, Fr, phi_Fr,
                          Fa,
                          M_xz, M_xy, M, phi_M,
                          v_xz, v_xy, v, phi_v,
                          u,
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

            phi_Fr    = np.arctan2(Fr_xy,    Fr_xz)
            phi_M     = np.arctan2(M_xy,     M_xz)
            phi_v     = np.arctan2(v_xy,     v_xz)
            phi_theta = np.arctan2(theta_xy, theta_xz)

            result[b.label] = {
                'Fr_xz'    : Fr_xz,
                'Fr_xy'    : Fr_xy,
                'Fr'       : Fr,
                'phi_Fr'   : phi_Fr,
                'Fa'       : Fa,
                'M_xz'     : M_xz,
                'M_xy'     : M_xy,
                'M'        : M,
                'phi_M'    : phi_M,
                'v_xz'     : v_xz,
                'v_xy'     : v_xy,
                'v'        : v,
                'phi_v'    : phi_v,
                'u'        : u,
                'theta_xz' : theta_xz,
                'theta_xy' : theta_xy,
                'theta'    : theta,
                'phi_theta': phi_theta,
            }

        return result

    # ------------------------------------------------------------------
    # Newton-Raphson -- internal load distribution
    # ------------------------------------------------------------------

    def _solve_bearing_internal(self,
                             bearing: Bearing,
                             Fr_xz: float,
                             Fr_xy: float,
                             Fa: float,
                             Mz_xz: float,
                             Mz_xy: float,
                             delta_r_xz_init: float,
                             delta_r_xy_init: float,
                             delta_a_init: float,
                             psi_xz_init: float,
                             psi_xy_init: float) -> LoadDistributionResult:
        """
        XZ: 3x3 Newton  (delta_r_xz, delta_a, psi_xz)
        XY: 2x2 Newton  (delta_r_xy, psi_xy)  -- delta_a fixed from XZ

        Requires bearing.cp, bearing.A, bearing.alpha_0,
                 bearing.phi_j, bearing.Dpw, bearing.Ri to exist.
        """
        delta_r_xz, delta_a, psi_xz, K_xz, delta_j_xz, alpha_j_xz, \
            Mz_xz_res, n_iter_xz, res_xz = self._newton_xz(
                bearing, Fr_xz, Fa, Mz_xz,
                delta_r_init = delta_r_xz_init,
                delta_a_init = delta_a_init,
                psi_init     = psi_xz_init,
            )

        delta_r_xy, psi_xy, K_xy, delta_j_xy, alpha_j_xy, \
            Mz_xy_res, n_iter_xy, res_xy = self._newton_xy(
                bearing, Fr_xy, Mz_xy, delta_a,
                delta_r_init = delta_r_xy_init,
                psi_init     = psi_xy_init,
            )

        return LoadDistributionResult(
            delta_r_xz=delta_r_xz, delta_a=delta_a,   psi_xz=psi_xz,
            K_xz=K_xz, delta_j_xz=delta_j_xz, alpha_j_xz=alpha_j_xz,
            Mz_xz=Mz_xz_res, n_iter_xz=n_iter_xz, residual_xz=res_xz,
            delta_r_xy=delta_r_xy, psi_xy=psi_xy,
            K_xy=K_xy, delta_j_xy=delta_j_xy, alpha_j_xy=alpha_j_xy,
            Mz_xy=Mz_xy_res, n_iter_xy=n_iter_xy, residual_xy=res_xy,
        )

    def _newton_xz(self, bearing, Fr_xz, Fa, Mz_xz,
               delta_r_init, delta_a_init, psi_init):
        """
        Newton-Raphson 3x3 -- XZ plane.
        u = [delta_r_xz, delta_a, psi_xz]

        R0 = Fr_xz - cp * sum(dj^1.5 * cos(aj) * cos(phi_j))
        R1 = Fa    - cp * sum(dj^1.5 * sin(aj))
        R2 = Mz_xz - Dpw/2 * cp * sum(dj^1.5 * sin(aj) * cos(phi_j))
        """
        cp      = bearing.cp
        A       = bearing.A
        alpha_0 = bearing.alpha_0
        phi_j   = bearing.phi_j
        Dpw     = bearing.Dpw
        Ri      = bearing.Ri
        Z       = bearing.Z

        # fallback: FEM rígido dá v=0 no rolamento -> usar estimativa Hertziana
        if abs(delta_r_init) < 1e-6:
            delta_r = (abs(Fr_xz) / (cp * Z)) ** (2.0 / 3.0) if Fr_xz != 0.0 else 1e-4
        else:
            delta_r = delta_r_init
        delta_a  = delta_a_init
        psi     = psi_init

        d32 = np.zeros(Z)
        sa  = np.zeros(Z)
        ca  = np.zeros(Z)
        K   = np.zeros((3, 3))
        residual = float('inf')

        for n_iter in range(self.max_iter):
            r_j     = Ri * np.cos(psi)
            U_j     = A * np.cos(alpha_0) + delta_r * np.cos(phi_j)
            V_j     = A * np.sin(alpha_0) + delta_a + Ri * np.sin(psi) * np.cos(phi_j)
            Sigma_j = np.sqrt(U_j**2 + V_j**2)
            delta_j = np.maximum(Sigma_j - A, 0.0)
            mask    = delta_j > 0.0

            alpha_j = np.where(mask, np.arctan2(V_j, U_j), 0.0)
            ca      = np.cos(alpha_j)
            sa      = np.sin(alpha_j)
            cp_j    = np.cos(phi_j)
            d32     = np.where(mask, delta_j ** 1.5, 0.0)
            d12     = np.where(mask, delta_j ** 0.5, 0.0)
            w       = 1.5 * cp * d12

            R0 = Fr_xz - cp * np.sum(d32 * ca * cp_j) # Fext - Fint por isso é += na 367
            R1 = Fa    - cp * np.sum(d32 * sa)
            R2 = Mz_xz - (Dpw / 2.0) * cp * np.sum(d32 * sa * cp_j)

            residual = float(np.sqrt(R0**2 + R1**2 + R2**2))
            if residual < self.tol:
                break

            K = np.array([
                [np.sum(w * ca**2    * cp_j**2),
                 np.sum(w * sa * ca  * cp_j),
                 np.sum(w * r_j * ca * cp_j**2)],

                [np.sum(w * sa * ca  * cp_j),
                 np.sum(w * sa**2),
                 np.sum(w * r_j * sa * cp_j)],

                [(Dpw/2) * np.sum(w * sa * ca  * cp_j**2),
                 (Dpw/2) * np.sum(w * sa**2    * cp_j),
                 (Dpw/2) * np.sum(w * r_j * sa * cp_j**2)],
            ])

            try:
                du = np.linalg.solve(K, np.array([R0, R1, R2]))
            except np.linalg.LinAlgError:
                break

            delta_r += du[0]
            delta_a += du[1]
            psi     += du[2]

        Mz_res = (Dpw / 2.0) * cp * float(np.sum(d32 * sa * np.cos(phi_j)))
        return delta_r, delta_a, psi, K, delta_j, alpha_j, Mz_res, n_iter + 1, residual

    def _newton_xy(self, bearing, Fr_xy, Mz_xy, delta_a_fixed,
               delta_r_init, psi_init):
        """
        Newton-Raphson 2x2 -- XY plane.
        u = [delta_r_xy, psi_xy]
        delta_a fixed from XZ solve.

        R0 = Fr_xy - cp * sum(dj^1.5 * cos(aj) * cos(phi_j))
        R1 = Mz_xy - Dpw/2 * cp * sum(dj^1.5 * sin(aj) * cos(phi_j))
        """
        cp      = bearing.cp
        A       = bearing.A
        alpha_0 = bearing.alpha_0
        phi_j   = bearing.phi_j
        Dpw     = bearing.Dpw
        Ri      = bearing.Ri
        Z       = bearing.Z

        if abs(delta_r_init) < 1e-6:
            delta_r = (abs(Fr_xy) / (cp * Z)) ** (2.0 / 3.0) if Fr_xy != 0.0 else 1e-4
        else:
            delta_r = delta_r_init
        psi     = psi_init

        d32 = np.zeros(Z)
        sa  = np.zeros(Z)
        K   = np.zeros((2, 2))
        residual = float('inf')

        for n_iter in range(self.max_iter):
            r_j     = Ri * np.cos(psi)
            U_j     = A * np.cos(alpha_0) + delta_r * np.cos(phi_j)
            V_j     = A * np.sin(alpha_0) + delta_a_fixed + Ri * np.sin(psi) * np.cos(phi_j)
            Sigma_j = np.sqrt(U_j**2 + V_j**2)
            delta_j = np.maximum(Sigma_j - A, 0.0)
            mask    = delta_j > 0.0

            alpha_j = np.where(mask, np.arctan2(V_j, U_j), 0.0)
            ca      = np.cos(alpha_j)
            sa      = np.sin(alpha_j)
            cp_j    = np.cos(phi_j)
            d32     = np.where(mask, delta_j ** 1.5, 0.0)
            d12     = np.where(mask, delta_j ** 0.5, 0.0)
            w       = 1.5 * cp * d12

            R0 = Fr_xy - cp * np.sum(d32 * ca * cp_j)
            R1 = Mz_xy - (Dpw / 2.0) * cp * np.sum(d32 * sa * cp_j)

            residual = float(np.sqrt(R0**2 + R1**2))
            if residual < self.tol:
                break

            K = np.array([
                [np.sum(w * ca**2  * cp_j**2),
                 np.sum(w * r_j * ca * cp_j**2)],

                [(Dpw/2) * np.sum(w * sa * ca  * cp_j**2),
                 (Dpw/2) * np.sum(w * r_j * sa * cp_j**2)],
            ])

            try:
                du = np.linalg.solve(K, np.array([R0, R1]))
            except np.linalg.LinAlgError:
                break

            delta_r += du[0]
            psi     += du[1]

        Mz_res = (Dpw / 2.0) * cp * float(np.sum(d32 * sa * np.cos(phi_j)))
        return delta_r, psi, K, delta_j, alpha_j, Mz_res, n_iter + 1, residual

    # ------------------------------------------------------------------
    # FEM with bearing stiffness injected into global K
    # ------------------------------------------------------------------

    def _solve_with_bearing_stiffness(self,
                                    shaft_system: ShaftSystem,
                                    K_map: dict[str, tuple]) -> SimpleFEMSolver:
        fem = SimpleFEMSolver()
        fem.solve(shaft_system)

        x_nodes  = fem.x_nodes
        K_global = fem.K.copy()

        for b in shaft_system.bearings:
            if b.label not in K_map:
                continue
            i           = Elem.find_node_index(x_nodes, b.position)
            K_xz, K_xy = K_map[b.label]

            dofs_xz = [3*i+1, 3*i, 3*i+2]
            K_global[np.ix_(dofs_xz, dofs_xz)] += K_xz

            dofs_xy = [3*i+1, 3*i+2]
            K_global[np.ix_(dofs_xy, dofs_xy)] += K_xy

        # free_dofs do FEM rígido + DOF v dos rolamentos (agora livres com mola)
        free_dofs_updated = list(fem.free_dofs)

        for b in shaft_system.bearings:
            i    = Elem.find_node_index(x_nodes, b.position)
            dof_v = 3 * i + 1
            if dof_v not in free_dofs_updated:
                free_dofs_updated.append(dof_v)

        free_dofs_updated = sorted(free_dofs_updated)

        K_red = K_global[np.ix_(free_dofs_updated, free_dofs_updated)]
        n     = 3 * len(x_nodes)
        d_xz  = np.zeros(n)
        d_xy  = np.zeros(n)
        d_xz[free_dofs_updated] = np.linalg.solve(K_red, fem.f_xz_ext[free_dofs_updated])
        d_xy[free_dofs_updated] = np.linalg.solve(K_red, fem.f_xy_ext[free_dofs_updated])

        fem.d_total_xz = d_xz
        fem.d_total_xy = d_xy
        fem.f_xz_total = K_global @ d_xz
        fem.f_xy_total = K_global @ d_xy
        fem.K          = K_global

        return fem

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------

    def _converged(self,
                   Fr_old: dict[str, tuple],
                   Fr_new: dict[str, tuple]) -> bool:
        """True if max |DeltaFr| across all bearings and planes < self.tol."""
        for label in Fr_old:
            xz_old, xy_old = Fr_old[label]
            xz_new, xy_new = Fr_new[label]
            if abs(xz_new - xz_old) > self.tol:
                return False
            if abs(xy_new - xy_old) > self.tol:
                return False
        return True