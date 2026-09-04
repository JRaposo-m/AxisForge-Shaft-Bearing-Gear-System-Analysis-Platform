"""axisforge/solvers/machine_elements/bearings/load_distribution/single_row/iso_16281/ball_bearing/solver.py

ISO/TS 16281 internal load distribution SOLVER for POINT-CONTACT bearings
(deep groove and angular contact ball bearings -- BearingType.DEEP_GROOVE_BALL
and BearingType.ANGULAR_CONTACT). alpha_0 is a bearing attribute, not
hardcoded, so both families are covered by the same kinematics.

Scope of this file: only things that PRODUCE a BallLoadDistributionResult
live here. Anything that CONSUMES an already-computed result to derive a
further quantity lives in postprocessing.py, which imports from this file
only for typing; this file never imports it.

Capacity (Q_ci/Q_ce, Cr/Ca) lives in core/ -- every subtype's BearingFamily
knows which of its own capacity formulas applies
(bearing.family.per_element_dynamic_capacity(bearing, Cr=...) /
(bearing, Ca=...)). debug_radial_capacity() below is a manual cross-check,
sourced from bearing.family too.

Single-row only, see contracts/bearing/ball_bearing_results.py --
solve() takes the shaft's already-read ShaftResults directly (a contracts
type) rather than a fixtures registry: solvers never import from fixtures/
(one-way dependency rule), so the caller -- the future bearing study
orchestrator in fixtures/studies/bearing/ -- is what holds the registry
and extracts shaft_results before calling this.

The solve is performed in the plane of the resultant radial force:

    phi_Fr = arctan2(Fr_xy, Fr_xz)
    Fr     = sqrt(Fr_xz^2 + Fr_xy^2)
    psi    = psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr)   [prescribed misalignment]

A 2-equation root (delta_r, delta_a) enforces static equilibrium (Sec 4.2.2.1),
using the point-contact load-deflection law Q_j = cp * delta_j^1.5. Ball
positions in the global frame (output/plots only):

    phi_j_global = (bearing.phi_j + phi_Fr) % 2*pi

References
----------
ISO/TS 16281:2008 Sec 4.2, eq.(12)-(15); Sec 4.3.1, eq.(19)-(28)
Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch. 6
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import brentq

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem
from axisforge.results.fem_results.shaft_results import ShaftResults
from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.numerics import run_root
from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.validation import check_bearing_ready, warn_if_floating_loaded
from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.dispatch import register_contact_solver
from axisforge.results.bearings.load_distribution.single_row.ball_bearing_results import (
    BallLoadDistributionResult,
    BallBearingResult,
)
from axisforge.config import SOLVER_TOLERANCE


# ---------------------------------------------------------------------------
# ISO16281BallSolver
# ---------------------------------------------------------------------------

REQUIRED_ATTRS = ("A", "alpha_0", "phi_j", "Ri", "cp", "Dpw", "Z")


@register_contact_solver
class ISO16281BallSolver:
    """ISO/TS 16281 internal load distribution solver, point contact
    (deep groove / angular contact ball bearings).

    Takes bearing reactions and shaft slopes from a ShaftResults (already
    read from the shaft FEM registry by the caller) and solves the
    per-bearing internal load distribution via a 2-equation root problem in
    the resultant-force plane.

    Only solving lives on this class -- Q_j(), phi_j_global(),
    contact_distribution() and bearing_stiffness() live in
    postprocessing.py as free functions. Capacity lives in core/.

    Dispatched by the bearing study orchestrator via CAPABILITY +
    REQUIRED_ATTRS (dispatch.py) for any family that declares
    "point_contact" -- DGBB, angular contact, self-aligning, single-row
    thrust ball.

    Parameters
    ----------
    tol       float   residual tolerance for scipy.optimize.root
    psi_input bool    False (default): psi is always projected from FEM shaft slopes.
                      True: per-bearing psi values supplied via psi_override in
                      solve() take precedence when present; any bearing not
                      covered by psi_override falls back to the FEM projection.

                      psi_override values must already be expressed in the
                      resultant-force plane of the CURRENT load case -- i.e. the
                      caller is responsible for projecting onto
                      phi_Fr = arctan2(Fr_xy, Fr_xz), which is recomputed from
                      the FEM reactions on every solve() call. A fixed
                      geometric slope is not a valid substitute unless phi_Fr
                      is also fixed across the cases being compared.
    """

    CAPABILITY = "point_contact"
    REQUIRED_ATTRS = REQUIRED_ATTRS
    MULTIROW_SOLVER: type | None = None

    def __init__(self, tol: float = SOLVER_TOLERANCE, psi_input: bool = False):
        self.tol       = tol
        self.psi_input = psi_input

    # ------------------------------------------------------------------
    # Public entry point -- batch solve over a shaft's bearing set
    # ------------------------------------------------------------------

    def solve(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              shaft_results: ShaftResults,
              psi_override: dict[str, float] | None = None,
              ) -> dict[str, BallBearingResult]:
        """Solve the internal load distribution for all point-contact bearings
        in `bearings`.

        Each bearing must have setup_internal_geometry() and
        compute_hertz_point_contact() already called before this method.

        Parameters
        ----------
        shaft_system   : ShaftSystem -- fully resolved
        bearings       : {label: Bearing} -- all point-contact
        shaft_results  : ShaftResults for shaft_system, already read by the
                         caller from its own registry (ShaftResultsReader output)
        psi_override   : {label: psi [rad]} -- user-supplied misalignment values.
                         Used only when self.psi_input is True. See class
                         docstring for the plane convention required. Labels not
                         present here fall back to the FEM projection; labels
                         present here but not in `bearings` trigger a warning
                         (likely a typo).

        Returns
        -------
        dict[str, BallBearingResult] -- one per label in `bearings`, each
        wrapping exactly 1 row. The caller (the bearing study orchestrator)
        merges this into the final BearingResultsLibrary; this method never
        touches a registry itself.
        """
        if self.psi_input and psi_override:
            unknown = set(psi_override) - set(bearings)
            if unknown:
                warnings.warn(
                    f"psi_override has labels not in `bearings`: {sorted(unknown)} -- "
                    f"check for typos; these entries are ignored."
                )

        node_by_label = {n.label: n for n in shaft_results.bearing_nodes}

        results: dict[str, BallBearingResult] = {}
        for label, b in bearings.items():
            check_bearing_ready(b, label, REQUIRED_ATTRS)
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

            row = self.solve_contact(
                b,
                Fr_xz        = Fr_xz,
                Fr_xy        = Fr_xy,
                Fa           = node.Fa,
                delta_r_init = float(np.hypot(node.v_xz, node.v_xy)),
                delta_a_init = node.u,
                psi          = psi,
                phi_Fr       = phi_Fr,
            )
            results[label] = BallBearingResult.single(row)

        return results

    # ------------------------------------------------------------------
    # Element kinematics -- ISO/TS 16281 eq.(12)/(15), Sec 4.2.2.1
    # Public: also called directly by shared-displacement multi-row
    # solvers for THRUST_BALL, once per row per residual evaluation, with
    # no nested root-find.
    # ------------------------------------------------------------------

    @staticmethod
    def elements(bearing: Bearing,
                 delta_r: float,
                 delta_a: float,
                 Vpsi: np.ndarray):
        """Per-element elastic deflection and effective contact angle.
        phi_j is local: 0 aligned with the resultant radial force.

        delta_j is already zero-floored for unloaded elements, so alpha_j and
        d32 need no separate masking.
        """
        A, alpha_0, phi_j = bearing.A, bearing.alpha_0, bearing.phi_j
        cp_j = np.cos(phi_j)

        U_j     = A * np.cos(alpha_0) + delta_r * cp_j
        V_j     = A * np.sin(alpha_0) + delta_a + Vpsi
        delta_j = np.maximum(np.sqrt(U_j**2 + V_j**2) - A, 0.0)

        alpha_j = np.arctan2(V_j, U_j)
        d32     = delta_j ** 1.5   # point-contact load-deflection exponent

        return delta_j, alpha_j, np.cos(alpha_j), np.sin(alpha_j), cp_j, d32

    def _initial_delta_r(self, bearing: Bearing, Fr: float, hint: float) -> float:
        """Seed for delta_r. Trusts the FEM hint whenever non-negligible; only
        falls back to a physically-derived estimate (unloaded contact gap +
        Hertz deflection) when the hint is ~0.
        """
        if hint > 1e-6:
            return hint
        gap   = bearing.A * (1.0 - np.cos(bearing.alpha_0))
        hertz = (abs(Fr) / (bearing.cp * bearing.Z)) ** (2.0/3.0) if Fr != 0.0 else 1e-4
        return gap + hertz

    def _initial_delta_a(self, bearing: Bearing, Fa: float, hint: float) -> float:
        """Seed for delta_a, mirroring _initial_delta_r."""
        if abs(hint) > 1e-6:
            return hint
        if Fa == 0.0:
            return 0.0
        hertz = (abs(Fa) / (bearing.cp * bearing.Z)) ** (2.0/3.0)
        return np.copysign(hertz, Fa)

    # ------------------------------------------------------------------
    # Single-row/single-bearing solve -- THE SEAM.
    # ------------------------------------------------------------------

    def solve_contact(self,
                      bearing: Bearing,
                      Fr_xz: float, Fr_xy: float, Fa: float,
                      delta_r_init: float, delta_a_init: float,
                      psi: float, phi_Fr: float) -> BallLoadDistributionResult:
        """2-equation root (delta_r, delta_a) in the resultant-force plane, for
        ONE point-contact raceway.

        psi and phi_Fr are pre-computed by the caller -- this method only
        runs the root problem and assembles the result.

        Equilibrium (Sec 4.2.2.1):
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

        delta_j, alpha_j, _, sa, cp_j, d32 = self.elements(
            bearing, delta_r, delta_a, Vpsi)
        Mz = (bearing.Dpw / 2.0) * cp * float(np.sum(d32 * sa * cp_j))

        return BallLoadDistributionResult(
            delta_r=delta_r, delta_a=delta_a, psi=psi, phi_Fr=phi_Fr,
            delta_j=delta_j, alpha_j=alpha_j,
            Mz=Mz, n_iter=nfev, residual=res, ok=ok,
        )

    # ------------------------------------------------------------------
    # Minimum axial preload -- still solving (iterates solve_contact over
    # Fa), not post-processing an already-fixed result.
    # ------------------------------------------------------------------

    def minimum_axial_load(self,
                            bearing: Bearing,
                            Fr_xz: float, Fr_xy: float,
                            psi: float,
                            delta_r_init: float = 0.0,
                            delta_a_init: float = 0.0,
                            Fa_bracket: tuple[float, float] = (0.0, 5.0e4),
                            xtol: float = 1e-6) -> tuple[float, BallLoadDistributionResult]:
        """Minimum axial preload [N] such that delta_a >= 0 (contact closure).
        Uses brentq on delta_a(Fa). Raises ValueError if the upper bracket is
        too small to contain a zero crossing.

        Parameters
        ----------
        psi : misalignment in the resultant-force plane [rad] -- already projected
              (use psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr) if coming from FEM slopes)
        """
        check_bearing_ready(bearing, bearing.label, REQUIRED_ATTRS)
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


# ---------------------------------------------------------------------------
# DEBUG / VALIDATION UTILITY -- not used in production pipelines
# ---------------------------------------------------------------------------

def debug_radial_capacity(bearing: Bearing, Cr: float, i: int | None = None,
                          label: str = "") -> None:
    """Print Q_ci/Q_ce for a radial ball bearing -- manual cross-check only,
    sourced from bearing.family.per_element_dynamic_capacity(), so it can
    never silently drift from what production returns.
    """
    _lbl = label or bearing.label
    Q_ci, Q_ce = bearing.family.per_element_dynamic_capacity(bearing, Cr=Cr)

    print(f"\n  +-- Q_ci/Q_ce debug -- {_lbl} {'-'*30}+")
    print(f"  |  INPUT")
    print(f"  |    Cr          = {Cr:.2f} N")
    print(f"  |    Z           = {bearing.Z}")
    print(f"  |    i (rows)    = {i if i is not None else getattr(bearing, 'i', 1)}")
    print(f"  |    alpha_0     = {np.degrees(bearing.alpha_0):.6f} deg")
    print(f"  |    ri          = {bearing.ri:.6f} mm")
    print(f"  |    re          = {bearing.re:.6f} mm")
    print(f"  |    Dw          = {bearing.Dw:.6f} mm")
    print(f"  |    Dpw         = {bearing.Dpw:.6f} mm")
    print(f"  |    gamma       = {bearing.gamma:.8f}")
    print(f"  |  OUTPUT (via bearing.family.per_element_dynamic_capacity)")
    print(f"  |    Q_ci        = {Q_ci:.4f} N")
    print(f"  |    Q_ce        = {Q_ce:.4f} N")
    print(f"  +{'-'*58}+")