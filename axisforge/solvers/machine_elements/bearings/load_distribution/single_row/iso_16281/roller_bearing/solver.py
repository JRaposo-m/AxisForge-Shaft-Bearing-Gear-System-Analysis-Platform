"""axisforge/solvers/machine_elements/bearings/load_distribution/single_row/iso_16281/roller_bearing/solver.py

ISO/TS 16281 internal load distribution SOLVER for LINE-CONTACT radial
roller bearings (cylindrical, NU/N-type, zero nominal contact angle).
Implements the Sec 5.2 lamina model rather than the Sec 4.2 point-contact
model used for balls: each roller is sliced into n_s identical laminae
along its effective length L_we, and the load carried by each lamina is
derived from the LOCAL elastic approach at that lamina, corrected for the
roller's logarithmic profile (crowning) so a purely cylindrical roller's
theoretical edge-stress singularity does not appear in the model.

Scope of this file: only things that RUN the iterative solve live here --
ISO16281RollerSolver. The result shapes it hands back,
RollerLoadDistributionResult and RollerBearingResult, live in
contracts/bearing/roller_bearing_results.py -- pure data, consumed by
postprocessing.py. This file never imports postprocessing.py.

Capacity (Q_ci/Q_ce, per-lamina q_ci/q_ce) lives in core/ --
CylindricalRollerFamily / ThrustCylindricalRollerFamily /
ThrustNeedleRollerFamily each own their lambda_v and dispatch their own
formula via bearing.family.per_element_dynamic_capacity(bearing, ...).
debug_radial_capacity() below is a manual cross-check, sourced from the
same call.

Why psi is an INPUT here, not a second unknown solved jointly with delta_r
------------------------------------------------------------------------
ISO/TS 16281 Sec 5.2.4 poses eq.(45) (radial force balance) and eq.(46)
(moment balance) together as "the equation system ... solved by iteration",
which in the standard's own generic derivation leaves both delta_r and psi
as unknowns. In this codebase's architecture the shaft FEM already supplies
BOTH quantities eq.(46) would otherwise be used to find: the prescribed
ring misalignment and a moment reaction, exactly as ISO16281BallSolver
already treats psi as prescribed. So here: psi is taken from FEM (or from
psi_override, mirroring the ball solver's psi_input mechanism), eq.(45) is
solved for the single unknown delta_r, and eq.(46) is evaluated afterwards
as a DIAGNOSTIC output (LoadDistributionResult.Mz) -- usable to sanity-check
against the FEM's own M_xz/M_xy, not a solve constraint.

Scope limitation -- cylindrical only
------------------------------------
This module covers radial cylindrical roller bearings (CAPABILITY =
"line_contact", REQUIRED_ATTRS below). Tapered and spherical roller
bearings share the lamina mechanics but need an additional coordinate
transform (cone half-angle for tapered; crown/osculation for spherical) --
they would declare the same "line_contact" capability with a different
REQUIRED_ATTRS set, so dispatch.resolve_solver_cls() routes them to their
own solver once one exists, not to this one. Multi-row radial roller
bearings are out of scope (see contracts/bearing/roller_bearing_results.py).

References
----------
ISO/TS 16281:2008 Sec 5.2, eq.(34)-(46) (lamina model, static equilibrium)
              Sec 5.3.1.2, eq.(47)-(49) (rating life, radial roller bearing)
Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch. 6
"""
from __future__ import annotations

import warnings

import numpy as np

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem
from axisforge.results.fem_results.shaft_results import ShaftResults
from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.numerics import run_root
from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.validation import check_bearing_ready, warn_if_floating_loaded
from axisforge.solvers.machine_elements.bearings.load_distribution.single_row.iso_16281.dispatch import register_contact_solver
from axisforge.results.bearings.load_distribution.single_row.roller_bearing_results import (
    RollerLoadDistributionResult,
    RollerBearingResult,
)
from axisforge.config import SOLVER_TOLERANCE


# ---------------------------------------------------------------------------
# Lamina geometry constants -- ISO/TS 16281 Sec 5.2.2-.3
# ---------------------------------------------------------------------------

_MIN_LAMINAE = 30                # Sec 5.2.2 -- "the number of laminae shall be at least n_s = 30"

# x_k (lamina midpoints) and P_xk (Sec 6.2 roller profile, eq.42-44) are not
# computed by this solver -- they're precomputed on the bearing at assembly
# time by each subtype's own geometry setup, mirroring bearing.phi_j for
# ball bearings. This solver only reads them.


# ---------------------------------------------------------------------------
# ISO16281RollerSolver
# ---------------------------------------------------------------------------

REQUIRED_ATTRS = ("Z", "Dwe", "Lwe", "Dpw", "phi_j", "s", "n_s", "x_k", "cL", "cs", "alpha_0",
                  "P_xk")


@register_contact_solver
class ISO16281RollerSolver:
    """ISO/TS 16281 Sec 5.2 lamina-model internal load distribution solver,
    line contact (radial cylindrical roller bearings, zero nominal contact
    angle, no axial capacity).

    Takes bearing reactions and shaft slopes from a ShaftResults (already
    read from the shaft FEM registry by the caller), exactly like
    ISO16281BallSolver. The internal load distribution reduces to a single
    unknown, delta_r, solved from the radial force balance eq.(45); see
    the module docstring for why psi is treated as an input.

    Self-contained: does not import or depend on ISO16281BallSolver.
    Dispatched by the bearing study orchestrator via CAPABILITY +
    REQUIRED_ATTRS (dispatch.py) for any family that declares
    "line_contact".

    Parameters
    ----------
    tol       float   residual tolerance for scipy.optimize.root
    psi_input bool    False (default): psi is always projected from FEM shaft
                      slopes, i.e. psi = psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr).
                      True: per-bearing psi values supplied via psi_override in
                      solve() take precedence when present; any bearing not
                      covered by psi_override falls back to the FEM projection.
                      Same convention as ISO16281BallSolver.
    """

    CAPABILITY = "line_contact"
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
              ) -> dict[str, RollerBearingResult]:
        """Solve the internal load distribution for all line-contact roller
        bearings in `bearings`.

        Each bearing must already carry: Z, Dwe, Lwe, Dpw, phi_j, s, n_s,
        x_k (lamina midpoints, precomputed by the bearing's own geometry
        setup; n_s >= 30 and len(x_k) == n_s are checked at solve time, see
        _check_lamina_count()), cL, cs (eq.(35), eq.(37)), alpha_0.

        Parameters
        ----------
        shaft_system   : ShaftSystem -- fully resolved
        bearings       : {label: Bearing} -- all line-contact (cylindrical roller)
        shaft_results  : ShaftResults for shaft_system, already read by the
                         caller from its own registry (ShaftResultsReader output)
        psi_override   : {label: psi [rad]} -- see class docstring. Labels not
                         present here fall back to the FEM projection; labels
                         present here but not in `bearings` trigger a warning.

        Returns
        -------
        dict[str, RollerBearingResult] -- one per label in `bearings`, each
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

        results: dict[str, RollerBearingResult] = {}
        for label, b in bearings.items():
            check_bearing_ready(b, label, REQUIRED_ATTRS)
            self._check_lamina_count(b, label)
            node   = node_by_label[label]
            Fr_xz  = node.Fr_xz
            Fr_xy  = node.Fr_xy
            phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

            # Radial roller bearings (NU/N-type) carry no axial load by design.
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
                delta_r_init = float(np.hypot(node.v_xz, node.v_xy)),
                psi          = psi,
                phi_Fr       = phi_Fr,
            )
            results[label] = RollerBearingResult.single(row)

        return results

    @staticmethod
    def _check_lamina_count(bearing: Bearing, label: str) -> None:
        """Sec 5.2.2 requires at least _MIN_LAMINAE (30) laminae, and
        bearing.x_k must have one entry per bearing.n_s -- this solver never
        constructs x_k itself, so a mismatch here is a bearing-setup bug.
        """
        n_s = bearing.n_s
        if n_s < _MIN_LAMINAE:
            raise ValueError(
                f"Bearing '{label}': n_s = {n_s} < {_MIN_LAMINAE} -- "
                f"ISO/TS 16281 Sec 5.2.2 requires at least {_MIN_LAMINAE} laminae."
            )
        if len(bearing.x_k) != n_s:
            raise ValueError(
                f"Bearing '{label}': len(x_k) = {len(bearing.x_k)} != n_s = {n_s} -- "
                f"x_k must have exactly one lamina midpoint per n_s."
            )

    # ------------------------------------------------------------------
    # Element/lamina kinematics -- ISO/TS 16281 eq.(36)-(41), Sec 5.2.1-.4
    # Static (does not touch self) so it can be called either as
    # self.elements(...) or directly as ISO16281RollerSolver.elements(...).
    # ------------------------------------------------------------------

    @staticmethod
    def elements(bearing: Bearing, delta_r: float, psi: float):
        """Per-roller and per-lamina deflection/force for given delta_r, psi.

        Returns
        -------
        delta_j   ndarray(Z,)     roller-centreline deflection, eq.(38)
        psi_j     ndarray(Z,)     per-roller local misalignment, eq.(39)
        delta_jk  ndarray(Z,n_s)  per-lamina deflection, eq.(41) (>= 0)
        q_jk      ndarray(Z,n_s)  per-lamina contact force, eq.(36)
        cp_j      ndarray(Z,)     cos(phi_j), reused by the caller
        """
        phi_j = bearing.phi_j
        x_k   = bearing.x_k
        s     = bearing.s

        cp_j    = np.cos(phi_j)
        delta_j = delta_r * cp_j - s / 2.0                          # eq.(38)
        psi_j   = np.arctan(np.tan(psi) * cp_j)                     # eq.(39)

        P_xk = bearing.P_xk   # eq.(42)-(44) -- cached on the bearing (Sec 6.2 profile)

        delta_jk = (delta_j[:, None]
                    - x_k[None, :] * np.tan(psi_j)[:, None]
                    - 2.0 * P_xk[None, :])                          # eq.(41)
        delta_jk = np.maximum(delta_jk, 0.0)

        q_jk = bearing.cs * delta_jk ** (10.0 / 9.0)                # eq.(36)

        return delta_j, psi_j, delta_jk, q_jk, cp_j

    def _initial_delta_r(self, bearing: Bearing, Fr: float, hint: float) -> float:
        """Seed for delta_r. Trusts the FEM hint whenever non-negligible;
        otherwise falls back to a physically-derived estimate (clearance gap
        + a lumped-roller estimate from eq.(34), Q = cL*delta^(10/9)).
        """
        if hint > 1e-6:
            return hint
        gap = bearing.s / 2.0
        line_hertz = (abs(Fr) / (bearing.cL * bearing.Z)) ** (9.0 / 10.0) if Fr != 0.0 else 1e-4
        return gap + line_hertz

    # ------------------------------------------------------------------
    # Single-row/single-bearing solve -- THE SEAM.
    # ------------------------------------------------------------------

    def solve_contact(self,
                      bearing: Bearing,
                      Fr_xz: float, Fr_xy: float,
                      delta_r_init: float,
                      psi: float, phi_Fr: float) -> RollerLoadDistributionResult:
        """1-equation root (delta_r) in the resultant-force plane, psi
        prescribed. Mz is evaluated afterwards from the converged state as a
        diagnostic (eq.(46)) -- see module docstring.

        Equilibrium (Sec 5.2.4.1, eq.(45)):
            Fr = sum_j cos(phi_j) * sum_k q_j,k
        """
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

        # eq.(46) -- diagnostic, not a solve constraint (see module docstring)
        Mz = float(np.sum(cp_j * np.sum(x_k[None, :] * q_jk, axis=1)))

        alpha_j = np.full(bearing.Z, bearing.alpha_0, dtype=float)

        return RollerLoadDistributionResult(
            delta_r=delta_r, delta_a=0.0, psi=psi, phi_Fr=phi_Fr,
            delta_j=delta_j, alpha_j=alpha_j,
            Mz=Mz, n_iter=nfev, residual=res, ok=ok,
            x_k=x_k, psi_j=psi_j, delta_jk=delta_jk, q_jk=q_jk,
        )


# ---------------------------------------------------------------------------
# DEBUG / VALIDATION UTILITY -- not used in production pipelines
# ---------------------------------------------------------------------------

def debug_radial_capacity(bearing: Bearing, Cr: float, i: int | None = None,
                          lambda_v: float | None = None, label: str = "") -> None:
    """Print Q_ci/Q_ce/q_ci/q_ce for a radial roller bearing -- manual
    cross-check only, sourced from bearing.family.per_element_dynamic_capacity().

    lambda_v defaults to the bearing's own subtype value (e.g.
    CylindricalRollerFamily.LAMBDA_V_RADIAL) when not overridden. `i` is
    informational only in the printout -- it never changes the computed
    Q_ci/Q_ce.
    """
    _lbl = label or bearing.label
    Q_ci, Q_ce = bearing.family.per_element_dynamic_capacity(
        bearing, Cr=Cr, lambda_v=lambda_v,
    )
    q_ci, q_ce = bearing.family.per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce)

    print(f"\n  +-- Q_ci/Q_ce debug (roller) -- {_lbl} {'-'*20}+")
    print(f"  |  INPUT")
    print(f"  |    Cr          = {Cr:.2f} N")
    print(f"  |    Z           = {bearing.Z}")
    print(f"  |    i (rows)    = {i if i is not None else bearing.i}")
    print(f"  |    alpha_0     = {np.degrees(bearing.alpha_0):.6f} deg")
    print(f"  |    Dwe         = {bearing.Dwe:.6f} mm")
    print(f"  |    Dpw         = {bearing.Dpw:.6f} mm")
    print(f"  |    gamma       = {bearing.gamma:.8f}")
    print(f"  |  OUTPUT (via bearing.family.per_element_dynamic_capacity)")
    print(f"  |    Q_ci        = {Q_ci:.4f} N")
    print(f"  |    Q_ce        = {Q_ce:.4f} N")
    print(f"  |  LAMINA (eq.56-57, n_s={bearing.n_s})")
    print(f"  |    q_ci        = {q_ci:.4f} N")
    print(f"  |    q_ce        = {q_ce:.4f} N")
    print(f"  +{'-'*58}+")