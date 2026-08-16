"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/roller_bearing.py

ISO/TS 16281 internal load distribution SOLVER for LINE-CONTACT radial
roller bearings (BearingType.CYLINDRICAL_ROLLER — NU/N-type, zero nominal
contact angle). Implements the §5.2 lamina model rather than the §4.2
point-contact model used for balls: each roller is sliced into n_s
identical laminae along its effective length L_we, and the load carried by
each lamina is derived from the LOCAL elastic approach at that lamina,
corrected for the roller's logarithmic profile (crowning) so a purely
cylindrical roller's theoretical edge-stress singularity does not appear
in the model. See ISO/TS 16281:2008 §5.2, eq.(34)-(46).

Scope of this file: only things that PRODUCE a LoadDistributionResult (or a
capacity value from bearing geometry alone) live here — mirrors the split
used by Ball_Bearing/ball_bearing.py. Anything that CONSUMES
an already-computed result to derive a further quantity (per-roller contact
force, contact/lamina-pressure distributions, secant stiffness, dynamic
equivalent load) lives in roller_bearing_postprocessing.py instead.
That file imports from this one only for typing and for the
RollerLoadDistributionResult class; this file never imports from it.

Why psi is an INPUT here, not a second unknown solved jointly with delta_r
------------------------------------------------------------------------
ISO/TS 16281 §5.2.4 poses eq.(45) (radial force balance) and eq.(46)
(moment balance) together as "the equation system ... solved by iteration",
which in the standard's own generic derivation leaves both delta_r and psi
as unknowns. In this codebase's architecture the shaft FEM already supplies
BOTH quantities eq.(46) would otherwise be used to find: the prescribed
ring misalignment (BearingNodeData.psi_xz/psi_xy — the shaft centreline
slope across the bearing seat, exactly the input ISO16281BallSolver already
treats as prescribed) and a moment reaction (BearingNodeData.M_xz/M_xy).
Mirroring ISO16281BallSolver's treatment of psi keeps both per-type solvers
consistent and keeps the shaft <-> bearing coupling one-directional (FEM
drives the bearing solve, not the reverse). So here: psi is taken from FEM
(or from psi_override, exactly like the ball solver's psi_input mechanism),
eq.(45) is solved for the single unknown delta_r, and eq.(46) is evaluated
afterwards as a DIAGNOSTIC output (LoadDistributionResult.Mz) — usable to
sanity-check against the FEM's own M_xz/M_xy, exactly as the ball solver's
Mz field is a byproduct, not a constraint. If a future case genuinely needs
psi solved jointly with delta_r (e.g. an isolated bearing test rig with a
prescribed external moment rather than a coupled shaft), that is a distinct
mode this file does not implement — flag it if it comes up rather than
silently repurposing this solver for it.

Scope limitation — cylindrical only
------------------------------------
This module covers BearingType.CYLINDRICAL_ROLLER only. Tapered and
spherical roller bearings share the lamina mechanics but need an additional
coordinate transform (cone half-angle for tapered; crown/osculation for
spherical) this module does not implement — they are NOT wired into
rolling_bearing_solver.RollingBearingSolver's dispatch table yet.

References
----------
ISO/TS 16281:2008 §5.2, eq.(34)-(46) (lamina model, static equilibrium)
              §5.3.1.2, eq.(47)-(49) (rating life, radial roller bearing)
Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch. 6
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    SimpleFEMResultsLibrary,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    LoadDistributionResult,
    check_bearing_ready,
    warn_if_floating_loaded,
    run_root,
)
from axisforge.config import SOLVER_TOLERANCE


# ---------------------------------------------------------------------------
# Lamina geometry constants — ISO/TS 16281 §5.2.2-.3
# ---------------------------------------------------------------------------

_MIN_LAMINAE = 30                # §5.2.2 — "the number of laminae shall be at least n_s = 30"
_LOG_ARG_EPS = 1e-12             # floor for the log() argument in the profile function

# x_k (the lamina midpoints) is NOT computed by this solver — it is one of
# _REQUIRED_ATTRS below, precomputed by the bearing's own geometry setup
# (mirrors bearing.phi_j being precomputed for ball bearings) and checked
# for §5.2.2's n_s >= 30 minimum by _check_lamina_count() at solve time.
# roller_profile() (eq.42-44) DOES live as a static method on
# ISO16281RollerSolver below, since it is genuinely internal to the solve
# — _elements() calls it every iteration — with no other consumer.


# ---------------------------------------------------------------------------
# RollerElementCapacity — ISO/TS 16281 §5.3.1.2, eq.(47)-(49)
# ---------------------------------------------------------------------------

_LAMBDA_V_RADIAL = 0.83   # eq.(49)
_LAMBDA_V_TRUST  = 0.73 # eq.(52)    


@dataclass(frozen=True)
class RollerElementCapacity:
    """
    Per-roller dynamic load capacity Q_ci (inner) / Q_ce (outer) for a
    radial roller bearing — ISO/TS 16281 §5.3.1.2, eq.(47)-(48) — plus the
    per-LAMINA dynamic load rating q_ci / q_ce, §5.3.2, eq.(56)-(57):

        q_ci = Q_ci * (1/n_s)^(7/9)
        q_ce = Q_ce * (1/n_s)^(7/9)

    q_ci/q_ce are what a lamina's dynamic equivalent load (see
    roller_bearing_postprocessing.LaminaDynamicEquivalentLoad, eq.61-64)
    is meant to be compared against — Q_ci/Q_ce are the whole-roller
    figures and are not the right denominator for a per-lamina check.

    Line contact only. Cr is supplied externally (ISO 281 catalogue); this
    module does not resolve it. Requires bearing.n_s for q_ci/q_ce (same
    n_s _check_lamina_count() already validates for the solve).

    Attributes
    ----------
    label     str
    Q_ci      float [N]        eq.(47), whole roller, inner raceway
    Q_ce      float [N]        eq.(48), whole roller, outer raceway
    q_ci      float [N]        eq.(56), per lamina,   inner raceway
    q_ce      float [N]        eq.(57), per lamina,   outer raceway
    Cr        float [N]
    i         int     number of roller rows
    lambda_v  float   stress concentration factor, eq.(49)
    """
    label    : str
    Q_ci     : float
    Q_ce     : float
    q_ci     : float
    q_ce     : float
    Cr       : float
    i        : int
    lambda_v : float

    @classmethod
    def per_lamina(cls, bearing: Bearing, Q_ci: float, Q_ce: float):

        n_s  = bearing.n_s
        q_ci = Q_ci * (1.0 / n_s) ** (7.0 / 9.0)   # eq.(56)
        q_ce = Q_ce * (1.0 / n_s) ** (7.0 / 9.0)   # eq.(57)

        return cls(q_ci=q_ci, q_ce=q_ce)

    @classmethod
    def radial(cls, bearing: Bearing, Cr: float, i: int = 1,
               lambda_v: float = _LAMBDA_V_RADIAL,
               label: str = "") -> "RollerElementCapacity":
        """
        Radial roller bearing capacity — eq.(47)-(48), plus per-lamina
        eq.(56)-(57). Requires bearing.Dwe, bearing.Dpw, bearing.Z,
        bearing.alpha_0, bearing.n_s.
        """
        Z, alpha = bearing.Z, bearing.alpha_0
        Dwe, Dpw = bearing.Dwe, bearing.Dpw
        gamma = Dwe * np.cos(alpha) / Dpw

        _lbl = label or bearing.label
        if not (0.0 < gamma < 1.0):
            raise ValueError(
                f"Bearing '{_lbl}': gamma = Dwe*cos(alpha)/Dpw = {gamma:.6f} "
                f"is outside (0, 1) — check Dwe/Dpw/alpha_0."
            )

        base    = 1.038 * ((1.0 - gamma) / (1.0 + gamma)) ** (143.0 / 108.0)
        denom_a = np.cos(alpha) * (i ** (7.0 / 9.0))

        Q_ci = (1.0 / lambda_v) * (Cr / (0.378 * Z * denom_a)) * (1.0 + base ** (9.0 / 2.0)) ** (2.0 / 9.0)
        Q_ce = (1.0 / lambda_v) * (Cr / (0.364 * Z * denom_a)) * (1.0 + base ** (-9.0 / 2.0)) ** (2.0 / 9.0)

        q_ci, q_ce = cls.per_lamina(bearing, Q_ci, Q_ce)

        return cls(label=_lbl, Q_ci=Q_ci, Q_ce=Q_ce, q_ci=q_ci, q_ce=q_ce,
                   Cr=Cr, i=i, lambda_v=lambda_v)

    @classmethod
    def thrust_nonzero_alpha(cls, bearing: Bearing, Ca: float,
               lambda_v: float = _LAMBDA_V_TRUST,
               label: str = "") -> "RollerElementCapacity":

        Z, alpha = bearing.Z, bearing.alpha_0
        Dwe, Dpw = bearing.Dwe, bearing.Dpw
        gamma = Dwe * np.cos(alpha) / Dpw

        _lbl = label or bearing.label
        if not (0.0 < gamma < 1.0):
            raise ValueError(
                f"Bearing '{_lbl}': gamma = Dwe*cos(alpha)/Dpw = {gamma:.6f} "
                f"is outside (0, 1) — check Dwe/Dpw/alpha_0."
            )

        base = ((1.0 - gamma) / (1.0 + gamma)) ** (143.0 / 108.0)
        denom_a = Z * np.sin(alpha)

        Q_ci = 1.0 / lambda_v * (Ca / denom_a) * (1.0 + base ** (9.0 / 2.0)) ** (2.0 / 9.0)
        Q_ce = 1.0 / lambda_v * (Ca / denom_a) * (1.0 + base ** (-9.0 / 2.0)) ** (2.0 / 9.0)

        q_ci, q_ce = cls.per_lamina(bearing, Q_ci, Q_ce)

        return cls(label=_lbl, Q_ci=Q_ci, Q_ce=Q_ce, q_ci=q_ci, q_ce=q_ce,
                   Ca=Ca, lambda_v=lambda_v)

    @classmethod
    def thrust_90deg(cls, bearing: Bearing, Ca: float,
               lambda_v: float = _LAMBDA_V_TRUST,
               label: str = "") -> "RollerElementCapacity":

        Z, alpha = bearing.Z, bearing.alpha_0
        Dwe, Dpw = bearing.Dwe, bearing.Dpw
        gamma = Dwe * np.cos(alpha) / Dpw

        _lbl = label or bearing.label
        if not (0.0 < gamma < 1.0):
            raise ValueError(
                f"Bearing '{_lbl}': gamma = Dwe*cos(alpha)/Dpw = {gamma:.6f} "
                f"is outside (0, 1) — check Dwe/Dpw/alpha_0."
            )

        Q_ci = 1.0 / lambda_v * (Ca / Z) * 2 ** (2.0 / 9.0)
        Q_ce = 1.0 / lambda_v * (Ca / Z) * 2 ** (2.0 / 9.0)

        q_ci, q_ce = cls.per_lamina(bearing, Q_ci, Q_ce)

        return cls(label=_lbl, Q_ci=Q_ci, Q_ce=Q_ce, q_ci=q_ci, q_ce=q_ce,
                   Ca=Ca, lambda_v=lambda_v)

    
# ---------------------------------------------------------------------------
# RollerLoadDistributionResult — LoadDistributionResult + per-lamina data
# ---------------------------------------------------------------------------

class RollerLoadDistributionResult(LoadDistributionResult):
    """
    LoadDistributionResult extended with the per-lamina data the §5.2 lamina
    model produces, which the shared (Z,)-shaped delta_j/alpha_j slots in
    library.py cannot hold (that class is deliberately kept free of any
    contact-type-specific physics — see its module docstring). Defined here
    rather than in library.py so line-contact-specific shape stays local to
    this contact type.

    Base-class fields keep their documented meaning, adapted for line
    contact:
      delta_r  : radial ring displacement [mm] — the one unknown solved
      delta_a  : always 0.0 — radial roller bearings (NU/N-type) carry no
                 axial load
      delta_j  : roller-centreline deflection per roller (Z,) [mm],
                 eq.(38), BEFORE the lamina/profile correction — the
                 per-lamina deflection is delta_jk, not this
      alpha_j  : bearing.alpha_0 broadcast to (Z,) — for a cylindrical
                 roller the contact normal stays radial regardless of tilt,
                 unlike a ball's alpha_j, which genuinely varies with load
      Mz       : diagnostic reaction moment, eq.(46), evaluated at the
                 converged (delta_r, psi) — NOT a solve constraint here,
                 see module docstring

    Extra fields (this subclass only)
    ----------------------------------
    x_k       ndarray(n_s,) [mm]     lamina positions, eq.(38)-figure 3
    psi_j     ndarray(Z,)   [rad]    per-roller local misalignment, eq.(39)
    delta_jk  ndarray(Z,n_s)[mm]     per-lamina elastic deflection, eq.(41)
    q_jk      ndarray(Z,n_s)[N]      per-lamina contact force, eq.(36)
    """
    __slots__ = ("x_k", "psi_j", "delta_jk", "q_jk")

    def __init__(self, *, x_k, psi_j, delta_jk, q_jk, **kwargs):
        super().__init__(**kwargs)
        self.x_k      = x_k
        self.psi_j    = psi_j
        self.delta_jk = delta_jk
        self.q_jk     = q_jk


# ---------------------------------------------------------------------------
# ISO16281RollerSolver
# ---------------------------------------------------------------------------

_REQUIRED_ATTRS = ("Z", "Dwe", "Lwe", "Dpw", "phi_j", "s", "n_s", "x_k", "cL", "cs", "alpha_0")


class ISO16281RollerSolver:
    """
    ISO/TS 16281 §5.2 lamina-model internal load distribution solver, line
    contact (radial cylindrical roller bearings — BearingType.CYLINDRICAL_
    ROLLER, zero nominal contact angle, no axial capacity).

    Takes bearing reactions and shaft slopes from a SimpleFEMResultsLibrary
    (pre-populated by ShaftResultsReader), exactly like ISO16281BallSolver.
    The internal load distribution reduces to a single unknown, delta_r,
    solved from the radial force balance eq.(45); see the module docstring
    for why psi is treated as an input rather than a second unknown.

    Self-contained: does not import or depend on ISO16281BallSolver or any
    other contact-type solver. rolling_bearing_solver.RollingBearingSolver
    dispatches to this class for BearingType.CYLINDRICAL_ROLLER; it can also
    be used directly when the caller already knows all bearings on a shaft
    are this type.

    Parameters
    ----------
    tol       float   residual tolerance for scipy.optimize.root
    psi_input bool    False (default): psi is always projected from FEM shaft
                      slopes, i.e. psi = psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr).
                      True: per-bearing psi values supplied via psi_override in
                      solve() take precedence when present; any bearing not
                      covered by psi_override falls back to the FEM projection.
                      Same convention as ISO16281BallSolver — see its docstring.
    """

    def __init__(self, tol: float = SOLVER_TOLERANCE, psi_input: bool = False):
        self.tol       = tol
        self.psi_input = psi_input

    # ------------------------------------------------------------------
    # Lamina profile — ISO/TS 16281 §5.2.3, eq.(42)-(44)
    # ------------------------------------------------------------------

    @staticmethod
    def roller_profile(x_k: np.ndarray, Dwe: float, Lwe: float) -> np.ndarray:
        """
        Roller profile function P(x_k) [mm] — ISO/TS 16281 §5.2.3, eq.(42)-(44).

        This is the crowning DEPTH; eq.(41) subtracts 2*P(x_k) from the raw
        lamina deflection so a purely cylindrical roller's theoretical edge
        stress singularity (from loading a truly flat-ended cylinder) does
        not appear in the model.

        x_k        : ndarray [mm] — bearing.x_k, the lamina midpoints the
                     bearing's own geometry setup already computed (this
                     solver does not construct x_k itself — see
                     _REQUIRED_ATTRS / _check_lamina_count()). Must lie
                     strictly within (-Lwe/2, Lwe/2), which a midpoint
                     construction guarantees by never landing exactly on
                     the roller ends.
        Dwe, Lwe   : effective roller diameter / length [mm]

        Two regimes, per the standard:
          Lwe <= 2.5*Dwe : full-length logarithmic crown, eq.(42)
          Lwe >  2.5*Dwe : flat centre + logarithmic crown only near the
                           ends, eq.(43)-(44) (stepwise)

        NOTE (per the standard): these are reference geometries giving
        approximate values — actual manufacturer roller profiles, based on
        the manufacturer's own expertise, can deviate significantly. Swap
        this method for a manufacturer-supplied profile if better data
        exists.
        """
        x_k = np.asarray(x_k, dtype=float)
        P = np.zeros_like(x_k)

        if Lwe <= 2.5 * Dwe:
            arg = 1.0 - (2.0 * x_k / Lwe) ** 2
            arg = np.maximum(arg, _LOG_ARG_EPS)
            P = 0.000350 * Dwe * np.log(1.0 / arg)
        else:
            half_flat = (Lwe - 2.5 * Dwe) / 2.0
            edge = np.abs(x_k) > half_flat
            if np.any(edge):
                xe = x_k[edge]
                arg = 1.0 - ((2.0 * np.abs(xe) - (Lwe - 2.5 * Dwe)) / (2.5 * Dwe)) ** 2
                arg = np.maximum(arg, _LOG_ARG_EPS)
                P[edge] = 0.000500 * Dwe * np.log(1.0 / arg)
            # flat centre region (|x_k| <= half_flat) stays 0.0 — eq.(43)

        return P

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              library: SimpleFEMResultsLibrary,
              psi_override: dict[str, float] | None = None,
              ) -> dict[str, RollerLoadDistributionResult]:
        """
        Solve the internal load distribution for all line-contact roller
        bearings in `bearings`.

        Each bearing must already carry: Z, Dwe, Lwe, Dpw, phi_j, s, n_s,
        x_k (lamina midpoints, precomputed by the bearing's own geometry
        setup — mirrors bearing.phi_j being precomputed for ball bearings;
        n_s >= 30 and len(x_k) == n_s are checked at solve time, see
        _check_lamina_count()), cL, cs (eq.(35), eq.(37)), alpha_0.

        Parameters
        ----------
        shaft_system  : ShaftSystem — fully resolved
        bearings      : {label: Bearing} — all line-contact (cylindrical roller)
        library       : SimpleFEMResultsLibrary — must contain results for
                        shaft_system.name (populated by ShaftResultsReader.read())
        psi_override  : {label: psi [rad]} — see class docstring. Labels not
                        present here fall back to the FEM projection; labels
                        present here but not in `bearings` trigger a warning.

        Returns
        -------
        {label: RollerLoadDistributionResult}
        """
        if self.psi_input and psi_override:
            unknown = set(psi_override) - set(bearings)
            if unknown:
                warnings.warn(
                    f"psi_override has labels not in `bearings`: {sorted(unknown)} — "
                    f"check for typos; these entries are ignored."
                )

        shaft_results = library.get(shaft_system.name)
        node_by_label = {n.label: n for n in shaft_results.bearing_nodes}

        results: dict[str, RollerLoadDistributionResult] = {}
        for label, b in bearings.items():
            check_bearing_ready(b, label, _REQUIRED_ATTRS)
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

            results[label] = self._solve_bearing(
                b,
                Fr_xz        = Fr_xz,
                Fr_xy        = Fr_xy,
                delta_r_init = float(np.hypot(node.v_xz, node.v_xy)),
                psi          = psi,
                phi_Fr       = phi_Fr,
            )

        return results

    @staticmethod
    def _check_lamina_count(bearing: Bearing, label: str) -> None:
        """
        §5.2.2 requires at least _MIN_LAMINAE (30) laminae, and bearing.x_k
        (whatever precomputed it) must actually have one entry per bearing.n_s
        — this solver never constructs x_k itself, so a mismatch here is a
        bearing-setup bug, not something _elements() can catch cleanly deep
        inside a root-finding call.
        """
        n_s = bearing.n_s
        if n_s < _MIN_LAMINAE:
            raise ValueError(
                f"Bearing '{label}': n_s = {n_s} < {_MIN_LAMINAE} — "
                f"ISO/TS 16281 §5.2.2 requires at least {_MIN_LAMINAE} laminae."
            )
        if len(bearing.x_k) != n_s:
            raise ValueError(
                f"Bearing '{label}': len(x_k) = {len(bearing.x_k)} != n_s = {n_s} — "
                f"x_k must have exactly one lamina midpoint per n_s."
            )

    # ------------------------------------------------------------------
    # Element/lamina kinematics — ISO/TS 16281 eq.(36)-(41), §5.2.1-.4
    # ------------------------------------------------------------------

    def _elements(self, bearing: Bearing, delta_r: float, psi: float):
        """
        Per-roller and per-lamina deflection/force for given delta_r, psi.

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

        P_xk = self.roller_profile(x_k, bearing.Dwe, bearing.Lwe)   # eq.(42)-(44)

        delta_jk = (delta_j[:, None]
                    - x_k[None, :] * np.tan(psi_j)[:, None]
                    - 2.0 * P_xk[None, :])                          # eq.(41)
        delta_jk = np.maximum(delta_jk, 0.0)

        q_jk = bearing.cs * delta_jk ** (10.0 / 9.0)                # eq.(36)

        return delta_j, psi_j, delta_jk, q_jk, cp_j

    def _initial_delta_r(self, bearing: Bearing, Fr: float, hint: float) -> float:
        """
        Seed for delta_r. Trusts the FEM hint whenever it is non-negligible;
        otherwise falls back to a physically-derived estimate (clearance gap
        + a lumped-roller estimate from eq.(34), Q = cL*delta^(10/9)).
        """
        if hint > 1e-6:
            return hint
        gap = bearing.s / 2.0
        line_hertz = (abs(Fr) / (bearing.cL * bearing.Z)) ** (9.0 / 10.0) if Fr != 0.0 else 1e-4
        return gap + line_hertz

    # ------------------------------------------------------------------
    # Single-bearing solve
    # ------------------------------------------------------------------

    def _solve_bearing(self,
                       bearing: Bearing,
                       Fr_xz: float, Fr_xy: float,
                       delta_r_init: float,
                       psi: float, phi_Fr: float) -> RollerLoadDistributionResult:
        """
        1-equation root (delta_r) in the resultant-force plane, psi
        prescribed. Mz is evaluated afterwards from the converged state as a
        diagnostic (eq.(46)) — see module docstring.

        Equilibrium (§5.2.4.1, eq.(45)):
            Fr = sum_j cos(phi_j) * sum_k q_j,k
        """
        Fr = float(np.hypot(Fr_xz, Fr_xy))

        def residual(u):
            dr = u[0]
            _, _, _, q_jk, cp_j = self._elements(bearing, dr, psi)
            Fr_internal = float(np.sum(cp_j * np.sum(q_jk, axis=1)))
            return np.array([Fr - Fr_internal])

        dr0              = self._initial_delta_r(bearing, Fr, delta_r_init)
        x, nfev, res, ok = run_root(residual, [dr0], self.tol)
        delta_r          = float(x[0])

        delta_j, psi_j, delta_jk, q_jk, cp_j = self._elements(bearing, delta_r, psi)
        x_k = bearing.x_k

        # eq.(46) — diagnostic, not a solve constraint (see module docstring)
        Mz = float(np.sum(cp_j * np.sum(x_k[None, :] * q_jk, axis=1)))

        alpha_j = np.full(bearing.Z, bearing.alpha_0, dtype=float)

        return RollerLoadDistributionResult(
            delta_r=delta_r, delta_a=0.0, psi=psi, phi_Fr=phi_Fr,
            delta_j=delta_j, alpha_j=alpha_j,
            Mz=Mz, n_iter=nfev, residual=res, ok=ok,
            x_k=x_k, psi_j=psi_j, delta_jk=delta_jk, q_jk=q_jk,
        )


# ---------------------------------------------------------------------------
# DEBUG / VALIDATION UTILITIES — not used in production pipelines
# ---------------------------------------------------------------------------

def debug_radial_capacity(bearing: Bearing, Cr: float, i: int = 1,
                          lambda_v: float = _LAMBDA_V_RADIAL, label: str = "") -> None:
    """
    Print all intermediate values for Q_ci / Q_ce (eq.47-49) — manual
    cross-check only. ASCII only, per project console-output convention.
    """
    Z, alpha = bearing.Z, bearing.alpha_0
    Dwe, Dpw = bearing.Dwe, bearing.Dpw
    gamma    = Dwe * np.cos(alpha) / Dpw
    base     = 1.038 * ((1.0 - gamma) / (1.0 + gamma)) ** (143.0 / 108.0)
    denom_a  = np.cos(alpha) * (i ** (7.0 / 9.0))
    inner_term = base ** (9.0 / 2.0)
    outer_term = base ** (-9.0 / 2.0)
    denom_ci = 0.378 * Z * denom_a
    denom_ce = 0.364 * Z * denom_a
    factor_ci = (1.0 + inner_term) ** (2.0 / 9.0)
    factor_ce = (1.0 + outer_term) ** (2.0 / 9.0)
    Q_ci = (1.0 / lambda_v) * (Cr / denom_ci) * factor_ci
    Q_ce = (1.0 / lambda_v) * (Cr / denom_ce) * factor_ce
    _lbl = label or bearing.label

    print(f"\n  +-- Q_ci/Q_ce debug (roller) -- {_lbl} {'-'*20}+")
    print(f"  |  INPUT")
    print(f"  |    Cr          = {Cr:.2f} N")
    print(f"  |    Z           = {Z}")
    print(f"  |    i (rows)    = {i}")
    print(f"  |    lambda_v    = {lambda_v:.4f}")
    print(f"  |    alpha_0     = {np.degrees(alpha):.6f} deg")
    print(f"  |    Dwe         = {Dwe:.6f} mm")
    print(f"  |    Dpw         = {Dpw:.6f} mm")
    print(f"  |  INTERMEDIATE")
    print(f"  |    gamma           = {gamma:.8f}   [Dwe*cos(alpha)/Dpw]")
    print(f"  |    base            = {base:.8f}   [1.038*((1-g)/(1+g))^(143/108)]")
    print(f"  |    base^(+9/2)     = {inner_term:.8f}   -> Q_ci eq.(47)")
    print(f"  |    base^(-9/2)     = {outer_term:.8f}   -> Q_ce eq.(48)")
    print(f"  |  CAPACITY")
    print(f"  |    denom_ci  = 0.378*{Z}*{denom_a:.6f} = {denom_ci:.6f}")
    print(f"  |    denom_ce  = 0.364*{Z}*{denom_a:.6f} = {denom_ce:.6f}")
    print(f"  |    factor_ci = (1 + {inner_term:.6f})^(2/9) = {factor_ci:.8f}")
    print(f"  |    factor_ce = (1 + {outer_term:.6f})^(2/9) = {factor_ce:.8f}")
    print(f"  |    Q_ci      = (1/{lambda_v:.4f})*{Cr:.2f}/{denom_ci:.4f} * {factor_ci:.6f} = {Q_ci:.4f} N")
    print(f"  |    Q_ce      = (1/{lambda_v:.4f})*{Cr:.2f}/{denom_ce:.4f} * {factor_ce:.6f} = {Q_ce:.4f} N")

    n_s  = bearing.n_s
    q_ci = Q_ci * (1.0 / n_s) ** (7.0 / 9.0)
    q_ce = Q_ce * (1.0 / n_s) ** (7.0 / 9.0)
    print(f"  |  LAMINA (eq.56-57, n_s={n_s})")
    print(f"  |    q_ci = Q_ci*(1/{n_s})^(7/9) = {q_ci:.4f} N")
    print(f"  |    q_ce = Q_ce*(1/{n_s})^(7/9) = {q_ce:.4f} N")
    print(f"  +{'-'*58}+")