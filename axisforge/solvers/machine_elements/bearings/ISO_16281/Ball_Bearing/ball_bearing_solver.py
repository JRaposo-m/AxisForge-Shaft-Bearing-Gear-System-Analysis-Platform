"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_solver.py

ISO/TS 16281 internal load distribution SOLVER for POINT-CONTACT bearings
(deep groove and angular contact ball bearings — BearingType.DEEP_GROOVE_BALL
and BearingType.ANGULAR_CONTACT). alpha_0 is a bearing attribute, not
hardcoded, so both families are covered by the same kinematics.

Scope of this file: only things that PRODUCE a BallLoadDistributionResult
(or a capacity value from bearing geometry alone) live here. Anything that
CONSUMES an already-computed BallLoadDistributionResult / BallBearingStiffness
to derive a further quantity — contact force distribution, ball positions,
secant stiffness, dynamic equivalent load — lives in
ball_bearing_postprocessing.py instead. That file imports from this
one only for typing; this file never imports from it.

solve() returns a BallLoadDistributionLibrary (ball_bearing_results.py) —
a LOCAL, single-type registry — not a bare dict. This mirrors
ISO16281RollerSolver.solve() returning a RollerLoadDistributionLibrary on
the roller side. rolling_bearing_solver.RollingBearingSolver.solve() is
what reads labels back out of it and merges it with any other per-type
local library; this file never touches the final, cross-type
BearingResultsLibrary in the global library.py.

This module is one of several per-bearing-type solvers dispatched by
rolling_bearing_solver.RollingBearingSolver (see BearingType in
bearing_types.py). It lives in its own Ball_Bearing/ subfolder, sibling to
Roller_Bearing/ — it does not import, and is not imported by, any other
contact-type solver. The two are independent and only share the neutral
data contract, generic utilities, and results registry one level up, in
bearings/ISO_16281/library.py.

The solve is performed in the plane of the resultant radial force:

    phi_Fr = arctan2(Fr_xy, Fr_xz)
    Fr     = sqrt(Fr_xz^2 + Fr_xy^2)
    psi    = psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr)   [prescribed misalignment]

A 2-equation root (delta_r, delta_a) enforces static equilibrium (§4.2.2.1),
using the point-contact load-deflection law Q_j = cp * delta_j^1.5. Ball
positions in the global frame (output/plots only):

    phi_j_global = (bearing.phi_j + phi_Fr) % 2*pi

Inputs (Fr, Fa, psi_xz, psi_xy) come from a SimpleFEMResultsLibrary populated
by ShaftResultsReader. The bearing solve is performed once per load case —
the resulting stiffness values are outputs, not fed back into FEM.

References
----------
ISO/TS 16281:2008 §4.2, eq.(12)-(15); §4.3.1, eq.(19)-(28)
Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch. 6
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    SimpleFEMResultsLibrary,
    BearingNodeData,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.library import (
    check_bearing_ready,
    warn_if_floating_loaded,
    run_root,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing.ball_bearing_results import (
    BallLoadDistributionResult,
    BallLoadDistributionLibrary,
)
from axisforge.config import SOLVER_TOLERANCE


# ---------------------------------------------------------------------------
# Geometry helpers — ISO/TS 16281 §4.3.1 (point-contact groove geometry)
# ---------------------------------------------------------------------------

def _geometry_bracket(gamma: float, ri: float, re: float, Dw: float) -> float:
    """
    Groove geometry factor for Q_ci / Q_ce (§4.3.1.2-.3):

        1.044 * ((1-gamma)/(1+gamma))^1.72 * (ri/re * (2re-Dw)/(2ri-Dw))^0.41
    """
    radii_ratio = (ri / re) * ((2.0 * re - Dw) / (2.0 * ri - Dw))
    return 1.044 * ((1.0 - gamma) / (1.0 + gamma)) ** 1.72 * radii_ratio ** 0.41


def _check_geometry(ri: float, re: float, Dw: float, label: str) -> None:
    if 2.0 * ri <= Dw:
        raise ValueError(
            f"Bearing '{label}': 2*ri ({2*ri:.4f}) <= Dw ({Dw:.4f}) — "
            f"inner groove radius too small."
        )
    if 2.0 * re <= Dw:
        raise ValueError(
            f"Bearing '{label}': 2*re ({2*re:.4f}) <= Dw ({Dw:.4f}) — "
            f"outer groove radius too small."
        )


# ---------------------------------------------------------------------------
# RollingElementCapacity — ISO/TS 16281 §4.3.1 (point contact)
#
# Stays here, not in postprocessing: depends only on bearing geometry + a
# catalogue Cr/Ca, never on a LoadDistributionResult.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RollingElementCapacity:
    """
    Per-element dynamic load capacity Q_ci (inner) / Q_ce (outer).

    ISO/TS 16281 §4.3.1.2  radial ball bearings      eq.(19)-(20)
                 §4.3.1.3  thrust ball, alpha != 90°  eq.(21)-(22)
                 §4.3.1.4  thrust ball, alpha = 90°   eq.(23)-(24)

    Point-contact only. Cr / Ca are supplied externally (ISO 281 catalogue);
    this module does not resolve them.

    Attributes
    ----------
    label         str
    Q_ci          float [N]
    Q_ce          float [N]
    bearing_class str    "radial" | "thrust_nonzero_alpha" | "thrust_90deg"
    Cr            float | None  [N]
    Ca            float | None  [N]
    """
    label         : str
    Q_ci          : float
    Q_ce          : float
    bearing_class : str
    Cr            : float | None
    Ca            : float | None
    i             : int = 1

    @classmethod
    def radial(cls, bearing: Bearing, Cr: float, i: int = 1,
            label: str = "") -> "RollingElementCapacity":
        """
        Radial ball bearing capacity — §4.3.1.2 eq.(19)-(20).
        Requires bearing.setup_internal_geometry() already called.
        """
        Z, alpha = bearing.Z, bearing.alpha_0
        ri, re, Dw = bearing.ri, bearing.re, bearing.Dw
        gamma = bearing.gamma

        _lbl = label or bearing.label
        _check_geometry(ri, re, Dw, _lbl)

        bracket      = _geometry_bracket(gamma, ri, re, Dw)
        cos_alpha_07 = np.cos(alpha) * i ** 0.7

        Q_ci = (Cr / (0.407 * Z * cos_alpha_07)) * (1.0 + bracket ** (10.0/3.0)) ** 0.3
        Q_ce = (Cr / (0.389 * Z * cos_alpha_07)) * (1.0 + bracket ** (-10.0/3.0)) ** 0.3

        return cls(label=_lbl, Q_ci=Q_ci, Q_ce=Q_ce, i=i,
                bearing_class="radial", Cr=Cr, Ca=None)

    @classmethod
    def thrust_nonzero_alpha(cls, bearing: Bearing, Ca: float,
                            label: str = "") -> "RollingElementCapacity":
        """Thrust ball bearing (alpha != 90°) — §4.3.1.3 eq.(21)-(22)."""
        Z, alpha = bearing.Z, bearing.alpha_0
        ri, re, Dw = bearing.ri, bearing.re, bearing.Dw
        gamma = bearing.gamma

        _lbl = label or bearing.label
        _check_geometry(ri, re, Dw, _lbl)

        bracket = _geometry_bracket(gamma, ri, re, Dw)
        sin_a   = np.sin(alpha)

        Q_ci = (Ca / (Z * sin_a)) * (1.0 + bracket ** (10.0/3.0)) ** 0.3
        Q_ce = (Ca / (Z * sin_a)) * (1.0 + bracket ** (-10.0/3.0)) ** 0.3

        return cls(label=_lbl, Q_ci=Q_ci, Q_ce=Q_ce,
                bearing_class="thrust_nonzero_alpha", Cr=None, Ca=Ca)

    @classmethod
    def thrust_90deg(cls, bearing: Bearing, Ca: float,
                    label: str = "") -> "RollingElementCapacity":
        """
        Thrust ball bearing (alpha = 90°) — §4.3.1.4 eq.(23)-(24).
        At alpha=90°, gamma=0; the (1-gamma)/(1+gamma) term vanishes and the
        geometry bracket reduces to the groove radii ratio alone.
        """
        Z, ri, re, Dw = bearing.Z, bearing.ri, bearing.re, bearing.Dw

        _lbl = label or bearing.label
        _check_geometry(ri, re, Dw, _lbl)

        ratio = (ri / re) * ((2.0*re - Dw) / (2.0*ri - Dw))

        Q_ci = (Ca / Z) * (1.0 + (ratio**0.41) ** (10.0/3.0)) ** 0.3
        Q_ce = (Ca / Z) * (1.0 + (ratio**0.41) ** (-10.0/3.0)) ** 0.3

        return cls(label=_lbl, Q_ci=Q_ci, Q_ce=Q_ce,
                bearing_class="thrust_90deg", Cr=None, Ca=Ca)


# ---------------------------------------------------------------------------
# ISO16281BallSolver
# ---------------------------------------------------------------------------

_REQUIRED_ATTRS = ("A", "alpha_0", "phi_j", "Ri", "cp", "Dpw", "Z")


class ISO16281BallSolver:
    """
    ISO/TS 16281 internal load distribution solver, point contact
    (deep groove / angular contact ball bearings).

    Takes bearing reactions and shaft slopes from a SimpleFEMResultsLibrary
    (pre-populated by ShaftResultsReader) and solves the per-bearing internal
    load distribution via a 2-equation root problem in the resultant-force plane.

    Only solving lives on this class — Q_j(), phi_j_global(),
    contact_distribution() and bearing_stiffness() (all of which read an
    already-computed LoadDistributionResult rather than producing one) moved
    to ball_bearing_postprocessing.py as free functions.

    Self-contained: does not import or depend on any other contact-type
    solver. rolling_bearing_solver.RollingBearingSolver dispatches to this
    class for BearingType.DEEP_GROOVE_BALL and BearingType.ANGULAR_CONTACT;
    it can also be used directly when the caller already knows all bearings
    on a shaft are point-contact.

    Parameters
    ----------
    tol       float   residual tolerance for scipy.optimize.root
    psi_input bool    False (default): psi is always projected from FEM shaft slopes.
                      True: per-bearing psi values supplied via psi_override in
                      solve() take precedence when present; any bearing not
                      covered by psi_override falls back to the FEM projection.

                      psi_override values must already be expressed in the
                      resultant-force plane of the CURRENT load case — i.e. the
                      caller is responsible for projecting onto
                      phi_Fr = arctan2(Fr_xy, Fr_xz), which is recomputed from
                      the FEM reactions on every solve() call. A fixed
                      geometric slope is not a valid substitute unless phi_Fr
                      is also fixed across the cases being compared.
    """

    def __init__(self, tol: float = SOLVER_TOLERANCE, psi_input: bool = False):
        self.tol       = tol
        self.psi_input = psi_input

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              library: SimpleFEMResultsLibrary,
              psi_override: dict[str, float] | None = None,
              ) -> BallLoadDistributionLibrary:
        """
        Solve the internal load distribution for all point-contact bearings
        in `bearings`.

        Each bearing must have setup_internal_geometry() and
        compute_hertz_point_contact() already called before this method.

        Parameters
        ----------
        shaft_system  : ShaftSystem — fully resolved
        bearings      : {label: Bearing} — all point-contact
        library       : SimpleFEMResultsLibrary — must contain results for
                        shaft_system.name (populated by ShaftResultsReader.read())
        psi_override  : {label: psi [rad]} — user-supplied misalignment values.
                        Used only when self.psi_input is True. See class
                        docstring for the plane convention required. Labels not
                        present here fall back to the FEM projection; labels
                        present here but not in `bearings` trigger a warning
                        (likely a typo).

        Returns
        -------
        BallLoadDistributionLibrary — local registry, one entry per label in
        `bearings`. rolling_bearing_solver.RollingBearingSolver.solve() is
        what reads this back out and merges it with any other per-type
        local library into the orchestrator's own result; this method never
        touches the final, cross-type BearingResultsLibrary itself.
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

        results = BallLoadDistributionLibrary()
        for label, b in bearings.items():
            check_bearing_ready(b, label, _REQUIRED_ATTRS)
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

            results.set(label, self._solve_bearing(
                b,
                Fr_xz        = Fr_xz,
                Fr_xy        = Fr_xy,
                Fa           = node.Fa,
                delta_r_init = float(np.hypot(node.v_xz, node.v_xy)),
                delta_a_init = node.u,
                psi          = psi,
                phi_Fr       = phi_Fr,
            ))

        return results

    # ------------------------------------------------------------------
    # Element kinematics — ISO/TS 16281 eq.(12)/(15), §4.2.2.1
    # ------------------------------------------------------------------

    @staticmethod
    def _elements(bearing: Bearing,
                  delta_r: float,
                  delta_a: float,
                  Vpsi: np.ndarray):
        """
        Per-element elastic deflection and effective contact angle.
        phi_j is local: 0 aligned with the resultant radial force.

        delta_j is already zero-floored for unloaded elements, so alpha_j and
        d32 need no separate masking — they are always multiplied by
        delta_j-derived terms downstream, which vanish there on their own.
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
        """
        Seed for delta_r. Trusts the FEM hint (from shaft nodal displacements)
        whenever it is non-negligible; only falls back to a physically-derived
        estimate (unloaded contact gap + Hertz deflection) when the hint is
        ~0, e.g. on the first solve of a load case with no prior estimate.
        """
        if hint > 1e-6:
            return hint
        gap   = bearing.A * (1.0 - np.cos(bearing.alpha_0))
        hertz = (abs(Fr) / (bearing.cp * bearing.Z)) ** (2.0/3.0) if Fr != 0.0 else 1e-4
        return gap + hertz

    def _initial_delta_a(self, bearing: Bearing, Fa: float, hint: float) -> float:
        """
        Seed for delta_a, mirroring _initial_delta_r: trust a non-negligible
        FEM hint, else fall back to a Hertz-scale estimate signed with Fa
        (0.0 when Fa is itself 0 — no axial engagement to seed).
        """
        if abs(hint) > 1e-6:
            return hint
        if Fa == 0.0:
            return 0.0
        hertz = (abs(Fa) / (bearing.cp * bearing.Z)) ** (2.0/3.0)
        return np.copysign(hertz, Fa)

    # ------------------------------------------------------------------
    # Single-bearing solve
    # ------------------------------------------------------------------

    def _solve_bearing(self,
                       bearing: Bearing,
                       Fr_xz: float, Fr_xy: float, Fa: float,
                       delta_r_init: float, delta_a_init: float,
                       psi: float, phi_Fr: float) -> BallLoadDistributionResult:
        """
        2-equation root (delta_r, delta_a) in the resultant-force plane.

        psi and phi_Fr are pre-computed by solve() — this method only runs
        the root problem and assembles the result.

        Equilibrium (§4.2.2.1):
            Fr = cp * sum(delta_j^1.5 * cos(alpha_j) * cos(phi_j))
            Fa = cp * sum(delta_j^1.5 * sin(alpha_j))
        """
        Fr   = float(np.hypot(Fr_xz, Fr_xy))
        Vpsi = bearing.Ri * np.sin(psi) * np.cos(bearing.phi_j)
        cp   = bearing.cp

        def residual(u):
            dr, da = u
            _, _, ca, sa, cp_j, d32 = self._elements(bearing, dr, da, Vpsi)
            return np.array([
                Fr - cp * np.sum(d32 * ca * cp_j),
                Fa - cp * np.sum(d32 * sa),
            ])

        dr0               = self._initial_delta_r(bearing, Fr, delta_r_init)
        da0               = self._initial_delta_a(bearing, Fa, delta_a_init)
        x, nfev, res, ok  = run_root(residual, [dr0, da0], self.tol)
        delta_r, delta_a  = float(x[0]), float(x[1])

        delta_j, alpha_j, _, sa, cp_j, d32 = self._elements(
            bearing, delta_r, delta_a, Vpsi)
        Mz = (bearing.Dpw / 2.0) * cp * float(np.sum(d32 * sa * cp_j))

        return BallLoadDistributionResult(
            delta_r=delta_r, delta_a=delta_a, psi=psi, phi_Fr=phi_Fr,
            delta_j=delta_j, alpha_j=alpha_j,
            Mz=Mz, n_iter=nfev, residual=res, ok=ok,
        )

    # ------------------------------------------------------------------
    # Minimum axial preload — still solving (iterates _solve_bearing over
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
        """
        Minimum axial preload [N] such that delta_a >= 0 (contact closure).
        Uses brentq on delta_a(Fa). Raises ValueError if the upper bracket is
        too small to contain a zero crossing.

        Parameters
        ----------
        psi : misalignment in the resultant-force plane [rad] — already projected
              (use psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr) if coming from FEM slopes)
        """
        check_bearing_ready(bearing, bearing.label, _REQUIRED_ATTRS)
        phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

        res0 = self._solve_bearing(bearing, Fr_xz, Fr_xy, 0.0,
                                    delta_r_init, delta_a_init, psi, phi_Fr)
        if res0.delta_a >= 0.0:
            return 0.0, res0

        def g(Fa: float) -> float:
            return self._solve_bearing(bearing, Fr_xz, Fr_xy, Fa,
                                        delta_r_init, delta_a_init, psi, phi_Fr).delta_a

        lo, hi = Fa_bracket
        g_hi = g(hi)
        if g_hi < 0.0:
            raise ValueError(
                f"minimum_axial_load: delta_a = {g_hi:.4e} mm at Fa = {hi:.1f} N — "
                f"widen Fa_bracket."
            )

        Fa_min = brentq(g, lo, hi, xtol=xtol)
        result = self._solve_bearing(bearing, Fr_xz, Fr_xy, Fa_min,
                                      delta_r_init, delta_a_init, psi, phi_Fr)
        return Fa_min, result


# ---------------------------------------------------------------------------
# DEBUG / VALIDATION UTILITIES — not used in production pipelines
# ---------------------------------------------------------------------------

def debug_radial_capacity(bearing: Bearing, Cr: float, label: str = "") -> None:
    """
    Print all intermediate values for Q_ci / Q_ce (S4.3.1.2) — manual cross-check only.

    Same box-print layout as the original ISO_16281_ball_bearing.py debug
    helper, with alpha/degree symbols swapped for ASCII equivalents (project
    convention: exploratory/console output stays ASCII-only for Windows
    PowerShell cp1252 — see project instructions).
    """
    Z, alpha_i = bearing.Z, bearing.alpha_0
    ri, re, Dw, Dpw = bearing.ri, bearing.re, bearing.Dw, bearing.Dpw
    gamma        = Dw * np.cos(alpha_i) / Dpw
    radii_ratio  = (ri / re) * ((2.0*re - Dw) / (2.0*ri - Dw))
    bracket      = _geometry_bracket(gamma, ri, re, Dw)
    cos_alpha_07 = np.cos(alpha_i) ** 0.7
    inner_term   = bracket ** (10.0/3.0)
    outer_term   = bracket ** (-10.0/3.0)
    denom_ci     = 0.407 * Z * cos_alpha_07
    denom_ce     = 0.389 * Z * cos_alpha_07
    factor_ci    = (1.0 + inner_term) ** 0.3
    factor_ce    = (1.0 + outer_term) ** 0.3
    Q_ci         = (Cr / denom_ci) * factor_ci
    Q_ce         = (Cr / denom_ce) * factor_ce
    _lbl         = label or bearing.label

    print(f"\n  +-- Q_ci/Q_ce debug -- {_lbl} {'-'*30}+")
    print(f"  |  INPUT")
    print(f"  |    Cr          = {Cr:.2f} N")
    print(f"  |    Z           = {Z}")
    print(f"  |    alpha_0     = {np.degrees(alpha_i):.6f} deg")
    print(f"  |    ri          = {ri:.6f} mm")
    print(f"  |    re          = {re:.6f} mm")
    print(f"  |    Dw          = {Dw:.6f} mm")
    print(f"  |    Dpw         = {Dpw:.6f} mm")
    print(f"  |  INTERMEDIATE")
    print(f"  |    gamma            = {gamma:.8f}   [Dw*cos(alpha_0)/Dpw]")
    print(f"  |    radii_ratio      = {radii_ratio:.8f}   [ri/re*(2re-Dw)/(2ri-Dw)]")
    print(f"  |    bracket          = {bracket:.8f}   [1.044*(gamma-term)^1.72*ratio^0.41]")
    print(f"  |    cos(alpha_0)^0.7 = {cos_alpha_07:.8f}")
    print(f"  |    bracket^(+10/3)  = {inner_term:.8f}   -> Q_ci eq.(19)")
    print(f"  |    bracket^(-10/3)  = {outer_term:.8f}   -> Q_ce eq.(20)")
    print(f"  |  CAPACITY")
    print(f"  |    denom_ci  = 0.407*{Z}*{cos_alpha_07:.6f} = {denom_ci:.6f}")
    print(f"  |    denom_ce  = 0.389*{Z}*{cos_alpha_07:.6f} = {denom_ce:.6f}")
    print(f"  |    factor_ci = (1 + {inner_term:.6f})^0.3 = {factor_ci:.8f}")
    print(f"  |    factor_ce = (1 + {outer_term:.6f})^0.3 = {factor_ce:.8f}")
    print(f"  |    Q_ci      = {Cr:.2f}/{denom_ci:.4f} * {factor_ci:.6f} = {Q_ci:.4f} N")
    print(f"  |    Q_ce      = {Cr:.2f}/{denom_ce:.4f} * {factor_ce:.6f} = {Q_ce:.4f} N")
    print(f"  +{'-'*58}+")