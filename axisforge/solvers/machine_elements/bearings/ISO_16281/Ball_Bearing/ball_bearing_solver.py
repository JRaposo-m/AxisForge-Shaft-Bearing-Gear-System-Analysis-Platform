"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/ball_bearing_solver.py

ISO/TS 16281 internal load distribution SOLVER for POINT-CONTACT bearings
(deep groove and angular contact ball bearings -- BearingType.DEEP_GROOVE_BALL
and BearingType.ANGULAR_CONTACT). alpha_0 is a bearing attribute, not
hardcoded, so both families are covered by the same kinematics -- and, via
solve_contact() below, so is a single row of a multi-row thrust ball
bearing (BearingType.THRUST_BALL): the 2-equation point-contact problem
this file solves does not know or care whether alpha_0 is a radial
bearing's small contact angle or a thrust bearing's 90deg -- only the
_REQUIRED_ATTRS geometry matters, and MultiRowThrustBallFamily's row dicts
already carry it (see ball_bearing_multirow_solver.py).

Scope of this file: only things that PRODUCE a BallLoadDistributionResult
live here. Anything that CONSUMES an already-computed BallLoadDistributionResult
/ BallBearingResult / BallBearingStiffness to derive a further quantity --
contact force distribution, ball positions, secant stiffness, dynamic
equivalent load -- lives in ball_bearing_postprocessing.py instead. That
file imports from this one only for typing; this file never imports from it.

Capacity (Q_ci/Q_ce, Cr/Ca) moved out of this file -- core cleanup
------------------------------------------------------------------
This file used to carry its own RollingElementCapacity dataclass plus
_geometry_bracket()/_check_geometry() helpers, duplicating eq.(19)-(24)
against core/machine_elements/Bearings/families/ball/{radial,thrust}/
functions/capacity.py -- which now owns the SAME formulas, tested, and is
the one every new ball subtype (self-aligning, thrust) is already wired
through via bearing.family.per_element_dynamic_capacity(bearing, Cr=...)
/ (bearing, Ca=...). Recomputing eq.(19)-(24) here risked drifting from
that single source of truth, so the dataclass and its geometry helpers are
gone; rolling_bearing_solver.py now calls bearing.family directly (see its
module docstring). debug_radial_capacity() below is kept as a manual
cross-check but also sources its numbers from bearing.family, not from a
local reimplementation -- see its own docstring.

solve() returns a BallLoadDistributionLibrary (ball_bearing_results.py) --
a LOCAL, single-type registry, one BallBearingResult per label -- not a
bare dict. This mirrors ISO16281RollerSolver.solve() returning a
RollerLoadDistributionLibrary on the roller side. rolling_bearing_solver.
RollingBearingSolver.solve() is what reads labels back out of it and merges
it with any other per-type local library; this file never touches the
final, cross-type BearingResultsLibrary in the global library.py.

RECONSTRUCTED, this turn -- the single-row/multi-row seam is now public
-------------------------------------------------------------------------
_solve_bearing() is renamed to solve_contact() and is now PUBLIC. It was
already being called from outside this class -- ball_bearing_multirow_solver.
py reached into self._row_solver._solve_bearing() to run each row's inner
solve -- so the underscore was never actually protecting anything; it was
just an undeclared cross-module contract. solve_contact() makes that
contract explicit: it is THE seam a multi-row (or any future per-row)
orchestration composes against, one row/bearing at a time, taking raw
scalar loads (Fr_xz, Fr_xy, Fa, psi, phi_Fr) rather than a ShaftSystem/
library batch. _REQUIRED_ATTRS is renamed REQUIRED_ATTRS for the same
reason -- ball_bearing_multirow_solver.py imports it to validate each row's
geometry via the SAME check_bearing_ready() this class's own solve() uses,
so it is part of that same public contract now, not a private detail.

_elements(), _initial_delta_r(), _initial_delta_a() stay private -- they
are internal to how ONE solve_contact() call converges, not part of the
seam anything outside this class composes against.

This module is one of several per-bearing-type solvers dispatched by
rolling_bearing_solver.RollingBearingSolver (see BearingType in
bearing_types.py). It lives in its own Ball_Bearing/ subfolder, sibling to
Roller_Bearing/ -- it does not import, and is not imported by, any other
contact-type solver. The two are independent and only share the neutral
data contract, generic utilities, and results registry one level up, in
bearings/ISO_16281/library.py.

The solve is performed in the plane of the resultant radial force:

    phi_Fr = arctan2(Fr_xy, Fr_xz)
    Fr     = sqrt(Fr_xz^2 + Fr_xy^2)
    psi    = psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr)   [prescribed misalignment]

A 2-equation root (delta_r, delta_a) enforces static equilibrium (Sec 4.2.2.1),
using the point-contact load-deflection law Q_j = cp * delta_j^1.5. Ball
positions in the global frame (output/plots only):

    phi_j_global = (bearing.phi_j + phi_Fr) % 2*pi

Inputs (Fr, Fa, psi_xz, psi_xy) come from a SimpleFEMResultsLibrary populated
by ShaftResultsReader. The bearing solve is performed once per load case --
the resulting stiffness values are outputs, not fed back into FEM.

References
----------
ISO/TS 16281:2008 Sec 4.2, eq.(12)-(15); Sec 4.3.1, eq.(19)-(28)
Harris & Kotzalas, "Rolling Bearing Analysis", 5th ed., Ch. 6
"""
from __future__ import annotations

import warnings

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
    BallBearingResult,
    BallLoadDistributionLibrary,
)
from axisforge.config import SOLVER_TOLERANCE


# ---------------------------------------------------------------------------
# ISO16281BallSolver
# ---------------------------------------------------------------------------

# PUBLIC, this turn -- formerly _REQUIRED_ATTRS. Part of the seam
# ball_bearing_multirow_solver.py composes against (see module docstring).
REQUIRED_ATTRS = ("A", "alpha_0", "phi_j", "Ri", "cp", "Dpw", "Z")


class ISO16281BallSolver:
    """
    ISO/TS 16281 internal load distribution solver, point contact
    (deep groove / angular contact ball bearings).

    Takes bearing reactions and shaft slopes from a SimpleFEMResultsLibrary
    (pre-populated by ShaftResultsReader) and solves the per-bearing internal
    load distribution via a 2-equation root problem in the resultant-force plane.

    Only solving lives on this class -- Q_j(), phi_j_global(),
    contact_distribution() and bearing_stiffness() (all of which read an
    already-computed BallBearingResult rather than producing one) live in
    ball_bearing_postprocessing.py as free functions. Capacity (Q_ci/Q_ce,
    Cr/Ca) lives in core/ -- see module docstring.

    Self-contained: does not import or depend on any other contact-type
    solver. rolling_bearing_solver.RollingBearingSolver dispatches to this
    class for BearingType.DEEP_GROOVE_BALL and BearingType.ANGULAR_CONTACT;
    it can also be used directly when the caller already knows all bearings
    on a shaft are point-contact. solve_contact() (the per-row/per-bearing
    seam) is also called directly by ISO16281MultiRowBallSolver, once per
    row per outer iteration -- see ball_bearing_multirow_solver.py.

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

    def __init__(self, tol: float = SOLVER_TOLERANCE, psi_input: bool = False):
        self.tol       = tol
        self.psi_input = psi_input

    # ------------------------------------------------------------------
    # Public entry point -- batch solve over a shaft's bearing set
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
        shaft_system  : ShaftSystem -- fully resolved
        bearings      : {label: Bearing} -- all point-contact
        library       : SimpleFEMResultsLibrary -- must contain results for
                        shaft_system.name (populated by ShaftResultsReader.read())
        psi_override  : {label: psi [rad]} -- user-supplied misalignment values.
                        Used only when self.psi_input is True. See class
                        docstring for the plane convention required. Labels not
                        present here fall back to the FEM projection; labels
                        present here but not in `bearings` trigger a warning
                        (likely a typo).

        Returns
        -------
        BallLoadDistributionLibrary -- local registry, one BallBearingResult
        per label in `bearings` (each wrapping exactly 1 row -- this method
        only ever produces single-row results; see BallBearingResult.single()
        in ball_bearing_results.py). rolling_bearing_solver.
        RollingBearingSolver.solve() is what reads this back out and merges
        it with any other per-type local library into the orchestrator's
        own result; this method never touches the final, cross-type
        BearingResultsLibrary itself.
        """
        if self.psi_input and psi_override:
            unknown = set(psi_override) - set(bearings)
            if unknown:
                warnings.warn(
                    f"psi_override has labels not in `bearings`: {sorted(unknown)} -- "
                    f"check for typos; these entries are ignored."
                )

        shaft_results = library.get(shaft_system.name)
        node_by_label = {n.label: n for n in shaft_results.bearing_nodes}

        results = BallLoadDistributionLibrary()
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
            results.set(label, BallBearingResult.single(row))

        return results

    # ------------------------------------------------------------------
    # Element kinematics -- ISO/TS 16281 eq.(12)/(15), Sec 4.2.2.1
    # Private: internal to how one solve_contact() call converges, not
    # part of the cross-module seam.
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
        d32 need no separate masking -- they are always multiplied by
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
        (0.0 when Fa is itself 0 -- no axial engagement to seed).
        """
        if abs(hint) > 1e-6:
            return hint
        if Fa == 0.0:
            return 0.0
        hertz = (abs(Fa) / (bearing.cp * bearing.Z)) ** (2.0/3.0)
        return np.copysign(hertz, Fa)

    # ------------------------------------------------------------------
    # Single-row/single-bearing solve -- THE SEAM. Public: this is the
    # per-row entry point ISO16281MultiRowBallSolver composes against
    # (once per row per outer iteration) -- see ball_bearing_multirow_
    # solver.py's module docstring. Formerly _solve_bearing().
    # ------------------------------------------------------------------

    def solve_contact(self,
                      bearing: Bearing,
                      Fr_xz: float, Fr_xy: float, Fa: float,
                      delta_r_init: float, delta_a_init: float,
                      psi: float, phi_Fr: float) -> BallLoadDistributionResult:
        """
        2-equation root (delta_r, delta_a) in the resultant-force plane, for
        ONE point-contact raceway -- a whole single-row bearing, or one row
        of a multi-row bearing (bearing may be a real Bearing, or any
        object exposing REQUIRED_ATTRS, e.g. the SimpleNamespace row views
        ball_bearing_multirow_solver.py builds from bearing.rows[j]).

        psi and phi_Fr are pre-computed by the caller (solve() above, or
        ISO16281MultiRowBallSolver.solve_bearing()) -- this method only runs
        the root problem and assembles the result.

        Equilibrium (Sec 4.2.2.1):
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
        """
        Minimum axial preload [N] such that delta_a >= 0 (contact closure).
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
    """
    Print Q_ci/Q_ce for a radial ball bearing -- manual cross-check only.

    Sources the numbers from bearing.family.per_element_dynamic_capacity(),
    the SAME call rolling_bearing_solver.postprocess_and_record() makes --
    this file no longer carries its own copy of eq.(19)-(20), so a debug
    print here can never silently drift from what production returns.
    `i` is informational only (DGBB/ACB/self-aligning already read row
    count off bearing.i internally); pass it just to echo it in the
    printout if it differs from bearing.i for some reason.
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