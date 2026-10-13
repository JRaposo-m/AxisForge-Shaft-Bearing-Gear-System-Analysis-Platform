"""Internal load distribution of rolling bearings after ISO/TS 16281.

Static equilibrium of one bearing under the loads of its shaft FEM node:
ball bearings with the point-contact model of ISO/TS 16281, 4.2 (unknowns
delta_r and delta_a), radial roller bearings with the lamina model of 5.2
(unknown delta_r). The misalignment psi is an input taken from the shaft
slopes. ``SolverBase.solve`` runs every bearing of a shaft
and, with ``postprocess=True``, adds the stiffness and the reference rating
life of ``contact_postprocessing``. The multi-row "shared displacement"
solvers are an extension that is not part of the standard.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq

from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.families.family import is_thrust_family
from axisforge.core.mechanical_system.shaft_system import ShaftSystem
from axisforge.results.fem_results.shaft_results import ShaftResults
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.numerics import run_root
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.validation import (
    check_bearing_ready
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.dispatch import register_contact_solver
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281 import (
    contact_postprocessing as pp,
)
from axisforge.results.bearings.load_distribution.load_distribution_results import (
    BearingResult,
    BallLoadDistributionResult, BallBearingResult,
    RollerLoadDistributionResult, RollerBearingResult,
)
from axisforge.results.bearings.life.basic_life_results import BasicReferenceRatingLifeResult
from axisforge.results.bearings.bearing_analysis_result import BearingAnalysisResult
from axisforge.config import SOLVER_TOLERANCE


# =====================================================================
# ---- SolverBase / LineContactSolverBase -- abstract contract --------
# =====================================================================

class SolverBase(ABC):
    """Common batch loop of the ISO/TS 16281 contact solvers.

    ``solve`` is the same for point and line contact: for each bearing it
    checks readiness, reads the loads of the FEM bearing node, computes the
    direction of the radial load and the misalignment in the load plane, calls
    the concrete solver and wraps the result. The equilibrium itself is in
    ``solve_contact`` of each concrete class, reached through
    ``_solve_one_bearing``.

    Parameters
    ----------
    tol : float, optional
        Tolerance passed unchanged to scipy.optimize.root (through
        ``run_root``). Default ``SOLVER_TOLERANCE``. It is not compared with
        the force residual; the residual norm [N] is stored in the result.
    psi_input : bool, optional
        If True, ``psi_override`` given to ``solve`` replaces the misalignment
        computed from the FEM slopes. Default False.
    postprocess : bool, optional
        If True, a single-row solver attaches the bearing stiffness and the
        basic reference rating life (L10r, Pref) to each result. Default
        False: ``basic_life`` is then None.

    Attributes
    ----------
    CAPABILITY : str
        "point_contact" or "line_contact"; written on the concrete classes by
        ``register_contact_solver``.
    MULTIROW_SOLVER : type or None
        Multi-row sibling of a single-row solver; None on every other class.
    _result_cls : type or None
        Bearing-level result class of the concrete solver.
    """

    CAPABILITY: str = ""
    MULTIROW_SOLVER: type | None = None
    _result_cls: type | None = None

    def __init__(self, tol: float = SOLVER_TOLERANCE, psi_input: bool = False,
                 postprocess: bool = False):
        """Store the solver options (see the class docstring)."""
        self.tol         = tol
        self.psi_input   = psi_input
        self.postprocess = postprocess

    def solve(self,
              shaft_system: ShaftSystem,
              bearings: dict[str, Bearing],
              shaft_results: ShaftResults,
              psi_override: dict[str, float] | None = None,
              ) -> dict[str, BearingAnalysisResult]:
        """Solve the internal load distribution of every bearing in ``bearings``.

        For each bearing the loop takes the radial load components of its FEM
        node, the load direction phi_Fr = arctan2(Fr_xy, Fr_xz) and the
        misalignment in the load plane
        psi = psi_xz*cos(phi_Fr) + psi_xy*sin(phi_Fr) (or the override), and calls
        the concrete solver.

        Parameters
        ----------
        shaft_system : ShaftSystem
            Not used.
        bearings : dict[str, Bearing]
            Bearings to solve, keyed by label. Every label must have a node in
            ``shaft_results.bearing_nodes``.
        shaft_results : ShaftResults
            FEM results of the shaft. From each bearing node the loop reads Fr_xz,
            Fr_xy [N] and the slopes psi_xz, psi_xy [rad]; the concrete solver also
            reads Fa [N] (ball bearings only) and the displacements u, v_xz, v_xy
            [mm] as initial guesses.
        psi_override : dict[str, float], optional
            Misalignment psi [rad] per bearing label. Used only by a solver built
            with ``psi_input=True``.

        Returns
        -------
        dict[str, BearingAnalysisResult]
            One entry per label. ``load_distribution`` is the bearing-level result
            (1 row, or N rows for a multi-row solver); ``basic_life`` is set only
            with ``postprocess=True``.

        Raises
        ------
        NotImplementedError
            If the class does not set ``_result_cls``; from a multi-row solver
            built with ``postprocess=True``.
        RuntimeError
            From ``check_bearing_ready``: the family does not match the solver, or
            the bearing has no ISO/TS 16281 stiffness.
        ValueError
            From ``_extra_ready_checks`` (lamina or row checks); with
            ``postprocess=True``, from ``BallBasicReferenceRatingLife.from_loads``
            when no ball is loaded.
        KeyError
            If a label has no FEM bearing node.

        Warns
        -----
        UserWarning
            With ``psi_input=True``, when ``psi_override`` has labels that are not
            in ``bearings`` (those entries are ignored).

        Notes
        -----
        psi is the projection of the two shaft slopes on the line of the radial
        load; the tilt component normal to that line is dropped.

        Conventions (D-0010): x is the shaft axis; _xz is the component in the
        horizontal plane (along Z), _xy in the vertical plane (along Y).
        phi_Fr = arctan2(Fr_xy, Fr_xz) [rad] is measured from +Z toward +Y. It
        is not the load angle theta of the AxisForge README (measured from +Y
        toward +Z): phi_Fr = pi/2 - theta. Fr_xz, Fr_xy and Fa are the
        reactions stored on the FEM bearing node, with the sign they have
        there.

        phi_j starts at 0 on the load line (the families use
        linspace(0, 2*pi, Z, endpoint=False)), so one rolling element is always
        exactly on the load line: the element set turns with phi_Fr.
        Harris2007_vol2, 1.2 and 1.3.3 use the same origin (azimuth measured
        from the most heavily loaded rolling element).
        """
        if self._result_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set _result_cls to its "
                f"*BearingResult class."
            )

        if self.psi_input and psi_override:
            unknown = set(psi_override) - set(bearings)
            if unknown:
                warnings.warn(
                    f"psi_override has labels not in `bearings`: {sorted(unknown)} -- "
                    f"check for typos; these entries are ignored."
                )

        node_by_label = {n.label: n for n in shaft_results.bearing_nodes}

        results: dict[str, BearingAnalysisResult] = {}
        for label, b in bearings.items():
            check_bearing_ready(b, label, self.CAPABILITY)
            self._extra_ready_checks(b, label)

            node   = node_by_label[label]
            Fr_xz  = node.Fr_xz
            Fr_xy  = node.Fr_xy
            phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

            if (self.psi_input
                    and psi_override is not None
                    and label in psi_override):
                psi = float(psi_override[label])
            else:
                psi = node.psi_xz * np.cos(phi_Fr) + node.psi_xy * np.sin(phi_Fr)

            result = self._wrap_result(self._solve_one_bearing(b, node, phi_Fr, psi))

            if self.postprocess:
                results[label] = self._attach_postprocessing(b, result)
            else:
                results[label] = BearingAnalysisResult(label=result.label,
                                                       load_distribution=result)

        return results

    def _extra_ready_checks(self, bearing: Bearing, label: str) -> None:
        """Hook for readiness checks beyond ``check_bearing_ready``; no check here.

        Overridden by ``LineContactSolverBase`` and ``MultiRowSolverBase``.
        """
        return None

    def _wrap_result(self, row):
        """Turn the output of ``_solve_one_bearing`` into the bearing-level result.

        Single-row solvers already return the complete result, so it is returned
        unchanged. ``MultiRowSolverBase`` overrides this.
        """
        return row

    def _attach_postprocessing(self, bearing: Bearing, result: BearingResult
                               ) -> BearingAnalysisResult:
        """Attach stiffness and rating life to a result: not implemented here.

        Overridden by each concrete single-row solver.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            f"{type(self).__name__}: postprocessing not implemented."
        )

    @abstractmethod
    def _solve_one_bearing(self, bearing: Bearing, node, phi_Fr: float, psi: float):
        """Solve one bearing from its FEM node data (abstract).

        Parameters
        ----------
        bearing : Bearing
            Bearing with the ISO/TS 16281 attributes set by its family.
        node
            FEM bearing node: loads [N], displacements [mm], slopes [rad].
        phi_Fr : float
            Direction of the resultant radial load [rad].
        psi : float
            Misalignment between the rings in the load plane [rad].

        Returns
        -------
        object
            The bearing-level result of ``solve_contact``, or, for a multi-row
            solver, what its ``_wrap_result`` expects.
        """
        ...

    @staticmethod
    def _Cr_Ca(bearing: Bearing) -> tuple[float | None, float | None]:
        """Basic dynamic load rating of the bearing as the pair (Cr, Ca).

        ``bearing.C`` is the only rating stored: it is the radial rating Cr of a
        radial family and the axial rating Ca of a thrust family
        (``is_thrust_family``).

        Returns
        -------
        tuple[float or None, float or None]
            (Cr, Ca) [N]: (None, C) for a thrust family, (C, None) otherwise.
        """
        if is_thrust_family(bearing.family):
            return None, bearing.C
        return bearing.C, None


class LineContactSolverBase(SolverBase):
    """Base of the line-contact (lamina model) solvers.

    Adds the checks of the lamina discretisation to ``check_bearing_ready``.

    Attributes
    ----------
    MIN_LAMINAE : int
        Minimum number of laminae per roller, 30 (ISO_TS_16281_2008, 5.2.2).
    """

    MIN_LAMINAE = 30  # Sec 5.2.2 -- "the number of laminae shall be at least n_s = 30"

    def _extra_ready_checks(self, bearing: Bearing, label: str) -> None:
        """Check the lamina discretisation of the bearing.

        Raises
        ------
        ValueError
            If ``n_s < MIN_LAMINAE`` or ``len(x_k) != n_s``.
        """
        n_s = bearing.n_s
        if n_s < self.MIN_LAMINAE:
            raise ValueError(
                f"Bearing '{label}': n_s = {n_s} < {self.MIN_LAMINAE} -- "
                f"ISO/TS 16281 Sec 5.2.2 requires at least {self.MIN_LAMINAE} laminae."
            )
        if len(bearing.x_k) != n_s:
            raise ValueError(
                f"Bearing '{label}': len(x_k) = {len(bearing.x_k)} != n_s = {n_s} -- "
                f"x_k must have exactly one lamina midpoint per n_s."
            )


# =====================================================================
# ---- MultiRowSolverBase -- shared-displacement multi-row ------------
# ---- (NOT reviewed for row-to-row coupling -- see MultiRowSolverBase) -
# =====================================================================

class MultiRowSolverBase(SolverBase, ABC):
    """Base of the multi-row "shared displacement" solvers.

    All rows see the same delta_r (and delta_a) and the same psi. One
    root-find balances the external load against the sum of the row reactions;
    each row is computed with the ``elements`` kinematics of its single-row
    sibling and with its own geometry and spring constant. The rows are treated
    as co-located: no axial offset between rows enters the equations.

    The result is one bearing-level result with N rows. The shared state
    (delta_r, delta_a, psi, phi_Fr, applied load, convergence data) is stored
    once; each row carries its own reactions Fr_row and Fa_row.

    Attributes
    ----------
    SINGLE_ROW_SOLVER : type or None
        Single-row sibling class whose ``elements`` is reused for each row.

    Warnings
    --------
    Not part of ISO/TS 16281: the standard prints no multi-row model. Clause 1
    says that the analysis for multi-row bearings "can be derived from the
    equations given" and that "the load distribution for each individual row
    has to be considered". Not reviewed for row-to-row coupling.

    The two multi-row families of AxisForge (``ThrustBallMultiRowFamily``,
    ``RollerThrustMultiRowFamily``) return ``rows`` without an
    ``iso16281_analysis`` flag, so ``check_bearing_ready`` rejects their
    bearings before a multi-row solver runs.

    ``postprocess=True`` is refused (NotImplementedError).

    References
    ----------
    (source to find) -- shared-displacement model for multi-row bearings.
    """

    #: Sibling single-row solver class, reused for elements() + per-row
    #: attribute validation. Set by concrete subclasses.
    SINGLE_ROW_SOLVER: type[SolverBase] | None = None

    @staticmethod
    def _row_views(bearing: Bearing) -> list:
        """Rows of a multi-row bearing as attribute-style objects.

        Returns
        -------
        list
            ``bearing.rows``; a row given as a dict is wrapped in a
            SimpleNamespace so that ``elements`` can read ``row.cp``,
            ``row.phi_j`` and the other attributes.

        Raises
        ------
        ValueError
            If the bearing has fewer than 2 rows.
        """
        rows = getattr(bearing, "rows", None) or []
        if len(rows) < 2:
            raise ValueError(
                f"Bearing '{getattr(bearing, 'label', '?')}': multi-row solver "
                f"needs >= 2 rows in bearing.rows; got {len(rows)}."
            )
        return [SimpleNamespace(**row) if isinstance(row, dict) else row for row in rows]

    def _extra_ready_checks(self, bearing: Bearing, label: str) -> None:
        """Refuse ``postprocess=True`` and check every row of the bearing.

        Raises
        ------
        NotImplementedError
            If the solver was built with ``postprocess=True``.
        ValueError
            From ``_row_views`` or ``_check_row_ready``.
        """
        if self.postprocess:
            raise NotImplementedError(
                f"Bearing '{label}': postprocess=True is not supported on "
                f"multi-row solvers ({type(self).__name__}) -- row-level "
                f"postprocessing linkage (Cr/Ca per row, L10r row combination) "
                f"has not been designed/reviewed for shared-displacement "
                f"multi-row bearings. Use postprocess=False and call "
                f"contact_postprocessing.py's functions directly if needed."
            )
        for j, row in enumerate(self._row_views(bearing)):
            row_label = f"{label}[row{j}]"
            self._check_row_ready(row, row_label)

    @abstractmethod
    def _check_row_ready(self, row, row_label: str) -> None:
        """Readiness check of one row (abstract).

        Capability and ``iso16281_analysis`` are not checked per row.
        """
        ...

    def _wrap_result(self, row):
        """Build the multi-row bearing-level result from (row_results, shared state)."""
        row_results, shared = row
        return self._result_cls.multirow(row_results, **shared)

    def _solve_one_bearing(self, bearing: Bearing, node, phi_Fr: float, psi: float):
        """Solve all rows of one bearing from its FEM node data.

        Fr_xz, Fr_xy [N] are read from the node; Fa [N] is read if the node has
        it, otherwise 0.0. The FEM displacements are not used as initial guesses.

        Returns
        -------
        tuple
            (row_results, shared): the row results and the dict of shared state
            for ``_wrap_result``.
        """
        row_views = self._row_views(bearing)
        row_results, delta_r, delta_a, Fa, n_iter, residual_norm, ok = self._solve_rows(
            row_views,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy, Fa=getattr(node, "Fa", 0.0),
            psi=psi, phi_Fr=phi_Fr,
        )
        shared = dict(
            label=bearing.label,
            delta_r=delta_r, delta_a=delta_a, psi=psi, phi_Fr=phi_Fr,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy, Fa=Fa,
            n_iter=n_iter, residual=residual_norm, ok=ok,
        )
        return row_results, shared

    @abstractmethod
    def _solve_rows(self, row_views, Fr_xz: float, Fr_xy: float, Fa: float,
                     psi: float, phi_Fr: float):
        """Solve the shared-displacement equilibrium over all rows (abstract).

        Parameters
        ----------
        row_views : list
            Rows from ``_row_views``.
        Fr_xz, Fr_xy : float
            Radial load components in the two shaft planes [N].
        Fa : float
            Axial load [N].
        psi : float
            Misalignment in the load plane [rad], prescribed.
        phi_Fr : float
            Direction of the resultant radial load [rad].

        Returns
        -------
        tuple
            (row_results, delta_r, delta_a, Fa, n_iter, residual_norm, ok):
            one row result per row; shared displacements delta_r, delta_a [mm];
            axial load carried by the bearing [N] (0.0 for radial roller rows);
            number of residual evaluations; Euclidean norm of the force residual
            [N]; success flag of the root-finder. Ball rows solve two unknowns
            (delta_r, delta_a), roller rows one (delta_r).
        """
        ...


# =====================================================================
# ---- ISO16281BallSolver -- point contact -----------------------------
# =====================================================================

@register_contact_solver(capability="point_contact")
class ISO16281BallSolver(SolverBase):
    """Load distribution of a ball bearing (point contact), ISO/TS 16281, 4.2.

    Two equilibrium equations (radial and axial force) are solved for the
    relative displacements delta_r and delta_a of the rings, in the plane of
    the resultant radial load, with the misalignment psi prescribed. The
    kinematics are in ``elements``, the equilibrium in ``solve_contact``.

    Warnings
    --------
    The standard prints three equilibrium equations, Eq. (16)-(18), and does
    not say which quantities are given. Harris2007_vol2, 1.2, Eq. 1.22-1.24
    prints the same three equations with the unknowns delta_a, delta_r and
    the misalignment theta (the moment is a load) [2]. Here psi is
    prescribed instead, Eq. (16)-(17)
    are solved for delta_r and delta_a, and Eq. (18) gives Mz for that psi.

    cp: see the Warnings of ``PointContactStiffness``.

    Scope: clause 1 names single-row radial ball bearings and 4.1 names radial
    and thrust ball bearings; Eq. (12) is written for a radial ball bearing
    with alpha_0 = arccos(1 - s/(2A)). This solver is registered for every
    point-contact family of AxisForge and applies Eq. (12)-(18) unchanged to
    all of them.

    References
    ----------
    .. [1] ISO_TS_16281_2008, clause 1 (scope); 4.1; 4.2.2, Eq. (12)-(15);
       4.2.2.1, Eq. (16)-(17); 4.2.2.2, Eq. (18).
    .. [2] Harris2007_vol2, 1.2, Eq. 1.22-1.24.
    """

    _result_cls = BallBearingResult

    def _solve_one_bearing(self, bearing: Bearing, node, phi_Fr: float, psi: float
                            ) -> BallBearingResult:
        """Solve one ball bearing with the loads and displacements of its FEM node.

        Loads Fr_xz, Fr_xy, Fa [N] from the node. Initial guesses: delta_r from
        hypot(v_xz, v_xy) [mm], delta_a from the axial displacement u [mm].

        Returns
        -------
        BallBearingResult
            Result of ``solve_contact``.
        """
        return self.solve_contact(
            bearing,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy, Fa=node.Fa,
            delta_r_init=float(np.hypot(node.v_xz, node.v_xy)),
            delta_a_init=node.u,
            psi=psi, phi_Fr=phi_Fr,
        )

    @staticmethod
    def _attach_postprocessing(bearing: Bearing, result: BallBearingResult
                               ) -> BearingAnalysisResult:
        """Add bearing stiffness and basic reference rating life to a ball result.

        Returns
        -------
        BearingAnalysisResult
            ``load_distribution`` with the secant stiffness attached, and
            ``basic_life`` with L10r [10^6 rev] and Pref [N].

        Raises
        ------
        ValueError
            From ``BallBasicReferenceRatingLife.from_loads`` when no ball is
            loaded (equivalent loads equal to zero).

        Notes
        -----
        Steps: secant stiffness (``bearing_stiffness``); Q_ci, Q_ce [N] from the
        family (Eq. (19)-(24); Eq. (19)-(20) as corrected by Cor. 1:2009); equivalent rolling-element loads Q_ei, Q_ee, Eq. (25)-(28); L10r,
        Eq. (29); Pref, Eq. (30)-(31), with
        ``bearing.C`` as Cr or Ca. The ring rotation flags keep their defaults
        (inner ring rotating, outer ring stationary relative to the load): the
        bearing has no attribute for them.

        References
        ----------
        .. [1] ISO_TS_16281_2008, 4.3.2-4.3.4, Eq. (25)-(31).
        .. [2] ISO_TS_16281_2008_Cor1_2009.
        """
        stiffness = pp.bearing_stiffness(bearing, result,
                                         Fr_xz=result.Fr_xz, Fr_xy=result.Fr_xy, Fa=result.Fa)
        result = result.with_stiffness(stiffness)

        Q_ci, Q_ce = bearing.family.per_element_dynamic_capacity(bearing)
        equiv = pp.DynamicEquivalentRollingElementLoad.from_distribution(
            bearing, result.row, label=bearing.label,
        )
        life = pp.BallBasicReferenceRatingLife.from_loads(
            label=bearing.label, Q_ci=Q_ci, Q_ei=equiv.Q_ei, Q_ce=Q_ce, Q_ee=equiv.Q_ee,
        )

        Cr, Ca = SolverBase._Cr_Ca(bearing)
        pref = pp.BallDynamicEquivalentReferenceLoad.from_L10r(
            label=bearing.label, L10r=life.L10r, Cr=Cr, Ca=Ca,
        )

        basic_life = BasicReferenceRatingLifeResult(
            label=bearing.label, rows=(life,), L10r=life.L10r, pref=pref,
        )
        return BearingAnalysisResult(label=result.label, load_distribution=result,
                                     basic_life=basic_life)

    @staticmethod
    def elements(bearing: Bearing, delta_r: float, delta_a: float, Vpsi: np.ndarray):
        """Elastic deflection and operating contact angle of every ball.

        Parameters
        ----------
        bearing : Bearing
            Needs A [mm], alpha_0 [rad] and phi_j [rad]. A row view with the same
            attributes works too.
        delta_r : float
            Relative radial displacement of the rings along the load line [mm].
        delta_a : float
            Relative axial displacement of the rings [mm].
        Vpsi : np.ndarray
            Axial displacement at each ball due to the misalignment,
            Ri*sin(psi)*cos(phi_j) [mm], shape (Z,).

        Returns
        -------
        delta_j : np.ndarray
            Total elastic deflection of both contacts of ball j [mm], >= 0,
            shape (Z,).
        alpha_j : np.ndarray
            Operating contact angle of ball j [rad].
        cos_alpha_j : np.ndarray
            cos(alpha_j) [-].
        sin_alpha_j : np.ndarray
            sin(alpha_j) [-].
        cos_phi_j : np.ndarray
            cos(phi_j) [-]; named ``cp_j`` in the code (not the spring constant
            ``cp``).
        d32 : np.ndarray
            delta_j^1.5 [mm^1.5].

        Notes
        -----
        Equations as coded::

            U_j     = A*cos(alpha_0) + delta_r*cos(phi_j)
            V_j     = A*sin(alpha_0) + delta_a + Vpsi_j
            delta_j = max(sqrt(U_j^2 + V_j^2) - A, 0)                  Eq. (12)
            alpha_j = arctan2(V_j, U_j)                                Eq. (15)

        A is the distance between the raceway groove curvature centres (Eq. (13))
        and alpha_0 the initial contact angle. The radial clearance s does not
        appear: alpha_0 carries it (deep groove family:
        alpha_0 = arccos(1 - s/(2*A)); the other families take alpha_0 as input).

        Conventions: phi_j is the local angular position, zero on the line of the
        resultant radial load. delta_a and Vpsi add to the axial component of the
        centre distance, in the direction of A*sin(alpha_0). The standard writes
        arctan and gives the signs only in its Figure 1; the code uses arctan2 and
        accepts delta_a of either sign.

        References
        ----------
        .. [1] ISO_TS_16281_2008, 4.2.2, Eq. (12)-(15).
        """
        A, alpha_0, phi_j = bearing.A, bearing.alpha_0, bearing.phi_j
        cp_j = np.cos(phi_j)

        U_j     = A * np.cos(alpha_0) + delta_r * cp_j
        V_j     = A * np.sin(alpha_0) + delta_a + Vpsi
        delta_j = np.maximum(np.sqrt(U_j**2 + V_j**2) - A, 0.0)

        alpha_j = np.arctan2(V_j, U_j)
        d32     = delta_j ** 1.5

        return delta_j, alpha_j, np.cos(alpha_j), np.sin(alpha_j), cp_j, d32

    def _initial_delta_r(self, bearing: Bearing, Fr: float, hint: float) -> float:
        """Initial guess of delta_r for the root-find.

        Returns
        -------
        float
            ``hint`` [mm] (FEM radial displacement) if it is above 1e-6 mm;
            otherwise A*(1 - cos(alpha_0)) + (abs(Fr)/(cp*Z))^(2/3) [mm], Fr in
            N, the second term being 1e-4 mm when Fr = 0. Heuristic of this code,
            not from the standard.
        """
        if hint > 1e-6:
            return hint
        gap   = bearing.A * (1.0 - np.cos(bearing.alpha_0))
        hertz = (abs(Fr) / (bearing.cp * bearing.Z)) ** (2.0/3.0) if Fr != 0.0 else 1e-4
        return gap + hertz

    def _initial_delta_a(self, bearing: Bearing, Fa: float, hint: float) -> float:
        """Initial guess of delta_a for the root-find.

        Returns
        -------
        float
            ``hint`` [mm] (FEM axial displacement) if abs(hint) is above 1e-6 mm;
            otherwise 0.0 when Fa = 0, else (abs(Fa)/(cp*Z))^(2/3) with the sign
            of Fa [mm], Fa in N. Heuristic of this code, not from the standard.
        """
        if abs(hint) > 1e-6:
            return hint
        if Fa == 0.0:
            return 0.0
        hertz = (abs(Fa) / (bearing.cp * bearing.Z)) ** (2.0/3.0)
        return np.copysign(hertz, Fa)

    def solve_contact(self,
                      bearing: Bearing,
                      Fr_xz: float, Fr_xy: float, Fa: float,
                      delta_r_init: float, delta_a_init: float,
                      psi: float, phi_Fr: float) -> BallBearingResult:
        """Solve the radial and axial force equilibrium of one ball bearing.

        Parameters
        ----------
        bearing : Bearing
            Needs A [mm], alpha_0 [rad], phi_j [rad], cp [N/mm^1.5], Ri [mm],
            Dpw [mm], Z and label.
        Fr_xz, Fr_xy : float
            Radial load components in the two shaft planes [N]. Only their
            resultant Fr = hypot(Fr_xz, Fr_xy) enters the equilibrium.
        Fa : float
            Axial load [N], of either sign.
        delta_r_init : float
            Initial guess of delta_r [mm]; a value not above 1e-6 is replaced by
            the heuristic of ``_initial_delta_r``.
        delta_a_init : float
            Initial guess of delta_a [mm]; a magnitude not above 1e-6 is replaced
            by the heuristic of ``_initial_delta_a``.
        psi : float
            Misalignment between the rings in the load plane [rad], prescribed.
        phi_Fr : float
            Direction of the resultant radial load [rad]. Stored in the result;
            it does not enter the equations.

        Returns
        -------
        BallBearingResult
            Single-row result. Shared state: delta_r, delta_a [mm], psi, phi_Fr
            [rad], applied Fr_xz, Fr_xy, Fa [N]. Row: phi_j [rad] (local), delta_j
            [mm], alpha_j [rad], Q_j = cp*delta_j^1.5 [N], reactions Fr_row,
            Fa_row [N] and moment Mz [N mm]. ``n_iter`` holds the number of
            residual evaluations (nfev), not iterations, and ``ok`` the success
            flag, both of the attempt that ``run_root`` kept; ``residual`` is the
            Euclidean norm of the two force residuals [N].

        Warnings
        --------
        Convergence is not enforced: check ``ok`` and ``residual``.

        Notes
        -----
        Equations as coded, with delta_j and alpha_j from ``elements`` and
        Vpsi_j = Ri*sin(psi)*cos(phi_j)::

            Fr - cp * sum_j(delta_j^1.5 * cos(alpha_j) * cos(phi_j)) = 0   Eq. (16)
            Fa - cp * sum_j(delta_j^1.5 * sin(alpha_j)) = 0                Eq. (17)
            Mz = (Dpw/2) * cp * sum_j(delta_j^1.5 * sin(alpha_j) * cos(phi_j))
                                                                           Eq. (18)

        The lever arm of the tilt is Ri and the arm of the moment is Dpw/2, as in
        the standard. The system is solved with ``run_root`` (scipy.optimize.root,
        hybr with lm fallback) and the tolerance ``self.tol``.

        References
        ----------
        .. [1] ISO_TS_16281_2008, 4.2.2, Eq. (12)-(15); 4.2.2.1, Eq. (16)-(17);
           4.2.2.2, Eq. (18).
        """
        Fr   = float(np.hypot(Fr_xz, Fr_xy))
        Vpsi = bearing.Ri * np.sin(psi) * np.cos(bearing.phi_j)
        cp   = bearing.cp

        def residual(u):
            """Radial and axial force residuals [N] for u = (delta_r, delta_a) [mm]; zero at equilibrium."""
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

        delta_j, alpha_j, ca, sa, cp_j, d32 = self.elements(bearing, delta_r, delta_a, Vpsi)
        Mz = (bearing.Dpw / 2.0) * cp * float(np.sum(d32 * sa * cp_j))

        row = BallLoadDistributionResult(
            phi_j=np.asarray(bearing.phi_j, dtype=float),
            delta_j=delta_j, alpha_j=alpha_j,
            Q_j=cp * d32,
            Fr_row=float(cp * np.sum(d32 * ca * cp_j)),
            Fa_row=float(cp * np.sum(d32 * sa)),
            Mz=Mz,
        )
        return BallBearingResult.single(
            row, label=bearing.label,
            delta_r=delta_r, delta_a=delta_a, psi=psi, phi_Fr=phi_Fr,
            Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa,
            n_iter=nfev, residual=res, ok=ok,
        )

    def minimum_axial_load(self,
                            bearing: Bearing,
                            Fr_xz: float, Fr_xy: float,
                            psi: float,
                            delta_r_init: float = 0.0,
                            delta_a_init: float = 0.0,
                            Fa_bracket: tuple[float, float] = (0.0, 5.0e4),
                            xtol: float = 1e-6) -> tuple[float, BallBearingResult]:
        """Smallest axial load for which the axial displacement is not negative.

        The bearing is solved at Fa = 0. If delta_a >= 0 there, the answer is 0.
        Otherwise the root of delta_a(Fa) = 0 is found by brentq inside
        ``Fa_bracket``; each evaluation is a full ``solve_contact``.

        Parameters
        ----------
        bearing : Bearing
            Ball bearing ready for the ISO/TS 16281 solver.
        Fr_xz, Fr_xy : float
            Radial load components in the two shaft planes [N].
        psi : float
            Misalignment between the rings in the load plane [rad].
        delta_r_init : float, optional
            Initial guess of delta_r [mm]. Default 0.0 (heuristic guess).
        delta_a_init : float, optional
            Initial guess of delta_a [mm]. Default 0.0 (heuristic guess).
        Fa_bracket : tuple[float, float], optional
            Lower and upper axial load of the search [N]. Default (0.0, 5.0e4).
        xtol : float, optional
            Absolute tolerance of brentq on Fa [N]. Default 1e-6.

        Returns
        -------
        Fa_min : float
            Minimum axial load [N]; 0.0 if delta_a >= 0 without axial load.
        result : BallBearingResult
            Load distribution at Fa_min.

        Raises
        ------
        RuntimeError
            From ``check_bearing_ready``.
        ValueError
            If delta_a is still negative at the upper end of ``Fa_bracket``;
            from brentq, if delta_a is not negative at a lower end of
            ``Fa_bracket`` above 0.

        Warnings
        --------
        Geometric criterion of this code (ring back at the reference position
        of Eq. (12)); not a minimum load against skidding. No library source
        prints a minimum axial load for ball bearings.

        Notes
        -----
        The load direction is phi_Fr = arctan2(Fr_xy, Fr_xz). delta_a = 0 is the
        reference position of Eq. (12): the centre distance has components
        A*cos(alpha_0), A*sin(alpha_0).

        References
        ----------
        (source to find) -- criterion of this code; not in ISO/TS 16281.
        """
        check_bearing_ready(bearing, bearing.label, self.CAPABILITY)
        phi_Fr = float(np.arctan2(Fr_xy, Fr_xz))

        res0 = self.solve_contact(bearing, Fr_xz, Fr_xy, 0.0,
                                  delta_r_init, delta_a_init, psi, phi_Fr)
        if res0.delta_a >= 0.0:
            return 0.0, res0

        def g(Fa: float) -> float:
            """Axial displacement delta_a [mm] of the solved contact for the axial load Fa [N]."""
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


@register_contact_solver(capability="point_contact")
class ISO16281MultiRowBallSolverSharedDisplacement(MultiRowSolverBase):
    """Multi-row ball bearing with delta_r and delta_a shared by all rows.

    Assumptions and status are in ``MultiRowSolverBase``: co-located rows, not
    part of ISO/TS 16281, not reviewed for row-to-row coupling.
    """

    _result_cls        = BallBearingResult
    SINGLE_ROW_SOLVER  = ISO16281BallSolver

    def _check_row_ready(self, row, row_label: str) -> None:
        """Require a positive spring constant cp [N/mm^1.5] on the row.

        Raises
        ------
        ValueError
            If ``row.cp <= 0``.
        """
        if row.cp <= 0:
            raise ValueError(f"Bearing '{row_label}': cp must be positive; got {row.cp}.")

    def _initial_delta_r(self, row_views, Fr: float) -> float:
        """Initial guess of the shared delta_r for the root-find.

        Returns
        -------
        float
            A*(1 - cos(alpha_0)) + (abs(Fr)/sum(cp*Z))^(2/3) [mm], Fr in N, with
            A and alpha_0 of the first row and the sum over the rows; the second
            term is 1e-4 mm when Fr = 0. Heuristic of this code, not from the
            standard.
        """
        # Pooled heuristic only (root-finder converges regardless of x0
        # quality) -- not an ISO/TS 16281 formula.
        A0, alpha0 = row_views[0].A, row_views[0].alpha_0
        cpZ_total  = sum(rv.cp * rv.Z for rv in row_views)
        gap   = A0 * (1.0 - np.cos(alpha0))
        hertz = (abs(Fr) / cpZ_total) ** (2.0/3.0) if Fr != 0.0 else 1e-4
        return gap + hertz

    def _initial_delta_a(self, row_views, Fa: float) -> float:
        """Initial guess of the shared delta_a for the root-find.

        Returns
        -------
        float
            0.0 when Fa = 0, else (abs(Fa)/sum(cp*Z))^(2/3) with the sign of Fa
            [mm], Fa in N, the sum over the rows. Heuristic of this code, not from
            the standard.
        """
        if Fa == 0.0:
            return 0.0
        cpZ_total = sum(rv.cp * rv.Z for rv in row_views)
        hertz     = (abs(Fa) / cpZ_total) ** (2.0/3.0)
        return np.copysign(hertz, Fa)

    def _solve_rows(self, row_views, Fr_xz, Fr_xy, Fa, psi, phi_Fr):
        """Solve the force equilibrium of all ball rows for the shared displacements.

        The residuals are those of ``ISO16281BallSolver.solve_contact`` (Eq. (16)
        and (17) of ISO_TS_16281_2008) with the sums taken over all rows, each row
        with its own cp, Ri and phi_j. Mz of each row is evaluated after the solve.
        Parameters and returns as in ``MultiRowSolverBase._solve_rows``; Fa is
        returned unchanged.
        """
        Fr       = float(np.hypot(Fr_xz, Fr_xy))
        elements = self.SINGLE_ROW_SOLVER.elements

        def residual(u):
            """Radial and axial force residuals [N], summed over all rows, for u = (delta_r, delta_a) [mm]."""
            dr, da = u
            res_r, res_a = Fr, Fa
            for rv in row_views:
                Vpsi_rv = rv.Ri * np.sin(psi) * np.cos(rv.phi_j)
                _, _, ca, sa, cp_j, d32 = elements(rv, dr, da, Vpsi_rv)
                res_r -= rv.cp * np.sum(d32 * ca * cp_j)
                res_a -= rv.cp * np.sum(d32 * sa)
            return np.array([res_r, res_a])

        dr0 = self._initial_delta_r(row_views, Fr)
        da0 = self._initial_delta_a(row_views, Fa)
        x, nfev, res, ok = run_root(residual, [dr0, da0], self.tol)
        delta_r, delta_a = float(x[0]), float(x[1])

        row_results = []
        for rv in row_views:
            Vpsi_rv = rv.Ri * np.sin(psi) * np.cos(rv.phi_j)
            delta_j, alpha_j, ca, sa, cp_j, d32 = elements(rv, delta_r, delta_a, Vpsi_rv)
            row_results.append(BallLoadDistributionResult(
                phi_j=np.asarray(rv.phi_j, dtype=float),
                delta_j=delta_j, alpha_j=alpha_j,
                Q_j=rv.cp * d32,
                Fr_row=float(rv.cp * np.sum(d32 * ca * cp_j)),
                Fa_row=float(rv.cp * np.sum(d32 * sa)),
                Mz=(rv.Dpw / 2.0) * rv.cp * float(np.sum(d32 * sa * cp_j)),
            ))
        return row_results, delta_r, delta_a, Fa, nfev, res, ok


# =====================================================================
# ---- ISO16281RollerSolver -- line contact -----------------------------
# =====================================================================

@register_contact_solver(capability="line_contact")
class ISO16281RollerSolver(LineContactSolverBase):
    """Load distribution of a radial cylindrical roller bearing (line contact).

    Lamina model of ISO/TS 16281, 5.2: each roller is cut into n_s laminae with
    their own approach and load. One equilibrium equation (radial force) is
    solved for delta_r in the plane of the resultant radial load, with the
    misalignment psi prescribed. The kinematics are in ``elements``, the
    equilibrium in ``solve_contact``.

    Warnings
    --------
    The standard prints two equilibrium equations, Eq. (45)-(46), and does not
    say which quantities are given. Harris2007_vol2, 1.3, solves its
    equivalent system for delta_r, the misalignment theta and the roller
    deflections. Here psi is prescribed, Eq. (45) is solved
    for delta_r, and Eq. (46) gives Mz for that psi.

    The axial load is ignored without a warning: the Fa of the FEM node is
    never read, and Fa, delta_a and Fa_row are 0.0 in the result. The standard
    covers roller bearings under pure radial load (clause 1).

    Registered for every line-contact family of AxisForge, although Eq. (38)
    is the kinematics of a radial bearing. Today only
    ``CylindricalRollerFamily`` reaches this solver: the thrust roller
    families do not set ``iso16281_analysis`` and are rejected by
    ``check_bearing_ready``.

    Steel only: the spring constants cL and cs contain no modulus of
    elasticity (see ``LineContactStiffness``).

    References
    ----------
    .. [1] ISO_TS_16281_2008, clause 1 (scope); 5.2.1, Eq. (34)-(35); 5.2.2,
       Eq. (36)-(41); 5.2.3, Eq. (42)-(44); 5.2.4.1, Eq. (45); 5.2.4.2,
       Eq. (46).
    """

    _result_cls = RollerBearingResult

    def _solve_one_bearing(self, bearing: Bearing, node, phi_Fr: float, psi: float
                            ) -> RollerBearingResult:
        """Solve one roller bearing with the loads and displacements of its FEM node.

        Loads Fr_xz, Fr_xy [N] from the node; the node Fa is not read. Initial
        guess of delta_r from hypot(v_xz, v_xy) [mm].

        Returns
        -------
        RollerBearingResult
            Result of ``solve_contact``.
        """
        return self.solve_contact(
            bearing,
            Fr_xz=node.Fr_xz, Fr_xy=node.Fr_xy,
            delta_r_init=float(np.hypot(node.v_xz, node.v_xy)),
            psi=psi, phi_Fr=phi_Fr,
        )

    @staticmethod
    def _attach_postprocessing(bearing: Bearing, result: RollerBearingResult
                               ) -> BearingAnalysisResult:
        """Add bearing stiffness and basic reference rating life to a roller result.

        Returns
        -------
        BearingAnalysisResult
            ``load_distribution`` with the secant stiffness attached, and
            ``basic_life`` with L10r [10^6 rev] and Pref [N].

        Notes
        -----
        Steps: secant stiffness with Fa = 0 (``bearing_stiffness``); lamina
        equivalent loads q_kei, q_kee with the stress-riser factor, Eq. (60)-(64);
        L10r, Eq. (65), with the two scalar lamina ratings q_kci, q_kce [N] of the
        family used for every lamina; Pref, Eq. (66)-(67), with ``bearing.C`` as Cr or Ca. The ring
        rotation flags keep their defaults (inner ring rotating, outer ring
        stationary relative to the load).

        References
        ----------
        .. [1] ISO_TS_16281_2008, 5.3.2-5.3.6, Eq. (56)-(67).
        """
        stiffness = pp.bearing_stiffness(bearing, result,
                                         Fr_xz=result.Fr_xz, Fr_xy=result.Fr_xy, Fa=0.0)
        result = result.with_stiffness(stiffness)

        q_kci, q_kce = bearing.family.per_lamina_dynamic_capacity(bearing)
        equiv = pp.LaminaDynamicEquivalentLoad.from_distribution(
            bearing, result.row, label=bearing.label,
        )
        q_kci_arr = np.full_like(equiv.q_kei, q_kci)
        q_kce_arr = np.full_like(equiv.q_kee, q_kce)
        life = pp.RollerBasicReferenceRatingLife.from_loads(
            label=bearing.label, q_kci=q_kci_arr, q_kei=equiv.q_kei,
            q_kce=q_kce_arr, q_kee=equiv.q_kee,
        )

        Cr, Ca = SolverBase._Cr_Ca(bearing)
        pref = pp.RollerDynamicEquivalentReferenceLoad.from_L10r(
            label=bearing.label, L10r=life.L10r, Cr=Cr, Ca=Ca,
        )

        basic_life = BasicReferenceRatingLifeResult(
            label=bearing.label, rows=(life,), L10r=life.L10r, pref=pref,
        )
        return BearingAnalysisResult(label=result.label, load_distribution=result,
                                     basic_life=basic_life)

    @staticmethod
    def elements(bearing: Bearing, delta_r: float, psi: float):
        """Approach and load of every roller and of every lamina.

        Parameters
        ----------
        bearing : Bearing
            Needs phi_j [rad], x_k [mm], s [mm], P_xk [mm] and cs [N/mm^(10/9)].
            A row view with the same attributes works too.
        delta_r : float
            Relative radial displacement of the rings along the load line [mm].
        psi : float
            Misalignment between the rings in the load plane [rad].

        Returns
        -------
        delta_j : np.ndarray
            Approach of roller j at its mid-length [mm], shape (Z,). Not clamped:
            negative for a roller out of contact.
        psi_j : np.ndarray
            Misalignment in the plane of roller j [rad], shape (Z,).
        delta_jk : np.ndarray
            Elastic deflection of lamina k of roller j [mm], >= 0, shape (Z, n_s).
        q_jk : np.ndarray
            Load on lamina k of roller j [N], shape (Z, n_s).
        cos_phi_j : np.ndarray
            cos(phi_j) [-]; named ``cp_j`` in the code.

        Notes
        -----
        Equations as coded::

            delta_j  = delta_r*cos(phi_j) - s/2                        Eq. (38)
            psi_j    = arctan(tan(psi)*cos(phi_j))                     Eq. (39)
            delta_jk = max(delta_j - x_k*tan(psi_j) - 2*P(x_k), 0)     Eq. (41)
            q_jk     = cs * delta_jk^(10/9)                            Eq. (36)

        s is the radial operating clearance, x_k the position of the lamina centre
        from the roller mid-length and P(x_k) the roller profile stored on the
        bearing as ``P_xk`` (Eq. (42)-(44)).

        Conventions: phi_j is the local angular position, zero on the line of the
        resultant radial load. A positive psi_j reduces the approach at positive
        x_k. No edge-stress correction is applied to q_jk here.

        References
        ----------
        .. [1] ISO_TS_16281_2008, 5.2.2, Eq. (36), (38)-(41).
        """
        phi_j = bearing.phi_j
        x_k   = bearing.x_k
        s     = bearing.s

        cp_j    = np.cos(phi_j)
        delta_j = delta_r * cp_j - s / 2.0                          # eq.(38)
        psi_j   = np.arctan(np.tan(psi) * cp_j)                     # eq.(39)

        P_xk = bearing.P_xk   # eq.(42)-(44)

        delta_jk = (delta_j[:, None]
                    - x_k[None, :] * np.tan(psi_j)[:, None]
                    - 2.0 * P_xk[None, :])                          # eq.(41)
        delta_jk = np.maximum(delta_jk, 0.0)

        q_jk = bearing.cs * delta_jk ** (10.0 / 9.0)                # eq.(36)

        return delta_j, psi_j, delta_jk, q_jk, cp_j

    def _initial_delta_r(self, bearing: Bearing, Fr: float, hint: float) -> float:
        """Initial guess of delta_r for the root-find.

        Returns
        -------
        float
            ``hint`` [mm] (FEM radial displacement) if it is above 1e-6 mm;
            otherwise s/2 + (abs(Fr)/(cL*Z))^(9/10) [mm], Fr in N, the second term
            being 1e-4 mm when Fr = 0. Heuristic of this code, not from the
            standard.
        """
        if hint > 1e-6:
            return hint
        gap = bearing.s / 2.0
        line_hertz = (abs(Fr) / (bearing.cL * bearing.Z)) ** (9.0 / 10.0) if Fr != 0.0 else 1e-4
        return gap + line_hertz

    def solve_contact(self,
                      bearing: Bearing,
                      Fr_xz: float, Fr_xy: float,
                      delta_r_init: float,
                      psi: float, phi_Fr: float) -> RollerBearingResult:
        """Solve the radial force equilibrium of one radial roller bearing.

        Parameters
        ----------
        bearing : Bearing
            Needs phi_j [rad], x_k [mm], s [mm], P_xk [mm], cs and cL
            [N/mm^(10/9)], alpha_0 [rad], Z and label.
        Fr_xz, Fr_xy : float
            Radial load components in the two shaft planes [N]. Only their
            resultant Fr = hypot(Fr_xz, Fr_xy) enters the equilibrium.
        delta_r_init : float
            Initial guess of delta_r [mm]; a value not above 1e-6 is replaced by
            the heuristic of ``_initial_delta_r``.
        psi : float
            Misalignment between the rings in the load plane [rad], prescribed.
        phi_Fr : float
            Direction of the resultant radial load [rad]. Stored in the result;
            it does not enter the equations.

        Returns
        -------
        RollerBearingResult
            Single-row result. Shared state: delta_r [mm], psi, phi_Fr [rad],
            applied Fr_xz, Fr_xy [N]; delta_a and Fa are 0.0. Row: phi_j [rad]
            (local), delta_j [mm] (not clamped), Q_j = sum_k(q_jk) [N], Fr_row [N],
            Fa_row = 0.0, Mz [N mm], and the lamina data x_k [mm], psi_j [rad],
            delta_jk [mm], q_jk [N]. alpha_j is ``bearing.alpha_0`` for every
            roller (the lamina model has no contact angle). ``n_iter`` holds the
            number of residual evaluations (nfev), not iterations, and ``ok`` the
            success flag, both of the attempt that ``run_root`` kept; ``residual``
            is the absolute force residual [N].

        Warnings
        --------
        Convergence is not enforced: check ``ok`` and ``residual``.

        Notes
        -----
        Equations as coded, with q_jk from ``elements``::

            Fr - sum_j(cos(phi_j) * sum_k(q_jk)) = 0                   Eq. (45)
            Mz = sum_j(cos(phi_j) * sum_k(x_k * q_jk))                 Eq. (46)

        The system is solved with ``run_root`` (scipy.optimize.root, hybr with lm
        fallback) and the tolerance ``self.tol``.

        References
        ----------
        .. [1] ISO_TS_16281_2008, 5.2.4.1, Eq. (45); 5.2.4.2, Eq. (46).
        """
        Fr = float(np.hypot(Fr_xz, Fr_xy))

        def residual(u):
            """Radial force residual [N] (one-element array) for u = (delta_r,) [mm]; zero at equilibrium."""
            dr = u[0]
            _, _, _, q_jk, cp_j = self.elements(bearing, dr, psi)
            Fr_internal = float(np.sum(cp_j * np.sum(q_jk, axis=1)))
            return np.array([Fr - Fr_internal])

        dr0              = self._initial_delta_r(bearing, Fr, delta_r_init)
        x, nfev, res, ok = run_root(residual, [dr0], self.tol)
        delta_r          = float(x[0])

        delta_j, psi_j, delta_jk, q_jk, cp_j = self.elements(bearing, delta_r, psi)
        x_k = np.asarray(bearing.x_k, dtype=float)
        Q_j = np.sum(q_jk, axis=1)

        Mz = float(np.sum(cp_j * np.sum(x_k[None, :] * q_jk, axis=1)))  # eq.(46), diagnostic
        alpha_j = np.full(bearing.Z, bearing.alpha_0, dtype=float)

        row = RollerLoadDistributionResult(
            phi_j=np.asarray(bearing.phi_j, dtype=float),
            delta_j=delta_j, alpha_j=alpha_j,
            Q_j=Q_j,
            Fr_row=float(np.sum(cp_j * Q_j)),
            Fa_row=0.0,
            Mz=Mz,
            x_k=x_k, psi_j=psi_j, delta_jk=delta_jk, q_jk=q_jk,
        )
        return RollerBearingResult.single(
            row, label=bearing.label,
            delta_r=delta_r, delta_a=0.0, psi=psi, phi_Fr=phi_Fr,
            Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=0.0,
            n_iter=nfev, residual=res, ok=ok,
        )


@register_contact_solver(capability="line_contact")
class ISO16281MultiRowRollerSolverSharedDisplacement(MultiRowSolverBase):
    """Multi-row radial roller bearing with delta_r shared by all rows.

    Only one unknown: there is no axial load and no delta_a. Assumptions and
    status are in ``MultiRowSolverBase``: co-located rows, not part of
    ISO/TS 16281.

    Warnings
    --------
    Not validated: no AxisForge family assembles a multi-row radial roller
    bearing to check it against.
    """

    _result_cls        = RollerBearingResult
    SINGLE_ROW_SOLVER  = ISO16281RollerSolver

    def _check_row_ready(self, row, row_label: str) -> None:
        """Check the lamina discretisation of the row.

        Raises
        ------
        ValueError
            If ``row.n_s`` is below ``MIN_LAMINAE`` of the single-row solver, or
            ``len(row.x_k) != row.n_s``.
        """
        min_laminae = self.SINGLE_ROW_SOLVER.MIN_LAMINAE
        if row.n_s < min_laminae:
            raise ValueError(
                f"Bearing '{row_label}': n_s = {row.n_s} < {min_laminae} -- "
                f"ISO/TS 16281 Sec 5.2.2 requires at least {min_laminae} laminae."
            )
        if len(row.x_k) != row.n_s:
            raise ValueError(
                f"Bearing '{row_label}': len(x_k) = {len(row.x_k)} != n_s = {row.n_s}."
            )

    def _initial_delta_r(self, row_views, Fr: float) -> float:
        """Initial guess of the shared delta_r for the root-find.

        Returns
        -------
        float
            s/2 + (abs(Fr)/sum(cL*Z))^(9/10) [mm], Fr in N, with s of the first
            row and the sum over the rows; the second term is 1e-4 mm when Fr = 0.
            Heuristic of this code, not from the standard.
        """
        s0         = row_views[0].s
        cLZ_total  = sum(rv.cL * rv.Z for rv in row_views)
        gap        = s0 / 2.0
        line_hertz = (abs(Fr) / cLZ_total) ** (9.0/10.0) if Fr != 0.0 else 1e-4
        return gap + line_hertz

    def _solve_rows(self, row_views, Fr_xz, Fr_xy, Fa, psi, phi_Fr):
        """Solve the radial equilibrium of all roller rows for the shared delta_r.

        The residual is that of ``ISO16281RollerSolver.solve_contact`` (Eq. (45) of
        ISO_TS_16281_2008) with the sum taken over all rows. Mz of each row is
        evaluated after the solve. Parameters and returns as in
        ``MultiRowSolverBase._solve_rows``; Fa is ignored, and delta_a and Fa are
        returned as 0.0.
        """
        # Fa ignored on purpose -- radial roller, no axial capacity (returned as 0.0).
        Fr       = float(np.hypot(Fr_xz, Fr_xy))
        elements = self.SINGLE_ROW_SOLVER.elements

        def residual(u):
            """Radial force residual [N] (one-element array), summed over all rows, for u = (delta_r,) [mm]."""
            dr = u[0]
            res_r = Fr
            for rv in row_views:
                _, _, _, q_jk, cp_j = elements(rv, dr, psi)
                res_r -= float(np.sum(cp_j * np.sum(q_jk, axis=1)))
            return np.array([res_r])

        dr0 = self._initial_delta_r(row_views, Fr)
        x, nfev, res, ok = run_root(residual, [dr0], self.tol)
        delta_r = float(x[0])

        row_results = []
        for rv in row_views:
            delta_j, psi_j, delta_jk, q_jk, cp_j = elements(rv, delta_r, psi)
            x_k = np.asarray(rv.x_k, dtype=float)
            Q_j = np.sum(q_jk, axis=1)
            row_results.append(RollerLoadDistributionResult(
                phi_j=np.asarray(rv.phi_j, dtype=float),
                delta_j=delta_j,
                alpha_j=np.full(rv.Z, rv.alpha_0, dtype=float),
                Q_j=Q_j,
                Fr_row=float(np.sum(cp_j * Q_j)),
                Fa_row=0.0,
                Mz=float(np.sum(cp_j * np.sum(x_k[None, :] * q_jk, axis=1))),
                x_k=x_k, psi_j=psi_j, delta_jk=delta_jk, q_jk=q_jk,
            ))
        return row_results, delta_r, 0.0, 0.0, nfev, res, ok


# Link each single-row solver to its multi-row sibling.
ISO16281BallSolver.MULTIROW_SOLVER   = ISO16281MultiRowBallSolverSharedDisplacement
ISO16281RollerSolver.MULTIROW_SOLVER = ISO16281MultiRowRollerSolverSharedDisplacement