"""
ISO/TS 16281 internal load distribution -- result shapes, data only.
Every equation lives in solvers/ (contact_solver.py, contact_postprocessing.py).

Two levels
----------
BearingResult           one per bearing. Shared state of the solve
                        (delta_r, delta_a, psi, phi_Fr), node force (K d) at the
                        FEM node, convergence of THE solve, bearing-level
                        stiffness. n_rows >= 1 -- single-row is n_rows == 1,
                        no separate class.
LoadDistributionResult  one per row. Only what differs between rows:
                        element kinematics, contact forces, row reactions.

Why shared state lives on the bearing: the multi-row solver is co-located /
shared-displacement (contact_solver.MultiRowSolverBase) -- every row sees
identical delta_r, delta_a, psi by construction, and there is ONE root-find,
so ONE n_iter/residual/ok. If axial row offset is modelled later, per-row
LOCAL displacements are added to the row as optional fields; the bearing
level stays unchanged.

Contact type (ball / roller) is the only axis with concrete classes.

References
----------
ISO/TS 16281:2008 Sec 4.2.1 eq.(10), 4.2.2 eq.(12)-(18) -- point contact
ISO/TS 16281:2008 Sec 5.2 eq.(36)-(46) -- line contact, lamina model
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import numpy as np

from axisforge.results._base import DC, check_finite, check_shape, reject_if_abstract

if TYPE_CHECKING:
    # ContactBearingStiffness lives in contact_postprocessing.py (with its
    # from_result()). Type-only import: no runtime dependency on solvers/.
    from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_postprocessing import (
        ContactBearingStiffness,
    )


# =====================================================================
# ---- Row level --------------------------------------------------------
# =====================================================================

@dataclass(**DC)
class LoadDistributionResult(ABC):
    """Load distribution of one row of rolling elements (abstract).

    Kinematics and contact forces of the Z rolling elements of one row, plus the reactions of
    the row. Arrays have shape (Z,); Z may differ between rows. Data only: the solver computes
    the values, this class checks shapes and signs.

    Parameters
    ----------
    phi_j : ndarray, shape (Z,)
        Angular position of each rolling element in the LOCAL frame of the bearing [rad]. The
        origin (0) is the direction in which the solver applies the radial force to the inner
        ring (the direction of delta_r, as F_r in ISO_TS_16281_2008 Figure 4); the most loaded
        element is at phi_j = 0. See ``BearingResult.phi_j_global`` and its Warnings for the
        global angle.
    delta_j : ndarray, shape (Z,)
        Total elastic deflection of the rolling element, both contacts together (approach of
        the two raceways along the contact line) [mm]. Balls: ISO_TS_16281_2008 Eq. (12), never
        negative. Rollers: Eq. (38), not clamped, negative for a roller out of contact.
    alpha_j : ndarray, shape (Z,)
        Contact angle of each element [rad]. Balls: operating contact angle,
        ISO_TS_16281_2008 Eq. (15). Rollers: the nominal angle ``bearing.alpha_0`` repeated for
        every roller (the lamina model has no contact angle); it is a filler, not a result.
    Q_j : ndarray, shape (Z,)
        Normal contact force of each rolling element [N], >= 0.
    Fr_row : float
        Radial reaction of the row, along phi_Fr [N].
    Fa_row : float
        Axial reaction of the row [N], signed; 0.0 for a radial roller bearing.
    Mz : float
        Tilting moment of the row in the plane of the load line (the plane of psi) [N mm]:
        ISO_TS_16281_2008 Eq. (18) for balls (arm D_pw/2), Eq. (46) for rollers. The subscript z
        is the symbol of the standard, not the global Z axis of AxisForge.

    Raises
    ------
    TypeError
        If the class is abstract, or ``phi_j``, ``alpha_j`` or ``Q_j`` (and the lamina arrays)
        is not an ndarray.
    ValueError
        If ``delta_j`` is not a 1-D array, the arrays do not share the shape (Z,), any ``Q_j``
        is negative, or a scalar is not finite.

    Notes
    -----
    ISO_TS_16281_2008 gives phi_j only as the "angular position of rolling element j" (clause 3)
    and draws it in Figure 4 from the direction of F_r (Figure 4 is a roller bearing; for ball
    bearings the origin is only implied by the cos phi_j terms of Eq. (12) and (16)). The standard has a single load plane;
    the global angle (phi_Fr) is a construct of this code (D-0010). Angles are in rad here; the
    standard prints degrees.

    References
    ----------
    .. [1] ISO_TS_16281_2008, clause 3 (symbols and units); (ball) 4.2.1, Eq. (10); 4.2.2,
       Eq. (12)-(18); (roller) 5.2.2-5.2.4, Eq. (36)-(46).
    """
    _ABSTRACT      : ClassVar[bool] = True
    IS_LINE_CONTACT: ClassVar[bool]

    phi_j   : np.ndarray   # (Z,) element angular position, LOCAL frame (0 = phi_Fr) [rad]
    delta_j : np.ndarray   # (Z,) element deflection [mm] (roller: raw approach eq.38, may be < 0)
    alpha_j : np.ndarray   # (Z,) operating contact angle [rad]
    Q_j     : np.ndarray   # (Z,) element contact force [N] -- plots / reports
    Fr_row  : float        # row reaction along phi_Fr [N]
    Fa_row  : float        # row axial reaction [N] -- signed; 0.0 for radial roller
    Mz      : float        # row moment reaction [N mm]

    def __post_init__(self):
        """Check the array shapes, Q_j >= 0 and the finiteness of the row scalars."""
        reject_if_abstract(self)
        owner = type(self).__name__
        if not isinstance(self.delta_j, np.ndarray) or self.delta_j.ndim != 1:
            raise ValueError(f"{owner}: delta_j must be a 1-D ndarray (Z,).")
        Z = (self.delta_j.shape[0],)
        for name in ("phi_j", "alpha_j", "Q_j"):
            check_shape(owner, name, getattr(self, name), Z)
        if np.any(self.Q_j < 0.0):
            raise ValueError(f"{owner}: Q_j must be >= 0.")
        check_finite(owner, Fr_row=self.Fr_row, Fa_row=self.Fa_row, Mz=self.Mz)

    @property
    def Z(self) -> int:
        """Number of rolling elements in the row [-]."""
        return self.delta_j.shape[0]

    @property
    def n_loaded(self) -> int:
        """Number of rolling elements in contact (Q_j > 0), the size of the loaded zone [-]."""
        return int(np.count_nonzero(self.Q_j > 0.0))


@dataclass(**DC)
class BallLoadDistributionResult(LoadDistributionResult):
    """Load distribution of one row of balls (point contact).

    Same fields as ``LoadDistributionResult``. The solver computes the ball force from the
    Hertz point-contact law Q_j = cp * delta_j^1.5 (cp [N/mm^1.5], delta_j [mm]); this class
    does not check it.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 4.2.1, Eq. (10) (Q = c_P delta^(3/2)); 4.2.2, Eq. (12) (delta_j)
       and Eq. (15) (alpha_j).
    """
    IS_LINE_CONTACT: ClassVar[bool] = False


@dataclass(**DC)
class LineContactLoadDistributionResult(LoadDistributionResult):
    """Load distribution of one row of rollers (line contact, lamina model; abstract).

    Adds the lamina data to ``LoadDistributionResult``. The roller force is the sum of the
    lamina loads, Q_j = sum_k q_jk; it is stored redundantly (for a uniform row interface
    across contact types) and checked here (relative tolerance 1e-9).

    Parameters
    ----------
    x_k : ndarray, shape (n_s,)
        Mid-position of each lamina along the roller axis [mm].
    psi_j : ndarray, shape (Z,)
        Tilt of each roller, ISO_TS_16281_2008 Eq. (39) [rad].
    delta_jk : ndarray, shape (Z, n_s)
        Approach of each lamina of each roller, Eq. (41) [mm].
    q_jk : ndarray, shape (Z, n_s)
        Load of each lamina of each roller, Eq. (36) [N].

    Raises
    ------
    ValueError
        If ``x_k`` is not 1-D, an array has the wrong shape, or Q_j differs from
        ``q_jk.sum(axis=1)``. The checks of ``LoadDistributionResult`` also apply.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 5.2.2-5.2.4, Eq. (36)-(46).
    """
    _ABSTRACT      : ClassVar[bool] = True
    IS_LINE_CONTACT: ClassVar[bool] = True
    _Q_J_RTOL      : ClassVar[float] = 1e-9

    x_k      : np.ndarray   # (n_s,)   lamina mid-positions [mm]
    psi_j    : np.ndarray   # (Z,)     roller tilt, eq.(39) [rad]
    delta_jk : np.ndarray   # (Z, n_s) lamina deflection, eq.(41) [mm]
    q_jk     : np.ndarray   # (Z, n_s) lamina load, eq.(36) [N]

    def __post_init__(self):
        """Check the lamina array shapes and Q_j = sum_k q_jk."""
        super().__post_init__()
        owner = type(self).__name__
        if not isinstance(self.x_k, np.ndarray) or self.x_k.ndim != 1:
            raise ValueError(f"{owner}: x_k must be a 1-D ndarray (n_s,).")
        check_shape(owner, "psi_j", self.psi_j, (self.Z,))
        check_shape(owner, "delta_jk", self.delta_jk, (self.Z, self.n_s))
        check_shape(owner, "q_jk", self.q_jk, (self.Z, self.n_s))
        if not np.allclose(self.Q_j, self.q_jk.sum(axis=1), rtol=self._Q_J_RTOL, atol=0.0):
            raise ValueError(f"{owner}: Q_j != q_jk.sum(axis=1).")

    @property
    def n_s(self) -> int:
        """Number of laminae per roller [-]."""
        return self.x_k.shape[0]


@dataclass(**DC)
class RollerLoadDistributionResult(LineContactLoadDistributionResult):
    """Load distribution of one row of radial cylindrical rollers (NU/N).

    Zero nominal contact angle; same fields as ``LineContactLoadDistributionResult``.
    """


# =====================================================================
# ---- Bearing level ----------------------------------------------------
# =====================================================================

RowT = TypeVar("RowT", bound=LoadDistributionResult)


@dataclass(**DC)
class BearingResult(ABC, Generic[RowT]):
    """Load distribution of a whole bearing: shared solve state, node force and rows (abstract).

    One object per bearing (one or more rows). It holds the state of the single root find of
    the ISO/TS 16281 solver: the unknowns delta_r and delta_a (delta_r only for radial rollers)
    and the prescribed tilt psi, the force of the FEM bearing node used as bearing load, the
    convergence of the solve and the per-row results. Data only.

    Parameters
    ----------
    label : str
        Bearing label [-].
    rows : tuple of RowT
        Non-empty tuple of row results, all of type ``ROW_TYPE``.
    delta_r : float
        Radial approach of the rings along phi_Fr [mm].
    delta_a : float
        Axial approach of the rings [mm]; 0.0 for a radial roller bearing.
    psi : float
        Tilt (misalignment) of the rings, projected on the plane of phi_Fr [rad], or the value
        of ``psi_override`` when the solver was run with ``psi_input=True``.
    phi_Fr : float
        Direction of the radial force of the bearing node in the GLOBAL frame [rad],
        phi_Fr = arctan2(Fr_xy, Fr_xz): zero along +Z (the XZ component), positive toward +Y
        (the XY component). It is not the load angle theta of the README (measured from +Y
        toward +Z): phi_Fr = pi/2 - theta.
    Fr_xz, Fr_xy : float
        Radial force components stored on the FEM bearing node (``BearingNodeData.Fr_xz``,
        ``Fr_xy``), with the sign they have there [N]; _xz is along Z, _xy along Y.
    Fa : float
        Axial force of the FEM bearing node (``BearingNodeData.Fa``: row of ``K d``, non-zero
        only for a ``"locating"`` bearing), with the sign it has there [N]. Ball solvers use it
        as the axial load; the radial roller solver does not read it and stores 0.0.
    n_iter : int
        Number of residual evaluations (``nfev``) of the root find, of the attempt that was kept
        [-]. It is not a count of iterations, despite the name.
    residual : float
        Euclidean norm of the force residual at the solution [N].
    ok : bool
        Convergence flag of the root find; convergence is not enforced, check it.
    stiffness : ContactBearingStiffness or None, optional
        Secant stiffness F/delta [N/mm] of the bearing, set by the post-processing; None before
        it. For the stiffness equations and limitations see ``ContactBearingStiffness``. The
        solver builds it with the axial load Fa of the result for balls and with Fa = 0.0 for
        radial rollers.

    Raises
    ------
    ValueError
        If ``rows`` is empty or not a tuple, one of delta_r, delta_a, psi, phi_Fr, Fr_xz, Fr_xy,
        Fa is not finite (``residual`` is not checked), or ``stiffness.label`` differs from
        ``label``.
    TypeError
        If the class is abstract or a row is not an instance of ``ROW_TYPE``.

    Warnings
    --------
    ``Fr_xz``, ``Fr_xy`` and ``Fa`` are rows of ``K d`` of the shaft FEM at the bearing node:
    the force of the support ON THE SHAFT (see ``BearingNodeData``). The load of the shaft on
    the inner ring is the opposite. ``phi_Fr`` is therefore the direction of the support
    reaction, and ``phi_j_global`` places the most loaded element at ``phi_Fr``, 180 deg from
    the side where the shaft pushes the inner ring. Open review item, not corrected in the
    code.

    Notes
    -----
    Frames (D-0010): the element angle ``rows[i].phi_j`` is LOCAL (0 on the radial load line);
    ``phi_Fr`` is global; the global element angle is
    ``phi_j_global(i) = (phi_j + phi_Fr) mod 2 pi``, in [0, 2 pi). ISO_TS_16281_2008 has a single
    load plane and gives phi_j only as "angular position of rolling element j" (clause 3,
    Figure 4); ``phi_Fr`` and the global angle are constructs of this code. The solver gets
    ``psi = psi_xz cos(phi_Fr) + psi_xy sin(phi_Fr)``.

    All rows share delta_r, delta_a and psi (co-located rows, one root find). ISO/TS 16281
    gives the equations for single-row bearings only; for multi-row bearings clause 1 says the
    analysis 'can be derived from the equations' with each row considered separately. The
    shared-displacement rows of this code (co-located rows, one root find) are such a
    derivation, not printed in the standard. Angles are in rad here and in the records (keys
    ``*_rad``); only the printed element table shows degrees.

    References
    ----------
    .. [1] ISO_TS_16281_2008, clause 1 and 3 (symbols, units); (ball) 4.2.1, Eq. (10); 4.2.2,
       Eq. (12)-(18); (roller) 5.2.2-5.2.4, Eq. (36)-(46).
    """
    _ABSTRACT: ClassVar[bool] = True
    ROW_TYPE : ClassVar[type[LoadDistributionResult]]
    _F_EPS   : ClassVar[float] = 1e-9    # [N] below this a total load counts as zero

    label    : str
    rows     : tuple[RowT, ...]
    # --- shared state: unknowns / prescribed inputs of the solve ---
    delta_r  : float        # [mm] along phi_Fr
    delta_a  : float        # [mm]
    psi      : float        # [rad] tilt, projected on phi_Fr plane
    phi_Fr   : float        # [rad] direction of Fr in global frame
    # --- force of the FEM bearing node (K d), used as bearing load ---
    Fr_xz    : float        # [N]
    Fr_xy    : float        # [N]
    Fa       : float        # [N]
    # --- convergence of THE solve (one root-find, 1 or N rows) ---
    n_iter   : int          # NOTE: holds nfev (number of residual evaluations), not iterations
    residual : float        # [N] norm of the force residual
    ok       : bool
    # --- bearing-level postprocessing ---
    stiffness: ContactBearingStiffness | None = None

    def __post_init__(self):
        """Check the rows, the finiteness of the shared state and the stiffness label."""
        reject_if_abstract(self)
        owner = f"{type(self).__name__}({self.label!r})"
        if not isinstance(self.rows, tuple) or len(self.rows) == 0:
            raise ValueError(f"{owner}: rows must be a non-empty tuple.")
        if not all(isinstance(r, self.ROW_TYPE) for r in self.rows):
            raise TypeError(f"{owner}: every row must be {self.ROW_TYPE.__name__}.")
        check_finite(owner, delta_r=self.delta_r, delta_a=self.delta_a, psi=self.psi,
                     phi_Fr=self.phi_Fr, Fr_xz=self.Fr_xz, Fr_xy=self.Fr_xy, Fa=self.Fa)
        if self.stiffness is not None and self.stiffness.label != self.label:
            raise ValueError(f"{owner}: stiffness.label {self.stiffness.label!r} != label.")

    # ---- constructors -------------------------------------------------
    @classmethod
    def single(cls, row: RowT, **shared):
        """Build a single-row bearing result from one row and the shared fields (keywords)."""
        return cls(rows=(row,), **shared)

    @classmethod
    def multirow(cls, rows, **shared):
        """Build a multi-row (>= 2) bearing result from the rows and the shared fields.

        Raises
        ------
        ValueError
            If fewer than two rows are given.
        """
        rows = tuple(rows)
        if len(rows) < 2:
            raise ValueError(f"{cls.__name__}.multirow: need >= 2 rows; got {len(rows)}.")
        return cls(rows=rows, **shared)

    def with_stiffness(self, stiffness: ContactBearingStiffness):
        """Copy of this result with the secant stiffness [N/mm] attached."""
        return replace(self, stiffness=stiffness)

    # ---- structure ----------------------------------------------------
    @property
    def n_rows(self) -> int:
        """Number of bearing rows [-]."""
        return len(self.rows)

    @property
    def is_single(self) -> bool:
        """True for a single-row bearing."""
        return len(self.rows) == 1

    @property
    def is_line_contact(self) -> bool:
        """True for line contact (rollers), False for point contact (balls)."""
        return self.ROW_TYPE.IS_LINE_CONTACT

    @property
    def row(self) -> RowT:
        """The row of a single-row bearing. Raises on multi-row -- use .rows."""
        if not self.is_single:
            raise AttributeError(f"{self.label}: .row is single-row only; "
                                 f"this bearing has {self.n_rows} rows -- use .rows[i].")
        return self.rows[0]

    # ---- loads (arithmetic on stored data, no physics) ----------------
    @property
    def Fr(self) -> float:
        """Magnitude of the radial node force, hypot(Fr_xz, Fr_xy) [N]."""
        return float(np.hypot(self.Fr_xz, self.Fr_xy))

    @property
    def Mz(self) -> float:
        """Tilting moment of the bearing in the plane of the load line, the sum of the row moments [N mm].

        ``Mz`` is the symbol of ISO_TS_16281_2008 (Eq. (18), (46)), not a moment about the global Z axis.
        """
        return float(sum(r.Mz for r in self.rows))

    @property
    def f_r(self) -> tuple[float, ...] | None:
        """Share of Fr per row, Fr_row / Fr [-]. None when Fr ~ 0 -- a share of nothing is undefined."""
        if self.Fr < self._F_EPS:
            return None
        return tuple(r.Fr_row / self.Fr for r in self.rows)

    @property
    def f_a(self) -> tuple[float, ...] | None:
        """Share of Fa per row, Fa_row / Fa [-]. None when Fa ~ 0 -- read .rows[i].Fa_row instead:
        e.g. a preloaded back-to-back pair has Fa = 0 but rows carry +/-Fa_row."""
        if abs(self.Fa) < self._F_EPS:
            return None
        return tuple(r.Fa_row / self.Fa for r in self.rows)

    @property
    def equilibrium_error(self) -> tuple[float, float]:
        """Static equilibrium error of the bearing: (Fr - sum Fr_row, Fa - sum Fa_row) [N]."""
        return (self.Fr - sum(r.Fr_row for r in self.rows),
                self.Fa - sum(r.Fa_row for r in self.rows))

    # ---- geometry -----------------------------------------------------
    def phi_j_global(self, i: int = 0) -> np.ndarray:
        """Angular positions of the elements of row ``i`` in the GLOBAL frame [rad].

        Parameters
        ----------
        i : int
            Row index (0 for single-row bearings).

        Returns
        -------
        phi : ndarray, shape (Z,)
            ``(phi_j + phi_Fr) mod 2 pi``, in [0, 2 pi) [rad]: the local element angle
            (0 on the radial load line) rotated by phi_Fr, the direction of the node force
            (see the Warnings of the class: 180 deg from the load on the inner ring). The
            global angle is zero along +Z and positive toward +Y.
        """
        return (self.rows[i].phi_j + self.phi_Fr) % (2.0 * np.pi)


@dataclass(**DC)
class BallBearingResult(BearingResult[BallLoadDistributionResult]):
    """Bearing-level load distribution of a ball bearing (point contact; rows are
    ``BallLoadDistributionResult``). Fields as in ``BearingResult``."""
    ROW_TYPE: ClassVar[type] = BallLoadDistributionResult


@dataclass(**DC)
class RollerBearingResult(BearingResult[RollerLoadDistributionResult]):
    """RADIAL cylindrical roller (NU/N): no axial capacity. Fa, delta_a and
    every Fa_row are 0.0 by definition -- not "0 by solve".

    NOTE (to review): thrust roller bearings are the mirror case -- Fa only,
    no Fr/delta_r. This class models radial roller only; a thrust roller
    needs its own result class (or a duty axis), not a reuse of this one."""
    ROW_TYPE: ClassVar[type] = RollerLoadDistributionResult

    def __post_init__(self):
        """Run the base checks, then require Fa = delta_a = Fa_row = 0.0 (no axial load)."""
        super().__post_init__()
        if self.Fa != 0.0 or self.delta_a != 0.0 or any(r.Fa_row != 0.0 for r in self.rows):
            raise ValueError(f"RollerBearingResult({self.label!r}): Fa, delta_a and Fa_row "
                             f"must be 0.0 -- radial roller bearings carry no axial load.")