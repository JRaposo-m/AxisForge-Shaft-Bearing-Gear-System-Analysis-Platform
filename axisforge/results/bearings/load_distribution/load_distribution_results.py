"""
axisforge/results/bearings/load_distribution/load_distribution_results.py

ISO/TS 16281 internal load distribution -- result shapes, data only.
Every equation lives in solvers/ (contact_solver.py, contact_postprocessing.py).

Two levels
----------
BearingResult           one per bearing. Shared state of the solve
                        (delta_r, delta_a, psi, phi_Fr), applied load at the
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
ISO/TS 16281:2008 Sec 4.2 eq.(12)-(15) -- point contact
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
    """One row. Arrays are (Z,) -- Z may differ between rows."""
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
        return self.delta_j.shape[0]

    @property
    def n_loaded(self) -> int:
        """Elements in contact (Q_j > 0) -- loaded-zone size."""
        return int(np.count_nonzero(self.Q_j > 0.0))


@dataclass(**DC)
class BallLoadDistributionResult(LoadDistributionResult):
    """Point contact, Sec 4.2. Q_j = cp * delta_j^1.5 (computed by the solver)."""
    IS_LINE_CONTACT: ClassVar[bool] = False


@dataclass(**DC)
class LineContactLoadDistributionResult(LoadDistributionResult):
    """Line contact, lamina model Sec 5.2. Q_j = sum_k q_jk -- checked, since
    it is stored redundantly for a uniform row API across contact types."""
    _ABSTRACT      : ClassVar[bool] = True
    IS_LINE_CONTACT: ClassVar[bool] = True
    _Q_J_RTOL      : ClassVar[float] = 1e-9

    x_k      : np.ndarray   # (n_s,)   lamina mid-positions [mm]
    psi_j    : np.ndarray   # (Z,)     roller tilt, eq.(39) [rad]
    delta_jk : np.ndarray   # (Z, n_s) lamina deflection, eq.(41) [mm]
    q_jk     : np.ndarray   # (Z, n_s) lamina load, eq.(36) [N]

    def __post_init__(self):
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
        return self.x_k.shape[0]


@dataclass(**DC)
class RollerLoadDistributionResult(LineContactLoadDistributionResult):
    """Radial cylindrical roller (NU/N), zero nominal contact angle."""


# =====================================================================
# ---- Bearing level ----------------------------------------------------
# =====================================================================

RowT = TypeVar("RowT", bound=LoadDistributionResult)


@dataclass(**DC)
class BearingResult(ABC, Generic[RowT]):
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
    # --- applied load at the FEM bearing node ---
    Fr_xz    : float        # [N]
    Fr_xy    : float        # [N]
    Fa       : float        # [N]
    # --- convergence of THE solve (one root-find, 1 or N rows) ---
    n_iter   : int          # NOTE: run_root returns nfev -- confirm naming
    residual : float
    ok       : bool
    # --- bearing-level postprocessing ---
    stiffness: ContactBearingStiffness | None = None

    def __post_init__(self):
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
        return cls(rows=(row,), **shared)

    @classmethod
    def multirow(cls, rows, **shared):
        rows = tuple(rows)
        if len(rows) < 2:
            raise ValueError(f"{cls.__name__}.multirow: need >= 2 rows; got {len(rows)}.")
        return cls(rows=rows, **shared)

    def with_stiffness(self, stiffness: ContactBearingStiffness):
        return replace(self, stiffness=stiffness)

    # ---- structure ----------------------------------------------------
    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def is_single(self) -> bool:
        return len(self.rows) == 1

    @property
    def is_line_contact(self) -> bool:
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
        return float(np.hypot(self.Fr_xz, self.Fr_xy))

    @property
    def Mz(self) -> float:
        return float(sum(r.Mz for r in self.rows))

    @property
    def f_r(self) -> tuple[float, ...] | None:
        """Share of Fr per row. None when Fr ~ 0 -- a share of nothing is undefined."""
        if self.Fr < self._F_EPS:
            return None
        return tuple(r.Fr_row / self.Fr for r in self.rows)

    @property
    def f_a(self) -> tuple[float, ...] | None:
        """Share of Fa per row. None when Fa ~ 0 -- read .rows[i].Fa_row instead:
        e.g. a preloaded back-to-back pair has Fa = 0 but rows carry +/-Fa_row."""
        if abs(self.Fa) < self._F_EPS:
            return None
        return tuple(r.Fa_row / self.Fa for r in self.rows)

    @property
    def equilibrium_error(self) -> tuple[float, float]:
        """(Fr - sum Fr_row, Fa - sum Fa_row) [N]."""
        return (self.Fr - sum(r.Fr_row for r in self.rows),
                self.Fa - sum(r.Fa_row for r in self.rows))

    # ---- geometry -----------------------------------------------------
    def phi_j_global(self, i: int = 0) -> np.ndarray:
        """Row i element positions in the global frame, wrapped to [0, 2 pi)."""
        return (self.rows[i].phi_j + self.phi_Fr) % (2.0 * np.pi)


@dataclass(**DC)
class BallBearingResult(BearingResult[BallLoadDistributionResult]):
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
        super().__post_init__()
        if self.Fa != 0.0 or self.delta_a != 0.0 or any(r.Fa_row != 0.0 for r in self.rows):
            raise ValueError(f"RollerBearingResult({self.label!r}): Fa, delta_a and Fa_row "
                             f"must be 0.0 -- radial roller bearings carry no axial load.")