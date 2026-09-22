"""axisforge/results/bearing/load_distribution/single_row/roller_bearing_results.py

Result shapes for a CYLINDRICAL_ROLLER bearing ISO/TS 16281 solve:
RollerLoadDistributionResult (one row's raw solve output, with the §5.2
lamina fields) and RollerBearingResult (the per-bearing wrapper). Pure
data -- no registry behavior. Mirrors ball_bearing_results.py. Single-row
only -- see that module's docstring; CylindricalRollerFamily itself has no
multi-row variant (its `i` is a capacity multiplier, not a rows list).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class RollerLoadDistributionResult:
    """Raw output of one line-contact solve (ISO16281RollerSolver.solve_contact()).

    Base fields mirror BallLoadDistributionResult (delta_a is always 0.0 --
    radial roller bearings carry no axial load). Extra per-lamina fields:

    x_k       ndarray(n_s,)     lamina positions, eq.(38)-figure 3
    psi_j     ndarray(Z,)       per-roller local misalignment, eq.(39)
    delta_jk  ndarray(Z,n_s)    per-lamina elastic deflection, eq.(41)
    q_jk      ndarray(Z,n_s)    per-lamina contact force, eq.(36)

    Optional postprocessing fields -- all default None, only populated
    when the solver that produced this row was run with postprocess=True
    (ISO16281RollerSolver(postprocess=True)). See contact_solver.py's
    _attach_postprocessing() and contact_postprocessing.py for what each
    one is computed from.

    Q_j        ndarray(Z,) [N]   per-roller total contact force, roller_Q_j()
    stiffness  ContactBearingStiffness   secant stiffness, bearing_stiffness()
    L10r       float [Mrev]      basic reference rating life, eq.(65)
    Pref_r     float | None [N]  dynamic equivalent reference load, radial, eq.(66)
    Pref_a     float | None [N]  dynamic equivalent reference load, axial, eq.(67)
    """
    __slots__ = (
        "delta_r", "delta_a", "psi", "phi_Fr",
        "delta_j", "alpha_j",
        "Mz", "n_iter", "residual", "ok",
        "x_k", "psi_j", "delta_jk", "q_jk",
        "Q_j", "stiffness", "L10r", "Pref_r", "Pref_a",
    )

    def __init__(self, *, delta_r, delta_a, psi, phi_Fr,
                 delta_j, alpha_j, Mz, n_iter, residual, ok,
                 x_k, psi_j, delta_jk, q_jk,
                 Q_j=None, stiffness=None, L10r=None, Pref_r=None, Pref_a=None):
        self.delta_r  = delta_r
        self.delta_a  = delta_a
        self.psi      = psi
        self.phi_Fr   = phi_Fr
        self.delta_j  = delta_j
        self.alpha_j  = alpha_j
        self.Mz       = Mz
        self.n_iter   = n_iter
        self.residual = residual
        self.ok       = ok
        self.x_k      = x_k
        self.psi_j    = psi_j
        self.delta_jk = delta_jk
        self.q_jk     = q_jk
        self.Q_j        = Q_j
        self.stiffness  = stiffness
        self.L10r       = L10r
        self.Pref_r     = Pref_r
        self.Pref_a     = Pref_a


class RollerBearingResult:
    """Result of solving a roller bearing's internal load distribution.

    `rows` holds exactly 1 entry (single-row only, see package docstring).
    delta_r / delta_a / psi read rows[0].

    Attributes
    ----------
    rows   list[RollerLoadDistributionResult]   length 1, always
    """
    __slots__ = ("rows",)

    def __init__(self, rows: Sequence[RollerLoadDistributionResult]):
        rows = list(rows)
        if len(rows) != 1:
            raise ValueError(
                f"RollerBearingResult is single-row only, got {len(rows)} rows."
            )
        self.rows = rows

    @classmethod
    def single(cls, row: RollerLoadDistributionResult) -> "RollerBearingResult":
        return cls(rows=[row])

    @property
    def is_single(self) -> bool:
        return True

    @property
    def delta_r(self) -> float:
        return self.rows[0].delta_r

    @property
    def delta_a(self) -> float:
        return self.rows[0].delta_a

    @property
    def psi(self) -> float:
        return self.rows[0].psi
