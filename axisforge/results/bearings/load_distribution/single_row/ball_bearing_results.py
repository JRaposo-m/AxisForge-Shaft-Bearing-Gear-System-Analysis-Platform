"""axisforge/results/bearings/load_distribution/single_row/ball_bearing_results.py

Result shapes for a point-contact (ball) bearing ISO/TS 16281 solve:
BallLoadDistributionResult (one row's raw solve output) and
BallBearingResult (the per-bearing wrapper). Pure data -- no registry
behavior. Single-row only: radial multi-row equilibrium is out of scope
for this codebase (see fixtures/studies/bearing); multi-row still exists
for THRUST families via their own dedicated solvers, untouched by this
file.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class BallLoadDistributionResult:
    """Raw output of one point-contact solve (ISO16281BallSolver.solve_contact()).

    Attributes
    ----------
    delta_r   float [mm]        radial ring displacement in the resultant-force plane
    delta_a   float [mm]        axial ring displacement
    psi       float [rad]       prescribed ring misalignment
    phi_Fr    float [rad]       angle of resultant Fr in the global frame
    delta_j   ndarray(Z,) [mm]  elastic deflection per rolling element
    alpha_j   ndarray(Z,) [rad] effective contact angle per element
    Mz        float [N*mm]      moment reaction
    n_iter    int               solver function evaluations
    residual  float [N]         final ||R||
    ok        bool              solver convergence flag
    """
    __slots__ = (
        "delta_r", "delta_a", "psi", "phi_Fr",
        "delta_j", "alpha_j",
        "Mz", "n_iter", "residual", "ok",
    )

    def __init__(self, *, delta_r, delta_a, psi, phi_Fr,
                 delta_j, alpha_j, Mz, n_iter, residual, ok):
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


class BallBearingResult:
    """Result of solving a ball bearing's internal load distribution.

    `rows` holds exactly 1 entry -- this codebase scopes radial multi-row
    equilibrium as out of scope (see package docstring); a multi-row
    THRUST_BALL bearing is solved by its own dedicated solver, not this one.

    delta_r / delta_a / psi read rows[0].

    Attributes
    ----------
    rows   list[BallLoadDistributionResult]   length 1, always
    """
    __slots__ = ("rows",)

    def __init__(self, rows: Sequence[BallLoadDistributionResult]):
        rows = list(rows)
        if len(rows) != 1:
            raise ValueError(
                f"BallBearingResult is single-row only, got {len(rows)} rows."
            )
        self.rows = rows

    @classmethod
    def single(cls, row: BallLoadDistributionResult) -> "BallBearingResult":
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