"""Public interface of the ISO/TS 16281 load-distribution solvers.

Re-exports, eagerly, the solver registry (``dispatch``), the root finder
(``numerics``), the readiness checks (``validation``), the post-processing of
the post-processing (``contact_postprocessing``: rolling-element loads and
basic reference rating life of ISO_TS_16281_2008, 4.3 and 5.3, and a secant
stiffness that is not in the standard) and the concrete ball and roller
solvers (``contact_solver``, which imports ``contact_postprocessing``).
"""
from __future__ import annotations

from .dispatch import (
    SolverDispatchError,
    register_contact_solver,
    resolve_solver_cls,
)
from .numerics import run_root
from .validation import check_bearing_ready
from .contact_postprocessing import (
    ContactBearingStiffness,
    bearing_stiffness,
    combine_row_L10r,
    DynamicEquivalentReferenceLoadBase,
    ball_Q_j,
    ball_phi_j_global,
    ball_contact_distribution,
    DynamicEquivalentRollingElementLoad,
    BallBasicReferenceRatingLife,
    ball_basic_reference_rating_life,
    BallDynamicEquivalentReferenceLoad,
    roller_Q_j,
    roller_phi_j_global,
    roller_contact_distribution,
    roller_lamina_distribution,
    stress_riser_factor,
    LaminaDynamicEquivalentLoad,
    RollerBasicReferenceRatingLife,
    roller_basic_reference_rating_life,
    RollerDynamicEquivalentReferenceLoad,
)
from .contact_solver import (
    SolverBase,
    LineContactSolverBase,
    MultiRowSolverBase,
    ISO16281BallSolver,
    ISO16281RollerSolver,
    ISO16281MultiRowBallSolverSharedDisplacement,
    ISO16281MultiRowRollerSolverSharedDisplacement,
)

__all__ = [
    # dispatch.py
    "SolverDispatchError", "register_contact_solver",
    "resolve_solver_cls",
    # numerics.py
    "run_root",
    # validation.py
    "check_bearing_ready",
    # contact_postprocessing.py
    "ContactBearingStiffness", "bearing_stiffness", "combine_row_L10r",
    "DynamicEquivalentReferenceLoadBase",
    "ball_Q_j", "ball_phi_j_global", "ball_contact_distribution",
    "DynamicEquivalentRollingElementLoad", "BallBasicReferenceRatingLife",
    "ball_basic_reference_rating_life", "BallDynamicEquivalentReferenceLoad",
    "roller_Q_j", "roller_phi_j_global", "roller_contact_distribution",
    "roller_lamina_distribution", "stress_riser_factor",
    "LaminaDynamicEquivalentLoad", "RollerBasicReferenceRatingLife",
    "roller_basic_reference_rating_life", "RollerDynamicEquivalentReferenceLoad",
    # contact_solver.py
    "SolverBase", "LineContactSolverBase", "MultiRowSolverBase",
    "ISO16281BallSolver", "ISO16281RollerSolver",
    "ISO16281MultiRowBallSolverSharedDisplacement",
    "ISO16281MultiRowRollerSolverSharedDisplacement",
]
