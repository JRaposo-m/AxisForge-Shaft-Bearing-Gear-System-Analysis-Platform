# axisforge/solvers/machine_elements/bearings/load_distribution/iso_16281/__init__.py
"""
axisforge/solvers/machine_elements/bearings/load_distribution/iso_16281/__init__.py

Ponto único de import para o subsistema ISO/TS 16281 -- estilo slippy,
tudo eager (mesmo padrão de core/machine_elements/bearings/__init__.py):
dispatch (registo/resolução de solver), numerics (root-finding),
validation (readiness checks), contact_postprocessing (Q_j, rigidez
secante, L10r, Pref) e contact_solver (os solvers concretos -- que já
importa contact_postprocessing por si, ver o próprio módulo, daí este
__init__ importar contact_postprocessing primeiro).
"""
from __future__ import annotations

from .dispatch import (
    SolverDispatchError,
    register_contact_solver,
    resolve_solver_cls_for_attrs,
    resolve_solver_cls,
)
from .numerics import run_root
from .validation import FA_FLOATING_EPS, check_bearing_ready, warn_if_floating_loaded
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
    "resolve_solver_cls_for_attrs", "resolve_solver_cls",
    # numerics.py
    "run_root",
    # validation.py
    "FA_FLOATING_EPS", "check_bearing_ready", "warn_if_floating_loaded",
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
