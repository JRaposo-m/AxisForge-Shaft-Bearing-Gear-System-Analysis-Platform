"""
axisforge/solvers/machine_elements/shaft/fem_solvers/numerics/numerical_guards.py

Beam-theory-agnostic and stateless -- kept separate only so
rigid_support.py stays cleaner, not for any deeper reason.
"""

from __future__ import annotations

import numpy as np


def check_conditioning(K_red: np.ndarray) -> None:
    """
    Raises if the reduced stiffness matrix is (near-)singular -- almost
    always a sign of a mechanism (not enough / badly placed bearings)
    rather than a numerical-precision issue, given the 1e14 threshold.
    """
    if np.linalg.cond(K_red) > 1e14:
        raise np.linalg.LinAlgError(
            "RigidSupportFEMSolver: reduced stiffness matrix is (near-)singular — "
            "check bearing count/positions (mechanism / insufficient constraints)."
        )