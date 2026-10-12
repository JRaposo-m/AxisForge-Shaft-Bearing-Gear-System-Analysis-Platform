"""Numerical guard of the shaft FEM solve: rejects a (near-)singular reduced stiffness matrix.
"""

from __future__ import annotations

import numpy as np


def check_conditioning(K_red: np.ndarray) -> None:
    """Raise if the reduced stiffness matrix is (near-)singular.

    A condition number above the threshold is almost always a mechanism (too few or badly placed
    bearings) rather than a rounding problem.

    Parameters
    ----------
    K_red : ndarray, shape (m, m)
        Stiffness matrix after elimination of the constrained DOFs [N/mm, N, N mm/rad].

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``cond(K_red) > 1e14``.

    Notes
    -----
    The threshold 1e14 is a choice of this code (source to find).
    """
    if np.linalg.cond(K_red) > 1e14:
        raise np.linalg.LinAlgError(
            "RigidSupportFEMSolver: reduced stiffness matrix is (near-)singular — "
            "check bearing count/positions (mechanism / insufficient constraints)."
        )