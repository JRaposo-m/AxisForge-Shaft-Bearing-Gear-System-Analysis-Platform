"""
axisforge/mesh/shaft/element_type/frame_element.py

Axial ("frame") behaviour for a 2-node beam element -- the piece that
turns a pure bending beam element into a full frame element by adding
the axial DOF (u_a, u_b).

Deliberately theory-independent: axial deformation does not depend on
beam_theory at all (Euler-Bernoulli vs Timoshenko bending theory has
no bearing on EA/L). Every Elem gets this behaviour unconditionally,
alongside whichever bending formulation elem.beam_theory selects --
there is no dispatch here, and no second "axial theory" to choose.

Shape functions are linear, identical in form to
TimoshenkoBeam.shape_functions() -- both are 2-node Lagrange
interpolation; the only difference is which physical quantity is being
interpolated (u here, v there).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.mesh.shaft.element_type.elem import Elem


class FrameElement:

    def shape_functions(self, zeta: float, elem: "Elem") -> np.ndarray:
        """
        Linear axial shape functions -- same form as
        TimoshenkoBeam.shape_functions(), interpolating u instead of v.
        """
        N = np.zeros(2)
        N[0] = 1 / 2 * (1 - zeta)  # N_1 -> u_a
        N[1] = 1 / 2 * (1 + zeta)  # N_2 -> u_b
        return N

    def axial_strain_matrix(self, elem: "Elem") -> np.ndarray:
        """
        B = du/dx -- constant, since N is linear: [-1/le, 1/le].
        """
        le = elem.length
        return np.array([-1.0 / le, 1.0 / le])

    def stiffness_element(self, elem: "Elem") -> np.ndarray:
        """
        Axial bar stiffness: K = EA/L * [[1, -1], [-1, 1]].
        """
        E, A, le = elem.E, elem.A, elem.length
        k = E * A / le
        return np.array([[k, -k],
                          [-k, k]])