"""
axisforge/mesh/shaft/element_type/euler_bernoulli/two_noded.py

"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axisforge.mesh.shaft.element_type.elem import Elem

class EulerBernoulliBeam:
    
    def shape_functions(self, zeta: float, elem: Elem) -> np.ndarray:
        # Implementation for Euler-Bernoulli beam shape functions
        le = elem.length
        N = np.zeros(4) 
        N[0] = 1/4 * (2 - 3*zeta + zeta**3)  # transverse v
        N[1] = le/8 * (1 - zeta - zeta**2 + zeta**3)  # rotation θ   
        N[2] = 1/4 * (2 + 3*zeta - zeta**3)  # transverse v
        N[3] = le/8 * (-1 - zeta + zeta**2 + zeta**3)  # rotation θ
        return N 
        
    def bending_strain_matrix(self, zeta: float, elem: Elem) -> np.ndarray:
        le = elem.length
        B = np.zeros((4))
        
        B[0] = 6 * zeta / le**2
        B[1] = (-1 + 3 * zeta) / le
        B[2] = -6 * zeta / le**2
        B[3] = (1 + 3 * zeta) / le

        return B

    def stiffness_element(self, elem) -> np.ndarray:
        le = elem.length
        E  = elem.E
        I  = elem.I
        k  = np.zeros((4, 4))
        Bending_const  = E * I / (le**3)
        k[0, 0] = k[2, 2] = 12 * Bending_const
        k[0, 1] = k[0, 3] = k[1, 0] = k[3, 0] = 6 * Bending_const * le
        k[1, 1] = k[3, 3] = 4 * Bending_const * le**2
        k[0, 2] = k[2, 0] = - 12 * Bending_const
        k[1, 2] = k[2, 1] = k[2, 3] = k[3, 2] = - 6 * Bending_const * le
        k[1, 3] = k[3, 1] = 2 * Bending_const * le**2
        return k

    def natural_coordenates(self, x: float, elem: Elem) -> float:
        """
        Convert global coordinate x to natural coordinate zeta in [-1, 1].
        """
        x_a = elem.x_a
        x_b = elem.x_b
        x_c = (x_a + x_b) / 2
        le = elem.length

        zeta = 2 * (x - x_c) / le
        
        return zeta
    
    
        