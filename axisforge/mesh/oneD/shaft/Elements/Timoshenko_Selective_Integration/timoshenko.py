"""
mesh/oneD/shaft/Elements/Timoshenko_Selective_Integration/timoshenko.py
"""
from __future__ import annotations

import numpy as np

from axisforge.mesh.oneD.shaft.Elements.elem import Elem

class TimoshenkoBeam:
    def stiffness_element(self, elem: Elem) -> np.ndarray:
        le = elem.length
        E  = elem.E
        I  = elem.I
        A  = elem.A
        shear_factor = 5/6
        v = elem.v
        G = E / (2 * (1 + v))
        k  = np.zeros((6, 6))
        Rod_const   = E * A / le
        Bending_const  = E * I / (le)
        Shear_const = shear_factor * E * G * A / le
        k[0, 0] = k[3, 3] = Rod_const
        k[3, 0] = k[0, 3] = - Rod_const
        k[1, 1] = k[4, 4] = Shear_const
        k[1, 2] = k[1, 5] = k[2, 1] = k[5, 1] = 1/2 * Shear_const * le
        k[2, 2] = k[5, 5] = Bending_const + 1/4 * Shear_const * le**2
        k[4, 1] = k[1, 4] = - Shear_const
        k[4, 2] = k[2, 4] = k[5, 4] = k[4, 5] = - 1/2 * Shear_const * le
        k[5, 2] = k[2, 5] = - Bending_const + 1/4 * Shear_const * le**2
        return k
    
    def shape_functions(self, zeta: float, elem: Elem) -> np.ndarray:
        # Implementation for Timoshenko beam shape functions
        le = elem.length
        N = np.zeros(6)
        N[0] = 1/2 * (1 - zeta)  # axial  
        N[1] = 1/2 * (1 - zeta)  # transverse v
        N[2] = 1/2 * (1 - zeta)  # rotation θ
        N[3] = 1/2 * (1 + zeta)  # axial     
        N[4] = 1/2 * (1 + zeta)  # transverse v
        N[5] = 1/2 * (1 + zeta)  # rotation θ
        return N 
        
    def deformation_matrix(self, zeta: float, elem: Elem) -> np.ndarray:
        le = elem.length
        B = np.zeros((3, 6))
        B[0, 0] = -1/2  # axial strain ε_x = du/dx
        B[1, 1] = -1/2 # curvature κ = d²v/dx²]
        B[2, 2] = -1/2 # curvature κ = d²v/dx²
        B[0, 3] = 1/2
        B[1, 4] = 1/2 # curvature κ = d²v/dx²
        B[2, 5] = 1/2 # curvature κ = d²v/dx²
        return B
    
    def elasticity_matrix(self, elem: Elem) -> np.ndarray:
        E = elem.E
        A = elem.A
        I = elem.I
        shear_factor = 5/6
        D = np.zeros((3, 3))
        D[0, 0] = E
        D[1, 1] = E
        D[2, 2] = shear_factor * E * A / (2 * (1 + elem.v))
        return D