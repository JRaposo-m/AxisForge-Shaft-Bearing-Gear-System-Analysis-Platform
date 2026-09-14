"""
axisforge/mesh/shaft/element_type/timoshenko/exact_two_noded.py
"""

"""
axisforge/mesh/shaft/element_type/timoshenko/two_noded.py

"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.mesh.shaft.element_type.timoshenko.shear_factor import ShearFactor

class ExactTimoshenkoBeam:
    
    def shape_functions(self, zeta: float, elem: Elem) -> np.ndarray:
        # Implementation for Timoshenko beam shape functions
        le = elem.length
        N = np.zeros(2) 
        N[0] = 1/2 * (1 - zeta)  # N_1
        N[1] = 1/2 * (1 + zeta)  # N_2
        return N 
        
    def bending_strain_matrix(self, zeta: float, elem: Elem) -> np.ndarray:
        le = elem.length
        B_b = np.zeros((4))
        B_b = [0, -1/le, 0, 1/le]  # Bending strain matrix
        return B_b

    def shear_strain_matrix(self, zeta: float, elem: Elem, integration: str) -> np.ndarray:

        le = elem.length
        if integration == "single_point":
            B_s = np.zeros((4))
            B_s = [ -1/le, -1/2, 1/le, -1/2]  # Shear strain matrix
            return B_s
        elif integration == "exact":
            B_s = np.zeros((4))
            B_s = [ -1/le, -(1-zeta)/2, 1/le, -(1+zeta)/2]  # Shear strain matrix
            return B_s
        else:
            raise ValueError(f"Unknown integration method: '{integration}'. "
                             f"Expected 'single_point' or 'exact'.")

    def shear_factor(self, elem: Elem, 
                     shear_factor: ShearFactor,
                     shear_theory: str, *,
                     kGA_override: float | None = None,) -> float:

        v = elem.v
        ratio = elem.radius_ratio
        E = elem.E
        A = elem.A
        shear_theory = shear_theory

        return shear_factor.shear_correction_factor(v, ratio, E, A, shear_theory, kGA_override=kGA_override)    

    def stiffness_element(self, 
                          elem: Elem,
                          integration: str = "single_point",
                          shear_theory: str = "cowper", 
                          *, 
                          kGA_override: float | None = None) -> np.ndarray:

        le = elem.length
        E  = elem.E
        I  = elem.I
        D_b = E * I

        self.shear_correction_parameter = self.shear_factor(elem, ShearFactor(), shear_theory, kGA_override=kGA_override)
        A  = elem.A
        v  = elem.v
        G  = E / (2 * (1 + v))
        D_s = self.shear_correction_parameter * G * A

        if integration == "single_point":
            k_b = np.zeros((4, 4))
            k_s = np.zeros((4, 4))
            k   = np.zeros((4, 4))

            k_b[1, 1] = k_b[3, 3] = D_b / le
            k_b[1, 3] = k_b[3, 1] = -D_b / le

            k_s[0, 0] = k_s[2, 2] = D_s / le
            k_s[0, 2] = k_s[2, 0] = -D_s / le
            k_s[0, 1] = k_s[1, 0] = k_s[0, 3] = k_s[3, 0] = 1/2 * D_s
            k_s[1, 2] = k_s[2, 1] = k_s[2, 3] = k_s[3, 2] = -1/2 * D_s
            k_s[1, 1] = k_s[3, 3] = k_s[1, 3] = k_s[3, 1] = 1/4 * D_s * le

            k = k_b + k_s
            return k
        elif integration == "exact":
            k_b = np.zeros((4, 4))
            k_s = np.zeros((4, 4))
            k   = np.zeros((4, 4))

            k_b[1, 1] = k_b[3, 3] = D_b / le
            k_b[1, 3] = k_b[3, 1] = -D_b / le  

            k_s[0, 0] = k_s[2, 2] = D_s / le
            k_s[0, 2] = k_s[2, 0] = -D_s / le
            k_s[0, 1] = k_s[1, 0] = k_s[0, 3] = k_s[3, 0] = 1/2 * D_s
            k_s[1, 2] = k_s[2, 1] = k_s[2, 3] = k_s[3, 2] = -1/2 * D_s
            k_s[1, 1] = k_s[3, 3] = 1/3 * D_s * le
            k_s[1, 3] = k_s[3, 1] = 1/6 * D_s * le

            k = k_b + k_s
            return k
        else:
            raise ValueError(f"Unknown integration method: '{integration}'. "
                             f"Expected 'single_point' or 'exact'.")


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
    
    
        