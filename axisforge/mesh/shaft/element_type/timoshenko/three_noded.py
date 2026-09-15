"""
axisforge/mesh/shaft/element_type/timoshenko/three_noded.py
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

class QuadraticTimoshenkoBeam:
    
    def shape_functions(self, zeta: float, elem: Elem) -> np.ndarray:
        # Implementation for Timoshenko beam shape functions
        le = elem.length
        N = np.zeros(3) 
        N[0] = 1/2 * zeta * (zeta - 1)  # N_1
        N[1] = (1 - zeta**2)            # N_2
        N[2] = 1/2 * zeta * (zeta + 1)  # N_3
        return N 
        
    def bending_strain_matrix(self, zeta: float, elem: Elem) -> np.ndarray:
        le = elem.length
        B_b = np.zeros((6))
        B_b = 2/le * [0, zeta - 1/2, 0, -2*zeta, 0, zeta + 1/2]  # Bending strain matrix
        return B_b

    def shear_strain_matrix(self, zeta: float, elem: Elem) -> np.ndarray:

        le = elem.length
        B_s = np.zeros((6))
        B_s = 2*le *[zeta - 1/2, 
                        -le/4 *(zeta**2 - zeta), 
                        -2*zeta, 
                        -le/2 *(1-zeta), 
                        zeta + 1/2,
                        -le/4 * (zeta**2 + zeta)]  # Shear strain matrix
        return B_s


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
                        integration: str = "two_point",
                        shear_theory: str = "cowper",
                        *,
                        kGA_override: float | None = None) -> np.ndarray:

        if integration != "two_point":
            raise NotImplementedError(
                f"QuadraticTimoshenkoBeam: integration='{integration}' is not "
                "implemented. This element's k_b/k_s are derived with a "
                "uniform two-point Gauss quadrature; use "
                "integration='two_point' for element_type='three_nodes'."
            )

        le = elem.length
        E  = elem.E
        I  = elem.I
        D_b = E * I

        self.shear_correction_parameter = self.shear_factor(elem, ShearFactor(), shear_theory, kGA_override=kGA_override)
        A  = elem.A
        v  = elem.v
        G  = E / (2 * (1 + v))
        D_s = self.shear_correction_parameter * G * A

        """
        Stiffness matrix for a three-noded Timoshenko beam element.
        The stiffness matrices K_b(e) and K_s(e) for the 3-noded
        Timoshenko beam element obtained with a uniform two-point Gauss quadrature.
        DOF order per node: [w, theta] -> [w1, th1, w2, th2, w3, th3]
        """

        k_b = np.zeros((6, 6))
        k_s = np.zeros((6, 6))
        k   = np.zeros((6, 6))

        k_b[1, 1] = k_b[5, 5] = 7 * D_b / (3 * le)
        k_b[3, 1] = k_b[1, 3] = k_b[3, 5] = k_b[5, 3] = -8 * D_b / (3 * le)
        k_b[3, 3] = 16 * D_b / (3 * le)
        k_b[1, 5] = k_b[5, 1] = D_b / (3 * le)

        c = D_s / (9 * le)

        k_s[0, 0] = 21 * c
        k_s[0, 1] = k_s[1, 0] = -4.5 * le * c
        k_s[0, 2] = k_s[2, 0] = -24 * c
        k_s[0, 3] = k_s[3, 0] = -6 * le * c
        k_s[0, 4] = k_s[4, 0] = 3 * c
        k_s[0, 5] = k_s[5, 0] = 1.5 * le * c

        k_s[1, 1] = le**2 * c
        k_s[1, 2] = k_s[2, 1] = 6 * le * c
        k_s[1, 3] = k_s[3, 1] = le**2 * c
        k_s[1, 4] = k_s[4, 1] = -1.5 * le * c
        k_s[1, 5] = k_s[5, 1] = -0.5 * le**2 * c

        k_s[2, 2] = 48 * c
        k_s[2, 3] = k_s[3, 2] = 0.0
        k_s[2, 4] = k_s[4, 2] = -24 * c
        k_s[2, 5] = k_s[5, 2] = -6 * le * c

        k_s[3, 3] = 4 * le**2 * c
        k_s[3, 4] = k_s[4, 3] = 6 * le * c
        k_s[3, 5] = k_s[5, 3] = le**2 * c

        k_s[4, 4] = 21 * c
        k_s[4, 5] = k_s[5, 4] = 4.5 * le * c

        k_s[5, 5] = le**2 * c

        k = k_b + k_s
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
    
    
        