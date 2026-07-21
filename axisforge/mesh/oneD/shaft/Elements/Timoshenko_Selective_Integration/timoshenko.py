"""
mesh/oneD/shaft/Elements/Timoshenko_Selective_Integration/timoshenko.py
"""
from __future__ import annotations

import numpy as np
from typing import Callable
import math as math

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
    
    def global_to_natural_radial(self, x1: float, x2: float, elem: Elem) -> Callable[[float], float]:
        """
        Returns zeta -> x(zeta) mapping natural coordinate zeta ∈ [-1, 1]
        to global axial coordinate x ∈ [x1, x2].

            x(zeta) = x1 * N[1](zeta) + x2 * N[4](zeta)
        """
        return lambda zeta: x1 * self.shape_functions(zeta, elem)[1] + \
                            x2 * self.shape_functions(zeta, elem)[4]
    
    def vetor_global_to_natural(self, f: Callable[[float], float], x_map: Callable[[float], float]) -> Callable[[float], float]:
        """
        Rewrites f(x) as f(x(zeta)) by composing with x_map.

        Parameters
        ----------
        f     : any function of global coordinate x, e.g. q(x)
        x_map : zeta -> x, from global_to_natural_radial()

        Returns
        -------
        Callable[[float], float] : zeta -> f(x(zeta))
        """
        return lambda zeta: f(x_map(zeta))
    
    def jacobian(self, elem: Elem):
        
        le = elem.length
        return le/2
    
    def gauss_quadrature(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Gauss-Legendre points and weights on [-1, 1] for arbitrary n.
        Uses numpy.polynomial.legendre for n > 3.
        """
        if n == 1:
            return np.array([0.0]), np.array([2.0])
        elif n == 2:
            s = 1.0 / np.sqrt(3)
            return np.array([-s, s]), np.array([1.0, 1.0])
        elif n == 3:
            s = np.sqrt(3/5)
            return np.array([-s, 0.0, s]), np.array([5/9, 8/9, 5/9])
        else:
            pts, wts = np.polynomial.legendre.leggauss(n)
            return pts, wts
        
    def shape_function_degree(self, node_idx: int, elem: Elem) -> int:
        """
        Estimate polynomial degree of shape function N[node_idx] in zeta
        by sampling and fitting.

        Parameters
        ----------
        node_idx : index in the shape function vector (1 for v_a, 4 for v_b)
        """
        zetas = np.linspace(-1.0, 1.0, 6)
        ys    = np.array([self.shape_functions(z, elem)[node_idx] for z in zetas])

        deg = 3
        for d in range(4):
            coeffs   = np.polyfit(zetas, ys, d)
            residual = np.max(np.abs(np.polyval(coeffs, zetas) - ys))
            if residual < 1e-8:
                deg = d
                break

        return deg
    

    def _q_degree(self, q: Callable, x_lo: float, x_hi: float) -> int:
        """Estimate polynomial degree of q(x) over [x_lo, x_hi]."""
        xs = np.linspace(x_lo, x_hi, 6)
        ys = np.array([q(x) for x in xs])
        deg = 3
        for d in range(4):
            coeffs   = np.polyfit(xs, ys, d)
            residual = np.max(np.abs(np.polyval(coeffs, xs) - ys))
            if residual < 1e-8:
                deg = d
                break
        return deg

    def _theta_degree(self, theta_fn: Callable, x_lo: float, x_hi: float) -> int:
        """
        Estimate effective degree of cos(theta(x)) over [x_lo, x_hi]
        by sampling cos(theta(x)) directly.
        """
        xs = np.linspace(x_lo, x_hi, 6)
        ys = np.array([np.cos(np.radians(theta_fn(x))) for x in xs])
        deg = 3
        for d in range(4):
            coeffs   = np.polyfit(xs, ys, d)
            residual = np.max(np.abs(np.polyval(coeffs, xs) - ys))
            if residual < 1e-8:
                deg = d
                break
        return deg


    def gauss_order(self, q, x_lo, x_hi, elem, theta_fn=None) -> int:
        deg_q     = self._q_degree(q, x_lo, x_hi)
        deg_theta = self._theta_degree(theta_fn, x_lo, x_hi) if theta_fn else 0
        deg_N     = max(self.shape_function_degree(1, elem),
                        self.shape_function_degree(4, elem))
        p = deg_q + deg_theta + deg_N
        return max(1, math.ceil((p + 1) / 2))