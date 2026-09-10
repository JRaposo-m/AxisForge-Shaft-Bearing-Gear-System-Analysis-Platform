"""
mesh/shaft/element_type/timoshenko_selective_integration/timoshenko.py
"""
from __future__ import annotations

import numpy as np
from typing import Callable
import math as math

from axisforge.mesh.shaft.element_type.elem import Elem



class TimoshenkoBeam:

    def _cowper_factor(self, v: float, ratio: float = 0.0) -> float:
        """
        Cowper (1966) shear correction factor.
        ratio = inner_radius / outer_radius (0.0 for solid section).
        """
        if ratio == 0.0:
            # solid circular section
            return 6 * (1 + v) / (7 + 6 * v)
        else:
            # hollow circular section (Cowper 1966)
            m2 = ratio**2
            num = 6 * (1 + v) * (1 + m2)**2
            den = (7 + 6 * v) * (1 + m2)**2 + (20 + 12 * v) * m2
            return num / den

    def _hutchinson_factor(self, v: float, ratio: float = 0.0) -> float:
        """
        Hutchinson (2001) shear correction factor.
        ratio = inner_radius / outer_radius (0.0 for solid section).
        """
        if ratio == 0.0:
            # solid circular section — confirmed formula
            return 6 * (1 + v)**2 / (7 + 12 * v + 4 * v**2)
        else:
            raise NotImplementedError(
                "Hutchinson (2001) hollow-section formula ainda não foi "
                "verificada contra a fonte original neste código. "
                "Usa theory='cowper' para secções ocas, ou confirma a "
                "expressão em Hutchinson, ASME J. Appl. Mech. 68 (2001) 87-92 "
                "antes de ativar este ramo."
            )

    def _shear_correction_factor(
        self, elem: Elem, theory: str = "cowper", *,
        kGA_override: float | None = None,
    ) -> float:
        """
        Returns the shear correction factor k such that the transverse
        shear stiffness term is K*G*A = k * G * A.

        kGA_override : if given, back-computes the EFFECTIVE shear
            factor implied by that TARGET K*G*A value --
            k_eff = kGA_override / (G*A) -- instead of looking one up
            from cowper/hutchinson; `theory` is ignored in that case.
            This is the single place the override is resolved:
            stiffness_element() and elasticity_matrix() both route
            through this method rather than each special-casing
            kGA_override in its own formula -- see this method's own
            call sites for why that matters (a formula that computed
            its shear term a different way, or a future third formula
            that also needs the shear factor, would otherwise have to
            remember to special-case the override too, and could
            silently fall back to the un-overridden cowper/hutchinson
            value if it forgot).

            Use this to plug in a value read directly from an external
            solver -- e.g. Abaqus's own *Preprint, model=YES
            section-properties printout gives "K*G(23)*A"/"K*G(13)*A"
            for a *Beam Section, which already has Abaqus's own
            slenderness compensation factor baked into it (see this
            suite's own validation conversation on the ~9% Timoshenko-
            vs-Abaqus deflection gap). Plugging that exact number in
            here lets you test directly whether matching Abaqus's
            ACTUAL transverse shear stiffness -- not just AxisForge's
            own cowper/hutchinson-derived one -- closes the gap,
            isolating "is it the K*G*A value" from "is it something
            else in the element formulation".
        """
        if kGA_override is not None:
            E = elem.E
            A = elem.A
            v = elem.v
            G = E / (2 * (1 + v))
            return kGA_override / (G * A)

        ratio = getattr(elem, "radius_ratio", 0.0)  # ajusta ao teu atributo real de Elem

        if theory == "cowper":
            return self._cowper_factor(elem.v, ratio)
        elif theory == "hutchinson":
            return self._hutchinson_factor(elem.v, ratio)
        else:
            raise ValueError(f"Unknown shear correction theory: '{theory}'. "
                            f"Expected 'cowper' or 'hutchinson'.")

    def stiffness_element(
        self, elem: Elem, shear_theory: str = "cowper", *,
        kGA_override: float | None = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        shear_theory : "cowper" | "hutchinson" -- which shear correction
            factor to use when computing K*G*A internally. Ignored when
            kGA_override is given (see below).
        kGA_override : forwarded to _shear_correction_factor() -- see
            that method's own docstring for the full reasoning. None
            (default): normal path, unchanged behaviour, K*G*A computed
            from shear_theory exactly as before this parameter existed.

            NOTE: this only overrides the value used INSIDE this one
            element's stiffness matrix. The solver assembles k_e per
            element by calling this method once per element (see
            ShaftResultsReader._sweep_plane_from_elements() and
            RigidBearingFEMSolver's own assembly) -- to run a FULL
            system solve with this override (not just build one
            element's k_e by hand for a spot check), every call site
            that builds a stiffness_element() needs to forward
            kGA_override through, which is not wired up yet outside
            this class. Flag if you want that plumbed through
            RigidBearingFEMSolver/solve_system() as well; for a quick
            single-element sanity check (e.g. comparing k[1,1] against
            a hand calc using Abaqus's own K*G*A) this parameter alone
            is already enough.

        FIX (this version): bending and shear flexibility now combine
        IN SERIES via the standard Timoshenko shear parameter
        phi = 12*E*I / (kGA * le**2), i.e. every bending/shear-coupled
        term is divided by (1+phi) -- the classic "locking-free" 2-node
        Timoshenko beam element (see e.g. Przemieniecki's stiffness
        matrix formulation). The PREVIOUS version of this method summed
        Bending_const and Shear_const directly (k22 = Bending_const +
        1/4*Shear_const*le**2, with k11 = Shear_const alone) -- that is
        a PARALLEL combination, not a series one, and it is wrong: a
        Timoshenko beam segment's total tip deflection under a
        transverse load is the SUM of a pure-bending deflection and a
        pure-shear deflection (compliances add in series, i.e.
        1/k = le**3/(12EI) + le/(kGA)), not two stiffnesses acting in
        parallel. Omitting the (1+phi) reduction meant every
        bending/shear-coupled term (k11, k12/k15, and to a lesser
        extent k22/k25) came out too STIFF, increasingly so as the
        element got shorter (phi grows as 1/le**2) -- confirmed
        numerically against this suite's own case_axial_and_moment
        Abaqus comparison: at a representative refined-mesh element
        length near a point load, the old k11/k12 terms were ~11%
        too stiff and k22/k25 ~2-3% too stiff, matching almost exactly
        the ~13% v_xy/v_xz error and much smaller theta_xy/theta_xz
        error observed there (both errors concentrated exactly where
        the mesh is shortest, and both vanishing on longer elements
        away from the load -- phi -> 0 as le grows, so (1+phi) -> 1
        and the old and new formulas converge for slender/long
        elements, which is why the bug was invisible everywhere except
        near locally-refined mesh). u (axial, Rod_const) and the
        Euler-Bernoulli beam were never affected -- see this module's
        own validation-conversation writeup for the full derivation.
        """
        le = elem.length
        E  = elem.E
        I  = elem.I
        A  = elem.A
        v = elem.v
        shear_factor = self._shear_correction_factor(elem, shear_theory, kGA_override=kGA_override)
        G = E / (2 * (1 + v))
        kGA = shear_factor * G * A

        # phi: standard Timoshenko shear parameter, combines bending
        # and shear compliance IN SERIES over the element -- phi -> 0
        # recovers the pure Euler-Bernoulli (shear-rigid) element,
        # phi -> large as le shrinks and/or kGA drops (short/soft-in-
        # shear elements become shear-dominated, correctly, instead of
        # artificially over-stiff as the un-corrected formula gave).
        phi = 12 * E * I / (kGA * le**2)
        denom = 1 + phi

        k  = np.zeros((6, 6))
        Rod_const = E * A / le
        k[0, 0] = k[3, 3] = Rod_const
        k[3, 0] = k[0, 3] = - Rod_const

        c_v  = 12 * E * I / (le**3 * denom)       # v-v (same node / opposite node, sign below)
        c_vt = 6  * E * I / (le**2 * denom)       # v-theta coupling
        c_t1 = (4 + phi) * E * I / (le * denom)   # theta-theta, same node
        c_t2 = (2 - phi) * E * I / (le * denom)   # theta-theta, opposite node

        k[1, 1] = k[4, 4] = c_v
        k[1, 4] = k[4, 1] = -c_v

        k[1, 2] = k[2, 1] = k[1, 5] = k[5, 1] = c_vt
        k[4, 2] = k[2, 4] = k[5, 4] = k[4, 5] = -c_vt

        k[2, 2] = k[5, 5] = c_t1
        k[2, 5] = k[5, 2] = c_t2
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

    def elasticity_matrix(
        self, elem: Elem, shear_theory: str = "cowper", *,
        kGA_override: float | None = None,
    ) -> np.ndarray:
        """
        D[2,2] is the same K*G*A term stiffness_element() uses for its
        shear terms (shear_factor * E * A / (2*(1+v)) == shear_factor *
        G * A algebraically). Both methods route kGA_override through
        the SAME _shear_correction_factor() call rather than each
        special-casing the override in its own formula -- see that
        method's own docstring for why (guarantees D[2,2] here and
        Shear_const in stiffness_element() stay consistent with each
        other by construction, instead of two independently-maintained
        "if kGA_override: ... else: ..." branches that could drift).
        kGA_override behaves identically to stiffness_element() here:
        when given, shear_theory is ignored -- see that method's own
        docstring for the full reasoning and the caveat about this only
        affecting THIS call, not a whole system solve, unless every
        call site forwards it.

        NOT touched by this version's stiffness_element() fix: D is a
        pure material constitutive relation (stress/curvature <-> strain,
        no element length in it at all), so it never needed the
        bending/shear series-combination (1+phi) correction that
        stiffness_element() was missing -- that correction is about how
        a 2-node element's DISCRETE stiffness combines bending and
        shear deflection over its length, which has no counterpart
        here. Flagged as an assumption, not confirmed against wherever
        elasticity_matrix()'s D is actually consumed downstream (never
        seen in this conversation) -- if it turns out to feed into a
        length-dependent stiffness computation (e.g. an
        integral(B^T D B dx) alternative to stiffness_element(), which
        deformation_matrix()/shape_functions()/gauss_quadrature() below
        suggest may exist or have once existed), it would need the same
        scrutiny stiffness_element() just got.
        """
        E = elem.E
        A = elem.A
        I = elem.I
        shear_factor = self._shear_correction_factor(elem, shear_theory, kGA_override=kGA_override)
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