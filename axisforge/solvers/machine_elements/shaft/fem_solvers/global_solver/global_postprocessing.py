"""Post-processing of the global shaft solve: turns a ``RigidSupportSolution`` (raw nodal DOFs) into
a ``ShaftResults`` with moments, shear forces, deflections, stresses, torsion, bearing reactions
and bearing-node data. The moment and shear recovery reuses the element post-processors; torsion
comes from ``TorsionSolver``, called here and not from ``RigidSupportFEMSolver.solve`` itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from axisforge.results.fem_results.shaft_results import BearingNodeData, ShaftResults

from axisforge.solvers.machine_elements.shaft.fem_solvers.element_theories.element_postprocessing import (
    EulerBernoulliPostProcessing,
    TimoshenkoPostProcessing,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.torsion import TorsionSolver

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (
        RigidSupportSolution,
    )
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem


def _find_node(x_nodes: list[float], x: float, tol: float = 1e-6) -> int:
    """Index of the node at position ``x`` (absolute distance below ``tol``).

    Parameters
    ----------
    x_nodes : list of float
        Node positions [mm].
    x : float
        Requested position [mm].
    tol : float, optional
        Accepted distance [mm]. Default 1e-6.

    Returns
    -------
    int
        Node index.

    Raises
    ------
    ValueError
        If no node is within ``tol``.
    """
    for i, xi in enumerate(x_nodes):
        if abs(xi - x) < tol:
            return i
    raise ValueError(f"Position {x:.4f} mm not found in x_nodes.")


# ===========================================================================
# Theory-specific M/V recovery, specialized to RigidSupportSolution
# ===========================================================================

class GlobalEulerBernoulliPostProcessing(EulerBernoulliPostProcessing):
    """Euler-Bernoulli moment and shear recovery, typed for a ``RigidSupportSolution``.

    No adapter is needed: the field names of the solution already match what the element
    post-processor reads. The override only adds the type hint.
    """

    def recover_internal_forces(self, solution: "RigidSupportSolution", x_nodes: list[float], n: int):
        """Moment and shear force at every node (see ``ElementTheoryPostProcessor.recover_internal_forces``).

        Parameters
        ----------
        solution : RigidSupportSolution
            Raw solution of the global solve.
        x_nodes : list of float
            Node positions [mm].
        n : int
            Number of nodes.

        Returns
        -------
        M_xz, M_xy : ndarray, shape (n,)
            Bending moments [N mm].
        V_xz, V_xy : ndarray, shape (n,)
            Shear forces [N].
        """
        return super().recover_internal_forces(solution, x_nodes, n)


class GlobalTimoshenkoPostProcessing(TimoshenkoPostProcessing):
    """Timoshenko counterpart of ``GlobalEulerBernoulliPostProcessing``.
    """

    def recover_internal_forces(self, solution: "RigidSupportSolution", x_nodes: list[float], n: int):
        """Moment and shear force at every node (see ``ElementTheoryPostProcessor.recover_internal_forces``).

        Parameters
        ----------
        solution : RigidSupportSolution
            Raw solution of the global solve.
        x_nodes : list of float
            Node positions [mm].
        n : int
            Number of nodes.

        Returns
        -------
        M_xz, M_xy : ndarray, shape (n,)
            Bending moments [N mm].
        V_xz, V_xy : ndarray, shape (n,)
            Shear forces [N].
        """
        return super().recover_internal_forces(solution, x_nodes, n)


def postprocessor_for_global(
    solution: "RigidSupportSolution",
) -> "GlobalEulerBernoulliPostProcessing | GlobalTimoshenkoPostProcessing":
    """Select the moment/shear post-processor that matches the beam theory of the solution.

    Parameters
    ----------
    solution : RigidSupportSolution
        Raw solution; ``solution.elements[0].beam_theory`` decides.

    Returns
    -------
    GlobalEulerBernoulliPostProcessing or GlobalTimoshenkoPostProcessing

    Raises
    ------
    ValueError
        If the beam theory is neither ``"timoshenko"`` nor ``"euler_bernoulli"``.
    """
    beam_theory = solution.elements[0].beam_theory
    if beam_theory == "timoshenko":
        return GlobalTimoshenkoPostProcessing()
    elif beam_theory == "euler_bernoulli":
        return GlobalEulerBernoulliPostProcessing()
    raise ValueError(f"postprocessor_for_global: unknown beam_theory '{beam_theory}'")


# ===========================================================================
# Full-result assembly
# ===========================================================================

def build_shaft_result(solution: "RigidSupportSolution", shaft_system: "ShaftSystem") -> ShaftResults:
    """Assemble the ``ShaftResults`` of the whole shaft from the raw global solution.

    Parameters
    ----------
    solution : RigidSupportSolution
        Raw output of the global solve (displacements [mm, rad], forces [N, N mm]).
    shaft_system : ShaftSystem
        Provides the section properties, bearings and torque loads.

    Returns
    -------
    ShaftResults
        Values at the mesh nodes unless noted; units are listed in ``ShaftResults``.

    Warnings
    --------
    Moment and shear force are written node by node from the elements (an interior node holds the
    value of the element on its left); Onate2013_vol2 samples the bending moment at the Gauss points
    (Euler-Bernoulli) or M and Q at the element mid-point (Timoshenko), see the element
    post-processors.

    At a node on a change of section, ``d``, ``W``, ``Wt``, ``sigma_b`` and ``tau`` are those of the
    section to the RIGHT of the node (``Shaft.section_at``). The stress of the smaller section at a
    step up is therefore not evaluated and ``sigma_b_max`` can be underestimated (at the step the
    stored stress uses the larger W, ``sigma_b = M / W``).

    Notes
    -----
    Plane resultants are combined with the quadratic sum: ``M = hypot(M_xz, M_xy)`` [N mm],
    ``V = hypot(V_xz, V_xy)`` [N], ``v = hypot(v_xz, v_xy)`` [mm]. The nominal bending stress is
    ``sigma_b = M / W`` [MPa] with ``W`` = ``Shaft.W_at`` [mm^3] (no stress concentration, no
    transverse shear stress); Onate2013_vol2, Eq. (1.4)-(1.5) give ``sigma_x = -z E d2w/dx2`` and
    ``M``, from which the nominal form follows; ``sigma_b = M/W`` as one equation is not printed in the
    library (source to find). The torsional shear stress, torque and twist come from ``TorsionSolver``.
    The maxima (``M_max``, ``v_max``, ``sigma_b_max``, ``tau_max``, ``phi_max``) are taken at nodes.
    """
    x_nodes = solution.x_nodes
    n       = len(x_nodes)

    T_total, tau_total, phi_total, torsion_contributions = TorsionSolver().solve(shaft_system, x_nodes)

    postproc = postprocessor_for_global(solution)
    M_xz, M_xy, V_xz, V_xy = postproc.recover_internal_forces(solution, x_nodes, n)

    v_xz, v_xy             = _recover_deflections(solution, n)
    u, theta_xz, theta_xy  = _recover_axial_and_rotation(solution, n)
    d_arr, W_arr, Wt_arr   = _section_properties(shaft_system, x_nodes)

    M_eq    = np.hypot(M_xz, M_xy)
    V_eq    = np.hypot(V_xz, V_xy)
    v_eq    = np.hypot(v_xz, v_xy)
    sigma_b = M_eq / W_arr
    tau_arr = tau_total

    bearing_reactions = _bearing_reactions(solution, shaft_system, x_nodes)
    bearing_nodes     = _bearing_node_data(solution, shaft_system, x_nodes)

    R_xz_arr    = np.array([r["R_xz"]    for r in bearing_reactions])
    R_xy_arr    = np.array([r["R_xy"]    for r in bearing_reactions])
    R_arr       = np.array([r["R"]       for r in bearing_reactions])
    R_axial_arr = np.array([r["R_axial"] for r in bearing_reactions])
    brg_positions = [r["position"] for r in bearing_reactions]

    idx_M   = int(np.argmax(M_eq))
    idx_v   = int(np.argmax(v_eq))
    idx_sb  = int(np.argmax(sigma_b))
    idx_t   = int(np.argmax(np.abs(tau_arr)))
    idx_phi = int(np.argmax(np.abs(phi_total)))

    return ShaftResults(
        name = shaft_system.name,

        x_nodes  = x_nodes,
        elements = solution.elements,

        K                     = solution.K,
        free_dofs             = solution.free_dofs,
        constrained_dofs      = solution.constrained_dofs,
        d_total_xz            = solution.d_total_xz,
        d_total_xy            = solution.d_total_xy,
        f_xz_ext              = solution.f_xz_ext,
        f_xy_ext              = solution.f_xy_ext,
        f_xz_total            = solution.f_xz_total,
        f_xy_total            = solution.f_xy_total,
        f_xz_reaction         = solution.f_xz_reaction,
        f_xy_reaction         = solution.f_xy_reaction,
        T_total               = T_total,
        tau_total             = tau_total,
        phi_total             = phi_total,
        torsion_contributions = torsion_contributions,

        x        = np.array(x_nodes),
        M_xz     = M_xz,    M_xy    = M_xy,    M  = M_eq,
        V_xz     = V_xz,    V_xy    = V_xy,    V  = V_eq,
        v_xz     = v_xz,    v_xy    = v_xy,    v  = v_eq,
        u        = u,
        theta_xz = theta_xz, theta_xy = theta_xy,
        T        = T_total,
        phi      = phi_total,
        d        = d_arr,   W       = W_arr,   Wt = Wt_arr,
        sigma_b  = sigma_b, tau     = tau_arr,
        bearing_positions = brg_positions,
        R_xz     = R_xz_arr, R_xy  = R_xy_arr,
        R        = R_arr,    R_axial = R_axial_arr,
        M_max        = float(M_eq[idx_M]),      x_M_max       = float(x_nodes[idx_M]),
        v_max        = float(v_eq[idx_v]),      x_v_max       = float(x_nodes[idx_v]),
        sigma_b_max  = float(sigma_b[idx_sb]),  x_sigma_b_max = float(x_nodes[idx_sb]),
        tau_max      = float(np.abs(tau_arr[idx_t])), x_tau_max = float(x_nodes[idx_t]),
        phi_max      = float(np.abs(phi_total[idx_phi])), x_phi_max = float(x_nodes[idx_phi]),

        bearing_nodes = bearing_nodes,
    )


# ------------------------------------------------------------------
# Internal helpers -- module-level functions taking (solution,
# shaft_system) explicitly.
# ------------------------------------------------------------------

def _recover_deflections(solution: "RigidSupportSolution", n: int):
    """Transverse deflection of each bending plane at every node.

    Parameters
    ----------
    solution : RigidSupportSolution
        Raw solution.
    n : int
        Number of nodes.

    Returns
    -------
    v_xz, v_xy : ndarray, shape (n,)
        Deflection ``d[3i+1]`` of the XZ and XY planes [mm].
    """
    v_xz = np.array([solution.d_total_xz[3 * i + 1] for i in range(n)])
    v_xy = np.array([solution.d_total_xy[3 * i + 1] for i in range(n)])
    return v_xz, v_xy


def _recover_axial_and_rotation(solution: "RigidSupportSolution", n: int):
    """Axial displacement and section rotation of each plane at every node.

    ``u`` is read from the XZ solution only: the axial DOF is shared between the planes and only the
    XZ load vector receives ``AxialLoad``.

    Parameters
    ----------
    solution : RigidSupportSolution
        Raw solution.
    n : int
        Number of nodes.

    Returns
    -------
    u : ndarray, shape (n,)
        Axial displacement [mm].
    theta_xz, theta_xy : ndarray, shape (n,)
        Nodal rotation DOF of the XZ and XY planes [rad]. For Timoshenko elements this is the section
        rotation, not dv/dx.
    """
    u        = np.array([solution.d_total_xz[3 * i]     for i in range(n)])
    theta_xz = np.array([solution.d_total_xz[3 * i + 2] for i in range(n)])
    theta_xy = np.array([solution.d_total_xy[3 * i + 2] for i in range(n)])
    return u, theta_xz, theta_xy


def _section_properties(shaft_system: "ShaftSystem", x_nodes: list[float]):
    """Diameter and section moduli of the shaft at every node.

    Parameters
    ----------
    shaft_system : ShaftSystem
        Provides ``shaft.diameter_at``, ``shaft.W_at`` and ``shaft.Wt_at``.
    x_nodes : list of float
        Node positions [mm].

    Returns
    -------
    d_arr : ndarray
        Diameter [mm].
    W_arr : ndarray
        Bending section modulus [mm^3].
    Wt_arr : ndarray
        Torsion section modulus [mm^3].
    """
    shaft  = shaft_system.shaft
    d_arr  = np.array([shaft.diameter_at(x) for x in x_nodes])
    W_arr  = np.array([shaft.W_at(x)        for x in x_nodes])
    Wt_arr = np.array([shaft.Wt_at(x)       for x in x_nodes])
    return d_arr, W_arr, Wt_arr


def _bearing_reactions(
    solution: "RigidSupportSolution", shaft_system: "ShaftSystem", x_nodes: list[float],
) -> list[dict]:
    """Support reactions at the bearing nodes.

    Parameters
    ----------
    solution : RigidSupportSolution
        Raw solution; uses ``f_xz_reaction`` and ``f_xy_reaction`` (``K d - f_ext``).
    shaft_system : ShaftSystem
        Provides the bearings.
    x_nodes : list of float
        Node positions [mm].

    Returns
    -------
    list of dict
        One dict per bearing with ``"label"``, ``"position"`` [mm], ``"R_xz"`` and ``"R_xy"``
        (radial reactions of the two planes [N]), ``"R"`` (their ``hypot`` [N]) and ``"R_axial"``
        [N].

    Warnings
    --------
    ``R_axial`` is non-zero for ``"locating"`` bearings only; a ``"thrust"`` bearing constrains the
    axial DOF but reports 0.0.

    Sign: these are forces exerted by the support ON THE SHAFT, positive along +Z (``_xz``), +Y
    (``_xy``) and +X (axial); the load on the bearing is the opposite (1000 N along +Y between two
    bearings gives ``R_xy`` = -666.67 N and -333.33 N).
    """
    reactions = []
    for b in shaft_system.bearings:
        i    = _find_node(x_nodes, b.position)
        R_xz = float(solution.f_xz_reaction[3 * i + 1])
        R_xy = float(solution.f_xy_reaction[3 * i + 1])
        R_ax = float(solution.f_xz_reaction[3 * i]) if b.arrangement == "locating" else 0.0
        reactions.append({
            "label":    b.label or b.designation,
            "position": b.position,
            "R_xz":     R_xz,
            "R_xy":     R_xy,
            "R":        float(np.hypot(R_xz, R_xy)),
            "R_axial":  R_ax,
        })
    return reactions


def _bearing_node_data(
    solution: "RigidSupportSolution", shaft_system: "ShaftSystem", x_nodes: list[float],
) -> list[BearingNodeData]:
    """State of the shaft at every bearing node, the input of the bearing solver.

    Parameters
    ----------
    solution : RigidSupportSolution
        Raw solution.
    shaft_system : ShaftSystem
        Provides the bearings.
    x_nodes : list of float
        Node positions [mm].

    Returns
    -------
    list of BearingNodeData
        Displacements [mm], rotations ``psi_xz = theta_xz``, ``psi_xy = theta_xy`` [rad], forces
        ``Fr_xz``, ``Fr_xy``, ``Fa`` [N] and moments ``M_xz``, ``M_xy`` [N mm] at the node.

    Warnings
    --------
    ``Fr_xz``, ``Fr_xy`` and ``Fa`` are rows of ``K d`` at the bearing node, that is the support
    reaction PLUS the nodal load on that node (``Fr = R + f_ext``). They equal ``ShaftResults.R_*``
    (``K d - f_ext``) only if no load reaches the node. A point load on the node, or a distributed
    load on an element that touches the node (its consistent nodal force, about ``q le/2``, mesh
    dependent), makes ``|Fr|`` smaller than the force carried by the support (500 N on the node:
    ``Fr_xy`` = -333.33 N, ``R_xy`` = -833.33 N). ``M_xz``/``M_xy`` are the nodal moments of the
    load vector at the node: an ``ExternalMoment`` on the node and, for Euler-Bernoulli, the
    consistent moment ``q le^2/12`` of an adjacent distributed load. They are not reactions (the
    rotation is never constrained).
    ``Fa`` is non-zero for ``"locating"`` bearings only (a ``"thrust"`` bearing constrains u but
    reports 0.0).

    Sign (no load on the node): these are forces exerted by the support ON THE SHAFT, positive
    along +Z (``_xz``), +Y (``_xy``) and +X (axial); the load on the bearing is the opposite
    (1000 N along +Y between two bearings gives ``Fr_xy`` = -666.67 N and -333.33 N).
    ``arctan2(Fr_xy, Fr_xz)``, which the bearing solver builds as ``phi_Fr`` and uses as load
    direction, is therefore the direction of the support reaction, 180 deg from the direction in
    which the shaft pushes the inner ring (whether this is intended is a question for the owner).
    """
    nodes = []

    for b in shaft_system.bearings:
        i = _find_node(x_nodes, b.position)

        u        = float(solution.d_total_xz[3 * i])
        v_xz     = float(solution.d_total_xz[3 * i + 1])
        v_xy     = float(solution.d_total_xy[3 * i + 1])
        theta_xz = float(solution.d_total_xz[3 * i + 2])
        theta_xy = float(solution.d_total_xy[3 * i + 2])

        Fr_xz = float(solution.f_xz_total[3 * i + 1])
        Fr_xy = float(solution.f_xy_total[3 * i + 1])
        Fa    = float(solution.f_xz_total[3 * i]) if b.arrangement == "locating" else 0.0
        M_xz  = float(solution.f_xz_total[3 * i + 2])
        M_xy  = float(solution.f_xy_total[3 * i + 2])

        psi_xz, psi_xy = theta_xz, theta_xy

        nodes.append(BearingNodeData(
            label    = b.label or b.designation,
            position = b.position,
            u        = u,
            v_xz     = v_xz,
            v_xy     = v_xy,
            theta_xz = theta_xz,
            theta_xy = theta_xy,
            Fr_xz    = Fr_xz,
            Fr_xy    = Fr_xy,
            Fr       = float(np.hypot(Fr_xz, Fr_xy)),
            Fa       = Fa,
            M_xz     = M_xz,
            M_xy     = M_xy,
            psi_xz   = psi_xz,
            psi_xy   = psi_xy,
        ))

    return nodes