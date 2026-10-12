"""Result containers of the global shaft FEM: nodal state at a bearing and the full shaft results.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class BearingNodeData:
    """FEM nodal state of the shaft at a bearing position (input of the bearing solver).

    Parameters
    ----------
    label : str
        Bearing label.
    position : float
        Axial position of the bearing node [mm].
    u : float
        Axial displacement [mm].
    v_xz, v_xy : float
        Transverse displacement in the XZ and XY planes [mm].
    theta_xz, theta_xy : float
        Nodal rotation DOF in the XZ and XY planes [rad].
    Fr_xz, Fr_xy : float
        Transverse nodal force ``K d`` in the XZ and XY planes [N].
    Fr : float
        Resultant of ``Fr_xz`` and ``Fr_xy``, ``hypot`` [N].
    Fa : float
        Axial nodal force ``K d`` [N]; non-zero for ``"locating"`` bearings only.
    M_xz, M_xy : float
        Nodal moment ``K d`` at the rotation DOF in the XZ and XY planes [N mm].
    psi_xz, psi_xy : float
        Nodal rotation (same value as ``theta_xz``, ``theta_xy``) [rad].

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
    A ``"thrust"`` bearing constrains u but its ``Fa`` is 0.

    Sign (no load on the node): ``Fr_xz``, ``Fr_xy`` and ``Fa`` are forces exerted by the support
    ON THE SHAFT, positive along +Z (``_xz``), +Y (``_xy``) and +X (axial); the load on the bearing
    is the opposite (1000 N along +Y between two bearings gives ``Fr_xy`` = -666.67 N and
    -333.33 N).
    ``arctan2(Fr_xy, Fr_xz)``, which the bearing solver builds as ``phi_Fr`` and uses as
    load direction, is therefore the direction of the support reaction, 180 deg from the direction in
    which the shaft pushes the inner ring (whether this is intended is a question for the owner).
    """

    label:    str   = ""
    position: float = 0.0    # [mm]

    u:        float = 0.0    # axial [mm]
    v_xz:     float = 0.0    # transverse XZ [mm]
    v_xy:     float = 0.0    # transverse XY [mm]
    theta_xz: float = 0.0    # bending rotation XZ [rad]
    theta_xy: float = 0.0    # bending rotation XY [rad]

    Fr_xz:    float = 0.0    # transverse nodal force K d, XZ [N]
    Fr_xy:    float = 0.0    # transverse nodal force K d, XY [N]
    Fr:       float = 0.0    # resultant of Fr_xz, Fr_xy [N]
    Fa:       float = 0.0    # axial nodal force K d [N] (locating only)
    M_xz:     float = 0.0    # nodal moment K d, XZ [N mm]
    M_xy:     float = 0.0    # nodal moment K d, XY [N mm]

    psi_xz:   float = 0.0    # nodal rotation XZ [rad]
    psi_xy:   float = 0.0    # nodal rotation XY [rad]


@dataclass
class ShaftResults:
    """Results of the global shaft FEM for one ``ShaftSystem``.

    Produced by ``RigidSupportFEMSolver.solve`` (through ``build_shaft_result``). Arrays marked
    "per node" are aligned with ``x_nodes``. Bending is solved in two planes (XZ and XY) and combined
    with ``hypot``.

    Parameters
    ----------
    name : str
        Name of the shaft system.
    x_nodes : list of float
        Node positions [mm].
    elements : list
        Beam elements of the mesh.
    K : ndarray
        Global stiffness matrix [N/mm, N, N mm/rad], DOF order ``[u, v, theta]`` per node.
    free_dofs, constrained_dofs : list of int
        Global DOF indices kept in / eliminated from the solve.
    d_total_xz, d_total_xy : ndarray
        Nodal displacements of the XZ and XY planes [mm, mm, rad] per ``[u, v, theta]``.
    f_xz_ext, f_xy_ext : ndarray
        Applied nodal loads [N, N, N mm].
    f_xz_total, f_xy_total : ndarray
        ``K d`` [N, N, N mm]: applied load plus support reaction.
    f_xz_reaction, f_xy_reaction : ndarray
        ``K d - f_ext`` [N, N, N mm].
    T_total, T : ndarray
        Torque per node [N m] (same values).
    tau_total, tau : ndarray
        Nominal torsional shear stress per node [MPa].
    phi_total, phi : ndarray
        Twist angle per node, zero at the first node [rad] (same values).
    torsion_contributions : list
        Per node, the torque sources applied (dicts with ``"label"``, ``"T"`` [N m], ``"tau"`` [MPa]).
    x : ndarray
        Node positions [mm].
    M_xz, M_xy, M : ndarray
        Bending moment per node in the XZ plane, the XY plane and the resultant [N mm].
    V_xz, V_xy, V : ndarray
        Shear force per node in the XZ plane, the XY plane and the resultant [N]. ``V_xz``, ``V_xy``:
        force on the +x face, positive along +v; ``dM/dx = -V``; same sign for both beam theories.
    v_xz, v_xy, v : ndarray
        Transverse deflection per node in the XZ plane, the XY plane and the resultant [mm].
    u : ndarray
        Axial displacement per node [mm] (from the XZ solution).
    theta_xz, theta_xy : ndarray
        Nodal rotation DOF per node [rad].
    d : ndarray
        Shaft diameter per node [mm]; at a change of section, that of the section to the right of
        the node.
    W, Wt : ndarray
        Bending and torsion section modulus per node [mm^3]; same right-hand rule as ``d``.
    sigma_b : ndarray
        Nominal bending stress ``M / W`` per node [MPa] (no stress concentration); ``W`` is the
        right-hand one at a change of section (see Warnings).
    bearing_positions : list of float
        Bearing positions [mm].
    R_xz, R_xy, R, R_axial : ndarray
        Support reactions per bearing ``K d - f_ext``: radial in the XZ and XY planes, their
        resultant, and axial (non-zero for ``"locating"`` bearings only) [N]. Forces exerted by the
        support ON THE SHAFT, opposite to the applied load.
    M_max, x_M_max : float
        Largest resultant bending moment [N mm] and its node position [mm].
    v_max, x_v_max : float
        Largest resultant deflection [mm] and its node position [mm].
    sigma_b_max, x_sigma_b_max : float
        Largest nominal bending stress [MPa] and its node position [mm].
    tau_max, x_tau_max : float
        Largest absolute torsional shear stress [MPa] and its node position [mm].
    phi_max, x_phi_max : float
        Largest absolute twist angle [rad] and its node position [mm].
    bearing_nodes : list of BearingNodeData
        Nodal state at each bearing.

    Warnings
    --------
    The torque ``T`` is in N m, not N mm (``TorsionSolver``); the record writer in
    ``outputs/solvers/fem_results/shaft_results.py`` names it ``T_Nmm``.

    At a node on a change of section, ``d``, ``W``, ``Wt``, ``sigma_b`` and ``tau`` are those of the
    section to the RIGHT of the node (``Shaft.section_at``). The stress of the smaller section at a
    step up is therefore not evaluated and ``sigma_b_max`` can be underestimated (at the step the
    stored stress uses the larger W, ``sigma_b = M / W``).

    Notes
    -----
    Bending moment: ``M = (E I) B_b a_e``, with ``B_b @ a_e`` the bending strain [1/mm], ``d2v/dx2``
    for Euler-Bernoulli (``M = +E I d2v/dx2``, Onate2013_vol2, Eq. (1.5), (1.16c)) and ``dtheta/dx``
    for Timoshenko (``M = +E I dtheta/dx``, Eq. (2.4b), (2.17)). Maxima are taken over nodes.

    Shear force: ``V`` is the force on the +x face along +v and ``dM/dx = -V``; both formulations
    return the same sign (Onate2013_vol2, Fig. 1.2, Eq. (1.24); the last member of Eq. (1.24) is
    printed without the minus sign).
    """

    name: str = ""

    # mesh
    x_nodes:  list[float] = field(default_factory=list)
    elements: list        = field(default_factory=list)

    # FEM solution -- raw
    K:                np.ndarray = field(default_factory=lambda: np.array([]))
    free_dofs:        list[int]  = field(default_factory=list)
    constrained_dofs: list[int]  = field(default_factory=list)

    d_total_xz:    np.ndarray = field(default_factory=lambda: np.array([]))
    d_total_xy:    np.ndarray = field(default_factory=lambda: np.array([]))
    f_xz_ext:      np.ndarray = field(default_factory=lambda: np.array([]))
    f_xy_ext:      np.ndarray = field(default_factory=lambda: np.array([]))
    f_xz_total:    np.ndarray = field(default_factory=lambda: np.array([]))
    f_xy_total:    np.ndarray = field(default_factory=lambda: np.array([]))
    f_xz_reaction: np.ndarray = field(default_factory=lambda: np.array([]))
    f_xy_reaction: np.ndarray = field(default_factory=lambda: np.array([]))

    T_total:               np.ndarray = field(default_factory=lambda: np.array([]))
    tau_total:             np.ndarray = field(default_factory=lambda: np.array([]))
    # Twist angle, node-aligned like T_total/tau_total above.
    # Comes from TorsionSolver.solve() (global_solver/torsion.py),
    # integrated node-to-node from phi(x_nodes[0]) = 0.
    phi_total:             np.ndarray = field(default_factory=lambda: np.array([]))
    torsion_contributions: list       = field(default_factory=list)

    # post-processed engineering quantities
    x: np.ndarray = field(default_factory=lambda: np.array([]))

    M_xz: np.ndarray = field(default_factory=lambda: np.array([]))
    M_xy: np.ndarray = field(default_factory=lambda: np.array([]))
    M:    np.ndarray = field(default_factory=lambda: np.array([]))

    V_xz: np.ndarray = field(default_factory=lambda: np.array([]))
    V_xy: np.ndarray = field(default_factory=lambda: np.array([]))
    V:    np.ndarray = field(default_factory=lambda: np.array([]))

    v_xz: np.ndarray = field(default_factory=lambda: np.array([]))
    v_xy: np.ndarray = field(default_factory=lambda: np.array([]))
    v:    np.ndarray = field(default_factory=lambda: np.array([]))

    # Axial displacement + bending rotation, node-aligned like v_xz/v_xy above
    u:        np.ndarray = field(default_factory=lambda: np.array([]))  # axial [mm]
    theta_xz: np.ndarray = field(default_factory=lambda: np.array([]))  # bending rotation XZ [rad]
    theta_xy: np.ndarray = field(default_factory=lambda: np.array([]))  # bending rotation XY [rad]

    T:   np.ndarray = field(default_factory=lambda: np.array([]))
    phi: np.ndarray = field(default_factory=lambda: np.array([]))  # twist angle [rad], same values as phi_total

    d:  np.ndarray = field(default_factory=lambda: np.array([]))
    W:  np.ndarray = field(default_factory=lambda: np.array([]))
    Wt: np.ndarray = field(default_factory=lambda: np.array([]))

    sigma_b: np.ndarray = field(default_factory=lambda: np.array([]))
    tau:     np.ndarray = field(default_factory=lambda: np.array([]))

    bearing_positions: list[float] = field(default_factory=list)
    R_xz:    np.ndarray = field(default_factory=lambda: np.array([]))
    R_xy:    np.ndarray = field(default_factory=lambda: np.array([]))
    R:       np.ndarray = field(default_factory=lambda: np.array([]))
    R_axial: np.ndarray = field(default_factory=lambda: np.array([]))

    M_max:         float = 0.0
    x_M_max:       float = 0.0
    v_max:         float = 0.0
    x_v_max:       float = 0.0
    sigma_b_max:   float = 0.0
    x_sigma_b_max: float = 0.0
    tau_max:       float = 0.0
    x_tau_max:     float = 0.0
    phi_max:       float = 0.0  # max |phi|, same pattern as tau_max
    x_phi_max:     float = 0.0

    # bearing node data
    bearing_nodes: list[BearingNodeData] = field(default_factory=list)