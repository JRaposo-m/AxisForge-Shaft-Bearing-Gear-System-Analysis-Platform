"""axisforge/results/fem_results/shaft_results.py"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class BearingNodeData:
    """FEM nodal state at a bearing position."""

    label:    str   = ""
    position: float = 0.0    # [mm]

    u:        float = 0.0    # axial [mm]
    v_xz:     float = 0.0    # transverse XZ [mm]
    v_xy:     float = 0.0    # transverse XY [mm]
    theta_xz: float = 0.0    # bending rotation XZ [rad]
    theta_xy: float = 0.0    # bending rotation XY [rad]

    Fr_xz:    float = 0.0    # radial reaction XZ [N]
    Fr_xy:    float = 0.0    # radial reaction XY [N]
    Fr:       float = 0.0    # resultant radial [N]
    Fa:       float = 0.0    # axial reaction [N] (locating only)
    M_xz:     float = 0.0    # moment reaction XZ [N·mm]
    M_xy:     float = 0.0    # moment reaction XY [N·mm]

    psi_xz:   float = 0.0    # slope XZ [rad]
    psi_xy:   float = 0.0    # slope XY [rad]


@dataclass
class ShaftResults:
    """Output of RigidBearingFEMSolver + ShaftResultsReader for one ShaftSystem."""

    name: str = ""

    # mesh
    x_nodes:  list[float] = field(default_factory=list)
    elements: list        = field(default_factory=list)

    # FEM solution — raw
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

    # NEW -- axial displacement + bending rotation, node-aligned like v_xz/v_xy above
    u:        np.ndarray = field(default_factory=lambda: np.array([]))  # axial [mm]
    theta_xz: np.ndarray = field(default_factory=lambda: np.array([]))  # bending rotation XZ [rad]
    theta_xy: np.ndarray = field(default_factory=lambda: np.array([]))  # bending rotation XY [rad]

    T: np.ndarray = field(default_factory=lambda: np.array([]))

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

    # bearing node data
    bearing_nodes: list[BearingNodeData] = field(default_factory=list)