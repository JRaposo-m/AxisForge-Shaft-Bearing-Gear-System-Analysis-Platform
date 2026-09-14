"""
demo_forces_and_stiffness.py

Standalone demo/test script -- NOT part of the repo, just something you
run directly to sanity-check the assembly pieces built so far. Shows:

  1. building a small element list (BeamModelSettings + Elem.from_x_nodes,
     no ShaftSystem involved -- geometry is hardcoded so this has zero
     dependency on shaft_system.py / loads.py, which I don't have the
     current exact API for),
  2. requesting an element's stiffness matrix (bending, via
     elem.stiffness_element(), AND axial, via elem.axial_stiffness_element()),
  3. requesting the GLOBAL external force vector, built from three
     independent contributions:
       - a distributed external load (assemble_distributed_load_vector)
       - external moments (point injection)
       - an axial point load (point injection)

FLAGGED, per "flag don't silently fix": point-load injection (moments,
axial) is done here with a small inline function, NOT by importing your
real point_loads.py -- I don't have that file's current content/API in
this conversation, so I'm not guessing at its signature. The DOF
convention used below is the one documented everywhere else
(build_stiffness_matrix.py, distributed_loads.py): 3 DOF/node,
[axial, radial, moment] -> f[3*i], f[3*i+1], f[3*i+2]. If your real
point_loads.py uses a different call shape, swap _inject_point() out for
it -- the injection math itself (direct add at a node's DOF, no shape
functions) is the same regardless.
"""

from __future__ import annotations

import numpy as np

from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.build_stiffness_matrix import (
    StiffnessMatrixBuilder,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.load_assembly.distributed_loads import (
    assemble_distributed_load_vector,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.numerics.gauss_quadrature import (
    QuadratureOrderEstimator,
)


# ----------------------------------------------------------------------
# 1. Minimal "system": just x_nodes + per-element geometry, no
#    ShaftSystem/shaft.py involved -- keeps this demo self-contained.
# ----------------------------------------------------------------------

def build_demo_elements(settings: BeamModelSettings) -> tuple[list[float], list[Elem]]:
    x_nodes = [0.0, 100.0, 250.0, 400.0]   # mm, 3 elements

    E = 210_000.0   # MPa (N/mm^2), steel
    v = 0.3
    d = 40.0        # mm, solid round shaft, constant section for this demo
    A = np.pi / 4 * d**2
    I = np.pi / 64 * d**4

    elements: list[Elem] = []
    for j in range(len(x_nodes) - 1):
        x_a, x_b = x_nodes[j], x_nodes[j + 1]
        elements.append(Elem(
            length=x_b - x_a,
            E=E, I=I, A=A, v=v,
            idx_node_1=j, idx_node_2=j + 1,
            x_a=x_a, x_b=x_b,
            settings=settings,
        ))
    return x_nodes, elements


# ----------------------------------------------------------------------
# 2. Point-load injection stand-in (axial + moments).
#    Convention: 3 DOF/node [axial, radial, moment].
# ----------------------------------------------------------------------

def _inject_point(f: np.ndarray, x_nodes: list[float], x: float,
                   value: float, dof_slot: int, node_tol: float = 1e-6) -> None:
    """dof_slot: 0=axial, 1=radial, 2=moment. Direct injection, no shape
    functions -- point loads only exist at nodes, so this only works if
    x lands exactly (within tol) on a mesh node, same as your real
    point_loads.py must already assume."""
    node_idx = Elem.find_node_index(x_nodes, x, tol=node_tol)
    f[3 * node_idx + dof_slot] += value


def assemble_axial_point_loads(x_nodes: list[float], axial_cases: list[tuple[float, float]]) -> np.ndarray:
    f = np.zeros(3 * len(x_nodes))
    for x, F_axial in axial_cases:
        _inject_point(f, x_nodes, x, F_axial, dof_slot=0)
    return f


def assemble_moment_point_loads(x_nodes: list[float], moment_cases: list[tuple[float, float]]) -> np.ndarray:
    f = np.zeros(3 * len(x_nodes))
    for x, M in moment_cases:
        _inject_point(f, x_nodes, x, M, dof_slot=2)
    return f


# ----------------------------------------------------------------------
# 3. Distributed external load -- built directly as a case dict, same
#    shape load_cases.py would hand to assemble_distributed_load_vector
#    (bypassing DistributedRadialLoad/ShaftSystem, for the same reason
#    as above: I don't have that class's current exact API).
# ----------------------------------------------------------------------

def build_demo_distributed_case() -> list[dict]:
    q0 = -5.0   # N/mm, uniform downward load over [100, 250]
    return [{
        "x_lo": 100.0, "x_hi": 250.0,
        "q": lambda x: q0,          # constant q(x) -- degree 0
        "theta_fn": None,           # already projected onto this plane
    }]


# ----------------------------------------------------------------------
# 4. Run it
# ----------------------------------------------------------------------

def main() -> None:
    settings = BeamModelSettings(
        beam_theory="timoshenko",
        shear_theory="cowper",
        integration_method="exact",
    )

    x_nodes, elements = build_demo_elements(settings)

    # --- element stiffness matrix (bending + axial), for ONE element ---
    elem0 = elements[0]
    K_bend_elem0 = elem0.stiffness_element()          # 4x4, [v_a, th_a, v_b, th_b]
    K_axial_elem0 = elem0.axial_stiffness_element()   # 2x2, [u_a, u_b]

    K_frame_elem0 = StiffnessMatrixBuilder._element_stiffness_6x6(elem0)  # 6x6, [u_a, v_a, th_a, u_b, v_b, th_b]

    print("=== Element 0 stiffness ===")
    print(repr(elem0))
    print("K_bend (4x4, [v_a, th_a, v_b, th_b]):")
    print(K_bend_elem0)
    print("K_axial (2x2, [u_a, u_b]):")
    print(K_axial_elem0)
    print("K_frame (6x6, [u_a, v_a, th_a, u_b, v_b, th_b] -- same combination used by the builder):")
    with np.printoptions(precision=3, suppress=False, linewidth=220):
        print(K_frame_elem0)

    # --- global stiffness matrix, all elements, no bearings added ---
    builder = StiffnessMatrixBuilder(mesh=_FakeMesh(x_nodes), elements=elements, frame=True)
    K_global = builder.build()
    print("\n=== Global stiffness matrix ===")
    print(f"shape: {K_global.shape}  (3 DOF/node x {len(x_nodes)} nodes, "
          f"dof order per node = [axial, radial, moment])")
    with np.printoptions(precision=3, suppress=False, linewidth=220):
        print(K_global)

    # --- global external force vector: distributed + moments + axial ---
    estimator = QuadratureOrderEstimator(settings.beam_theory)
    distributed_cases = build_demo_distributed_case()
    f_distributed = assemble_distributed_load_vector(
        x_nodes, elements, distributed_cases, estimator,
    )

    moment_cases = [(250.0, 15_000.0)]   # N*mm, CCW-positive at x=250
    f_moments = assemble_moment_point_loads(x_nodes, moment_cases)

    axial_cases = [(400.0, -2_000.0)]    # N, compressive at the free end
    f_axial = assemble_axial_point_loads(x_nodes, axial_cases)

    f_global = f_distributed + f_moments + f_axial

    print("\n=== Global external force vector ===")
    print("f = f_distributed + f_moments + f_axial")
    for i, x in enumerate(x_nodes):
        print(f"node {i} (x={x:6.1f} mm): "
              f"axial={f_global[3*i]:10.3f} N, "
              f"radial={f_global[3*i+1]:10.3f} N, "
              f"moment={f_global[3*i+2]:10.3f} N*mm")


class _FakeMesh:
    """StiffnessMatrixBuilder only reads mesh.x_nodes / mesh.n_nodes --
    this stands in for Mesh1D so the demo doesn't need mesh_1D.py wired
    up to a real ShaftSystem."""
    def __init__(self, x_nodes: list[float]):
        self.x_nodes = x_nodes
        self.n_nodes = len(x_nodes)


if __name__ == "__main__":
    main()