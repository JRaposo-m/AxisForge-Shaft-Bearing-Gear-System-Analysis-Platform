"""
axisforge/solvers/shaft/oneD_analysis/build_stiffness_matrix.py
"""
import numpy as np

from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.mesh.shaft.element_type.timoshenko import TimoshenkoBeam
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.core.machine_elements.bearings.bearing import Bearing

class StiffnessMatrixBuilder:

    _BEAM_THEORIES = {
        "timoshenko": TimoshenkoBeam,
    }

    def __init__(self, theory: str = "timoshenko"):
        self.theory = theory.lower()
        if self.theory not in self._BEAM_THEORIES:
            raise ValueError(
                f"StiffnessMatrixBuilder: unknown theory '{theory}', "
                f"available: {list(self._BEAM_THEORIES)}"
            )
        
        self.beam = self._BEAM_THEORIES[self.theory]()


    def build_stiffness_matrix(self, mesh: Mesh1D, elements: list[Elem]) -> np.ndarray:

        n_nodes = mesh.n_nodes
        n_dofs = 3 * n_nodes
        K = np.zeros((n_dofs, n_dofs))
        beam = self.beam                           
        
        for elem in elements:
            K_elem = beam.stiffness_element(elem)
            dofs = [
                3*elem.idx_node_1,     # u1
                3*elem.idx_node_1 + 1, # v1
                3*elem.idx_node_1 + 2, # θ1
                3*elem.idx_node_2,     # u2
                3*elem.idx_node_2 + 1, # v2
                3*elem.idx_node_2 + 2, # θ2
            ]
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    K[gi, gj] += K_elem[i, j]
        return K