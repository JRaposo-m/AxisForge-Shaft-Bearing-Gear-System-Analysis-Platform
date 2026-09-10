"""
axisforge/solvers/shaft/oneD_analysis/build_stiffness_matrix.py
"""
import numpy as np

from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.mesh.shaft.element_type.euler_bernoulli import EulerBernoulliBeam
from axisforge.mesh.shaft.element_type.timoshenko import TimoshenkoBeam
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.core.machine_elements.bearings.bearing import Bearing

class StiffnessMatrixBuilder:

    _BEAM_THEORIES = {
        "timoshenko": TimoshenkoBeam,
        "euler": EulerBernoulliBeam,
    }

    def __init__(self, theory: str = "timoshenko"):
        self.theory = theory.lower()
        if self.theory not in self._BEAM_THEORIES:
            raise ValueError(
                f"StiffnessMatrixBuilder: unknown theory '{theory}', "
                f"available: {list(self._BEAM_THEORIES)}"
            )

        self.beam = self._BEAM_THEORIES[self.theory]()


    def build_stiffness_matrix(self,
                               mesh: Mesh1D,
                               elements: list[Elem],
                               shear_theory: str | None = None,
                               kGA_override: float | None = None) -> np.ndarray:
        """
        kGA_override : forwarded to TimoshenkoBeam.stiffness_element()
            for every element -- see that method's own docstring
            (replaces the transverse shear stiffness K*G*A with this
            exact value instead of computing it from shear_theory, e.g.
            to test against a value read directly from Abaqus's own
            *Preprint, model=YES section-properties printout). ONLY
            forwarded when self.theory == "timoshenko" -- EulerBernoulliBeam
            has no shear term at all (Euler-Bernoulli beam theory has no
            shear correction factor), so kGA_override is silently
            ignored for theory="euler" rather than raising, matching
            shear_theory's own existing "has no effect for euler" note
            in RigidBearingFEMSolver's docstring.
        """
        n_nodes = mesh.n_nodes
        n_dofs = 3 * n_nodes
        K = np.zeros((n_dofs, n_dofs))
        beam = self.beam

        for elem in elements:
            if shear_theory is not None and elem.shear_theory != shear_theory:
                raise ValueError(
                    f"build_stiffness_matrix: elem.shear_theory={elem.shear_theory!r} "
                    f"mas foi pedido shear_theory={shear_theory!r} -- elemento "
                    f"construído com a teoria errada, verifica Elem.from_mesh()."
                )
            theory_to_use = shear_theory if shear_theory is not None else elem.shear_theory

            if self.theory == "timoshenko":
                K_elem = beam.stiffness_element(
                    elem, shear_theory=theory_to_use, kGA_override=kGA_override,
                )
            else:
                K_elem = beam.stiffness_element(elem, shear_theory=theory_to_use)

            dofs = [
                3*elem.idx_node_1, 3*elem.idx_node_1 + 1, 3*elem.idx_node_1 + 2,
                3*elem.idx_node_2, 3*elem.idx_node_2 + 1, 3*elem.idx_node_2 + 2,
            ]
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    K[gi, gj] += K_elem[i, j]
        return K