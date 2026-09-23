"""
axisforge/solvers/machine_elements/shaft/fem_solvers/__init__.py


    RigidBearingFEMSolver  -> ".rigid_bearing"   (nunca existiu com este
                                                    nome; a classe real é
                                                    RigidSupportFEMSolver,
                                                    em global_solver/rigid_support.py)
    SubmodelResult,
    SubmodelSolver         -> ".sub_models"      (pasta moveu-se para
                                                    submodel_solver/;
                                                    SubmodelResult também
                                                    não é definido aqui,
                                                    ver nota abaixo)

"""
from __future__ import annotations

from .element_theories.element_postprocessing import (
    ElementTheoryPostProcessor,
    EulerBernoulliPostProcessing,
    TimoshenkoPostProcessing,
)

from .assembly.build_stiffness_matrix import StiffnessMatrixBuilder
from .assembly.load_assembly.point_loads import assemble_point_load_vector
from .assembly.load_assembly.distributed_loads import (
    element_distributed_force_vector,
    assemble_distributed_load_vector,
)
from .assembly.load_assembly.vector_external_forces import (
    is_distributed,
    build_load_cases,
)
from .assembly.numerics.gauss_quadrature import (
    gauss_points_weights,
    q_degree,
    theta_degree,
    QuadratureOrderEstimator,
)
from .assembly.numerics.numerical_guards import check_conditioning

from .constraints.boundary_conditions import boundary_dofs, boundary_dofs_explicit
from .constraints.submodel_extraction import extract_submodel_values

from .global_solver.rigid_support import RigidSupportSolution, RigidSupportFEMSolver
from .global_solver.torsion import TorsionSolver
from .global_solver.global_postprocessing import (
    GlobalEulerBernoulliPostProcessing,
    GlobalTimoshenkoPostProcessing,
    postprocessor_for_global,
    build_shaft_result,
)

from .submodel_solver.lagrange_multipliers import SubmodelSolution, SubmodelSolver
from .submodel_solver.submodel_postprocessing import (
    SubmodelEulerBernoulliPostProcessing,
    SubmodelTimoshenkoPostProcessing,
    postprocessor_for_submodel,
    build_submodel_result,
)

__all__ = [
    # element_theories/element_postprocessing.py
    "ElementTheoryPostProcessor", "EulerBernoulliPostProcessing", "TimoshenkoPostProcessing",
    # assembly/build_stiffness_matrix.py
    "StiffnessMatrixBuilder",
    # assembly/load_assembly/point_loads.py
    "assemble_point_load_vector",
    # assembly/load_assembly/distributed_loads.py
    "element_distributed_force_vector", "assemble_distributed_load_vector",
    # assembly/load_assembly/vector_external_forces.py
    "is_distributed", "build_load_cases",
    # assembly/numerics/gauss_quadrature.py
    "gauss_points_weights", "q_degree", "theta_degree", "QuadratureOrderEstimator",
    # assembly/numerics/numerical_guards.py
    "check_conditioning",
    # constraints/boundary_conditions.py
    "boundary_dofs", "boundary_dofs_explicit",
    # constraints/submodel_extraction.py
    "extract_submodel_values",
    # global_solver/rigid_support.py
    "RigidSupportSolution", "RigidSupportFEMSolver",
    # global_solver/torsion.py
    "TorsionSolver",
    # global_solver/global_postprocessing.py
    "GlobalEulerBernoulliPostProcessing", "GlobalTimoshenkoPostProcessing",
    "postprocessor_for_global", "build_shaft_result",
    # submodel_solver/lagrange_multipliers.py
    "SubmodelSolution", "SubmodelSolver",
    # submodel_solver/submodel_postprocessing.py
    "SubmodelEulerBernoulliPostProcessing", "SubmodelTimoshenkoPostProcessing",
    "postprocessor_for_submodel", "build_submodel_result",
]
