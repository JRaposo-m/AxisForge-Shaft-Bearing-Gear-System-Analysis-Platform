"""
axisforge/solvers/machine_elements/shaft/fem_solvers/__init__.py

Ponto único de import para o subsistema fem_solvers/ -- estilo slippy,
tudo eager (mesmo padrão de
core/machine_elements/bearings/load_distribution/iso_16281/__init__.py
e de core/machine_elements/bearings/__init__.py): element_theories
(base ABC + as duas implementações de teoria), assembly (stiffness
matrix, load assembly, numerics), constraints (boundary conditions,
extração para submodelo), global_solver (RigidSupportFEMSolver +
postprocessing + torção) e submodel_solver (SubmodelSolver +
postprocessing), todos importados aqui e listados em __all__.

REPLACES a versão anterior deste ficheiro, que era lazy (PEP 562
module __getattr__) e ficou desatualizada da reorganização de pastas já
feita -- apontava para módulos/nomes que já não existem:

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

Quem fizesse `from axisforge...fem_solvers import RigidBearingFEMSolver`
contra o ficheiro antigo tinha um ModuleNotFoundError assim que
__getattr__ tentasse `importlib.import_module(".rigid_bearing", ...)`
-- não um erro que apontasse para o problema real. Esta reescrita troca
o lazy import pelo padrão eager já usado no resto do projeto, e corrige
os caminhos/nomes.

NÃO reexportado aqui (deliberado, avisa se preferires diferente):
SubmodelResult e ShaftResults são dataclasses de RESULTADO, não
solvers -- vivem em axisforge.results.fem_results, um pacote diferente.
O ficheiro antigo misturava essa fronteira (expondo SubmodelResult como
se fosse um cidadão de fem_solvers/). Esta versão só expõe o que
fem_solvers/ define de facto.
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
