"""
axisforge/solvers/mesh/__init__.py

Ponto único de import para o subsistema solvers/mesh/ -- estilo slippy,
tudo eager (mesmo padrão de fem_solvers/__init__.py e do
iso_16281/__init__.py): o registo de formas de avaliar
(register_eval_form/resolve_eval_form), a matemática de Richardson GCI
(RichardsonGCI) e o bookkeeping de convergência (MeshConvergenceStudy/
build_mesh_refinement_result), todos de convergence_solver.py -- o
único módulo real que existe hoje em solvers/mesh/.

REPLACES a versão anterior, que era lazy (PEP 562 module __getattr__)
e apontava ConvergenceRecord/MeshRefinementResult/RichardsonGCI/
MeshConvergenceStudy para ".mesh_convergence_study" -- um módulo que já
não existe (só sobra o .pyc em cache; o código real ficou em
convergence_solver.py depois de uma passagem de reorganização
anterior, sem este ficheiro ser atualizado). Qualquer import contra a
versão antiga dava ModuleNotFoundError assim que __getattr__ tentasse
importar ".mesh_convergence_study", não um erro que apontasse para o
problema real -- mesmo padrão de staleness já corrigido em
fem_solvers/__init__.py.

NÃO reexportado aqui (deliberado, mesma razão do fem_solvers/__init__.py):
ConvergenceRecord e MeshRefinementResult são dataclasses de RESULTADO,
não solvers -- vivem em axisforge.results.fem_results.convergence_results,
um pacote diferente. A versão antiga misturava essa fronteira (expondo-as
como se fossem cidadãs de solvers/mesh/). Esta versão só expõe o que
solvers/mesh/ define de facto.
"""
from __future__ import annotations

from .convergence_solver import (
    register_eval_form,
    resolve_eval_form,
    RichardsonGCI,
    MeshConvergenceStudy,
    build_mesh_refinement_result,
)

__all__ = [
    "register_eval_form",
    "resolve_eval_form",
    "RichardsonGCI",
    "MeshConvergenceStudy",
    "build_mesh_refinement_result",
]
