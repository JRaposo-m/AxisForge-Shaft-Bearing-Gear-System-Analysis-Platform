# axisforge/results/__init__.py
"""
axisforge/results/__init__.py

Ponto único de import para as formas de resultado do AxisForge -- estilo
slippy, tudo eager (mesmo padrão de solvers/.../iso_16281/__init__.py).

Só formas de dados: nenhum módulo aqui importa de solvers/ em runtime
(os imports de solvers/ que existem são TYPE_CHECKING-only, para anotações).
Por isso importar este pacote nunca puxa um solver -- a dependência vai
sempre solvers -> results, nunca o contrário.

Ordem: _base primeiro (convenções partilhadas), depois as famílias sem
dependências internas (fem, convergence), depois bearings de baixo para
cima -- load_distribution e life antes do agregador bearing_analysis_result,
que é o único que conhece os dois.

_base (DC, is_abstract, check_*) é interno e não é reexportado.
"""
from __future__ import annotations

# ---- FEM do veio -------------------------------------------------------
from .fem_results.shaft_results import BearingNodeData, ShaftResults
from .fem_results.submodel_results import SubmodelResult

# ---- Convergência de malha --------------------------------------------
from .convergence_results.convergence_results import ConvergenceRecord, MeshRefinementResult

# ---- Rolamentos: distribuição de carga (ISO/TS 16281) ------------------
from .bearings.load_distribution.load_distribution_results import (
    LoadDistributionResult,
    BallLoadDistributionResult,
    LineContactLoadDistributionResult,
    RollerLoadDistributionResult,
    BearingResult,
    BallBearingResult,
    RollerBearingResult,
)

# ---- Rolamentos: vida (ISO/TS 16281) ----------------------------------
from .bearings.life.basic_life_results import BasicReferenceRatingLifeResult

# ---- Rolamentos: agregado por rolamento -------------------------------
from .bearings.bearing_analysis_result import BearingAnalysisResult

__all__ = [
    # fem_results/
    "BearingNodeData", "ShaftResults", "SubmodelResult",
    # convergence_results/
    "ConvergenceRecord", "MeshRefinementResult",
    # bearings/load_distribution/
    "LoadDistributionResult", "BallLoadDistributionResult",
    "LineContactLoadDistributionResult", "RollerLoadDistributionResult",
    "BearingResult", "BallBearingResult", "RollerBearingResult",
    # bearings/life/
    "BasicReferenceRatingLifeResult",
    # bearings/
    "BearingAnalysisResult",
]