"""Result containers of AxisForge: shaft FEM, sub-model, mesh convergence and bearing analysis.

Data shapes only: no module here imports from ``solvers`` at runtime (the ``solvers`` imports that
exist are ``TYPE_CHECKING``-only, for annotations), so the dependency is always solvers -> results.
The shared base module ``_base`` is internal and is not re-exported.
"""
from __future__ import annotations

# ---- Shaft FEM ---------------------------------------------------------
from .fem_results.shaft_results import BearingNodeData, ShaftResults
from .fem_results.submodel_results import SubmodelResult

# ---- Mesh convergence --------------------------------------------------
from .convergence_results.convergence_results import ConvergenceRecord, MeshRefinementResult

# ---- Bearings: load distribution (ISO/TS 16281) ------------------------
from .bearings.load_distribution.load_distribution_results import (
    LoadDistributionResult,
    BallLoadDistributionResult,
    LineContactLoadDistributionResult,
    RollerLoadDistributionResult,
    BearingResult,
    BallBearingResult,
    RollerBearingResult,
)

# ---- Bearings: life (ISO/TS 16281) -------------------------------------
from .bearings.life.basic_life_results import BasicReferenceRatingLifeResult

# ---- Bearings: aggregate per bearing -----------------------------------
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