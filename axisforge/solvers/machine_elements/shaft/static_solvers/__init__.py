"""
Static post-processing of shaft results (stress concentration factors).

The package is meant to hold the static-analysis layer on top of the shaft FEM
results. Only the submodule ``postprocessing`` has content; the names listed in
``__all__`` cannot be resolved (see the Warning on ``__getattr__``).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["BearingNodeData", "ShaftResults", "SimpleFEMResultsLibrary", "ShaftResultsReader" ]

_LAZY = {
    "BearingNodeData": ".static_analysis",
    "ShaftResults": ".static_analysis",
    "SimpleFEMResultsLibrary": ".static_analysis",
    "ShaftResultsReader": ".static_analysis",

}

def __getattr__(name: str):
    """
    Resolve the names in ``_LAZY`` on first attribute access.

    Warning
    -------
    Every name in ``_LAZY`` points to the module ``.static_analysis``, which does
    not exist in this package (it holds only ``postprocessing`` and
    ``static_failure``). Any access to ``BearingNodeData``, ``ShaftResults``,
    ``SimpleFEMResultsLibrary`` or ``ShaftResultsReader`` through this package
    raises ``ModuleNotFoundError``. Any other name raises
    ``AttributeError``.
    """
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    """Return the module globals plus the names in ``_LAZY``."""
    return sorted(list(globals().keys()) + list(_LAZY.keys()))

if TYPE_CHECKING:  # pragma: no cover
    # Warning: ``global_postprocessing`` defines ``ShaftResults`` and
    # ``BearingNodeData`` (imported there), but not ``SimpleFEMResultsLibrary``
    # or ``ShaftResultsReader`` (the latter was replaced by functions there).
    from ..fem_solvers.global_solver.global_postprocessing import BearingNodeData, ShaftResults, SimpleFEMResultsLibrary, ShaftResultsReader

