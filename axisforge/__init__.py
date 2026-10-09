# axisforge/__init__.py
"""
axisforge/__init__.py

Package root -- single top-level import point, slippy style (eager, no lazy loading), same
cascade as the levels below:

    config    module, NOT flattened -- axisforge.config.SOLVER_TOLERANCE
    core      materials, loads, mechanical system, machine elements
    mesh      discretisation (currently only shaft/)
    results   result shapes (data only)
    solvers   computation -- after the packages it depends on: core/mesh/results
    outputs   records, text and CSV of the objects above; module, NOT flattened --
              axisforge.outputs.solvers.bearings...

Order = dependency order. results/ does not import solvers/ at run time, so it comes first;
solvers/ imports results/, so it comes after. outputs/ reads core and result objects through
their public attributes and imports neither core/ nor solvers/, so it comes last.

Out of scope:
  ui/        pulls in PySide6 -- heavy, and breaks `import axisforge` in headless
             environments (CI, servers). Import it explicitly.
  fixtures/  study data and result libraries, not API.

config is not flattened: MAX_ITER, SOLVER_TOLERANCE, ... in the top-level namespace mixed
constants with classes and collided easily. outputs is not flattened for the same reason: its
functions are named after what they print (``system_text``, ``print_system``, ...) and are
meant to be reached through their module path.

Collisions: flattening four subpackages into one namespace can hide two DIFFERENT objects with
the same name -- the last `import *` silently wins. _check_no_collisions() runs at the end and
fails loudly (ImportError) instead of exporting the wrong object. The same object re-exported by
two subpackages (for example a class from results/ re-exported by solvers/) is not a collision
and is accepted.

Rule for internal code: NEVER `from axisforge import X` -- always the full path
(from axisforge.results.bearings... import X). While this __init__ runs the package is half
initialised, and a top-level import made from inside raises a circular ImportError.
"""
from __future__ import annotations

from . import config

from .core import *  # noqa: F401,F403 -- __all__ comes from core/__init__.py
from .core import __all__ as _core_names
from .mesh import *  # noqa: F401,F403 -- __all__ comes from mesh/__init__.py
from .mesh import __all__ as _mesh_names
from .results import *  # noqa: F401,F403 -- __all__ comes from results/__init__.py
from .results import __all__ as _results_names
from .solvers import *  # noqa: F401,F403 -- __all__ comes from solvers/__init__.py
from .solvers import __all__ as _solvers_names

from . import outputs  # module, not flattened (same treatment as config)

from . import core as _core, mesh as _mesh, results as _results, solvers as _solvers


def _check_no_collisions() -> None:
    """An exported name shared by more than one subpackage must be the SAME object in all of
    them. Otherwise the `import *` above has already silently overwritten one of them and the
    import is refused."""
    seen: dict[str, tuple[str, object]] = {}
    clashes = []
    for pkg in (_core, _mesh, _results, _solvers):
        for name in pkg.__all__:
            obj = getattr(pkg, name)
            if name in seen and seen[name][1] is not obj:
                clashes.append(f"{name!r}: {seen[name][0]} vs {pkg.__name__}")
            seen.setdefault(name, (pkg.__name__, obj))
    if clashes:
        raise ImportError(
            "axisforge: names exported by more than one subpackage with different "
            "objects -- rename one or stop re-exporting it:\n  "
            + "\n  ".join(clashes)
        )


_check_no_collisions()

# TODO (packaging): confirm the distribution name in pyproject.toml.
try:
    from importlib.metadata import PackageNotFoundError, version as _dist_version
    __version__ = _dist_version("axisforge")
except PackageNotFoundError:        # running from the source tree, without install
    __version__ = "0+unknown"

__all__ = list(dict.fromkeys([      # dedup (same object via 2 subpackages), order preserved
    "config", "outputs", "__version__",
    *_core_names, *_mesh_names, *_results_names, *_solvers_names,
]))

del _check_no_collisions, _core, _mesh, _results, _solvers