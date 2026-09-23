# axisforge/__init__.py
"""
axisforge/__init__.py

Raiz do pacote -- ponto único de import de topo, estilo slippy (eager,
sem lazy loading), mesma cascata dos níveis abaixo:

    config    módulo, NÃO achatado -- axisforge.config.SOLVER_TOLERANCE
    core      materiais, cargas, sistema mecânico, elementos de máquina
    mesh      discretização (hoje só shaft/)
    results   formas de resultado (só dados)
    solvers   cálculo -- por último: depende de core/mesh/results

Ordem = ordem de dependência. results/ não importa solvers/ em runtime,
por isso vem antes; solvers/ importa results/, por isso vem depois.

Fora de propósito:
  ui/        puxa PySide6 -- pesado, e parte `import axisforge` em
             ambientes headless (CI, servidores). Importa-se explicitamente.
  fixtures/  dados de estudo e bibliotecas de resultados, não API.

config não é achatado: MAX_ITER, SOLVER_TOLERANCE, ... no namespace de
topo misturavam constantes com classes e colidiam facilmente.

Colisões: achatar quatro subpacotes num só namespace pode esconder dois
objetos DIFERENTES com o mesmo nome -- o último `import *` ganha em
silêncio. _check_no_collisions() corre no fim e falha alto (ImportError)
em vez de exportar o objeto errado. O mesmo objeto reexportado por dois
subpacotes (ex.: uma classe de results/ reexportada por solvers/) não é
colisão e é aceite.

Regra para código interno: NUNCA `from axisforge import X` -- sempre o
caminho completo (from axisforge.results.bearings... import X). Enquanto
este __init__ corre o pacote está a meio de inicializar, e um import de
topo feito de dentro dá ImportError circular.
"""
from __future__ import annotations

from . import config

from .core import *  # noqa: F401,F403 -- __all__ vem de core/__init__.py
from .core import __all__ as _core_names
from .mesh import *  # noqa: F401,F403 -- __all__ vem de mesh/__init__.py
from .mesh import __all__ as _mesh_names
from .results import *  # noqa: F401,F403 -- __all__ vem de results/__init__.py
from .results import __all__ as _results_names
from .solvers import *  # noqa: F401,F403 -- __all__ vem de solvers/__init__.py
from .solvers import __all__ as _solvers_names

from . import core as _core, mesh as _mesh, results as _results, solvers as _solvers


def _check_no_collisions() -> None:
    """Um nome exportado por mais do que um subpacote tem de ser o MESMO
    objeto em todos. Caso contrário o `import *` acima já sobrescreveu um
    deles em silêncio -- recusa-se o import."""
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
            "axisforge: nomes exportados por mais do que um subpacote com "
            "objetos diferentes -- renomear um ou deixar de o reexportar:\n  "
            + "\n  ".join(clashes)
        )


_check_no_collisions()

# TODO (packaging): confirmar o nome da distribuição em pyproject.toml.
try:
    from importlib.metadata import PackageNotFoundError, version as _dist_version
    __version__ = _dist_version("axisforge")
except PackageNotFoundError:        # a correr da árvore de código, sem install
    __version__ = "0+unknown"

__all__ = list(dict.fromkeys([      # dedup (mesmo objeto via 2 subpacotes), ordem preservada
    "config", "__version__",
    *_core_names, *_mesh_names, *_results_names, *_solvers_names,
]))

del _check_no_collisions, _core, _mesh, _results, _solvers