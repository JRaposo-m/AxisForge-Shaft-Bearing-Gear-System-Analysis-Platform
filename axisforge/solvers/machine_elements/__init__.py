"""
axisforge/solvers/machine_elements/__init__.py

Ponto único de import para solvers/machine_elements/ -- junta os dois
subpacotes com conteúdo real hoje, shaft/ e bearings/, reexportando os
seus __all__ tal e qual, estilo slippy (eager, sem lazy loading) --
mesma cascata usada em shaft/__init__.py -> fem_solvers/__init__.py e
em bearings/__init__.py -> load_distribution/__init__.py, agora um
nível acima a juntar os dois. Um terceiro tipo de elemento de máquina
(ex.: gears/) junta-se aqui do mesmo modo, ao lado destes dois imports.
"""
from __future__ import annotations

from .shaft import *  # noqa: F401,F403 -- __all__ já vem de shaft/__init__.py
from .shaft import __all__ as _shaft_names

from .bearings import *  # noqa: F401,F403 -- __all__ já vem de bearings/__init__.py
from .bearings import __all__ as _bearings_names

__all__ = list(_shaft_names) + list(_bearings_names)