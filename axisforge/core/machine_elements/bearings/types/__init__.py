# axisforge/core/machine_elements/bearings/families/__init__.py
"""
axisforge/core/machine_elements/bearings/families/__init__.py

Rollup eager de todas as BearingFamily concretas -- estilo slippy:
import direto, sem lazy loading, sem __getattr__. `from .family import
_FAMILY_REGISTRY` já corre as classes todas (os decorators
@register_family executam ao definir cada classe); __all__ vem do
registo, não é escrito à mão, para nunca desalinhar dos nomes reais das
classes.
"""
from __future__ import annotations

from .family import _FAMILY_REGISTRY

globals().update(_FAMILY_REGISTRY)

__all__ = sorted(_FAMILY_REGISTRY)


def available_families() -> dict[str, type]:
    """Nome -> classe BearingFamily. Para listar/selecionar sem já
    saber os nomes de antemão (ex.: capabilities.py, UI de catálogo)."""
    return dict(_FAMILY_REGISTRY)