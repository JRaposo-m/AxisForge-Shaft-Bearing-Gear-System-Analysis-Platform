"""
Materials: elastic, strength and thermal property blocks and a registry.

Re-exports the classes and functions of ``base.py``. No catalogue of concrete
materials exists (there is no ``library/`` content), so
``get_material`` finds only what the caller has registered with ``register``.
"""
from __future__ import annotations

from .base import (
    ElasticBehavior, IsotropicElastic, OrthotropicElastic,
    StrengthProperties, ThermalProperties, Material,
    register, get_material, available_materials,
)

__all__ = [
    "ElasticBehavior", "IsotropicElastic", "OrthotropicElastic",
    "StrengthProperties", "ThermalProperties", "Material",
    "register", "get_material", "available_materials",
]