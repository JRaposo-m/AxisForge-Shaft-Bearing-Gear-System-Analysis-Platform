# axisforge/core/materials/__init__.py
"""
axisforge/core/materials/__init__.py

Ponto único de import para o subsistema de materiais -- estilo slippy,
eager. Por agora só existe base.py (Material, ElasticBehavior e os
blocos opcionais); quando as bibliotecas de materiais concretos
(steels.py, cast_iron.py, polymers.py, ...) existirem, importam-se aqui
também -- cada uma regista as suas instâncias via register() só por
serem importadas, por isso o import tem de ser eager, tal como
families/__init__.py faz com @register_family.
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