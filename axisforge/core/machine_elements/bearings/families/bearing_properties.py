"""
axisforge/core/machine_elements/bearings/families/bearing_properties.py  -- ESBOCO

Propriedades de um rolamento que nao sao geometria nem catalogo. Por agora
so os materiais; e aqui que entram, mais tarde, outras propriedades
(lubrificacao, temperatura, ...) sem tocar no base.py.

Os materiais vem de core/materials (Material). Este ficheiro so diz QUAIS
materiais um rolamento tem; cada familia decide o que faz com eles.
"""
from __future__ import annotations

from dataclasses import dataclass

from axisforge.core.materials.base import Material, get_material


@dataclass(frozen=True)
class BearingMaterials:
    """
    rolling_element : material da esfera / do rolo
    inner_ring      : material da pista interior
    outer_ring      : material da pista exterior; se None, igual a inner_ring

    Cada campo aceita um Material ou o seu material_id (str).
    O contacto de Hertz precisa de E e nu, por isso os tres tem de ser
    isotropicos (verificado aqui, ao construir).

    Falta agora fazer o agrupamento de superficies em contacto, ou tal pode ser 
    feito no repositorio de Axisforge-Design-Studies de forma a fazer a ligação
    direta a slippy/contact/hertz.py 
    """
    rolling_element: Material | str
    inner_ring: Material | str
    outer_ring: Material | str | None = None

    def __post_init__(self) -> None:
        outer = self.inner_ring if self.outer_ring is None else self.outer_ring
        for name, value in (("rolling_element", self.rolling_element),
                            ("inner_ring", self.inner_ring),
                            ("outer_ring", outer)):
            material = get_material(value) if isinstance(value, str) else value
            if not isinstance(material, Material):
                raise TypeError(f"BearingMaterials.{name}: expected Material or "
                                f"material_id, got {type(value).__name__}")
            material.require_isotropic()
            object.__setattr__(self, name, material)   # frozen: resolve em sitio