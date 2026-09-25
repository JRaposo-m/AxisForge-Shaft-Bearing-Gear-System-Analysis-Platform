"""
axisforge/core/machine_elements/bearings/families/bearing_properties.py

Bearing properties used by contact and lubrication analysis, expressed as
CONTACTING SURFACE PAIRS (rolling element <-> raceway):

    SurfacePair                    two surfaces (moduli, poisson, in the
                                   format expected by hertz_full; the
                                   reduced modulus is computed by slippy,
                                   not here)
    BearingSurfaces  (abstract)    common contract: pairs() -> {name: SurfacePair}
      `- RadialSurfaces            inner + outer (outer is None where the
                                   bearing type has no outer raceway)
         (future: ThrustSurfaces, ...)

Contact and lubrication code iterates over surfaces.pairs() without any
knowledge of the concrete bearing type; only the family itself uses the
type-specific names. Each BearingFamily declares which surface class it
expects (SURFACES = RadialSurfaces), in the same way it already declares
BEARING_TYPE and DUTY. Materials come from core/materials (Material,
isotropic).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from axisforge.core.materials.base import Material, get_material


def _resolve(value: Material | str, name: str) -> Material:
    material = get_material(value) if isinstance(value, str) else value
    if not isinstance(material, Material):
        raise TypeError(f"{name}: expected Material or material_id, got {type(value).__name__}")
    material.require_isotropic()          # contact analysis requires E and nu
    return material


# =====================================================================
# ---- SurfacePair ----------------------------------------------------
# =====================================================================

@dataclass(frozen=True)
class SurfacePair:
    """Two surfaces in contact. ``rolling_elem`` is the rolling element, ``race_way`` is the raceway."""
    rolling_elem: Material
    race_way: Material

    @property
    def moduli(self) -> list[float]:
        """``[E_rolling, E_raceway]`` [MPa] -- the format expected by ``hertz_full``."""
        return [self.rolling_elem.E, self.race_way.E]

    @property
    def poisson(self) -> list[float]:
        """``[nu_rolling, nu_raceway]`` -- the format expected by ``hertz_full``."""
        return [self.rolling_elem.poisson_ratio, self.race_way.poisson_ratio]

    @property
    def same_material(self) -> bool:
        return self.rolling_elem == self.race_way


# =====================================================================
# ---- BearingSurfaces (contract) ---------------------------------------
# =====================================================================

class BearingSurfaces(ABC):
    """
    Contract: every surface-pair container for a bearing must be able to
    list its ``SurfacePair`` instances by name, including only those
    that actually exist for the bearing type in question.
    """

    @abstractmethod
    def pairs(self) -> dict[str, SurfacePair]: ...


# =====================================================================
# ---- RadialSurfaces -----------------------------------------------------
# =====================================================================

@dataclass(frozen=True)
class RadialSurfaces(BearingSurfaces):
    """
    Radial bearings: rolling element vs. raceway. Inner and outer raceway
    are always the SAME material (ADR-001) -- one SurfacePair per bearing.
    Whether an outer raceway exists at all, and its curvature, is a
    GEOMETRY question answered by BearingFamily.curvature_outer_raceway()
    (which may return None) -- not a materials question, so there's no
    inner/outer split at this level.
    """
    surface: SurfacePair

    @classmethod
    def from_materials(cls, rolling_element: Material | str,
                        raceway: Material | str) -> "RadialSurfaces":
        rolling = _resolve(rolling_element, "rolling_element")
        race = _resolve(raceway, "raceway")
        return cls(surface=SurfacePair(rolling, race))

    def pairs(self) -> dict[str, SurfacePair]:
        return {"surface": self.surface}