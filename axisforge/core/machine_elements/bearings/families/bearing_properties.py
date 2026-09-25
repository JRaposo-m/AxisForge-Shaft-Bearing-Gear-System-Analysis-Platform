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

from axisforge.core.materials import Material, get_material


def _resolve(value: Material | str, name: str) -> Material:
    material = get_material(value) if isinstance(value, str) else value
    if not isinstance(material, Material):
        raise TypeError(f"{name}: expected Material or material_id, got {type(value).__name__}")
    material.require_isotropic()          # contact analysis requires E and nu
    return material


# =====================================================================
# ---- SurfacePair -------------------------------------------------------
# =====================================================================

@dataclass(frozen=True)
class SurfacePair:
    """Two surfaces in contact. ``a`` is the rolling element, ``b`` is the raceway."""
    a: Material
    b: Material

    @property
    def moduli(self) -> list[float]:
        """``[E_a, E_b]`` [MPa] -- the format expected by ``hertz_full``."""
        return [self.a.E, self.b.E]

    @property
    def poisson(self) -> list[float]:
        """``[nu_a, nu_b]`` -- the format expected by ``hertz_full`` (its ``v`` argument)."""
        return [self.a.poisson_ratio, self.b.poisson_ratio]

    @property
    def same_material(self) -> bool:
        return self.a == self.b


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
    Radial bearings: rolling element in contact with the inner raceway
    and, where present, the outer raceway. ``outer`` is ``None`` for
    bearing types that have no outer raceway.
    """
    inner: SurfacePair
    outer: SurfacePair | None = None

    @classmethod
    def from_materials(cls, rolling_element: Material | str,
                       inner_ring: Material | str,
                       outer_ring: Material | str | None = None) -> "RadialSurfaces":
        """
        ``outer_ring=None`` means the bearing has NO outer raceway; it
        does not mean "same material as the inner ring" -- for that,
        pass the same material explicitly, or use ``uniform()``.
        """
        re = _resolve(rolling_element, "rolling_element")
        ri = _resolve(inner_ring, "inner_ring")
        outer = None if outer_ring is None else SurfacePair(re, _resolve(outer_ring, "outer_ring"))
        return cls(inner=SurfacePair(re, ri), outer=outer)

    @classmethod
    def uniform(cls, material: Material | str) -> "RadialSurfaces":
        """Same material for the rolling element and both raceways."""
        return cls.from_materials(material, material, material)

    def require_outer(self) -> SurfacePair:
        """Return the outer surface pair, for families that require it."""
        if self.outer is None:
            raise ValueError("This bearing has no outer surface (outer_ring=None), "
                             "but the family requires one.")
        return self.outer

    def pairs(self) -> dict[str, SurfacePair]:
        out = {"inner": self.inner}
        if self.outer is not None:
            out["outer"] = self.outer
        return out