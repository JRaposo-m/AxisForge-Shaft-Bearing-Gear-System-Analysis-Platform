"""
core/materials.py

Material property library for AxisForge.

Two independent dataclasses:
  - Material     : structural/shaft materials (Shigley-based)
  - GearMaterial : gear materials (ISO 6336-5)

Phase 1: embedded data for common engineering steels and gear materials.
Phase 3+: database-backed material library.

All stress values in MPa, density in kg/m³, thermal in SI.

References:
  Shigley Table A-20        — shaft steel properties
  Shigley §6-2              — endurance limits
  ISO 6336-5:2016           — gear material fatigue limits
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ===========================================================================
# Material — shaft / structural
# ===========================================================================

@dataclass(frozen=True)
class Material:
    """
    Mechanical properties of a shaft material.

    Se_base is the standard specimen endurance limit (0.5*Sut for Sut < 1400 MPa).
    Actual component Se' is computed by StressSolver applying Marin factors.

    Attributes
    ----------
    material_id   : str     — unique identifier, e.g. 'S355', '42CrMo4'
    Sut           : float   — ultimate tensile strength [MPa]
    Sy            : float   — yield strength [MPa]
    E             : float   — Young's modulus [MPa]
    poisson_ratio : float   — Poisson's ratio [-]
    density       : float   — density [kg/m³]
    Se_base       : float   — specimen endurance limit [MPa] (optional)
    description   : str
    """
    material_id:   str
    Sut:           float
    Sy:            float
    E:             float
    density:       float
    poisson_ratio: float          = 0.3
    Se_base:       Optional[float] = None
    description:   str            = ""

    def __post_init__(self) -> None:
        if self.Sut <= 0:
            raise ValueError(f"Sut must be > 0, got {self.Sut}")
        if self.Sy <= 0:
            raise ValueError(f"Sy must be > 0, got {self.Sy}")
        if self.Sy > self.Sut:
            raise ValueError(f"Sy ({self.Sy}) cannot exceed Sut ({self.Sut})")
        if self.E <= 0:
            raise ValueError(f"E must be > 0, got {self.E}")
        if self.density <= 0:
            raise ValueError(f"density must be > 0, got {self.density}")

    @property
    def endurance_limit(self) -> float:
        """
        Specimen endurance limit Se [MPa].
        Uses stored value if provided; otherwise applies Shigley §6-2:
          Se = 0.5·Sut  (Sut ≤ 1400 MPa)
          Se = 700 MPa  (Sut > 1400 MPa)
        """
        if self.Se_base is not None:
            return self.Se_base
        return min(0.5 * self.Sut, 700.0)

    @property
    def shear_yield_strength(self) -> float:
        """Ssy ≈ 0.577·Sy [MPa] — von Mises distortion energy."""
        return 0.577 * self.Sy


# ===========================================================================
# GearMaterial — ISO 6336-5
# ===========================================================================

@dataclass(frozen=True)
class GearMaterial:
    """
    Material properties for gear analysis (ISO 6336-5).

    Attributes
    ----------
    material_id    : str   — unique identifier, e.g. 'GEAR_STEEL'
    E              : float — Young's modulus [MPa]
    poisson_ratio  : float — Poisson's ratio [-]
    density        : float — density [kg/m³]
    cp             : float — heat capacity [J/(kg·K)]
    k_thermal      : float — thermal conductivity [W/(m·K)]
    sigma_Hlim     : float — contact fatigue limit [MPa] (ISO 6336-5)
    sigma_Flim     : float — bending fatigue limit [MPa] (ISO 6336-5)
    material_class : str   — 'ML' | 'MQ' | 'ME'  (ISO 6336-5 quality grade)
    description    : str

    Note
    ----
    Polymer materials (POM, PA66) have temperature- and cycle-dependent
    sigma_Hlim / sigma_Flim — stored as 0.0 here; computed separately
    by the polymer fatigue solver when implemented.
    """
    material_id:    str
    E:              float
    poisson_ratio:  float
    density:        float
    cp:             float
    k_thermal:      float
    sigma_Hlim:     float
    sigma_Flim:     float
    material_class: str = "MQ"
    description:    str = ""

    def __post_init__(self) -> None:
        if self.E <= 0:
            raise ValueError(f"E must be > 0, got {self.E}")
        if self.poisson_ratio <= 0:
            raise ValueError(f"poisson_ratio must be > 0, got {self.poisson_ratio}")
        if self.density <= 0:
            raise ValueError(f"density must be > 0, got {self.density}")
        if self.sigma_Hlim < 0:
            raise ValueError(f"sigma_Hlim must be >= 0, got {self.sigma_Hlim}")
        if self.sigma_Flim < 0:
            raise ValueError(f"sigma_Flim must be >= 0, got {self.sigma_Flim}")
        if self.material_class not in ("ML", "MQ", "ME"):
            raise ValueError(
                f"material_class must be 'ML', 'MQ' or 'ME', got {self.material_class!r}"
            )

    def equivalent_modulus(self, other: "GearMaterial") -> float:
        """
        Hertzian reduced modulus E* [MPa] for a gear pair.
        E* = 1 / ((1-v1²)/E1 + (1-v2²)/E2)
        """
        return 1.0 / (
            (1 - self.poisson_ratio**2) / self.E
            + (1 - other.poisson_ratio**2) / other.E
        )


# ===========================================================================
# Embedded library — Material
# ===========================================================================

S355 = Material(
    material_id="S355",
    Sut=590.0, Sy=355.0, E=210_000.0, density=7850.0,
    description="EN 10025-2 S355 — structural steel",
)

CrMo42 = Material(
    material_id="42CrMo4",
    Sut=1000.0, Sy=800.0, E=210_000.0, density=7850.0,
    description="DIN 42CrMo4 (AISI 4140 equiv.) — heat-treated QT900",
)

AISI_1045 = Material(
    material_id="AISI_1045",
    Sut=570.0, Sy=310.0, E=207_000.0, density=7850.0,
    description="AISI 1045 normalised — Shigley Table A-20",
)

AISI_4340 = Material(
    material_id="AISI_4340",
    Sut=1460.0, Sy=1380.0, E=207_000.0, density=7850.0,
    Se_base=700.0,
    description="AISI 4340 OQT 600 — high-strength alloy steel",
)


# ===========================================================================
# Embedded library — GearMaterial
# ===========================================================================

GEAR_STEEL = GearMaterial(
    material_id="GEAR_STEEL",
    E=206_000.0, poisson_ratio=0.3, density=7830.0,
    cp=465.0, k_thermal=46.0,
    sigma_Hlim=1500.0, sigma_Flim=430.0,
    material_class="ME",
    description="Case-hardened steel — ISO 6336-5 ME grade",
)

GEAR_ADI = GearMaterial(
    material_id="GEAR_ADI",
    E=210_000.0, poisson_ratio=0.26, density=7850.0,
    cp=460.0, k_thermal=55.0,
    sigma_Hlim=700.0, sigma_Flim=250.0,
    material_class="MQ",
    description="Austempered ductile iron (ADI)",
)

GEAR_POM = GearMaterial(
    material_id="GEAR_POM",
    E=3_200.0, poisson_ratio=0.35, density=1415.0,
    cp=1465.0, k_thermal=0.3,
    sigma_Hlim=0.0, sigma_Flim=0.0,
    material_class="ML",
    description="POM polymer — σHlim/σFlim temperature and cycle dependent",
)

GEAR_PA66 = GearMaterial(
    material_id="GEAR_PA66",
    E=1_850.0, poisson_ratio=0.3, density=1140.0,
    cp=1670.0, k_thermal=0.26,
    sigma_Hlim=0.0, sigma_Flim=0.0,
    material_class="ML",
    description="PA66 polymer — σHlim/σFlim temperature and cycle dependent",
)


# ===========================================================================
# Lookup helpers
# ===========================================================================

_LIBRARY: dict[str, Material] = {
    m.material_id: m for m in [S355, CrMo42, AISI_1045, AISI_4340]
}

_GEAR_LIBRARY: dict[str, GearMaterial] = {
    m.material_id: m for m in [GEAR_STEEL, GEAR_ADI, GEAR_POM, GEAR_PA66]
}


def get_material(material_id: str) -> Material:
    """Retrieve a shaft material by ID. Raises KeyError if not found."""
    if material_id not in _LIBRARY:
        raise KeyError(
            f"Material '{material_id}' not found. "
            f"Available: {', '.join(_LIBRARY)}"
        )
    return _LIBRARY[material_id]


def get_gear_material(material_id: str) -> GearMaterial:
    """Retrieve a gear material by ID. Raises KeyError if not found."""
    if material_id not in _GEAR_LIBRARY:
        raise KeyError(
            f"GearMaterial '{material_id}' not found. "
            f"Available: {', '.join(_GEAR_LIBRARY)}"
        )
    return _GEAR_LIBRARY[material_id]


def available_materials() -> list[str]:
    """Return list of embedded shaft material IDs."""
    return list(_LIBRARY)


def available_gear_materials() -> list[str]:
    """Return list of embedded gear material IDs."""
    return list(_GEAR_LIBRARY)

# adicionar no fim de core/materials.py, antes da secção de embedded library (ou onde preferires)

__all__ = [
    "Material",
    "GearMaterial",
    "S355",
    "CrMo42",
    "AISI_1045",
    "AISI_4340",
    "GEAR_STEEL",
    "GEAR_ADI",
    "GEAR_POM",
    "GEAR_PA66",
    "get_material",
    "get_gear_material",
    "available_materials",
    "available_gear_materials",
]