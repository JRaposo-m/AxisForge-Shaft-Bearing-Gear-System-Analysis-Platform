"""
core/materials.py
Material properties for shaft analysis.

Phase 1: embedded data for common engineering steels.
Phase 3+: database-backed material library.

All values in SI: MPa for stress, dimensionless for ratios.

References:
  Shigley Table A-20 (steel properties)
  Shigley §6-2 (endurance limits)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Material:
    """
    Mechanical properties of a shaft material.

    Se_base is the standard specimen endurance limit (0.5*Sut for Sut < 1400 MPa).
    Actual component Se' is computed by StressSolver applying Marin factors.

    Attributes
    ----------
    material_id : str
        Unique identifier, e.g. 'S355', '42CrMo4'.
    Sut : float
        Ultimate tensile strength [MPa].
    Sy : float
        Yield strength [MPa].
    E : float
        Young's modulus [GPa].
    density : float
        Density [kg/m³].
    Se_base : float
        Specimen endurance limit [MPa]. If not provided, computed as 0.5*Sut
        (capped at 700 MPa per Shigley §6-2).
    description : str
        Optional human-readable description.
    """
    material_id: str
    Sut: float          # [MPa]
    Sy: float           # [MPa]
    E: float            # [GPa]
    density: float      # [kg/m³]
    poisson_ratio: float = 0.3
    Se_base: Optional[float] = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.Sut <= 0:
            raise ValueError(f"Sut must be > 0, got {self.Sut}")
        if self.Sy <= 0:
            raise ValueError(f"Sy must be > 0, got {self.Sy}")
        if self.Sy > self.Sut:
            raise ValueError(
                f"Sy ({self.Sy}) cannot exceed Sut ({self.Sut})"
            )
        if self.E <= 0:
            raise ValueError(f"E must be > 0, got {self.E}")
        if self.density <= 0:
            raise ValueError(f"density must be > 0, got {self.density}")

    @property
    def endurance_limit(self) -> float:
        """
        Specimen endurance limit Se [MPa].
        Uses stored value if provided; otherwise applies Shigley §6-2 rule:
          Se = 0.5 * Sut  (Sut ≤ 1400 MPa)
          Se = 700 MPa    (Sut > 1400 MPa)
        """
        if self.Se_base is not None:
            return self.Se_base
        return min(0.5 * self.Sut, 700.0)

    @property
    def shear_yield_strength(self) -> float:
        """Ssy ≈ 0.577 * Sy [MPa] — von Mises distortion energy."""
        return 0.577 * self.Sy


# ---------------------------------------------------------------------------
# Embedded material library — Phase 1
# ---------------------------------------------------------------------------

# Structural steel — EN 10025 S355 (general shaft use, low-alloy)
S355 = Material(
    material_id="S355",
    Sut=590.0,
    Sy=355.0,
    E=210.0,
    density=7850.0,
    description="EN 10025-2 S355 — structural steel",
)

# Heat-treatable alloy steel — DIN 42CrMo4 (common transmission shafts)
CrMo42 = Material(
    material_id="42CrMo4",
    Sut=1000.0,
    Sy=800.0,
    E=210.0,
    density=7850.0,
    description="DIN 42CrMo4 (AISI 4140 equiv.) — heat-treated, QT900",
)

# Carbon steel — AISI 1045 normalised (Shigley reference material)
AISI_1045 = Material(
    material_id="AISI_1045",
    Sut=570.0,
    Sy=310.0,
    E=207.0,
    density=7850.0,
    description="AISI 1045 normalised — Shigley reference (Table A-20)",
)

# High-strength alloy — 4340 OQT 600 (Shigley Table A-24)
AISI_4340 = Material(
    material_id="AISI_4340",
    Sut=1460.0,
    Sy=1380.0,
    E=207.0,
    density=7850.0,
    Se_base=700.0,   # capped per Shigley §6-2
    description="AISI 4340 OQT 600 — high-strength alloy steel",
)

# ---------------------------------------------------------------------------
# Lookup helper
# ---------------------------------------------------------------------------

_LIBRARY: dict[str, Material] = {
    m.material_id: m
    for m in [S355, CrMo42, AISI_1045, AISI_4340]
}


def get_material(material_id: str) -> Material:
    """
    Retrieve a material from the embedded library by ID.

    Raises
    ------
    KeyError
        If material_id is not found. Provides available IDs in message.
    """
    if material_id not in _LIBRARY:
        available = ", ".join(_LIBRARY.keys())
        raise KeyError(
            f"Material '{material_id}' not found. "
            f"Available: {available}"
        )
    return _LIBRARY[material_id]


def available_materials() -> list[str]:
    """Return list of embedded material IDs."""
    return list(_LIBRARY.keys())
