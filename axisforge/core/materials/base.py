"""
Material definition: density plus independent blocks of elastic, strength and thermal properties.

The module knows nothing about machine elements (shafts, gears, bearings): it only defines what a
material is. Each element reads from here what it needs; standard-specific data of an element (for
example the allowable stress numbers of a gear rating standard) live with that element and refer to the material by its id.
Units: E, G, stresses [MPa]; density [kg/m^3]; cp [J/(kg K)]; k_thermal [W/(m K)]; alpha [1/K].
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


def _raise_if(errors: list[str]) -> None:
    """Raise ValueError with the messages joined by newlines if the list is not empty."""
    if errors:
        raise ValueError("\n".join(errors))


# =====================================================================
# ---- Elastic behaviour ---------------------------------------------
# =====================================================================

@dataclass(frozen=True)
class ElasticBehavior(ABC):
    """
    Abstract elastic behaviour of a material.

    Each concrete behaviour states whether it is isotropic (``is_isotropic``) and lists its own
    validation errors (``errors``); construction raises ``ValueError`` if the list is not empty.
    """

    def __post_init__(self) -> None:
        """Raise ValueError if ``errors()`` returns any message."""
        _raise_if(self.errors())

    @property
    @abstractmethod
    def is_isotropic(self) -> bool:
        """True for an isotropic behaviour, False otherwise."""
        ...

    @abstractmethod
    def errors(self) -> list[str]:
        """Return the validation messages of this behaviour; empty if valid."""
        ...


@dataclass(frozen=True)
class IsotropicElastic(ElasticBehavior):
    """
    Linear-elastic isotropic behaviour (two independent constants).

    Parameters
    ----------
    E : float
        Young modulus [MPa], > 0.
    poisson_ratio : float
        Poisson ratio [-], 0 < nu < 0.5 (no default).

    Raises
    ------
    ValueError
        At construction, if E <= 0 or nu is outside (0, 0.5).
    """
    E: float
    poisson_ratio: float

    is_isotropic = True

    @property
    def G(self) -> float:
        """
        Shear modulus [MPa].

        Notes
        -----
        ``G = E / (2 (1 + nu))``, with ``E`` = ``E`` and ``nu`` = ``poisson_ratio``.

        References
        ----------
        .. [1] Onate2013_vol2, text under Eq. (2.3b); Harris2007_vol2, Eq. (4.34).
        """
        return self.E / (2.0 * (1.0 + self.poisson_ratio))

    def errors(self) -> list[str]:
        """Return the validation messages (E > 0, 0 < nu < 0.5); empty if valid."""
        errs = []
        if self.E <= 0:
            errs.append(f"IsotropicElastic: E must be > 0, got {self.E}")
        if not 0.0 < self.poisson_ratio < 0.5:
            errs.append(f"IsotropicElastic: poisson_ratio must be in (0, 0.5), "
                        f"got {self.poisson_ratio}")
        return errs


@dataclass(frozen=True)
class OrthotropicElastic(ElasticBehavior):
    """
    Linear-elastic orthotropic behaviour (three Young moduli, three shear moduli, three Poisson
    ratios).

    Parameters
    ----------
    E1, E2, E3 : float
        Young moduli along the three material axes [MPa], > 0.
    G12, G13, G23 : float
        Shear moduli in the three material planes [MPa], > 0.
    nu12, nu13, nu23 : float
        Poisson ratios [-]. Not validated.

    Notes
    -----
    Only the positivity of the six moduli is checked; the stability (positive definiteness) of
    the compliance matrix is not checked.
    """
    E1: float
    E2: float
    E3: float
    G12: float
    G13: float
    G23: float
    nu12: float
    nu13: float
    nu23: float

    is_isotropic = False

    def errors(self) -> list[str]:
        """Return the validation messages (the six moduli must be > 0); empty if valid."""
        errs = []
        for name in ("E1", "E2", "E3", "G12", "G13", "G23"):
            if getattr(self, name) <= 0:
                errs.append(f"OrthotropicElastic: {name} must be > 0, got {getattr(self, name)}")
        return errs


# =====================================================================
# ---- Optional property blocks ---------------------------------------
# =====================================================================

@dataclass(frozen=True)
class StrengthProperties:
    """
    Strength block (optional part of a material).

    Parameters
    ----------
    Sut : float or None
        Ultimate tensile strength [MPa], > 0 when given.
    Sy : float or None
        Yield strength [MPa], > 0 when given; ``Sy > Sut`` raises.
    Se : float or None
        Endurance (fatigue) limit [MPa], > 0 when given. Stored value only; the rules to estimate
        it live in the element that uses it.

    Raises
    ------
    ValueError
        At construction, if a given value is <= 0 or if Sy exceeds Sut.
    """
    Sut: Optional[float] = None
    Sy: Optional[float] = None
    Se: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate the fields; see the class docstring for what is checked."""
        errs = [f"StrengthProperties: {n} must be > 0, got {v}"
                for n, v in (("Sut", self.Sut), ("Sy", self.Sy), ("Se", self.Se))
                if v is not None and v <= 0]
        if self.Sut is not None and self.Sy is not None and self.Sy > self.Sut:
            errs.append(f"StrengthProperties: Sy ({self.Sy}) cannot exceed Sut ({self.Sut})")
        _raise_if(errs)


@dataclass(frozen=True)
class ThermalProperties:
    """
    Thermal block (optional part of a material).

    Parameters
    ----------
    cp : float or None
        Specific heat capacity [J/(kg K)], > 0 when given.
    k_thermal : float or None
        Thermal conductivity [W/(m K)], > 0 when given.
    alpha : float or None
        Coefficient of linear thermal expansion [1/K], > 0 when given.

    Raises
    ------
    ValueError
        At construction, if a given value is <= 0.
    """
    cp: Optional[float] = None
    k_thermal: Optional[float] = None
    alpha: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate the fields; see the class docstring for what is checked."""
        errs = [f"ThermalProperties: {n} must be > 0, got {v}"
                for n, v in (("cp", self.cp), ("k_thermal", self.k_thermal), ("alpha", self.alpha))
                if v is not None and v <= 0]
        _raise_if(errs)


# =====================================================================
# ---- Material --------------------------------------------------------
# =====================================================================

@dataclass(frozen=True, kw_only=True)
class Material:
    """
    Material of a machine element: density plus elastic, strength and thermal blocks.

    Parameters
    ----------
    material_id : str
        Unique non-empty identifier (key of the registry).
    density : float
        Mass density [kg/m^3], > 0.
    elastic : ElasticBehavior
        Elastic behaviour (isotropic or orthotropic).
    strength : StrengthProperties or None
        Strength block; None if not available.
    thermal : ThermalProperties or None
        Thermal block; None if not available.
    description : str
        Free text.

    Raises
    ------
    ValueError
        At construction, if ``material_id`` is empty or ``density`` <= 0.

    Notes
    -----
    All fields are keyword-only and the instance is frozen. The density is not read by any
    module of axisforge.
    """
    material_id: str
    density: float
    elastic: ElasticBehavior
    strength: Optional[StrengthProperties] = None
    thermal: Optional[ThermalProperties] = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the fields; see the class docstring for what is checked."""
        errs = []
        if not self.material_id:
            errs.append("Material: material_id must be non-empty")
        if self.density <= 0:
            errs.append(f"Material({self.material_id!r}): density must be > 0, got {self.density}")
        _raise_if(errs)

    # --- accessors that fail when the data an analysis needs is missing --
    def require_isotropic(self) -> IsotropicElastic:
        """Return the isotropic elastic block; raise TypeError if the material is not isotropic."""
        if not isinstance(self.elastic, IsotropicElastic):
            raise TypeError(f"Material '{self.material_id}' is not isotropic "
                            f"({type(self.elastic).__name__}).")
        return self.elastic

    @property
    def E(self) -> float:
        """Young modulus [MPa] of an isotropic material (TypeError otherwise)."""
        return self.require_isotropic().E

    @property
    def poisson_ratio(self) -> float:
        """Poisson ratio [-] of an isotropic material (TypeError otherwise)."""
        return self.require_isotropic().poisson_ratio

    def require_strength(self, *fields: str) -> StrengthProperties:
        """Return the strength block; raise ValueError if it or any named field (Sut, Sy, Se) is missing."""
        if self.strength is None or any(getattr(self.strength, f) is None for f in fields):
            raise ValueError(f"Material '{self.material_id}': missing strength "
                             f"data {fields or ''}")
        return self.strength

    def require_thermal(self, *fields: str) -> ThermalProperties:
        """Return the thermal block; raise ValueError if it or any named field (cp, k_thermal, alpha) is missing."""
        if self.thermal is None or any(getattr(self.thermal, f) is None for f in fields):
            raise ValueError(f"Material '{self.material_id}': missing thermal "
                             f"data {fields or ''}")
        return self.thermal


# =====================================================================
# ---- Registry / lookup -----------------------------------------------
# =====================================================================
# The registry starts empty and no catalogue module exists: it holds only the
# materials registered by the calling code.

_REGISTRY: dict[str, Material] = {}


def register(*materials: Material) -> None:
    """Add materials to the registry by material_id; raise ValueError if an id is already registered."""
    for m in materials:
        if m.material_id in _REGISTRY:
            raise ValueError(f"Material '{m.material_id}' already registered")
        _REGISTRY[m.material_id] = m


def get_material(material_id: str) -> Material:
    """Return the registered material; raise KeyError (listing the registered ids) if it is unknown."""
    if material_id not in _REGISTRY:
        raise KeyError(f"Material '{material_id}' not found. "
                       f"Available: {', '.join(_REGISTRY)}")
    return _REGISTRY[material_id]


def available_materials() -> list[str]:
    """Return the ids of the registered materials (empty until some are registered)."""
    return list(_REGISTRY)


__all__ = [
    "ElasticBehavior", "IsotropicElastic", "OrthotropicElastic",
    "StrengthProperties", "ThermalProperties", "Material",
    "register", "get_material", "available_materials",
]