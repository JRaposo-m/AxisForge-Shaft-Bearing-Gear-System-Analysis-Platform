# axisforge/core/machine_elements/bearings/base.py
"""Catalogue data and family contract of a rolling bearing.

``BearingCatalog`` holds the family-agnostic catalogue data of a bearing
(bore, outer diameter, width, position on the shaft, arrangement).
``BearingFamily`` is the contract that every bearing family implements so that
``Bearing.assemble()`` can build the geometry and evaluate the load ratings.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

# =====================================================================
# ---- BearingCatalog ---------------------------------------------------
# =====================================================================

@dataclass(frozen=True)
class BearingCatalog:
    """Family-agnostic catalogue data of a bearing.

    Frozen dataclass that validates itself at construction: ``__post_init__``
    calls ``validate_or_raise()``.

    Parameters
    ----------
    d : float
        Bore diameter [mm].
    D : float
        Outer diameter [mm].
    b : float, optional
        Width [mm].
    designation : str, optional
        Manufacturer designation, e.g. ``"6208"``.
    label : str, optional
        Identifier used for reporting and traceability.
    position : float, optional
        Axial coordinate along the shaft [mm].
    arrangement : {"locating", "floating", "non-locating", "thrust"}, optional
        Axial role of the bearing in the shaft arrangement; ``"thrust"`` is
        required by the thrust families and refused by the others.

    Raises
    ------
    ValueError
        At construction, if a value is invalid (see ``validate``).

    Notes
    -----
    The basic dynamic load rating is not catalogue data: it is computed by the
    family from the assembled geometry (``BearingFamily.dynamic_capacity``).
    The static rating is not implemented in any family.
    """
    d: float
    D: float
    b: float = 0.0
    designation: str = ""
    label: str = ""
    position: float = 0.0
    arrangement: Literal["locating", "floating", "non-locating", "thrust"] = "locating"

    def __post_init__(self) -> None:
        """Validate the fields; see the class docstring for what is checked."""
        self.validate_or_raise()

    def validate(self) -> list[str]:
        """Check the catalogue values (d > 0, D > d, b >= 0, position >= 0, known arrangement).

        Returns
        -------
        list[str]
            One message per violated condition; empty if the catalogue is valid.
        """
        errors: list[str] = []
        tag = self.label or self.designation or "BearingCatalog"

        if self.d <= 0:
            errors.append(f"{tag}: d must be > 0, got {self.d}")
        if self.D <= 0:
            errors.append(f"{tag}: D must be > 0, got {self.D}")
        if self.D <= self.d:
            errors.append(f"{tag}: D must be > d, got D={self.D}, d={self.d}")
        if self.b < 0:
            errors.append(f"{tag}: b must be >= 0, got {self.b}")
        if self.position < 0:
            errors.append(f"{tag}: position must be >= 0, got {self.position}")
        if self.arrangement not in ("locating", "floating", "non-locating", "thrust"):
            errors.append(
                f"{tag}: arrangement must be 'locating', 'floating', "
                f"'non-locating', or 'thrust' got '{self.arrangement}'"
            )
        return errors

    def validate_or_raise(self) -> None:
        """Raise if the catalogue is invalid.

        Raises
        ------
        ValueError
            With the messages of ``validate()`` joined by newlines.
        """
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


# =====================================================================
# ---- BearingFamily -----------------------------------------------------
# =====================================================================

class BearingFamily(ABC):
    """Contract that every bearing family implements to be used by ``Bearing.assemble()``.

    A family maps the principal dimensions of one bearing type to the geometry
    attributes stored on the ``Bearing`` and evaluates its load ratings. The
    contract is a plain instance contract; registration and classification
    (``@register_family``, ``@point_contact``, ``@line_contact``, ``@thrust``)
    are kept in ``families/family.py``.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Calls the default implementation only; no check is made at class definition."""
        super().__init_subclass__(**kwargs)

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier of the family, e.g. ``'deep_groove_ball'``."""

    @abstractmethod
    def assemble_geometry(self, catalog: BearingCatalog, **geometry_kwargs: Any) -> dict[str, Any]:
        """Map the raw geometry inputs of the bearing to the attributes stored on the ``Bearing``.

        Intended as a pure function: no side effects and no state cached on
        ``self``.

        Parameters
        ----------
        catalog : BearingCatalog
            Read-only; lets a family cross-check catalogue fields (e.g. the
            arrangement).
        **geometry_kwargs
            Family-specific inputs (diameters in [mm], angles as documented by
            each family).

        Returns
        -------
        dict
            Flat dictionary of attributes, each set on the ``Bearing`` by
            ``Bearing.assemble()``. The keys are family-specific.
        """

    @staticmethod
    @abstractmethod
    def dynamic_capacity(bearing) -> float:
        """Basic dynamic load rating of the assembled bearing [N]: Cr for radial duty, Ca for thrust duty.

        Computed from the assembled geometry (not catalogue data).
        """

    @staticmethod
    @abstractmethod
    def per_element_dynamic_capacity(bearing, capacity: float | None = None) -> tuple[float, float]:
        """Load capacities (Q_ci, Q_ce) of one rolling element at the inner and outer ring contact [N].

        ISO_TS_16281_2008, 4.3 (ball) and 5.3 (roller). The ``capacity``
        argument is accepted by the contract; the implementations name it Cr
        or Ca and ignore it. The multi-row families return one pair per row.
        """