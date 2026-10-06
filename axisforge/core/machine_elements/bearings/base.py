# axisforge/core/machine_elements/bearings/base.py
"""
axisforge/core/machine_elements/bearings/base.py

Defines the three foundational primitives on which the rest of the
``bearings`` package depends: ``BearingType``, ``BearingCatalog`` and
``BearingFamily``. They are kept in a single module not because the
package requires one file per concept, but because they are small,
stable, and mutually coupled: ``BearingFamily.assemble_geometry`` reads
a ``BearingCatalog``, and ``BearingFamily.BEARING_TYPE`` is itself a
``BearingType``.

``bearings/__init__.py`` re-exports all three directly from this module
(a plain import, without ``__getattr__``-based lazy loading, since they
are inexpensive to import and deferring them would offer no benefit).
``Bearing`` (see ``bearing.py``) remains lazily imported there, as it is
comparatively more expensive to load.
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
    """
    Generic, family-agnostic catalogue data for a bearing.

    Validates itself at construction time: ``__post_init__`` calls
    ``validate_or_raise()``. Previously, ``validate()``/
    ``validate_or_raise()`` existed but were opt-in, which allowed a
    physically impossible catalogue (``D < d``, negative ``b``) to be
    constructed and to circulate through the rest of the code
    unchecked until something happened to call
    ``validate_or_raise()`` explicitly.

    Attributes
    ----------
    d, D, b     : bore diameter, outer diameter, width [mm].
    designation : manufacturer designation, e.g. ``"6208"``.
    label       : identifier used for reporting and traceability.
    position    : axial coordinate along the shaft [mm].
    arrangement : ``"locating"`` | ``"floating"`` | ``"non-locating"``.

    Notes
    -----
    ``C`` and ``C0`` are intentionally not catalogue data: the dynamic
    load rating is computed by the family from the assembled geometry
    (see ``BearingFamily.dynamic_capacity``).
    """
    d: float
    D: float
    b: float = 0.0
    designation: str = ""
    label: str = ""
    position: float = 0.0
    arrangement: Literal["locating", "floating", "non-locating", "thrust"] = "locating"

    def __post_init__(self) -> None:
        self.validate_or_raise()

    def validate(self) -> list[str]:
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
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


# =====================================================================
# ---- BearingFamily -----------------------------------------------------
# =====================================================================

class BearingFamily(ABC):
    """
    Contract that every bearing family/subtype must implement in order
    to be pluggable into ``Bearing.assemble()``. This is a plain
    instance contract; it does not itself require registration
    (``families/family.py`` maintains its own ``@register_family``
    decorator, which is a separate concern from this contract).

    ``__init_subclass__`` (below) enforces, at class-definition time
    rather than only in tests, two consistency requirements that were
    previously caught only by a test suite: ``DUTY`` must agree with
    ``BEARING_TYPE.duty``, and the keys of ``REQUIRED_FOR`` must match
    ``CAPABILITIES`` exactly. A misdeclared family therefore fails at
    import time rather than at first use.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``'deep_groove_ball'``."""

    @abstractmethod
    def assemble_geometry(self, catalog: BearingCatalog, **geometry_kwargs: Any) -> dict[str, Any]:
        """
        Pure function mapping raw geometry inputs to a flat attribute
        dictionary that is mirrored onto the ``Bearing``. Must have no
        side effects and must not cache state on ``self``.

        Parameters
        ----------
        catalog : read-only; provided so that families that need to
            cross-check catalogue fields can do so (e.g. rejecting
            ``arrangement="locating"`` where it is not physically
            meaningful).

        Returns
        -------
        dict
            Must include every field referenced by this family's own
            ``REQUIRED_FOR``.
        """

    @staticmethod
    @abstractmethod
    def dynamic_capacity(bearing) -> float:
        """
        Basic dynamic load rating (``Cr`` for radial duty, ``Ca`` for
        thrust duty) [N], computed from the assembled geometry. This
        replaces the ``C`` value that previously came from the
        catalogue.
        """

    @staticmethod
    @abstractmethod
    def per_element_dynamic_capacity(bearing, capacity: float | None = None) -> tuple[float, float]:
        """Per-rolling-element dynamic capacity (``Q_ci``, ``Q_ce``)."""