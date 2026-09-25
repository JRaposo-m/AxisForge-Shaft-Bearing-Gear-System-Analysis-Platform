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
from enum import Enum
from typing import Any, ClassVar, Literal


# =====================================================================
# ---- BearingType ------------------------------------------------------
# =====================================================================

class BearingType(str, Enum):
    """
    Informational label and the single source of truth for a bearing
    type's duty (radial/thrust) — see the ``duty`` property below.

    Inherits from ``str`` rather than relying on ``enum.auto()``: values
    are persisted to SQLite, and an ``auto()``-assigned integer would
    silently shift if a member were ever inserted mid-list, whereas a
    string value remains stable indefinitely.

    Dispatch on the core side is no longer performed through this enum
    (see ``BearingFamily``); it is retained purely as a label so that
    reporting and GUI code can group bearings by type without importing
    each family class, and so that the solver-side dispatch tables
    (``ISO_16281/rolling_bearing_solver.py``) continue to function
    without modification.
    """
    DEEP_GROOVE_BALL          = "deep_groove_ball"
    ANGULAR_CONTACT           = "angular_contact"
    SELF_ALIGNING_BALL        = "self_aligning_ball"
    THRUST_BALL               = "thrust_ball"
    CYLINDRICAL_ROLLER        = "cylindrical_roller"
    TAPERED_ROLLER            = "tapered_roller"
    SPHERICAL_ROLLER          = "spherical_roller"
    THRUST_CYLINDRICAL_ROLLER = "thrust_cylindrical_roller"
    THRUST_NEEDLE_ROLLER      = "thrust_needle_roller"

    @property
    def duty(self) -> str:
        """
        Return ``'radial'`` or ``'thrust'``.

        Deliberately raises ``NotImplementedError`` for
        ``TAPERED_ROLLER`` and ``SPHERICAL_ROLLER``: tapered and
        spherical roller bearings typically carry combined radial and
        axial load, which does not reduce cleanly to either duty. The
        mapping for these types is left for the implementer to decide
        explicitly when they are supported, rather than being guessed
        here.
        """
        try:
            return _DUTY_BY_TYPE[self]
        except KeyError:
            raise NotImplementedError(
                f"BearingType.{self.name}: duty has not been decided yet -- "
                f"this type may carry combined load, see the class docstring."
            ) from None


_DUTY_BY_TYPE: dict[BearingType, str] = {
    BearingType.DEEP_GROOVE_BALL: "radial",
    BearingType.ANGULAR_CONTACT: "radial",
    BearingType.SELF_ALIGNING_BALL: "radial",
    BearingType.THRUST_BALL: "thrust",
    BearingType.CYLINDRICAL_ROLLER: "radial",
    BearingType.THRUST_CYLINDRICAL_ROLLER: "thrust",
    BearingType.THRUST_NEEDLE_ROLLER: "thrust",
    # TAPERED_ROLLER and SPHERICAL_ROLLER are deliberately left unmapped.
}


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
    arrangement: Literal["locating", "floating", "non-locating"] = "locating"

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
        if self.arrangement not in ("locating", "floating", "non-locating"):
            errors.append(
                f"{tag}: arrangement must be 'locating', 'floating', or "
                f"'non-locating', got '{self.arrangement}'"
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

    CAPABILITIES: ClassVar[frozenset[str]] = frozenset()
    REQUIRED_FOR: ClassVar[dict[str, frozenset[str]]] = {}
    BEARING_TYPE: ClassVar[BearingType | None] = None

    #: ``"radial"`` | ``"thrust"``. Kept as an explicit class attribute
    #: (rather than being derived solely from ``BEARING_TYPE.duty``) by
    #: design: it is read at the class level in several places (tests,
    #: reporting) without instantiating the family, and
    #: ``__init_subclass__`` guarantees it can never diverge from what
    #: ``BEARING_TYPE.duty`` would return.
    DUTY: ClassVar[Literal["radial", "thrust"] | None] = None

    #: The surface-pair container class (a ``BearingSurfaces`` subclass,
    #: e.g. ``RadialSurfaces``) that this family expects. Left as
    #: ``None`` for families that do not yet have one (thrust families;
    #: multi-row families, whose individual rows carry their own).
    #: Validated in ``register_family()`` (``families/family.py``),
    #: since ``base.py`` must not import from ``families/``, which in
    #: turn depends on this module.
    SURFACES: ClassVar[type | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if set(cls.REQUIRED_FOR) != set(cls.CAPABILITIES):
            raise TypeError(
                f"{cls.__name__}: REQUIRED_FOR keys {set(cls.REQUIRED_FOR)} "
                f"must exactly match CAPABILITIES {set(cls.CAPABILITIES)}"
            )

        if cls.BEARING_TYPE is not None and cls.DUTY is not None:
            expected = cls.BEARING_TYPE.duty
            if cls.DUTY != expected:
                raise TypeError(
                    f"{cls.__name__}: DUTY={cls.DUTY!r} does not match "
                    f"BEARING_TYPE.duty={expected!r} for {cls.BEARING_TYPE!r}"
                )

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