"""
core/machine_elements/bearings/bearing.py

``Bearing`` is the orchestrator: catalogue data combined with
family-derived geometry, assembled exactly once via ``Bearing.assemble()``
and immutable thereafter. It has no knowledge of load cases, X/Y factors,
service life, or any other analysis result; those belong in ``solvers/``,
each with its own dedicated result object.

Do not instantiate ``Bearing`` directly; always go through
``Bearing.assemble()``.

``C`` (the basic dynamic load rating) is computed by the family from the
assembled geometry (see the ``C`` property below); it is no longer
catalogue data. If the family supplies ``surfaces`` (a ``BearingSurfaces``
instance), it is exposed as ``bearing.surfaces`` in the same way as every
other attribute returned by ``assemble_geometry``.
"""
from __future__ import annotations
from typing import Any

from axisforge.core.machine_elements.bearings.base import BearingCatalog
from axisforge.core.machine_elements.bearings.base import BearingFamily


class Bearing:

    def __init__(self, catalog: BearingCatalog, family: BearingFamily):
        self.d = catalog.d
        self.D = catalog.D
        self.b = catalog.b
        self.designation = catalog.designation
        self.label = catalog.label
        self.position = catalog.position
        self.arrangement = catalog.arrangement
        self.dm = 0.5 * (catalog.d + catalog.D)
        self.surfaces = None   # BearingSurfaces; set by assemble() if the family provides it

        self._family = family
        self._enabled_analyses: frozenset[str] = frozenset()
        self._assembled = False   # __setattr__ refuses further writes once True

    @classmethod
    def assemble(cls,
                 family: BearingFamily,
                 catalog: BearingCatalog,
                 geometry: dict[str, Any],
                 analyses: dict[str, bool] | None = None) -> "Bearing":
        """
        Parameters
        ----------
        family   : a ``BearingFamily`` instance, passed directly.
        catalog  : a ``BearingCatalog`` instance.
        geometry : raw keyword arguments forwarded to
            ``family.assemble_geometry(catalog, **geometry)``.
        analyses : ``{name: True/False}``; validated against
            ``family.CAPABILITIES`` / ``REQUIRED_FOR`` but never
            dispatched from here.
        """
        catalog.validate_or_raise()
        bearing = cls(catalog, family)

        enabled = {name for name, on in (analyses or {}).items() if on}
        unsupported = enabled - family.CAPABILITIES
        if unsupported:
            raise NotImplementedError(
                f"{catalog.label or catalog.designation}: family "
                f"'{family.name}' does not support: {sorted(unsupported)}. "
                f"Supported: {sorted(family.CAPABILITIES)}"
            )

        computed = family.assemble_geometry(catalog, **geometry)
        for attr, value in computed.items():
            setattr(bearing, attr, value)
        bearing.duty = family.DUTY
        bearing.bearing_type = family.BEARING_TYPE

        missing_by_analysis: dict[str, list[str]] = {}
        for analysis in enabled:
            required = family.REQUIRED_FOR.get(analysis, frozenset())
            missing = [f for f in required if getattr(bearing, f, None) is None]
            if missing:
                missing_by_analysis[analysis] = missing
        if missing_by_analysis:
            details = "; ".join(f"{a}: missing {m}" for a, m in missing_by_analysis.items())
            raise RuntimeError(
                f"{catalog.label or catalog.designation}: geometry insufficient "
                f"for the requested analyses -- {details}"
            )

        bearing._enabled_analyses = frozenset(enabled)
        bearing._assembled = True
        return bearing

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_assembled", False):
            raise AttributeError(
                f"Bearing(label={getattr(self, 'label', '')!r}) is immutable "
                f"after assemble() -- cannot set '{name}'. Analysis results "
                f"belong in the solver's own result object, not on the Bearing."
            )
        super().__setattr__(name, value)

    def is_enabled(self, analysis: str) -> bool:
        return analysis in self._enabled_analyses

    def has_internal_geometry(self) -> bool:
        return self._assembled

    def is_locating(self) -> bool:
        return self.arrangement == "locating"

    @property
    def family(self) -> BearingFamily:
        """
        The ``BearingFamily`` instance this ``Bearing`` was assembled
        with. Exposed publicly so that callers (scripts, solvers) can
        dispatch family-specific methods (e.g.
        ``per_element_dynamic_capacity()``) off the bearing itself,
        without having to re-import or track the concrete subtype
        class: ``bearing.family.per_element_dynamic_capacity(bearing)``
        rather than
        ``DeepGrooveBallFamily.per_element_dynamic_capacity(bearing)``.
        """
        return self._family

    @property
    def C(self) -> float:
        """
        Basic dynamic load rating (``Cr`` for radial duty, ``Ca`` for
        thrust duty) [N], computed by the family from the assembled
        geometry. Read by the solvers' ``_Cr_Ca()``. Requires
        ``assemble()`` to have completed.
        """
        if not self._assembled:
            raise RuntimeError(
                f"Bearing '{self.label or self.designation}': C requires "
                f"Bearing.assemble() to have completed.")
        return self._family.dynamic_capacity(self)

    def validate(self) -> list[str]:
        """
        Provided for call-site compatibility with
        ``ShaftSystem.validate()``, which does
        ``errors.extend(f"{tag}.bearing: {e}" for e in b.validate())``
        for every bearing on the shaft — the pre-rewrite bearing
        classes exposed this as a post-hoc check.

        In normal use this method cannot actually surface anything new:
        ``catalog.validate_or_raise()`` has already run inside
        ``assemble()``, and the instance is immutable from that point
        on (``__setattr__`` refuses further writes), so any ``Bearing``
        that exists has already been proven valid and cannot have
        drifted since. The checks are repeated here regardless, against
        this instance's own copied attributes (the original
        ``BearingCatalog`` is discarded after ``assemble()``; only its
        field values survive on ``self``) as genuine defense-in-depth
        — for example against ``__setattr__`` being bypassed via
        ``object.__setattr__`` — rather than as an unconditional empty
        list.
        """
        errors: list[str] = []
        tag = self.label or self.designation or self.__class__.__name__

        if not self._assembled:
            errors.append(f"{tag}: not assembled -- Bearing.assemble() never completed")

        if self.position < 0:
            errors.append(f"{tag}: position must be >= 0, got {self.position}")
        if self.arrangement not in ("locating", "floating", "non-locating"):
            errors.append(
                f"{tag}: arrangement must be 'locating', 'floating', or "
                f"'non-locating', got '{self.arrangement}'"
            )
        if self.d <= 0:
            errors.append(f"{tag}: d must be > 0, got {self.d}")
        if self.D <= 0:
            errors.append(f"{tag}: D must be > 0, got {self.D}")
        if self.D <= self.d:
            errors.append(f"{tag}: D must be > d, got D={self.D}, d={self.d}")
        if self.b < 0:
            errors.append(f"{tag}: b must be >= 0, got {self.b}")

        for analysis in self._enabled_analyses:
            required = self._family.REQUIRED_FOR.get(analysis, frozenset())
            missing = [f for f in required if getattr(self, f, None) is None]
            if missing:
                errors.append(f"{tag}: enabled analysis '{analysis}' missing {missing}")

        return errors

    def summary(self) -> str:
        tag = self.label or self.designation or "Bearing"
        header = f"-- {tag} "
        c_txt = f"{self.C:.0f} N" if self._assembled else "n/a"
        lines = [
            header + "-" * max(0, 44 - len(header)),
            f"  family      : {self._family.name}",
            f"  bearing_type: {getattr(self, 'bearing_type', None)}",
            f"  duty        : {getattr(self, 'duty', None)}",
            f"  position    : {self.position:.2f} mm",
            f"  arrangement : {self.arrangement}",
            f"  d / D       : {self.d:.1f} mm / {self.D:.1f} mm",
            f"  b           : {self.b:.1f} mm",
            f"  C (computed): {c_txt}",
            f"  geometry    : {'assembled' if self._assembled else 'not assembled'}",
            f"  analyses    : {sorted(self._enabled_analyses) or '(none enabled)'}",
            "-" * 44,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"family={self._family.name!r}, "
            f"designation={self.designation!r}, "
            f"position={self.position:.2f} mm, "
            f"arrangement={self.arrangement!r})"
        )