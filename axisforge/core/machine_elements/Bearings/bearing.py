"""
core/machine_elements/Bearings/bearing.py

Bearing -- the orchestrator. Catalogue data + family-derived geometry,
assembled ONCE via Bearing.assemble() and immutable from then on. Never
learns about load cases, X/Y, life, or any analysis result -- those live
in solvers/, each with its own result object.

Do not instantiate directly -- always go through Bearing.assemble().
"""
from __future__ import annotations
from typing import Any

from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.Bearings.family import BearingFamily


class Bearing:

    def __init__(self, catalog: BearingCatalog, family: BearingFamily):
        self.d = catalog.d
        self.D = catalog.D
        self.b = catalog.b
        self.C = catalog.C
        self.C0 = catalog.C0
        self.designation = catalog.designation
        self.label = catalog.label
        self.position = catalog.position
        self.arrangement = catalog.arrangement
        self.dm = 0.5 * (catalog.d + catalog.D)

        self._family = family
        self._enabled_analyses: frozenset[str] = frozenset()
        self._assembled = False   # __setattr__ refuses writes once True

    @classmethod
    def assemble(cls,
                 family: BearingFamily,
                 catalog: BearingCatalog,
                 geometry: dict[str, Any],
                 analyses: dict[str, bool] | None = None) -> "Bearing":
        """
        family   : BearingFamily instance, passed directly
        catalog  : BearingCatalog
        geometry : raw kwargs -> family.assemble_geometry(catalog, **geometry)
        analyses : {name: True/False} -- validated against
                   family.CAPABILITIES / REQUIRED_FOR, never dispatched
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
        The BearingFamily instance this Bearing was assembled with. Public
        so callers (scripts, solvers) can dispatch family-specific methods
        (e.g. per_element_dynamic_capacity()) off the bearing itself,
        without re-importing/tracking the concrete subtype class --
        bearing.family.per_element_dynamic_capacity(bearing) instead of
        DeepGrooveBallFamily.per_element_dynamic_capacity(bearing).
        """
        return self._family

    def validate(self) -> list[str]:
        """
        Call-site compatibility with ShaftSystem.validate() (which does
        `errors.extend(f"{tag}.bearing: {e}" for e in b.validate())` for
        every bearing on the shaft) -- the old pre-rewrite bearing classes
        exposed this as a post-hoc check.

        In normal use this can never actually find anything:
        catalog.validate_or_raise() already ran inside assemble(), and the
        instance is immutable from that point on (__setattr__ refuses
        further writes), so a Bearing that exists has already been proven
        valid and cannot have drifted since. Re-checked here anyway,
        against this instance's own copied attributes (the original
        BearingCatalog is discarded after assemble(), only its fields
        survive on self) -- genuine defense-in-depth (e.g. against
        __setattr__ being bypassed via object.__setattr__), not an
        unconditional [].
        """
        errors: list[str] = []
        tag = self.label or self.designation or self.__class__.__name__

        if not self._assembled:
            errors.append(f"{tag}: not assembled -- Bearing.assemble() never completed")

        if self.position < 0:
            errors.append(f"{tag}: position must be >= 0, got {self.position}")
        if self.C < 0:
            errors.append(f"{tag}: C must be >= 0, got {self.C}")
        if self.C0 < 0:
            errors.append(f"{tag}: C0 must be >= 0, got {self.C0}")
        if self.arrangement not in ("locating", "floating", "non-locating"):
            errors.append(
                f"{tag}: arrangement must be 'locating', 'floating', or "
                f"'non-locating', got '{self.arrangement}'"
            )

        for analysis in self._enabled_analyses:
            required = self._family.REQUIRED_FOR.get(analysis, frozenset())
            missing = [f for f in required if getattr(self, f, None) is None]
            if missing:
                errors.append(f"{tag}: enabled analysis '{analysis}' missing {missing}")

        return errors

    def summary(self) -> str:
        tag = self.label or self.designation or "Bearing"
        header = f"-- {tag} "
        lines = [
            header + "-" * max(0, 44 - len(header)),
            f"  family      : {self._family.name}",
            f"  bearing_type: {getattr(self, 'bearing_type', None)}",
            f"  duty        : {getattr(self, 'duty', None)}",
            f"  position    : {self.position:.2f} mm",
            f"  arrangement : {self.arrangement}",
            f"  d / D       : {self.d:.1f} mm / {self.D:.1f} mm",
            f"  b           : {self.b:.1f} mm",
            f"  C / C0      : {self.C:.0f} N / {self.C0:.0f} N",
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