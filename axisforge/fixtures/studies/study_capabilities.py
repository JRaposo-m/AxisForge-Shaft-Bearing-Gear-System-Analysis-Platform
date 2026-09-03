"""axisforge/fixtures/studies/study_capabilities.py

Study-stage capabilities. Same shape as ConstructionCapabilities
(fixtures/construction/construction_capabilities.py): typed fields
holding capability strings, `_require()` resolving each to concrete
classes/functions, `resolve()` merging with collision detection.

`construction` holds the ConstructionCapabilities that built the system
this study runs against. validate() no longer walks prerequisites
itself -- it calls catalogue.verify() per requested capability, passing
{"construction": self.construction, "studies": self} as context. The
prerequisite DATA (which capability needs what, and from which stage)
lives in catalogue.py's CATALOGUE, next to each capability's own
description -- see that module's docstring for the exact/prefix match
rule.

Adding a configuration
-----------------------
1. Add a branch to `_require()`.
2. Register the capability's leaf (description + requires, if any) in
   catalogue.py's CATALOGUE, under its chapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from axisforge.fixtures.capabilities import CapabilityError, ConstructionCapabilities
from axisforge.fixtures.capabilities import catalogue

__all__ = ["StudyCapabilities"]


@dataclass(frozen=True)
class StudyCapabilities:
    """Each field (other than `construction`) is the tuple of capability
    strings to pull for that domain, e.g.
    StudyCapabilities(construction=my_construction,
                      shaft_fem=("shaft_fem.timoshenko_rigid",))."""

    construction: ConstructionCapabilities = field(default_factory=ConstructionCapabilities)
    shaft_fem: tuple[str, ...] = ()
    bearing_iso16281: tuple[str, ...] = ()

    _FIELD_DOMAINS: ClassVar[dict[str, str]] = {
        "shaft_fem": "shaft_fem",
        "bearing_iso16281": "bearing_iso16281",
    }

    # ------------------------------------------------------------------
    # Field access
    # ------------------------------------------------------------------

    def _fields(self) -> dict[str, tuple[str, ...]]:
        return {name: getattr(self, name) for name in self._FIELD_DOMAINS}

    def _all(self) -> set[str]:
        out: set[str] = set()
        for values in self._fields().values():
            out.update(values)
        return out

    def has_capability(self, capability: str) -> bool:
        """True if `capability` was requested somewhere in this request.
        Declarative only, mirrors ConstructionCapabilities.has_capability()."""
        return capability in self._all()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []

        for field_name, values in self._fields().items():
            prefix = self._FIELD_DOMAINS[field_name]
            for capability in values:
                if not capability.startswith(prefix + "."):
                    errors.append(
                        f"study.{field_name}: '{capability}' is not "
                        f"a '{prefix}.*' capability"
                    )

        context = {"construction": self.construction, "studies": self}
        for capability in self._all():
            errors.extend(catalogue.verify(capability, context))

        return errors

    def validate_chain(self) -> list[str]:
        # Cross-stage checks against `construction` already happen inside
        # validate() itself, via catalogue.verify() -- nothing further to
        # walk back to.
        return self.validate()

    def validate_or_raise(self) -> None:
        errors = self.validate_chain()
        if errors:
            raise CapabilityError("; ".join(errors))

    # ------------------------------------------------------------------
    # Resolution -- one capability string in, exact classes/functions out.
    # ------------------------------------------------------------------

    def _require(self, capability: str) -> dict:
        # ---- shaft_fem --------------------------------------------------
        if capability == "shaft_fem.timoshenko_rigid":
            from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_bearing import (
                RigidBearingFEMSolver,
            )
            from axisforge.solvers.machine_elements.shaft.static.results_reader import (
                ShaftResultsReader,
            )
            from axisforge.fixtures.studies.shafts.results_library import (
                RigidBearingFEMResultsLibrary,
            )
            from axisforge.fixtures.studies.shafts.fem_simple import solve_system
            return {
                "RigidBearingFEMSolver": RigidBearingFEMSolver,
                "ShaftResultsReader": ShaftResultsReader,
                "RigidBearingFEMResultsLibrary": RigidBearingFEMResultsLibrary,
                "solve_system": solve_system,
            }

        # ---- bearing_iso16281 --------------------------------------------
        if capability == "bearing_iso16281.single_row":
            from axisforge.fixtures.studies.bearings.load_distribution.no_lubrication.single_row.rolling_bearing_study import (
                RollingBearingSolver,
            )
            from axisforge.fixtures.studies.bearings.load_distribution.no_lubrication.results_library import (
                BearingResultsLibrary,
            )
            return {
                "RollingBearingSolver": RollingBearingSolver,
                "BearingResultsLibrary": BearingResultsLibrary,
            }

        raise CapabilityError(f"unknown capability: {capability!r}")

    def resolve(self) -> dict[str, object]:
        self.validate_or_raise()
        out: dict[str, object] = {}
        for capability in self._all():
            for name, value in self._require(capability).items():
                if name in out and out[name] is not value:
                    raise CapabilityError(
                        f"name collision on '{name}' resolving "
                        f"{capability!r}: already resolved to a "
                        f"different object"
                    )
                out[name] = value
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        used = {name: values for name, values in self._fields().items() if values}
        body = "; ".join(f"{name}={', '.join(v)}" for name, v in used.items())
        return f"study: {body or '(none)'}"