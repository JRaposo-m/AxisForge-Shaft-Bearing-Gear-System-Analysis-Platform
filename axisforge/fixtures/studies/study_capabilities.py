"""axisforge/fixtures/studies/study_capabilities.py

Study-stage capabilities for the shaft-FEM domain. Same shape as
ConstructionCapabilities (fixtures/capabilities/construction.py): typed
fields holding capability strings, `_require()` resolving each to
concrete classes/functions, `resolve()` merging with collision detection.

solve_system() takes the ConstructionCapabilities that built its
`system` argument as a second parameter, to confirm construction
requested "systems.parallel_axis_linear" (the only capability that
resolves the system -- gear-mesh loads + shaft positions -- before
returning it), via ConstructionCapabilities.has_capability(). Declarative
check, not runtime inspection of `system` itself.

Capability string names the exact solver configuration, not a generic
placeholder: "shaft_fem.timoshenko_rigid" -- RigidBearingFEMSolver's only
real configuration today (Timoshenko beam theory, every bearing a rigid
support). A future beam theory or bearing-constraint model gets its own
capability string (e.g. "shaft_fem.euler_rigid"), never a hidden
parameter folded inside this one.

Adding a configuration
-----------------------
1. Add a branch to `_require()`.
2. Register the capability string in catalogue.py, under CAPABILITIES["shaft_fem"].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from axisforge.fixtures.capabilities import CapabilityError

__all__ = ["StudyCapabilities"]


@dataclass(frozen=True)
class StudyCapabilities:
    """`shaft_fem` is the tuple of capability strings for the
    shaft-FEM-solve domain, e.g.
    StudyCapabilities(shaft_fem=("shaft_fem.timoshenko_rigid",))."""

    shaft_fem: tuple[str, ...] = ()

    _FIELD_DOMAINS: ClassVar[dict[str, str]] = {
        "shaft_fem": "shaft_fem",
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
        return errors

    def validate_chain(self) -> list[str]:
        # Study takes an already-built system directly at call time --
        # nothing upstream to chain to.
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