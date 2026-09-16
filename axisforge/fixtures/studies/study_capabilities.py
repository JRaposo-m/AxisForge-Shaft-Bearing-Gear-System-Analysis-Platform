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

shaft_fem capabilities each PIN a `theory` value into the `solve_system`
they return, via functools.partial -- see "shaft_fem.timoshenko_rigid"/
"shaft_fem.euler_bernoulli_rigid" below. Without pinning, the capability
string would mean nothing: fem_simple.solve_system()'s own `theory`
kwarg defaults to "timoshenko" regardless of which capability asked for
it, so two differently-named capabilities could silently resolve to
identical physics if either branch just re-exported the generic
function. A caller can still override `theory=` explicitly when calling
the returned `solve_system` -- partial() fixes a default, not a
lock -- but that would have to be a deliberate choice at the call site,
not an accident of which capability string was requested.

FLAGGED, not yet resolved: fem_simple.solve_system()'s own signature
has not been reviewed against BeamModelSettings (see
mesh/shaft/beam_model_settings.py and rigid_support.py) -- that class
requires beam_theory/shear_theory/integration_method all explicit, no
defaults. Validation case scripts already show `theory` pinned here
while `shear_theory` is passed by the caller at each call site (a
correct no-hidden-default pattern), but no case script observed so far
supplies `integration_method` anywhere -- fem_simple.py needs checking
directly before this pinning scheme can be called complete.

"shaft_fem.comparison" is a different SHAPE of capability from the
other two: it does not return a solve_system pinned to one theory, it
returns the run_comparison()/print_comparison()/write_comparison_report()
trio from comparison_study.py, which together solve the SAME system
TWICE (two theories) and report the difference. It still requires only
"systems.parallel_axis_linear" from construction, same as the other two
-- the extra requirement (two shaft_fem solves happening, not one) is
internal to comparison_study.run_comparison() itself, not something
catalogue.verify() needs to check, since this capability does not
depend on shaft_fem.timoshenko_rigid/euler_bernoulli_rigid also being
requested -- it calls fem_simple.solve_system() directly, with whatever
theory_a/theory_b the caller passes to run_comparison().

"shaft_fem.convergence.<region>.timoshenko" (3 capabilities: region in
{gears, external_distributed, total}) each PIN both a `theory` AND a
`regions` set into convergence_study.run_convergence(), same
functools.partial discipline as shaft_fem.timoshenko_rigid/
euler_bernoulli_rigid -- the capability string alone determines which
shaft regions get mesh-refined, never left to a caller-supplied kwarg.
Timoshenko-only for now: euler_bernoulli convergence is not yet
validated/implemented, so there is no
"shaft_fem.convergence.<region>.euler_bernoulli" branch here nor a
matching leaf in the catalogue -- add both together if/when that need
comes up, following the exact same pattern as the three timoshenko
ones below. "bearings" is deliberately not one of the three region
choices exposed here -- see _convergence_require()'s own docstring
below, and convergence_study.run_convergence()'s own top docstring,
for why: a bearing interval today is still a rigid point reaction, not
the real per-roller load distribution, so "converged" would describe a
physics model that is itself about to be replaced once a
roller-bearing solver exists. All three still require only
"systems.parallel_axis_linear", same reasoning as every other
shaft_fem capability -- the baseline global solve each one refines
around needs an already-resolved system.

Adding a configuration
-----------------------
1. Add a branch to `_require()`.
2. Register the capability's leaf (description + requires, if any) in
   catalogue.py's CATALOGUE, under its chapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from axisforge.fixtures.construction.construction_capabilities import (
    CapabilityError, ConstructionCapabilities,
)
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
                elif not catalogue.is_registered(capability):
                    # Domain prefix looks right but the exact string is
                    # not a leaf in CATALOGUE (typo, or a capability that
                    # was removed/renamed there but not here). Without
                    # this check catalogue.verify() would say nothing --
                    # it treats an unregistered capability as "no known
                    # requirements" -- and the only place this would ever
                    # surface is resolve() -> _require()'s own
                    # CapabilityError fallback, much later than validate().
                    errors.append(
                        f"study.{field_name}: '{capability}' is not "
                        f"registered in the capability catalogue"
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
            from functools import partial
            from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import (
                RigidSupportFEMSolver,
            )
            from axisforge.solvers.machine_elements.shaft.static_solvers.results_reader import (
                ShaftResultsReader,
            )
            from axisforge.fixtures.studies.shafts.fem_studies.results_library import (
                RigidSupportFEMResultsLibrary,
            )
            from axisforge.fixtures.studies.shafts.fem_studies.fem_simple import solve_system as _solve_system
            return {
                "RigidSupportFEMSolver": RigidSupportFEMSolver,
                "ShaftResultsReader": ShaftResultsReader,
                "RigidSupportFEMResultsLibrary": RigidSupportFEMResultsLibrary,
                "solve_system": partial(_solve_system, theory="timoshenko"),
            }

        if capability == "shaft_fem.euler_bernoulli_rigid":
            from functools import partial
            from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import (
                RigidSupportFEMSolver,
            )
            from axisforge.solvers.machine_elements.shaft.static_solvers.results_reader import (
                ShaftResultsReader,
            )
            from axisforge.fixtures.studies.shafts.fem_studies.results_library import (
                RigidSupportFEMResultsLibrary,
            )
            from axisforge.fixtures.studies.shafts.fem_studies.fem_simple import solve_system as _solve_system
            return {
                "RigidSupportFEMSolver": RigidSupportFEMSolver,
                "ShaftResultsReader": ShaftResultsReader,
                "RigidSupportFEMResultsLibrary": RigidSupportFEMResultsLibrary,
                "solve_system": partial(_solve_system, theory="euler_bernoulli"),
            }

        if capability == "shaft_fem.comparison":
            from axisforge.fixtures.studies.shafts.fem_studies.comparison_study import (
                run_comparison,
                print_comparison,
                write_comparison_report,
            )
            return {
                "run_comparison": run_comparison,
                "print_comparison": print_comparison,
                "write_comparison_report": write_comparison_report,
            }

        # ---- shaft_fem.convergence.<region>.<theory> ---------------------
        # region in {gears, external_distributed, total} -- "bearings" is
        # deliberately not offered here; see this class's own top
        # docstring and _convergence_require()'s docstring for why.
        if capability == "shaft_fem.convergence.gears.timoshenko":
            return self._convergence_require(theory="timoshenko", regions={"gears"})

        if capability == "shaft_fem.convergence.external_distributed.timoshenko":
            return self._convergence_require(
                theory="timoshenko", regions={"external_distributed"}
            )

        if capability == "shaft_fem.convergence.total.timoshenko":
            return self._convergence_require(
                theory="timoshenko", regions={"gears", "external_distributed"}
            )

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

    @staticmethod
    def _convergence_require(theory: str, regions: set[str]) -> dict:
        """
        Shared body for all three "shaft_fem.convergence.<region>.timoshenko"
        branches -- avoids repeating the same three imports three times.
        `theory`/`regions` are pinned into run_convergence() via
        functools.partial, exactly like shaft_fem.timoshenko_rigid/
        euler_bernoulli_rigid pin `theory` into fem_simple.solve_system().

        `regions` here is always one of {"gears"},
        {"external_distributed"}, or {"gears", "external_distributed"}
        -- called only from the six branches above, never with
        "bearings" in it. A bearing interval still represents a rigid
        point reaction rather than the real per-roller load
        distribution (no roller-bearing solver exists yet to resolve
        that distribution and feed it into the FEM), so "converging" a
        mesh around it today would validate a physics model that is
        itself about to be replaced -- see convergence_study.
        run_convergence()'s own top docstring for the full reasoning.
        Bearing convergence stays reachable only by calling
        MeshConvergenceStudy.intervals_from_shaft_system() directly
        with regions={"bearings", ...}, never through this capability
        layer, until that future solver exists.
        """
        from functools import partial
        from axisforge.fixtures.studies.shafts.convergence_studies.convergence_study import (
            run_convergence,
        )
        from axisforge.fixtures.studies.shafts.convergence_studies.convergence_library import (
            ConvergenceResultsLibrary,
        )
        return {
            "run_convergence": partial(run_convergence, theory=theory, regions=regions),
            "ConvergenceResultsLibrary": ConvergenceResultsLibrary,
        }

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