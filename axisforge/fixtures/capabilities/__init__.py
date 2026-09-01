"""
fixtures/capabilities/__init__.py

Two classes, one per pipeline stage, in ONE file on purpose:
`ConstructionCapabilities` and `ResolutionCapabilities`. They share the
same shape (typed fields holding capability strings; `_require()`
resolving each to concrete classes/functions; `resolve()` merging with
collision detection) and, since ResolutionCapabilities' own
solve_system() needs to check a ConstructionCapabilities instance's
requested capabilities (via `has_capability()`, defined on
ConstructionCapabilities below), the two are tightly coupled across the
Construction/Resolution boundary -- keeping them in the same file means
that coupling is one scroll away, not a cross-module import to trace.
Previously ResolutionCapabilities lived in its own resolution.py; moved
here for that reason. Future stages (MeshLoads, ElementAnalysis) land
the same way, as their own class in this file, when written.

The full listing of capability strings, with descriptions, lives apart
in catalogue.py -- pure metadata, no axisforge imports, safe to
introspect without pulling in any core/solvers code; catalogue.py is
shared across both classes below (its CAPABILITIES dict is not
Construction-only), not duplicated per stage.

`ConstructionCapabilities` covers Construction-domain capabilities only
(shafts; bearings/gears/systems land here the same way, one at a time,
as each is written). Nothing resolved by IT is solved -- MeshLoads/
Resolution/ElementAnalysis stay out of its `_require()` -- WITH ONE
DELIBERATE EXCEPTION: "systems.parallel_axis_linear" also calls
SpurHelicalGearSystem.resolve() before returning, on explicit user
decision (a system's whole purpose is to be resolved and positioned,
so shipping it unresolved was judged an incomplete deliverable). See
that branch's own comment in ConstructionCapabilities._require() below,
and fixtures/construction/systems/parallel_axis/spur_helical/
linear_chain_fixture.py's module docstring, for the full reasoning.
Every other Construction capability still stops at Construction. The
actual FEM solve lives in `ResolutionCapabilities` below (via its own
exported solve_system() -- see that class's own docstring).

Usage
-----
    construction = ConstructionCapabilities(
        shaft=("shafts.stepped_3section",),
    )
    construction.validate_or_raise()
    objs = construction.resolve()
    ShaftFixture          = objs["ShaftFixture"]
    make_stepped_3section = objs["factory"]

    resolution = ResolutionCapabilities(
        shaft_fem=("shaft_fem.timoshenko_rigid",),
    )
    objs = resolution.resolve()
    solve_system = objs["solve_system"]
    library = solve_system(system, construction)  # construction: the
                                                    # ConstructionCapabilities
                                                    # that built `system`

Adding a domain (either class)
--------------------------------
1. Add a branch to that class's own `_require()`, grouped under that
   domain's own "# ---- <domain> ----" section.
2. Register its capability strings (with one-line descriptions) in
   catalogue.py, under CAPABILITIES["<domain>"].
3. Construction only: if a capability only makes sense alongside
   another domain already requested -- e.g. "systems.parallel_axis_linear"
   needs already-built shaft/gear/bearing objects to assemble a
   ShaftSystem from -- add an entry to ConstructionCapabilities'
   `_PREREQUISITES`. See its own comment below for the worked example
   (no longer hypothetical -- that entry is live).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

__all__ = ["CapabilityError", "ConstructionCapabilities", "ResolutionCapabilities"]


class CapabilityError(Exception):
    """An unknown capability string, a request whose prerequisite
    domain is missing from the same request, a capability string
    that doesn't belong to the field it was put in, or a name
    collision between two capabilities resolving the same name to two
    different objects."""


# ===========================================================================
# ConstructionCapabilities
# ===========================================================================

@dataclass(frozen=True)
class ConstructionCapabilities:
    """Stage 1: Construction. Each field is the tuple of capability
    strings to pull for that domain -- e.g.
    `ConstructionCapabilities(shaft=("shafts.stepped_3section",),
    bearings=("bearings.dgbb_iso16281",))`. An empty tuple means the
    domain is not used. `validate_chain()` exists (mirroring the old
    stage-gated design) even though Construction is the first stage
    and has nothing to walk back to -- so a future MeshLoadsCapabilities
    can call `self.construction.validate_chain()` the same way every
    later stage will.

    Field name is singular (`shaft`, `system`) to match the old
    dataclass; the domain PREFIX a field's strings must carry is
    plural ("shafts.", "systems.") to match catalogue.py's domain
    keys -- see `_FIELD_DOMAINS`.
    """

    shaft: tuple[str, ...] = ()
    bearings: tuple[str, ...] = ()
    bearing_families: tuple[str, ...] = ()
    gears: tuple[str, ...] = ()
    system: tuple[str, ...] = ()

    # field name -> the capability-string domain prefix it accepts.
    _FIELD_DOMAINS: ClassVar[dict[str, str]] = {
        "shaft": "shafts",
        "bearings": "bearings",
        "bearing_families": "bearing_families",
        "gears": "gears",
        "system": "systems",
    }

    # capability string -> domain prefixes it needs present elsewhere
    # in the SAME ConstructionCapabilities request. This is the
    # within-Construction "can this resolve given what else is asked
    # for" check -- the equivalent of the old stage-gating, but keyed
    # per capability instead of per stage.
    _PREREQUISITES: ClassVar[dict[str, tuple[str, ...]]] = {
        "systems.parallel_axis_linear": ("shafts", "gears", "bearings"),
        #
        # Read as: requesting "systems.parallel_axis_linear" (in the
        # `system` field) is only valid if `shaft`, `gears` and
        # `bearings` also carry at least one capability starting with
        # "shafts.", "gears." and "bearings." respectively --
        # build_linear_system() receives already-built
        # Shaft/GearElement/Bearing objects (wrapped in ShaftSpec/
        # StageSpec), it does not construct them itself. Note this
        # uses the bare domain prefixes ("gears", not the earlier
        # placeholder "gears.parallel_axis", which never matches a
        # real capability string -- gears.* capabilities are
        # "gears.spur"/"gears.helical"/etc., not "gears.parallel_axis.*").
        #
        # Nothing else needs an entry here yet: shaft/bearings/gears
        # are independent of each other at Construction level -- only
        # a system (built from the others) depends on anything.
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
        """True if `capability` was requested somewhere in this
        request (any field). Purely declarative -- reads this
        instance's own fields, nothing else; does not check whether
        `resolve()` was actually called, or whether it succeeded.

        This is how a later stage checks a Construction prerequisite
        without holding a runtime reference to whatever object
        Construction eventually built. The motivating case:
        fixtures/solvers/fem_simple.py's solve_system() takes the
        ConstructionCapabilities that built its `system` argument
        alongside `system` itself, and calls
        construction.has_capability("systems.parallel_axis_linear") to
        confirm the system it was handed came from the one Construction
        capability that also resolves (see that capability's own note
        in `_require()` below) -- i.e. Resolution's prerequisite is
        "Construction was asked to produce an already-resolved system",
        checked here declaratively rather than by inspecting the
        SpurHelicalGearSystem object's own (private) resolved state.
        """
        return capability in self._all()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors: list[str] = []

        # Each field only accepts capabilities from its own domain --
        # catches a gears.* string typed into bearings=(...), etc.
        for field_name, values in self._fields().items():
            prefix = self._FIELD_DOMAINS[field_name]
            for capability in values:
                if not capability.startswith(prefix + "."):
                    errors.append(
                        f"construction.{field_name}: '{capability}' is not "
                        f"a '{prefix}.*' capability"
                    )

        requested = self._all()
        for capability in requested:
            for prefix in self._PREREQUISITES.get(capability, ()):
                if not any(c.startswith(prefix + ".") for c in requested):
                    errors.append(
                        f"{capability}: requires a '{prefix}.*' capability "
                        f"in the same request (none requested)"
                    )

        return errors

    def validate_chain(self) -> list[str]:
        # Construction is the first stage -- nothing to walk back to.
        return self.validate()

    def validate_or_raise(self) -> None:
        errors = self.validate_chain()
        if errors:
            raise CapabilityError("; ".join(errors))

    # ------------------------------------------------------------------
    # Resolution -- one capability string in, exact classes/factory out.
    # ------------------------------------------------------------------

    def _require(self, capability: str) -> dict:
        """Resolve one capability string. Does not check
        `_PREREQUISITES` -- that only matters once the whole request
        is considered together, so it lives in `validate()`, not here.
        """

        # ---- shafts --------------------------------------------------
        if capability == "shafts.generic":
            from axisforge.core.machine_elements.shaft.shaft import (
                Shaft, ShaftSection, Shoulder,
            )
            from axisforge.fixtures.construction.shafts.shaft_fixture import (
                SectionSpec, ShaftFixture, make_shaft,
            )
            return {
                "Shaft": Shaft,
                "ShaftSection": ShaftSection,
                "Shoulder": Shoulder,
                "SectionSpec": SectionSpec,
                "ShaftFixture": ShaftFixture,
                "factory": make_shaft,
            }

        if capability == "shafts.stepped":
            from axisforge.core.machine_elements.shaft.shaft import (
                Shaft, ShaftSection, Shoulder,
            )
            from axisforge.fixtures.construction.shafts.shaft_fixture import (
                SectionSpec, ShaftFixture, make_stepped_shaft,
            )
            return {
                "Shaft": Shaft,
                "ShaftSection": ShaftSection,
                "Shoulder": Shoulder,
                "SectionSpec": SectionSpec,
                "ShaftFixture": ShaftFixture,
                "factory": make_stepped_shaft,
            }

        # ---- bearings ----------------------------------------------------
        # 9 families total (core/machine_elements/bearings/families/):
        #   ball_bearing/radial:  DeepGrooveBallFamily, AngularContactFamily,
        #                         SelfAligningBallFamily
        #   ball_bearing/thrust:  SingleRowThrustBallFamily, MultiRowThrustBallFamily
        #   roller_bearing/radial: CylindricalRollerFamily
        #   roller_bearing/thrust: ThrustCylindricalRollerFamily,
        #                          MultiRowThrustCylindricalRollerFamily,
        #                          ThrustNeedleRollerFamily
        #
        # Capability suffix = the family's own `.name` property verbatim
        # (e.g. DeepGrooveBallFamily().name == "deep_groove_ball" ->
        # "bearings.deep_groove_ball") -- read off the real core source,
        # not chosen independently, so there is one name to keep in sync,
        # not two.
        #
        # Two domains, same suffix, different prefix -- both point at the
        # SAME family class (safe: resolve()'s collision guard only fires
        # on two DIFFERENT objects under one name, not two capabilities
        # agreeing on the same object):
        #   "bearings.<name>"          -> Bearing + BearingCatalog +
        #                                  make_<...>_bearing() factory
        #                                  (fully assembled Bearing)
        #   "bearing_families.<name>"  -> Bearing + BearingCatalog +
        #                                  the bare <...>Family class, for
        #                                  a script that wants to call
        #                                  Bearing.assemble() itself with
        #                                  full control (e.g. a custom
        #                                  geometry dict shape) instead of
        #                                  going through the factory.
        #
        # Every family already has a distinct class name AND a distinct
        # factory function name (make_deep_groove_ball_bearing vs.
        # make_angular_contact_bearing, ...), so even requesting every
        # bearings.* capability in one ConstructionCapabilities call can't
        # collide -- unlike shafts.generic/shafts.stepped sharing the
        # generic "factory" key, which is why gears (above) and bearings
        # (here) both use type-specific keys from the start.
        if capability == "bearings.deep_groove_ball":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.fixtures.construction.bearings.ball_radial_fixture import (
                make_deep_groove_ball_bearing,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "make_deep_groove_ball_bearing": make_deep_groove_ball_bearing,
            }

        if capability == "bearing_families.deep_groove_ball":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.core.machine_elements.bearings.families.ball_bearing.radial.subtypes.deep_groove import (
                DeepGrooveBallFamily,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "DeepGrooveBallFamily": DeepGrooveBallFamily,
            }

        if capability == "bearings.angular_contact":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.fixtures.construction.bearings.ball_radial_fixture import (
                make_angular_contact_bearing,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "make_angular_contact_bearing": make_angular_contact_bearing,
            }

        if capability == "bearing_families.angular_contact":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.core.machine_elements.bearings.families.ball_bearing.radial.subtypes.angular_contact import (
                AngularContactFamily,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "AngularContactFamily": AngularContactFamily,
            }

        if capability == "bearings.self_aligning":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.fixtures.construction.bearings.ball_radial_fixture import (
                make_self_aligning_ball_bearing,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "make_self_aligning_ball_bearing": make_self_aligning_ball_bearing,
            }

        if capability == "bearing_families.self_aligning":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.core.machine_elements.bearings.families.ball_bearing.radial.subtypes.self_aligning import (
                SelfAligningBallFamily,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "SelfAligningBallFamily": SelfAligningBallFamily,
            }

        if capability == "bearings.thrust_ball_single_row":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.fixtures.construction.bearings.ball_thrust_fixture import (
                make_thrust_ball_bearing,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "make_thrust_ball_bearing": make_thrust_ball_bearing,
            }

        if capability == "bearing_families.thrust_ball_single_row":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.core.machine_elements.bearings.families.ball_bearing.thrust.subtypes.thrust_single import (
                SingleRowThrustBallFamily,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "SingleRowThrustBallFamily": SingleRowThrustBallFamily,
            }

        if capability == "bearings.thrust_ball_multirow":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.fixtures.construction.bearings.ball_thrust_fixture import (
                make_thrust_ball_multirow_bearing,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "make_thrust_ball_multirow_bearing": make_thrust_ball_multirow_bearing,
            }

        if capability == "bearing_families.thrust_ball_multirow":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.core.machine_elements.bearings.families.ball_bearing.thrust.subtypes.thrust_multirow import (
                MultiRowThrustBallFamily,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "MultiRowThrustBallFamily": MultiRowThrustBallFamily,
            }

        if capability == "bearings.cylindrical_roller":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.fixtures.construction.bearings.roller_radial_fixture import (
                make_cylindrical_roller_bearing,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "make_cylindrical_roller_bearing": make_cylindrical_roller_bearing,
            }

        if capability == "bearing_families.cylindrical_roller":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.core.machine_elements.bearings.families.roller_bearing.radial.subtypes.cylindrical_roller import (
                CylindricalRollerFamily,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "CylindricalRollerFamily": CylindricalRollerFamily,
            }

        if capability == "bearings.thrust_cylindrical_roller":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.fixtures.construction.bearings.roller_thrust_fixture import (
                make_thrust_cylindrical_roller_bearing,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "make_thrust_cylindrical_roller_bearing": make_thrust_cylindrical_roller_bearing,
            }

        if capability == "bearing_families.thrust_cylindrical_roller":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.core.machine_elements.bearings.families.roller_bearing.thrust.subtypes.cylindrical_single import (
                ThrustCylindricalRollerFamily,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "ThrustCylindricalRollerFamily": ThrustCylindricalRollerFamily,
            }

        if capability == "bearings.thrust_cylindrical_roller_multirow":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.fixtures.construction.bearings.roller_thrust_fixture import (
                make_thrust_cylindrical_roller_multirow_bearing,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "make_thrust_cylindrical_roller_multirow_bearing": make_thrust_cylindrical_roller_multirow_bearing,
            }

        if capability == "bearing_families.thrust_cylindrical_roller_multirow":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            # Straight from the leaf module -- NOT via subtypes/__init__.py's
            # lazy re-export, whose _LAZY dict has a copy/paste bug for this
            # exact name (maps to .cylindrical_single instead of
            # .cylindrical_multirow). See roller_thrust_fixture.py's own
            # module docstring for the full note. Not touched -- core is
            # out of scope for Construction fixtures.
            from axisforge.core.machine_elements.bearings.families.roller_bearing.thrust.subtypes.cylindrical_multirow import (
                MultiRowThrustCylindricalRollerFamily,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "MultiRowThrustCylindricalRollerFamily": MultiRowThrustCylindricalRollerFamily,
            }

        if capability == "bearings.thrust_needle_roller":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.fixtures.construction.bearings.roller_thrust_fixture import (
                make_thrust_needle_roller_bearing,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "make_thrust_needle_roller_bearing": make_thrust_needle_roller_bearing,
            }

        if capability == "bearing_families.thrust_needle_roller":
            from axisforge.core.machine_elements.bearings.bearing import Bearing
            from axisforge.core.machine_elements.bearings.catalog import BearingCatalog
            from axisforge.core.machine_elements.bearings.families.roller_bearing.thrust.subtypes.needle_single import (
                ThrustNeedleRollerFamily,
            )
            return {
                "Bearing": Bearing,
                "BearingCatalog": BearingCatalog,
                "ThrustNeedleRollerFamily": ThrustNeedleRollerFamily,
            }

        # ---- gears ----------------------------------------------------
        # NOTE: gears uses TYPE-SPECIFIC keys (make_spur_gear/make_helical_
        # gear/make_internal_gear), not the generic "factory" key shafts
        # uses. Unlike shaft=(...), where requesting shafts.generic AND
        # shafts.stepped together in the same call is rare, requesting
        # more than one gear capability together (e.g. spur pinions +
        # internal ring gear for the same design) is the NORMAL case here
        # -- a generic "factory" key would collide across them (two
        # DIFFERENT functions resolving to the same name), tripping
        # resolve()'s own collision guard. Discovered by testing
        # gears=("gears.spur","gears.helical","gears.internal") together
        # before delivering this.
        # PATH UPDATE: external_gear_fixture.py / internal_gear_fixture.py
        # moved from fixtures/construction/gears/ to
        # fixtures/construction/gears/parallel_axis/fixed/ (confirmed via
        # device_list_dir against the real repo) -- the three branches
        # below import from the new nested path. The old flat path no
        # longer exists in the real repo.
        if capability == "gears.spur":
            from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import (
                SpurHelicalGear,
            )
            from axisforge.fixtures.construction.gears.parallel_axis.fixed.external_gear_fixture import (
                make_spur_gear,
            )
            return {
                "SpurHelicalGear": SpurHelicalGear,
                "make_spur_gear": make_spur_gear,
            }

        if capability == "gears.helical":
            from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import (
                SpurHelicalGear,
            )
            from axisforge.fixtures.construction.gears.parallel_axis.fixed.external_gear_fixture import (
                make_helical_gear,
            )
            return {
                "SpurHelicalGear": SpurHelicalGear,
                "make_helical_gear": make_helical_gear,
            }

        if capability == "gears.internal":
            from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.internal_gear import (
                InternalGear,
            )
            from axisforge.fixtures.construction.gears.parallel_axis.fixed.internal_gear_fixture import (
                make_internal_gear,
            )
            return {
                "InternalGear": InternalGear,
                "make_internal_gear": make_internal_gear,
            }

        # ---- gears: meshing (fixed-axis pairs) ------------------------
        # Same domain prefix as gear generation above ("gears.*") -- your
        # call, to avoid a 6th ConstructionCapabilities field for a
        # distinction that only matters in the capability STRING, not the
        # schema. Building a SpurHelicalGearMeshing/InternalGearMeshing is
        # still Construction (working geometry, contact ratios,
        # validate()) -- calling .forces() with a real torque is
        # MeshLoads, not exposed by either factory below.
        if capability == "gears.spur_helical_meshing":
            from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import (
                SpurHelicalGearMeshing,
            )
            from axisforge.fixtures.construction.gears.parallel_axis.fixed.spur_helical_meshing_fixture import (
                make_spur_helical_meshing,
            )
            return {
                "SpurHelicalGearMeshing": SpurHelicalGearMeshing,
                "make_spur_helical_meshing": make_spur_helical_meshing,
            }

        if capability == "gears.internal_meshing":
            from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.internal_meshing import (
                InternalGearMeshing,
            )
            from axisforge.fixtures.construction.gears.parallel_axis.fixed.internal_meshing_fixture import (
                make_internal_meshing,
            )
            return {
                "InternalGearMeshing": InternalGearMeshing,
                "make_internal_meshing": make_internal_meshing,
            }

        # ---- systems ----------------------------------------------------
        # First cut: linear chains only (no fan-out, no convergent merge --
        # see linear_chain_fixture.py's own module docstring). Composes
        # already-built Shaft/Bearing/SpurHelicalGear objects via
        # ShaftSpec/StageSpec -- does not construct any of them itself,
        # per the _PREREQUISITES note above. Type-specific keys, same
        # discipline as gears/bearings above (a generic "factory" key
        # would collide against shafts.* if a script ever requested both
        # in one ConstructionCapabilities call).
        #
        # UNLIKE every other branch in this method, build_linear_system()
        # (exported below) also RESOLVES the system before returning it
        # -- P [W] is a required kwarg on that function, rpm comes from
        # shaft_specs[0].speed_rpm, and the SpurHelicalGearSystem handed
        # back already carries its gear-mesh loads and shaft positions.
        # Deliberate exception to this module's own Construction-only
        # scope, on explicit user decision -- see this module's own
        # docstring (top of file) and linear_chain_fixture.py's module
        # docstring for the full reasoning. Nothing to change HERE for
        # that (this branch only re-exports names, it doesn't call
        # anything), but a reader stopping at this comment should not
        # assume "Construction only" holds for this one capability.
        if capability == "systems.parallel_axis_linear":
            from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
                GearElement, ShaftSystem,
            )
            from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
                SpurHelicalMeshLink, SpurHelicalGearSystem,
            )
            from axisforge.fixtures.construction.systems.parallel_axis.spur_helical.linear_chain_fixture import (
                ShaftSpec, StageSpec, build_linear_system,
            )
            return {
                "GearElement": GearElement,
                "ShaftSystem": ShaftSystem,
                "SpurHelicalMeshLink": SpurHelicalMeshLink,
                "SpurHelicalGearSystem": SpurHelicalGearSystem,
                "ShaftSpec": ShaftSpec,
                "StageSpec": StageSpec,
                "build_linear_system": build_linear_system,
            }

        raise CapabilityError(f"unknown capability: {capability!r}")

    def resolve(self) -> dict[str, object]:
        """
        Validate the whole request, then `_require()` every capability
        across all 5 fields and merge the results.

        Two capabilities resolving the same name to the SAME object
        (e.g. both "shafts.generic" and "shafts.stepped" return the
        core Shaft class under "Shaft") is not a collision -- only two
        DIFFERENT objects under the same name raises.
        """
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
        return f"construction: {body or '(none)'}"


# ===========================================================================
# ResolutionCapabilities
# ===========================================================================

@dataclass(frozen=True)
class ResolutionCapabilities:
    """Stage: Resolution. `shaft_fem` is the tuple of capability
    strings to pull for the shaft-FEM-solve domain -- e.g.
    `ResolutionCapabilities(shaft_fem=("shaft_fem.timoshenko_rigid",))`.
    An empty tuple means the domain is not used.

    No _PREREQUISITES on THIS class (nothing here depends on another
    shaft_fem.* capability being requested in the same call) and no
    stored reference to ConstructionCapabilities -- see this class's
    own top docstring for where the actual cross-stage prerequisite
    check lives instead (solve_system()'s own `construction` parameter,
    checked via ConstructionCapabilities.has_capability() above).

    Exposes the shaft-FEM solve pipeline (SimpleFEMSolver ->
    ShaftResultsReader -> SimpleFEMResultsLibrary) plus
    fixtures/solvers/fem_simple.py's own solve_system() convenience --
    the Resolution-stage counterpart to ConstructionCapabilities above
    (same shared shape: typed fields holding capability strings,
    `_require()` resolving each to concrete classes/functions,
    `resolve()` merging with collision detection).

    solve_system() takes the ConstructionCapabilities that built its
    `system` argument as a second required parameter, specifically to
    check Resolution's own prerequisite: construction must have
    requested "systems.parallel_axis_linear" (the only Construction
    capability that resolves the system -- gear-mesh loads + shaft
    positions -- before returning it), via
    ConstructionCapabilities.has_capability(). That check is
    declarative (reads construction's own requested capability strings)
    rather than inspecting `system`'s runtime state -- no core change
    needed, SpurHelicalGearSystem is untouched. See
    has_capability()'s own docstring above and fem_simple.py's own
    module docstring for the full reasoning.

    Naming note: this class's own resolve() is unrelated to
    SpurHelicalGearSystem.resolve() (propagates torque/positions) and
    to solve_system() (runs the actual FEM) -- same word, three
    different meanings across the codebase, each documented at its own
    definition. resolve() HERE means only "capability strings in,
    concrete classes/functions out" -- exactly what
    ConstructionCapabilities.resolve() means, nothing more.

    Capability string names the exact solver configuration, not a
    generic placeholder: "shaft_fem.timoshenko_rigid" -- SimpleFEMSolver's
    only real configuration today (Timoshenko beam theory; every
    bearing a rigid support). See fixtures/solvers/fem_simple.py's own
    module docstring for why "rigid" is accurate even though
    SimpleFEMSolver's own constraint_bearing parameter isn't actually
    wired to an alternative yet. A future beam theory or
    bearing-constraint model gets its OWN capability string (e.g.
    "shaft_fem.euler_rigid"), the same growth pattern bearings.*
    already uses in ConstructionCapabilities -- never a hidden
    parameter folded inside this one string.

    UNLIKE every Construction capability except systems.parallel_axis_linear,
    this class's own capability ALSO performs real work when its
    exported solve_system() is called -- not at resolve() time
    (resolve() only hands back the function, it never calls it), but
    the function itself runs the FEM solve rather than only assembling
    objects. Deliberate, on the same reasoning already accepted for
    systems.parallel_axis_linear, applied even more directly here:
    "Resolution" IS "solve" -- a Resolution capability that only
    returned inert classes without also offering something that solves
    would not actually resolve anything.

    Adding a configuration
    -----------------------
    1. Add a branch to `_require()` below.
    2. Register the capability string (with a one-line description) in
       catalogue.py, under CAPABILITIES["shaft_fem"].
    """

    shaft_fem: tuple[str, ...] = ()

    # field name -> the capability-string domain prefix it accepts.
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
                        f"resolution.{field_name}: '{capability}' is not "
                        f"a '{prefix}.*' capability"
                    )
        return errors

    def validate_chain(self) -> list[str]:
        # Resolution takes an already-built system directly at call
        # time (see this class's own top docstring) -- nothing
        # upstream to chain to.
        return self.validate()

    def validate_or_raise(self) -> None:
        errors = self.validate_chain()
        if errors:
            raise CapabilityError("; ".join(errors))

    # ------------------------------------------------------------------
    # Resolution -- one capability string in, exact classes/factory out.
    # ------------------------------------------------------------------

    def _require(self, capability: str) -> dict:
        """Resolve one capability string. Does not check any
        prerequisite -- there is none for this stage (see this class's
        own top docstring)."""

        # ---- shaft_fem --------------------------------------------------
        if capability == "shaft_fem.timoshenko_rigid":
            from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import (
                SimpleFEMSolver,
            )
            from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
                ShaftResultsReader,
                SimpleFEMResultsLibrary,
            )
            from axisforge.fixtures.solvers.fem_analysis.fem_simple import solve_system
            return {
                "SimpleFEMSolver": SimpleFEMSolver,
                "ShaftResultsReader": ShaftResultsReader,
                "SimpleFEMResultsLibrary": SimpleFEMResultsLibrary,
                "solve_system": solve_system,
            }

        raise CapabilityError(f"unknown capability: {capability!r}")

    def resolve(self) -> dict[str, object]:
        """
        Validate the whole request, then `_require()` every capability
        across all fields and merge the results. Same collision rule
        as ConstructionCapabilities.resolve() -- two capabilities
        resolving the same name to the SAME object is fine; two
        DIFFERENT objects under the same name raises.
        """
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
        return f"resolution: {body or '(none)'}"