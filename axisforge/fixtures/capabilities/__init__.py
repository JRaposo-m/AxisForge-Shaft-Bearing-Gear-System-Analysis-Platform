"""
fixtures/capabilities/__init__.py

The Construction stage, as one class: `ConstructionCapabilities` owns
both what a design script can ask for (5 typed fields, one per domain,
holding capability strings) and how each of those capability strings
resolves to exact classes/factories (`_require()`, a method, not a
free function elsewhere in the module) -- "quero construir X" and "o
que X significa em imports" live together on purpose, since both are
Construction's own concern and nothing outside Construction needs to
call `_require()` directly.

The full listing of capability strings, with descriptions, lives apart
in catalogue.py -- pure metadata, no axisforge imports, safe to
introspect without pulling in any core/solvers code. That is the only
other file in this package; see catalogue.py's own docstring for why
it stays separate from this one.

This currently covers Construction-domain capabilities only (shafts;
bearings/gears/systems land here the same way, one at a time, as each
is written). Nothing resolved here is solved -- MeshLoads/Resolution/
ElementAnalysis stay out of scope for this file.

Usage
-----
    construction = ConstructionCapabilities(
        shaft=("shafts.stepped_3section",),
    )
    construction.validate_or_raise()
    objs = construction.resolve()
    ShaftFixture          = objs["ShaftFixture"]
    make_stepped_3section = objs["factory"]

Adding a domain
----------------
1. Add a branch to `_require()` below, grouped under that domain's own
   "# ---- <domain> ----" section.
2. Register its capability strings (with one-line descriptions) in
   catalogue.py, under CAPABILITIES["<domain>"].
3. If a capability only makes sense alongside another domain already
   requested -- e.g. a future "systems.parallel_axis_fixed" needs
   already-built shaft/gear/bearing objects to assemble a ShaftSystem
   from -- add an entry to `_PREREQUISITES`. See its own comment below
   for the worked example.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

__all__ = ["CapabilityError", "ConstructionCapabilities"]


class CapabilityError(Exception):
    """An unknown capability string, a request whose prerequisite
    domain is missing from the same request, a capability string
    that doesn't belong to the field it was put in, or a name
    collision between two capabilities resolving the same name to two
    different objects."""


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
        # "systems.parallel_axis_fixed": ("shafts", "gears.parallel_axis", "bearings"),
        #
        # Read as: requesting "systems.parallel_axis_fixed" (in the
        # `system` field) is only valid if `shaft`, `gears` and
        # `bearings` also carry at least one capability starting with
        # "shafts.", "gears.parallel_axis." and "bearings."
        # respectively -- build_systems() receives already-built
        # Shaft/GearPair/Bearing objects, it does not construct them
        # itself.
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
        # not written yet.

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

        # ---- systems ----------------------------------------------------
        # not written yet -- this is the domain that will populate
        # _PREREQUISITES above once it exists.

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