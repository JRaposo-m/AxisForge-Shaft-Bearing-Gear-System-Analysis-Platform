"""axisforge/fixtures/construction/construction_capabilities.py

Stage 1: Construction. Declares capability strings per domain, resolves
each to concrete classes/factories. catalogue.py holds the full listing,
organised by chapter, with each capability's description AND its
`requires` (if any) on the same leaf -- validate() no longer keeps its
own prerequisite table, it calls catalogue.verify() per requested
capability, passing {"construction": self} as context.

Adding a domain
----------------
1. Add a branch to `_require()`, grouped under its own "# ---- <domain> ----".
2. Register the capability's leaf (description + requires, if any) in
   catalogue.py's CATALOGUE, under ["construction"][<domain>].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from axisforge.fixtures.capabilities import catalogue

__all__ = ["CapabilityError", "ConstructionCapabilities"]


class CapabilityError(Exception):
    """Unknown capability string, missing prerequisite, wrong-domain
    string, or a name collision between two capabilities resolving the
    same name to two different objects."""


@dataclass(frozen=True)
class ConstructionCapabilities:
    """Each field is the tuple of capability strings to pull for that
    domain, e.g. ConstructionCapabilities(shaft=("shafts.stepped_3section",)).
    Empty tuple = domain not used."""

    shaft: tuple[str, ...] = ()
    bearings: tuple[str, ...] = ()
    bearing_families: tuple[str, ...] = ()
    gears: tuple[str, ...] = ()
    system: tuple[str, ...] = ()

    _FIELD_DOMAINS: ClassVar[dict[str, str]] = {
        "shaft": "shafts",
        "bearings": "bearings",
        "bearing_families": "bearing_families",
        "gears": "gears",
        "system": "systems",
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
        Declarative only -- reads this instance's own fields, does not
        check whether resolve() ran. Lets a later stage (StudyCapabilities,
        via catalogue.verify()) confirm a Construction prerequisite
        without holding a runtime reference to what Construction built."""
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
                        f"construction.{field_name}: '{capability}' is not "
                        f"a '{prefix}.*' capability"
                    )

        context = {"construction": self}
        for capability in self._all():
            errors.extend(catalogue.verify(capability, context))

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
        """Resolve one capability string. Does not check requires --
        that's a whole-request concern, handled in validate()."""

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

        # ---- bearings --------------------------------------------------
        # Capability suffix = the family's own `.name` property verbatim.
        # "bearings.<name>"         -> Bearing + BearingCatalog + make_<...>_bearing()
        #                               factory (fully assembled Bearing).
        # "bearing_families.<name>" -> Bearing + BearingCatalog + the bare
        #                               <...>Family class, for a script calling
        #                               Bearing.assemble() itself.
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
            # Straight from the leaf module, NOT via subtypes/__init__.py's
            # lazy re-export -- its _LAZY dict has a copy/paste bug for this
            # name (maps to .cylindrical_single instead of .cylindrical_multirow).
            # Core is out of scope for Construction fixtures, so not touched here.
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

        # ---- gears --------------------------------------------------
        # Type-specific keys (make_spur_gear/make_helical_gear/make_internal_gear),
        # not a generic "factory" key -- requesting more than one gear
        # capability together (spur pinion + internal ring gear) is normal
        # here, and a shared "factory" key would collide across them.
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
        # Same "gears.*" prefix as gear generation above -- the distinction
        # only matters in the capability string, not the schema. Building
        # the meshing object is still Construction; calling .forces() with
        # a real torque is MeshLoads, not exposed by either factory below.
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

        # ---- systems --------------------------------------------------
        # Linear chains only. Composes already-built Shaft/Bearing/
        # SpurHelicalGear objects via ShaftSpec/StageSpec -- does not
        # construct them itself, per catalogue.py's own requires for this
        # capability.
        #
        # Deliberate exception to this class's Construction-only scope:
        # build_linear_system() also RESOLVES the system (torque/positions)
        # before returning it -- P [W] is a required kwarg, rpm comes from
        # shaft_specs[0].speed_rpm. See linear_chain_fixture.py's own
        # module docstring for the full reasoning.
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
        """Validate the whole request, then _require() every capability
        and merge. Two capabilities resolving the same name to the SAME
        object is fine; two DIFFERENT objects under the same name raises."""
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