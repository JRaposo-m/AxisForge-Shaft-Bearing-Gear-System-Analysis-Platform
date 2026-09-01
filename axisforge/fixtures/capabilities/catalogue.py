"""
fixtures/capabilities/catalogue.py

Single registry of every capability string accepted by a
fixtures/capabilities/<domain>.py `require()` function, organised by
domain and then by sub-division. This module is pure metadata -- it
imports nothing from axisforge, so it can be introspected (listing,
printing, building a menu) without pulling in any core/solvers code.

Each <domain>.py owns the *behaviour* for its own capability strings
(what `require()` returns); this module only owns the *listing* --
which strings exist and a one-line description of each. The two are
kept apart on purpose: a fixture author (or this file itself) can ask
"what can I build in bearings?" without importing axisforge.core at
all, and a new capability is registered here independently of writing
its require() branch -- forgetting one half is easy to spot, since
`require()` raises on an unknown string and this file simply won't
list one that was never added.

Capability string shape: "<domain>.<technique>" -- domain matches a
fixtures/capabilities/<domain>.py module name.
"""

from __future__ import annotations

CAPABILITIES: dict[str, dict[str, str]] = {
    "shafts": {
        "shafts.generic": (
            "N-section stepped shaft from an explicit SectionSpec list "
            "(make_shaft) -- full control over each section."
        ),
        "shafts.stepped": (
            "N-section stepped shaft, one call (make_stepped_shaft) -- "
            "sections carry no shoulders; each of the N-1 internal "
            "Shoulder objects is derived from the two adjacent "
            "SectionSpec.diameter values plus one fillet radius, never "
            "re-typed by the caller."
        ),
    },
    "bearings": {
        "bearings.deep_groove_ball": (
            "DGBB (make_deep_groove_ball_bearing) -- point contact, radial "
            "duty. Clearance-specified (s), not contact-angle-specified. "
            "i in {1, 2} rows."
        ),
        "bearings.angular_contact": (
            "ACB (make_angular_contact_bearing) -- point contact, radial "
            "duty. Contact-angle-specified (alpha_0_deg), bounded to "
            "(0, 45] -- above 45deg is thrust duty. i in {1, 2} rows."
        ),
        "bearings.self_aligning": (
            "Self-aligning ball bearing (make_self_aligning_ball_bearing) "
            "-- point contact, radial duty. Contact-angle-specified "
            "(alpha_0_deg), bounded to (0, 45]. Outer raceway radius re is "
            "derived from gamma(alpha_0), not a fixed ratio of Dw like the "
            "other two ball-radial families."
        ),
        "bearings.thrust_ball_single_row": (
            "Single-row thrust ball bearing (make_thrust_ball_bearing) -- "
            "point contact, thrust duty. Contact-angle-specified "
            "(alpha_0_deg), bounded to (45, 90] -- 90 is pure thrust. "
            "Always exactly one row; see bearings.thrust_ball_multirow "
            "for i >= 2."
        ),
        "bearings.thrust_ball_multirow": (
            "Multi-row (i >= 2) thrust ball bearing "
            "(make_thrust_ball_multirow_bearing) -- ONE catalog part / "
            "ONE Bearing built from a row_specs list, each row in "
            "SingleRowThrustBallFamily's own shape. Not i independent "
            "Bearings."
        ),
        "bearings.cylindrical_roller": (
            "NU/N-type cylindrical roller bearing "
            "(make_cylindrical_roller_bearing) -- line contact, radial "
            "duty. arrangement must be 'floating' or 'non-locating' (no "
            "flange to react axial load) -- the factory defaults to "
            "'floating' rather than BearingCatalog's own 'locating' "
            "default."
        ),
        "bearings.thrust_cylindrical_roller": (
            "Single-row cylindrical roller thrust bearing "
            "(make_thrust_cylindrical_roller_bearing) -- line contact, "
            "thrust duty. alpha_0_deg is a required kwarg (no class-level "
            "default)."
        ),
        "bearings.thrust_cylindrical_roller_multirow": (
            "Multi-row (i >= 2) cylindrical roller thrust bearing "
            "(make_thrust_cylindrical_roller_multirow_bearing) -- ONE "
            "catalog part / ONE Bearing built from a row_specs list, same "
            "shape convention as bearings.thrust_ball_multirow."
        ),
        "bearings.thrust_needle_roller": (
            "Needle roller thrust bearing (make_thrust_needle_roller_bearing) "
            "-- line contact, thrust duty. alpha_0 fixed at 90deg inside "
            "the family (flat-race); no alpha_0_deg parameter exists."
        ),
    },
    "bearing_families": {
        "bearing_families.deep_groove_ball": (
            "Bare DeepGrooveBallFamily class, for a script that wants to "
            "call Bearing.assemble() itself instead of going through "
            "bearings.deep_groove_ball's factory."
        ),
        "bearing_families.angular_contact": (
            "Bare AngularContactFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain exists."
        ),
        "bearing_families.self_aligning": (
            "Bare SelfAligningBallFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain exists."
        ),
        "bearing_families.thrust_ball_single_row": (
            "Bare SingleRowThrustBallFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain exists."
        ),
        "bearing_families.thrust_ball_multirow": (
            "Bare MultiRowThrustBallFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain exists."
        ),
        "bearing_families.cylindrical_roller": (
            "Bare CylindricalRollerFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain exists."
        ),
        "bearing_families.thrust_cylindrical_roller": (
            "Bare ThrustCylindricalRollerFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain exists."
        ),
        "bearing_families.thrust_cylindrical_roller_multirow": (
            "Bare MultiRowThrustCylindricalRollerFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain exists."
        ),
        "bearing_families.thrust_needle_roller": (
            "Bare ThrustNeedleRollerFamily class -- see "
            "bearing_families.deep_groove_ball for why this domain exists."
        ),
    },
    "gears": {
        "gears.spur": (
            "External spur gear (make_spur_gear) -- SpurHelicalGear with "
            "beta_n_deg pinned to 0.0, not exposed as a parameter. "
            "Construction only, no meshing."
        ),
        "gears.helical": (
            "External helical gear (make_helical_gear) -- SpurHelicalGear "
            "with beta_n_deg required and validated > 0 (0 is rejected, "
            "use gears.spur instead). Construction only, no meshing."
        ),
        "gears.internal": (
            "Internal (ring) gear (make_internal_gear) -- InternalGear, a "
            "different core class from external spur/helical. z is "
            "required to be NEGATIVE here (e.g. z=-60 for 60 teeth) -- "
            "the core's da/df formulas use z/abs(z) as a sign switch, and "
            "only z<0 makes the addendum genuinely shrink the bore "
            "(da<d), as a real internal gear's tip circle must. "
            "Construction only, no meshing."
        ),
        "gears.spur_helical_meshing": (
            "Fixed-axis pair of two external gears (make_spur_helical_meshing) "
            "-- SpurHelicalGearMeshing: working centre distance, working "
            "pressure angle, contact ratios (epsilon_alpha/beta/gamma), "
            "interference/undercut checks via .validate(). Takes two "
            "already-built SpurHelicalGear objects (gears.spur/gears.helical), "
            "does not build them. Does not call .forces() -- that needs a "
            "real driving torque, a MeshLoads-stage concern."
        ),
        "gears.internal_meshing": (
            "Fixed-axis external-pinion + internal-ring pair "
            "(make_internal_meshing) -- InternalGearMeshing: same working-"
            "geometry/contact-ratio outputs as gears.spur_helical_meshing, "
            "plus interference checks delegated to InternalGear.validate_mesh(). "
            "Takes an already-built SpurHelicalGear (pinion, gears.spur/"
            "helical) and InternalGear (ring, gears.internal; z<0 is "
            "re-checked here as a guard). Does not call .forces()."
        ),
    },
    "systems": {
        "systems.parallel_axis_linear": (
            "Linear N-stage parallel-axis chain, no fan-out/no convergent "
            "merge (build_linear_system) -- N+1 ShaftSpec + N StageSpec "
            "assembled into one SpurHelicalGearSystem (N+1 ShaftSystem, N "
            "SpurHelicalMeshLink). Takes already-built Shaft/Bearing/"
            "SpurHelicalGear/Load objects, does not construct them -- "
            "requires shafts.*, gears.* and bearings.* capabilities in "
            "the same request (see _PREREQUISITES). Spur/helical stages "
            "only (internal-gear stages deferred). UNLIKE every other "
            "capability in this catalogue, this one also RESOLVES the "
            "system: P [W] is a required kwarg, rpm is read from "
            "shaft_specs[0].speed_rpm, and the returned "
            "SpurHelicalGearSystem already carries its gear-mesh loads "
            "and shaft positions -- a deliberate exception, see "
            "linear_chain_fixture.py's own module docstring."
        ),
    },
    "shaft_fem": {
        "shaft_fem.timoshenko_rigid": (
            "1D Timoshenko beam FEM solve for every shaft in a "
            "SpurHelicalGearSystem (SimpleFEMSolver + ShaftResultsReader "
            "via fixtures/solvers/fem_simple.py's solve_system()) -- "
            "every bearing a rigid support (v=0 always; u=0 "
            "additionally for 'locating' bearings). SimpleFEMSolver's "
            "own constraint_bearing parameter is not yet wired to any "
            "alternative, so this name describes the only physics "
            "actually applied today, not a chosen switch -- a future "
            "beam theory or bearing-constraint model gets its own "
            "capability string, never a hidden parameter here. "
            "solve_system() takes the already-built system directly, "
            "plus the ConstructionCapabilities that built it (checked "
            "via has_capability('systems.parallel_axis_linear') -- "
            "Resolution's own prerequisite) -- no _PREREQUISITES on "
            "ResolutionCapabilities itself though, see its own "
            "docstring in fixtures/capabilities/__init__.py. UNLIKE "
            "every capability in shafts/bearings/"
            "gears/systems except systems.parallel_axis_linear, "
            "resolving this one hands back a function that performs "
            "the real FEM solve when called -- Resolution's whole "
            "purpose is to solve."
        ),
    },
}


def list_capabilities(domain: str | None = None) -> dict[str, str]:
    """
    All registered capability strings and their descriptions.

    Pass `domain` (e.g. "shafts") to filter to one domain's entries;
    omit it to get the flattened catalogue across every domain.

    Raises
    ------
    ValueError
        If `domain` is given but not registered.
    """
    if domain is None:
        out: dict[str, str] = {}
        for entries in CAPABILITIES.values():
            out.update(entries)
        return out
    if domain not in CAPABILITIES:
        raise ValueError(
            f"unknown domain: {domain!r}. Registered domains: "
            f"{', '.join(sorted(CAPABILITIES)) or '(none)'}"
        )
    return dict(CAPABILITIES[domain])


def print_catalogue() -> None:
    """ASCII listing of every domain and its capability strings."""
    if not CAPABILITIES:
        print("(no capabilities registered)")
        return
    for domain in sorted(CAPABILITIES):
        print(domain)
        entries = CAPABILITIES[domain]
        width = max(len(cap) for cap in entries) + 2
        for cap in sorted(entries):
            print(f"  {cap:<{width}}{entries[cap]}")