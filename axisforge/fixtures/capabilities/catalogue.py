"""
fixtures/capabilities/catalogue.py

Single registry of every capability string, organised by CHAPTER --
stage (construction / studies), then sub-chapter (mirrors the fixtures/
folder each domain lives under: studies/shafts, studies/bearings, ...),
then domain group, down to each capability leaf:

    {"description": "...", "requires": {stage: (token, ...), ...}}

`requires` lives on the leaf itself, not in a separate table -- read the
description and its prerequisites in the same place. Every `token`
inside one stage's tuple is ANDed against every other token in that
tuple. A token is one of:

  - WITHOUT a dot: a domain prefix, checked against everything
    requested for that stage (loose: "was ANY capability of this
    domain requested").
  - WITH a dot: one exact capability string, checked via
    has_capability() (strict, and always the cascade TERMINAL one --
    e.g. "systems.parallel_axis_linear" already guarantees
    Construction's own shafts/gears/bearings prerequisites were met via
    ITS OWN `requires`, so a study capability never re-lists those
    underneath it).
  - a NESTED tuple/list of tokens: an OR-group -- true if at least ONE
    of the tokens inside it holds, each resolved by the same two rules
    above (recursively). Reach for this when two or more capabilities
    are interchangeable for one prerequisite slot -- e.g.
    "gears.internal_meshing" needs an internal ring gear AND *some*
    external pinion, but the pinion can be EITHER gears.spur OR
    gears.helical, never both:
    `("gears.internal", ("gears.spur", "gears.helical"))`. A bare
    domain-prefix token (no dot, above) is already its own kind of OR
    -- across EVERY capability in that domain -- so reach for a nested
    tuple instead when the OR only covers a SUBSET of one domain.

This module is pure metadata -- it imports nothing from axisforge, so it
can be introspected without pulling in any core/solvers code. verify()
below is the one place that WALKS `requires` -- both ConstructionCapabilities
and StudyCapabilities call it from their own validate() instead of each
keeping its own prerequisite loop.

Adding a capability
--------------------
1. Add the leaf under its chapter/sub-chapter/group in CATALOGUE, with a
   "description" and (if it needs one) a "requires".
2. Add the matching branch to the owning class's `_require()`.
"""

from __future__ import annotations

CATALOGUE: dict = {
    "construction": {
        "shafts": {
            "shafts.generic": {
                "description": (
                    "N-section stepped shaft from an explicit SectionSpec "
                    "list (make_shaft) -- full control over each section."
                ),
            },
            "shafts.stepped": {
                "description": (
                    "N-section stepped shaft, one call (make_stepped_shaft) "
                    "-- sections carry no shoulders; each of the N-1 "
                    "internal Shoulder objects is derived from the two "
                    "adjacent SectionSpec.diameter values plus one fillet "
                    "radius, never re-typed by the caller."
                ),
            },
        },
        "bearings": {
            "bearings.deep_groove_ball": {
                "description": (
                    "DGBB (make_deep_groove_ball_bearing) -- point contact, "
                    "radial duty. Clearance-specified (s), not contact-"
                    "angle-specified. i in {1, 2} rows."
                ),
            },
            "bearings.angular_contact": {
                "description": (
                    "ACB (make_angular_contact_bearing) -- point contact, "
                    "radial duty. Contact-angle-specified (alpha_0_deg), "
                    "bounded to (0, 45] -- above 45deg is thrust duty. "
                    "i in {1, 2} rows."
                ),
            },
            "bearings.self_aligning": {
                "description": (
                    "Self-aligning ball bearing "
                    "(make_self_aligning_ball_bearing) -- point contact, "
                    "radial duty. Contact-angle-specified (alpha_0_deg), "
                    "bounded to (0, 45]. Outer raceway radius re is derived "
                    "from gamma(alpha_0), not a fixed ratio of Dw like the "
                    "other two ball-radial families."
                ),
            },
            "bearings.thrust_ball_single_row": {
                "description": (
                    "Single-row thrust ball bearing "
                    "(make_thrust_ball_bearing) -- point contact, thrust "
                    "duty. Contact-angle-specified (alpha_0_deg), bounded "
                    "to (45, 90] -- 90 is pure thrust. Always exactly one "
                    "row; see bearings.thrust_ball_multirow for i >= 2."
                ),
            },
            "bearings.thrust_ball_multirow": {
                "description": (
                    "Multi-row (i >= 2) thrust ball bearing "
                    "(make_thrust_ball_multirow_bearing) -- ONE catalog "
                    "part / ONE Bearing built from a row_specs list, each "
                    "row in SingleRowThrustBallFamily's own shape. Not i "
                    "independent Bearings."
                ),
            },
            "bearings.cylindrical_roller": {
                "description": (
                    "NU/N-type cylindrical roller bearing "
                    "(make_cylindrical_roller_bearing) -- line contact, "
                    "radial duty. arrangement must be 'floating' or "
                    "'non-locating' (no flange to react axial load) -- the "
                    "factory defaults to 'floating' rather than "
                    "BearingCatalog's own 'locating' default."
                ),
            },
            "bearings.thrust_cylindrical_roller": {
                "description": (
                    "Single-row cylindrical roller thrust bearing "
                    "(make_thrust_cylindrical_roller_bearing) -- line "
                    "contact, thrust duty. alpha_0_deg is a required kwarg "
                    "(no class-level default)."
                ),
            },
            "bearings.thrust_cylindrical_roller_multirow": {
                "description": (
                    "Multi-row (i >= 2) cylindrical roller thrust bearing "
                    "(make_thrust_cylindrical_roller_multirow_bearing) -- "
                    "ONE catalog part / ONE Bearing built from a row_specs "
                    "list, same shape convention as "
                    "bearings.thrust_ball_multirow."
                ),
            },
            "bearings.thrust_needle_roller": {
                "description": (
                    "Needle roller thrust bearing "
                    "(make_thrust_needle_roller_bearing) -- line contact, "
                    "thrust duty. alpha_0 fixed at 90deg inside the family "
                    "(flat-race); no alpha_0_deg parameter exists."
                ),
            },
        },
        "bearing_families": {
            "bearing_families.deep_groove_ball": {
                "description": (
                    "Bare DeepGrooveBallFamily class, for a script that "
                    "wants to call Bearing.assemble() itself instead of "
                    "going through bearings.deep_groove_ball's factory."
                ),
            },
            "bearing_families.angular_contact": {
                "description": (
                    "Bare AngularContactFamily class -- see "
                    "bearing_families.deep_groove_ball for why this domain "
                    "exists."
                ),
            },
            "bearing_families.self_aligning": {
                "description": (
                    "Bare SelfAligningBallFamily class -- see "
                    "bearing_families.deep_groove_ball for why this domain "
                    "exists."
                ),
            },
            "bearing_families.thrust_ball_single_row": {
                "description": (
                    "Bare SingleRowThrustBallFamily class -- see "
                    "bearing_families.deep_groove_ball for why this domain "
                    "exists."
                ),
            },
            "bearing_families.thrust_ball_multirow": {
                "description": (
                    "Bare MultiRowThrustBallFamily class -- see "
                    "bearing_families.deep_groove_ball for why this domain "
                    "exists."
                ),
            },
            "bearing_families.cylindrical_roller": {
                "description": (
                    "Bare CylindricalRollerFamily class -- see "
                    "bearing_families.deep_groove_ball for why this domain "
                    "exists."
                ),
            },
            "bearing_families.thrust_cylindrical_roller": {
                "description": (
                    "Bare ThrustCylindricalRollerFamily class -- see "
                    "bearing_families.deep_groove_ball for why this domain "
                    "exists."
                ),
            },
            "bearing_families.thrust_cylindrical_roller_multirow": {
                "description": (
                    "Bare MultiRowThrustCylindricalRollerFamily class -- "
                    "see bearing_families.deep_groove_ball for why this "
                    "domain exists."
                ),
            },
            "bearing_families.thrust_needle_roller": {
                "description": (
                    "Bare ThrustNeedleRollerFamily class -- see "
                    "bearing_families.deep_groove_ball for why this domain "
                    "exists."
                ),
            },
        },
        "gears": {
            "gears.spur": {
                "description": (
                    "External spur gear (make_spur_gear) -- SpurHelicalGear "
                    "with beta_n_deg pinned to 0.0, not exposed as a "
                    "parameter. Construction only, no meshing."
                ),
            },
            "gears.helical": {
                "description": (
                    "External helical gear (make_helical_gear) -- "
                    "SpurHelicalGear with beta_n_deg required and "
                    "validated > 0 (0 is rejected, use gears.spur instead). "
                    "Construction only, no meshing."
                ),
            },
            "gears.internal": {
                "description": (
                    "Internal (ring) gear (make_internal_gear) -- "
                    "InternalGear, a different core class from external "
                    "spur/helical. z is required to be NEGATIVE here "
                    "(e.g. z=-60 for 60 teeth) -- the core's da/df formulas "
                    "use z/abs(z) as a sign switch, and only z<0 makes the "
                    "addendum genuinely shrink the bore (da<d), as a real "
                    "internal gear's tip circle must. Construction only, "
                    "no meshing."
                ),
            },
            "gears.spur_helical_meshing": {
                "description": (
                    "Fixed-axis pair of two external gears "
                    "(make_spur_helical_meshing) -- SpurHelicalGearMeshing: "
                    "working centre distance, working pressure angle, "
                    "contact ratios (epsilon_alpha/beta/gamma), "
                    "interference/undercut checks via .validate(). Takes "
                    "two already-built SpurHelicalGear objects (gears.spur/"
                    "gears.helical), does not build them. Does not call "
                    ".forces() -- that needs a real driving torque, a "
                    "MeshLoads-stage concern."
                ),
                # Kept AND, exactly as drafted -- but flagging, not
                # silently fixing: the pair can be spur+spur,
                # helical+helical or one of each, so gears.spur alone
                # (or gears.helical alone) already covers "can build an
                # external gear" for BOTH slots. Requiring both tokens
                # forces a script that only ever wants two spur gears to
                # also request gears.helical for nothing. Same shape as
                # the gears.internal_meshing fix below -- say if you
                # want this one changed to
                # {"construction": (("gears.spur", "gears.helical"),)}
                # (an OR-group of one) too.
                "requires": {"construction": ("gears.helical", "gears.spur")},
            },
            "gears.internal_meshing": {
                "description": (
                    "Fixed-axis external-pinion + internal-ring pair "
                    "(make_internal_meshing) -- InternalGearMeshing: same "
                    "working-geometry/contact-ratio outputs as "
                    "gears.spur_helical_meshing, plus interference checks "
                    "delegated to InternalGear.validate_mesh(). Takes an "
                    "already-built SpurHelicalGear (pinion, gears.spur/"
                    "helical) and InternalGear (ring, gears.internal; z<0 "
                    "is re-checked here as a guard). Does not call "
                    ".forces()."
                ),
                # AND(gears.internal, OR(gears.spur, gears.helical)) --
                # the ring is always gears.internal, but the pinion is
                # EITHER an external spur OR helical gear, never both at
                # once, so the pinion slot is a nested OR-group rather
                # than two ANDed exact tokens.
                "requires": {
                    "construction": ("gears.internal", ("gears.spur", "gears.helical")),
                },
            },
        },
        "systems": {
            "systems.parallel_axis_linear": {
                "description": (
                    "Linear N-stage parallel-axis chain, no fan-out/no "
                    "convergent merge (build_linear_system) -- N+1 "
                    "ShaftSpec + N StageSpec assembled into one "
                    "SpurHelicalGearSystem (N+1 ShaftSystem, N "
                    "SpurHelicalMeshLink). Takes already-built Shaft/"
                    "Bearing/SpurHelicalGear/Load objects, does not "
                    "construct them. Spur/helical stages only (internal-"
                    "gear stages deferred). UNLIKE every other capability "
                    "in this catalogue, this one also RESOLVES the system: "
                    "P [W] is a required kwarg, rpm is read from "
                    "shaft_specs[0].speed_rpm, and the returned "
                    "SpurHelicalGearSystem already carries its gear-mesh "
                    "loads and shaft positions -- see linear_chain_fixture.py's "
                    "own module docstring. THE cascade terminal: every "
                    "downstream study capability's construction-side "
                    "prerequisite is just this one string, never the "
                    "shafts/gears/bearings underneath it."
                ),
                "requires": {"construction": ("shafts", "gears", "bearings")},
            },
        },
    },

# ---------------------------------------------------------------------------
# Studies -- every study capability is a leaf, with its own
# ---------------------------------------------------------------------------
    "studies": {
        "shafts": {
            "shaft_fem": {
                "shaft_fem.timoshenko_rigid": {
                    "description": (
                        "1D Timoshenko beam FEM solve for every shaft in a "
                        "SpurHelicalGearSystem (RigidBearingFEMSolver + "
                        "ShaftResultsReader via "
                        "fixtures/studies/shafts/fem_simple.py's "
                        "solve_system()) -- every bearing a rigid support "
                        "(v=0 always; u=0 additionally for 'locating' "
                        "bearings). RigidBearingFEMSolver's own "
                        "constraint_bearing parameter is not yet wired to "
                        "any alternative, so this name describes the only "
                        "physics actually applied today, not a chosen "
                        "switch."
                    ),
                    "requires": {"construction": ("systems.parallel_axis_linear",)},
                },
                "shaft_fem.euler_bernoulli_rigid": {
                    "description": (
                        "1D Euler-Bernoulli beam FEM solve for every shaft in a "
                        "SpurHelicalGearSystem (EulerBernoulliRigidBearingFEMSolver "
                        "+ ShaftResultsReader, same solve_system() entry point and "
                        "same rigid-bearing boundary conditions as "
                        "shaft_fem.timoshenko_rigid -- v=0 always, u=0 additionally "
                        "for 'locating' bearings). Differs only in element physics: "
                        "cubic Hermite shape functions, no independent shear-strain "
                        "field (equivalent to G*As -> infinity, i.e. shear rigidity "
                        "assumed infinite). Valid for slender shafts (L/D large, "
                        "no short stepped sections near supports); for anything "
                        "stubbier, shaft_fem.timoshenko_rigid is the one that does "
                        "not neglect shear flexibility. Mutually informative with "
                        "timoshenko_rigid -- same construction prerequisite, "
                        "different solver class, pick one per study run."
                    ),
                    "requires": {"construction": ("systems.parallel_axis_linear",)},
                },
                
                "shaft_fem.comparison": {
                    "description": (
                        "Two-solve theory comparison for every shaft in a "
                        "SpurHelicalGearSystem (comparison_study.run_comparison() + "
                        "print_comparison()/write_comparison_report(), fixtures/studies/"
                        "shafts/fem_studies/comparison_study.py) -- calls "
                        "fem_simple.solve_system() TWICE on the SAME already-resolved "
                        "system, once per theory (default 'timoshenko' vs 'euler', "
                        "overridable via run_comparison()'s own theory_a/theory_b "
                        "kwargs), then reports sigma_b_max/v_max/bearing-reaction "
                        "delta and delta%% per shaft via comparison_report.py's "
                        "shaft_comparison_block(). Genuinely theory-agnostic -- does "
                        "NOT itself require 'shaft_fem.timoshenko_rigid' or "
                        "'shaft_fem.euler_bernoulli_rigid' to also be requested; it "
                        "reaches RigidBearingFEMSolver's theory dispatch directly "
                        "through fem_simple.solve_system(), the same construction "
                        "prerequisite as those two, nothing more. Comparing two "
                        "ShaftResults that were NOT solved on the same `system` object "
                        "would silently produce a meaningless table -- see "
                        "comparison_report.py's own top docstring; this capability "
                        "does not (and cannot, from here) guard against that, it is "
                        "the caller's responsibility, same as for the other two "
                        "shaft_fem capabilities' `system` parameter."
                    ),
                    "requires": {"construction": ("systems.parallel_axis_linear",)},
                },
                
                "shaft_fem.convergence.gears.timoshenko": {
                    "description": (
                        "Mesh convergence study (Richardson GCI on resultant "
                        "transverse displacement) restricted to gear face-width "
                        "intervals, Timoshenko theory, for every shaft in a "
                        "SpurHelicalGearSystem (convergence_study.run_convergence(), "
                        "fixtures/studies/shafts/convergence_studies/convergence_study.py) "
                        "-- one fresh RigidBearingFEMSolver(theory='timoshenko') baseline "
                        "solve per shaft, refined via SubmodelSolver/RichardsonGCI "
                        "(convergence_solver.py) at each gear's face-width span, "
                        "published into a ConvergenceResultsLibrary keyed by shaft name. "
                        "Gears with no face width defined (b < "
                        "MIN_FACE_WIDTH_FOR_CONVERGENCE_MM) are skipped and reported "
                        "via warnings.warn(), never silently dropped."
                    ),
                    "requires": {"construction": ("systems.parallel_axis_linear",)},
                },
                "shaft_fem.convergence.external_distributed.timoshenko": {
                    "description": (
                        "Mesh convergence study restricted to DistributedRadialLoad "
                        "spans (any load registered on shaft_system."
                        "distributed_radial_loads, regardless of source= -- 'user' or "
                        "'gear_mesh'), Timoshenko theory. Same "
                        "convergence_study.run_convergence() entry point as "
                        "'shaft_fem.convergence.gears.timoshenko', just a different "
                        "`regions` set pinned underneath -- see that capability's own "
                        "description for the shared mechanics."
                    ),
                    "requires": {"construction": ("systems.parallel_axis_linear",)},
                },
                "shaft_fem.convergence.total.timoshenko": {
                    "description": (
                        "Mesh convergence study over BOTH gear face-width intervals "
                        "AND DistributedRadialLoad spans together (regions="
                        "{'gears', 'external_distributed'}), Timoshenko theory. Does "
                        "NOT include bearing intervals -- see this catalogue entry "
                        "group's own top-of-file note for why 'total' still excludes "
                        "them today."
                    ),
                    "requires": {"construction": ("systems.parallel_axis_linear",)},
                },
            },
        },
        "bearings": {
            "bearing_iso16281": {
                "bearing_iso16281.single_row": {
                    "description": (
                        "ISO/TS 16281 single-row internal load "
                        "distribution for point- and line-contact bearings "
                        "(RollingBearingSolver, dispatching per bearing to "
                        "ISO16281BallSolver/ISO16281RollerSolver) into a "
                        "BearingResultsLibrary. Multi-row (THRUST) bearings "
                        "are out of scope for this capability."
                    ),
                    "requires": {
                        "construction": ("systems.parallel_axis_linear",),
                        "studies": ("shaft_fem",),
                    },
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Lookup -- flatten CATALOGUE once, by capability string
# ---------------------------------------------------------------------------

def _is_leaf(node: object) -> bool:
    return isinstance(node, dict) and "description" in node


def _iter_leaves(node: dict, path: tuple[str, ...] = ()):
    for key, value in node.items():
        if _is_leaf(value):
            yield key, path, value
        else:
            yield from _iter_leaves(value, path + (key,))


_BY_CAPABILITY: dict[str, dict] = {cap: entry for cap, _path, entry in _iter_leaves(CATALOGUE)}
_PATH_BY_CAPABILITY: dict[str, tuple[str, ...]] = {cap: path for cap, path, _entry in _iter_leaves(CATALOGUE)}


def describe(capability: str) -> str:
    """One-line description of `capability`. Raises KeyError if unregistered."""
    return _BY_CAPABILITY[capability]["description"]


def chapter_of(capability: str) -> tuple[str, ...]:
    """Chapter path, e.g. ("studies", "bearings", "bearing_iso16281")."""
    return _PATH_BY_CAPABILITY[capability]


def requirements_of(capability: str) -> dict[str, tuple[str, ...]]:
    """This capability's own `requires` dict, or {} if it has none."""
    return _BY_CAPABILITY.get(capability, {}).get("requires", {})


# ---------------------------------------------------------------------------
# Verification -- the one place `requires` is walked
# ---------------------------------------------------------------------------

def _token_ok(token: object, caps_obj) -> bool:
    """Resolve ONE requirement token against `caps_obj`. A tuple/list is
    an OR-group -- true if ANY of its own tokens resolves true, each
    checked by this same function (so an OR-group can itself contain a
    domain-prefix or exact-capability token, recursively)."""
    if isinstance(token, (tuple, list)):
        return any(_token_ok(t, caps_obj) for t in token)
    return (caps_obj.has_capability(token) if "." in token
            else any(c.startswith(token + ".") for c in caps_obj._all()))


def verify(capability: str, context: dict[str, object]) -> list[str]:
    """
    Check `capability`'s `requires` against `context` -- {stage_name:
    capabilities_object}, e.g. {"construction": construction_caps,
    "studies": study_caps}. Each object must expose has_capability(str)
    -> bool and _all() -> set[str] (ConstructionCapabilities and
    StudyCapabilities both do).

    Every token in a stage's tuple is ANDed. A token WITH a dot is one
    exact capability string, checked via has_capability() (strict,
    cascade-terminal). A token WITHOUT a dot is a domain prefix, checked
    against that stage's own _all() (loose -- "was any capability of
    this domain requested"). A token that is itself a tuple/list is an
    OR-group -- true if at least one of ITS OWN tokens resolves true --
    for a prerequisite slot where several capabilities are
    interchangeable (see this module's own docstring, and
    "gears.internal_meshing" in CATALOGUE, for the worked example).

    `capability` unregistered in CATALOGUE -> no known requirements,
    returns [] (resolve()'s own _require() is the authority on whether
    the capability string exists at all).
    """
    errors: list[str] = []
    for stage, tokens in requirements_of(capability).items():
        caps_obj = context.get(stage)
        if caps_obj is None:
            errors.append(
                f"{capability}: needs '{stage}' capabilities passed in "
                f"context to verify against, none given"
            )
            continue
        for token in tokens:
            if not _token_ok(token, caps_obj):
                label = f"one of {token!r}" if isinstance(token, (tuple, list)) else f"'{token}'"
                errors.append(
                    f"{capability}: requires {label} in '{stage}' "
                    f"(none requested)"
                )
    return errors


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def list_capabilities(*chapter_path: str) -> dict[str, str]:
    """All capability descriptions under a chapter path, e.g.
    list_capabilities("studies", "bearings") -- or every capability if
    no path is given."""
    node = CATALOGUE
    for key in chapter_path:
        node = node[key]
    return {cap: entry["description"] for cap, _path, entry in _iter_leaves(node)}


def print_catalogue() -> None:
    """Indented listing of every chapter and its capability strings."""
    _print_node(CATALOGUE, indent=0)


def _print_node(node: dict, indent: int) -> None:
    for key, value in sorted(node.items()):
        if _is_leaf(value):
            print(f"{'  ' * indent}{key}")
        else:
            print(f"{'  ' * indent}{key}/")
            _print_node(value, indent + 1)