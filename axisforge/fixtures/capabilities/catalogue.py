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

This module is pure metadata -- it imports nothing from axisforge (only
from its own catalogue_data/ package, itself axisforge-free), so it can
be introspected without pulling in any core/solvers code. verify()
below is the one place that WALKS `requires` -- both ConstructionCapabilities
and StudyCapabilities call it from their own validate() instead of each
keeping its own prerequisite loop.

CATALOGUE data split (CHANGED)
-------------------------------
The actual per-domain leaf dicts (SHAFTS, BEARINGS, GEARS, SHAFT_FEM, ...)
no longer live inline in this file -- this file was growing without
bound as more studies get added (gear_iso6336, fatigue, lubrication,
failure_risk, ...). They now live one file per sub-chapter under
catalogue_data/ (catalogue_data/construction/shafts.py, ...), each
exporting one dict constant -- except shaft_fem, split one level
further into catalogue_data/studies/shaft_fem/{timoshenko,
euler_bernoulli,comparison}.py, one file per beam theory plus a
theory-agnostic one, merged back into a single dict below (see that
package's own __init__.py for why). This file imports each of those
and re-assembles CATALOGUE below, exactly as it looked before the
split -- every function past this point (describe, chapter_of,
requirements_of, verify, list_capabilities, print_catalogue) is
unchanged and still the one place other files call into; nothing
outside this module should import from catalogue_data/ directly.

Adding a capability
--------------------
1. Add the leaf under its chapter/sub-chapter/group in the matching
   catalogue_data/ file, with a "description" and (if it needs one) a
   "requires". Adding a whole new sub-chapter/domain (a new study type,
   say) means adding a new file there and importing it below.
2. Add the matching branch to the owning class's `_require()`.
"""

from __future__ import annotations

from axisforge.fixtures.capabilities.catalogue_data.construction import shafts as _construction_shafts
from axisforge.fixtures.capabilities.catalogue_data.construction import bearings as _construction_bearings
from axisforge.fixtures.capabilities.catalogue_data.construction import bearing_families as _construction_bearing_families
from axisforge.fixtures.capabilities.catalogue_data.construction import gears as _construction_gears
from axisforge.fixtures.capabilities.catalogue_data.construction import systems as _construction_systems
from axisforge.fixtures.capabilities.catalogue_data.studies.shaft_fem import timoshenko as _shaft_fem_timoshenko
from axisforge.fixtures.capabilities.catalogue_data.studies.shaft_fem import euler_bernoulli as _shaft_fem_euler_bernoulli
from axisforge.fixtures.capabilities.catalogue_data.studies.shaft_fem import comparison as _shaft_fem_comparison
from axisforge.fixtures.capabilities.catalogue_data.studies import bearing_iso16281 as _studies_bearing_iso16281

CATALOGUE: dict = {
    "construction": {
        "shafts": _construction_shafts.SHAFTS,
        "bearings": _construction_bearings.BEARINGS,
        "bearing_families": _construction_bearing_families.BEARING_FAMILIES,
        "gears": _construction_gears.GEARS,
        "systems": _construction_systems.SYSTEMS,
    },

    # -----------------------------------------------------------------
    # Studies -- every study capability is a leaf, with its own
    # -----------------------------------------------------------------
    "studies": {
        "shafts": {
            # shaft_fem itself is further split by theory (timoshenko.py /
            # euler_bernoulli.py) plus a theory-agnostic comparison.py --
            # see catalogue_data/studies/shaft_fem/__init__.py. Merged
            # back into one dict here so shaft_fem's own leaves (and the
            # "shaft_fem" domain-prefix token other capabilities' `requires`
            # already use, e.g. bearing_iso16281.single_row below) keep
            # working exactly as before the split.
            "shaft_fem": {
                **_shaft_fem_timoshenko.TIMOSHENKO,
                **_shaft_fem_euler_bernoulli.EULER_BERNOULLI,
                **_shaft_fem_comparison.COMPARISON,
            },
        },
        "bearings": {
            "bearing_iso16281": _studies_bearing_iso16281.BEARING_ISO16281,
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


def is_registered(capability: str) -> bool:
    """True if `capability` is an exact, known leaf in CATALOGUE.

    verify() itself does NOT call this -- an unregistered capability
    just has no known `requires`, so verify() silently returns [] for
    it (see verify()'s own docstring: "resolve()'s own _require() is
    the authority on whether the capability string exists at all").
    That leaves a typo'd/nonexistent capability string free to pass
    validate()/validate_or_raise() cleanly as long as it has a
    plausible-looking domain prefix, only to blow up much later inside
    resolve() -> _require()'s fallback `raise CapabilityError(...)`.
    Callers that want that caught at validate() time (StudyCapabilities.
    validate() does) should check every requested capability against
    this function themselves.
    """
    return capability in _BY_CAPABILITY


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
    "gears.internal_meshing" in catalogue_data/construction/gears.py,
    for the worked example).

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