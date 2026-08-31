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
            "required to be > 0 here (the standard convention); the "
            "negative-z escape hatch InternalGear itself documents is not "
            "exposed by this factory. Construction only, no meshing."
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