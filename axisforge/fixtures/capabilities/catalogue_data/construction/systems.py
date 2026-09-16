"""axisforge/fixtures/capabilities/catalogue_data/construction/systems.py

Leaf dict for CATALOGUE["construction"]["systems"]. See catalogue.py's
own module docstring for the leaf shape and `requires` token rules.
"""

from __future__ import annotations

SYSTEMS: dict = {
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
}