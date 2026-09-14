"""
axisforge/solvers/machine_elements/shaft/fem_solvers/assembly/load_assembly/vector_external_forces.py

MOCKUP / SKELETON -- extracted from
RigidSupportFEMSolver._build_load_cases() exactly as it was, with no
change in logic. Beam-theory-agnostic: decomposes the ShaftSystem's
loads (RadialLoad, distributed_radial_loads, AxialLoad, ExternalMoment)
into per-plane (XZ/XY) "load case" dictionaries, regardless of whether
the solver is using Timoshenko or Euler-Bernoulli elements -- hence it
lives in its own module rather than in one of the theory-specific
modules.

This module only decomposes and packages loads -- it never touches an
Elem, a mesh, or a quadrature rule. The per-element distributed-load
integration that consumes the "distributed_xz"/"distributed_xy"
entries built here lives in assembly/load_assembly/distributed_loads.py;
point-load entries ("radial_xz", "axial", "moments_xz", etc.) are
injected directly into the global vector by
assembly/load_assembly/point_loads.py, with no per-element step at all.

TorqueLoad is intentionally EXCLUDED here -- it has no DOF in this
element, and is handled separately in torsion.py.

Called by rigid_support.py::RigidSupportFEMSolver.solve() -- see that
module for the orchestrating entry point.
"""

from __future__ import annotations

from axisforge.core.loads import LoadPlane
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem


def is_distributed(label: str, source: str, *,
                    distribute_all: bool, distribute_labels: set[str]) -> bool:
    """
    Extracted from RigidSupportFEMSolver._is_distributed() -- decides
    whether a gear-mesh load (source="gear_mesh") should be treated as
    distributed over the tooth face width or as a point load.

    User loads (source != "gear_mesh") are ALWAYS point loads -- they
    never pass through here as distributed.
    """
    if source != "gear_mesh":
        return False
    if distribute_all:
        return True
    gear_label = label.split(":")[0]
    return gear_label in distribute_labels


def build_load_cases(
    shaft_system: ShaftSystem,
    *,
    distribute_all: bool,
    distribute_labels: set[str],
) -> list[dict]:
    """
    Extracted from RigidSupportFEMSolver._build_load_cases() -- logic
    unchanged, only now takes distribute_all/distribute_labels as
    parameters instead of reading self._distribute_all/self._distribute_labels
    (the solver still owns those two flags, it just calls this free
    function instead of a method of its own).

    TODO (you): decide whether this should become a method of a small
    LoadCaseBuilder class (if it ever needs more internal state) or
    stay a free function -- left as a free function for now since it
    held no state of its own, only read attributes off the solver.
    """
    cases: list[dict] = []

    for ld in shaft_system.radial_loads:
        if is_distributed(ld.label, ld.source,
                           distribute_all=distribute_all,
                           distribute_labels=distribute_labels):
            continue   # gear mesh load -- handled as distributed below
        cases.append({
            "label": ld.label or f"radial@{ld.position:.1f}",
            "source": ld.source,
            "radial_xz": [(ld.position, ld.component(LoadPlane.XZ))],
            "radial_xy": [(ld.position, ld.component(LoadPlane.XY))],
            "axial": [], "moments_xz": [], "moments_xy": [],
        })

    for ld in shaft_system.distributed_radial_loads:
        theta_fn = ld._theta_fn if ld._theta_variable else None
        cases.append({
            "label": ld.label or f"dist@[{ld.x_lo:.1f},{ld.x_hi:.1f}]",
            "source": ld.source,
            "distributed_xz": [{
                "x_lo": ld.x_lo, "x_hi": ld.x_hi,
                "q": lambda x, _ld=ld: _ld.component_intensity(x, LoadPlane.XZ),
                "theta_fn": theta_fn,
            }],
            "distributed_xy": [{
                "x_lo": ld.x_lo, "x_hi": ld.x_hi,
                "q": lambda x, _ld=ld: _ld.component_intensity(x, LoadPlane.XY),
                "theta_fn": theta_fn
            }],
            "radial_xz": [], "radial_xy": [],
            "axial": [], "moments_xz": [], "moments_xy": [],
        })

    for ld in shaft_system.axial_loads:
        cases.append({
            "label": ld.label or f"axial@{ld.position:.1f}",
            "source": ld.source,
            "radial_xz": [], "radial_xy": [],
            "axial": [(ld.position, ld.magnitude)],
            "moments_xz": [], "moments_xy": [],
        })

    for m in shaft_system.external_moments:
        cases.append({
            "label": m.label or f"moment@{m.position:.1f}",
            "source": m.source,
            "radial_xz": [], "radial_xy": [], "axial": [],
            "moments_xz": [(m.position, m.component(LoadPlane.XZ))],
            "moments_xy": [(m.position, m.component(LoadPlane.XY))],
        })

    return cases