"""Splits the loads of a ``ShaftSystem`` (radial, distributed radial, axial, external moments) into
per-plane (XZ and XY) load-case dictionaries, independent of the beam theory. It only packages the
loads: no element, mesh or quadrature is touched. ``TorqueLoad`` is excluded (it has no DOF in the
beam element and is handled by ``TorsionSolver``). Used by ``RigidSupportFEMSolver.solve``.
"""

from __future__ import annotations

from axisforge.core.loads import LoadPlane
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem


def is_distributed(label: str, source: str, *,
                    distribute_all: bool, distribute_labels: set[str]) -> bool:
    """Decide whether a gear-mesh load is treated as distributed over the face width or as a point load.

    Parameters
    ----------
    label : str
        Load label; the part before the first ``":"`` is taken as the gear label.
    source : str
        Load origin; only ``"gear_mesh"`` loads can be distributed.
    distribute_all : bool
        If ``True``, every gear-mesh load is distributed.
    distribute_labels : set of str
        Gear labels whose mesh loads are distributed.

    Returns
    -------
    bool
        ``True`` if the load is to be treated as distributed. User loads (``source != "gear_mesh"``)
        are always point loads.
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
    """Decompose the loads of the shaft system into one dictionary per load, by bending plane.

    Parameters
    ----------
    shaft_system : ShaftSystem
        Source of ``radial_loads``, ``distributed_radial_loads``, ``axial_loads`` and
        ``external_moments``.
    distribute_all : bool
        Treat every gear-mesh radial load as distributed (see ``is_distributed``).
    distribute_labels : set of str
        Gear labels whose mesh loads are treated as distributed.

    Returns
    -------
    list of dict
        One dictionary per load, with the keys ``"label"`` (str), ``"source"`` (str), ``"radial_xz"``
        and ``"radial_xy"`` (lists of ``(x [mm], F [N])``), ``"axial"`` (list of ``(x [mm], F [N])``),
        ``"moments_xz"`` and ``"moments_xy"`` (lists of ``(x [mm], M [N mm])``), and, for distributed
        loads only, ``"distributed_xz"`` and ``"distributed_xy"`` (lists of dict with ``"x_lo"``,
        ``"x_hi"`` [mm], ``"q"`` callable ``q(x)`` [N/mm] and ``"theta_fn"`` callable [deg] or None).
        Keys that do not apply to the load are empty lists or absent.

    Warnings
    --------
    The plane assigned to ``ExternalMoment.component`` is forwarded unchanged (``core/loads.py``); its
    agreement with the moment-axis convention of that module is an open review item.

    Notes
    -----
    Plane mapping: the XZ plane takes ``component(LoadPlane.XZ)`` (Fz) and the XY plane
    ``component(LoadPlane.XY)`` (Fy), as in the AxisForge README "Conventions" (D-0010). Axial loads go
    to the XZ vector only, when the solver assembles them.

    A gear-mesh ``RadialLoad`` flagged as distributed is skipped here; this function does not create
    its distributed counterpart, it only reads ``shaft_system.distributed_radial_loads``.
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