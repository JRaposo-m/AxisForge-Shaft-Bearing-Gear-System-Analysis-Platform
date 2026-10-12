"""Registry and selection of the ISO/TS 16281 contact solver for a bearing.

Solver classes register with a capability ("point_contact" or
"line_contact") through ``register_contact_solver``. ``resolve_solver_cls``
selects the solver class from the contact type of the bearing family, with
no table per bearing type. The capability strings belong to this module;
the families only state whether they are point-contact or line-contact.
"""
from __future__ import annotations

from axisforge.core.machine_elements.bearings.families.family import (
    is_point_contact_family, is_line_contact_family,
)


class SolverDispatchError(RuntimeError):
    """No ISO/TS 16281 contact solver can be selected for a bearing."""

    pass


_REGISTRY: list[type] = []

_CAPABILITY_CHECK = {
    "point_contact": is_point_contact_family,
    "line_contact": is_line_contact_family,
}


def family_matches_capability(family, capability: str) -> bool:
    """Tell whether a bearing family has the contact type of a capability.

    Parameters
    ----------
    family : BearingFamily
        Family of the bearing.
    capability : str
        "point_contact" or "line_contact".

    Returns
    -------
    bool
        True if the family is registered with that contact type. False for an
        unknown capability (no KeyError).
    """
    check = _CAPABILITY_CHECK.get(capability)
    return check is not None and check(family)


def register_contact_solver(*, capability: str):
    """Class decorator factory that registers an ISO/TS 16281 contact solver.

    Parameters
    ----------
    capability : str
        "point_contact" or "line_contact" (keyword only).

    Returns
    -------
    callable
        Decorator that writes ``CAPABILITY`` on the class, appends the class
        to the registry and returns the same class.

    Raises
    ------
    TypeError
        If ``capability`` is empty or unknown.
    """
    if not capability:
        raise TypeError("register_contact_solver: capability must be non-empty")
    if capability not in _CAPABILITY_CHECK:
        raise TypeError(
            f"register_contact_solver: unknown capability {capability!r} -- "
            f"expected one of {sorted(_CAPABILITY_CHECK)}."
        )
    def _decorator(cls: type) -> type:
        """Set ``CAPABILITY`` on ``cls``, add it to the registry and return it."""
        cls.CAPABILITY     = capability
        _REGISTRY.append(cls)
        return cls
    return _decorator


def resolve_solver_cls(bearing, *, label: str = "") -> type:
    """Select the contact solver class for a bearing.

    Three checks, in this order: (1) at least one registered solver has the
    contact type of the bearing family; (2) ``bearing.iso16281_analysis`` is
    true, i.e. the family stored the ISO/TS 16281 stiffness at assembly;
    (3) exactly one of the matching solvers has a ``MULTIROW_SOLVER`` (the
    single-row solver of that contact type).

    Parameters
    ----------
    bearing : Bearing
        Assembled bearing.
    label : str, optional
        Bearing label for the error messages (keyword only).

    Returns
    -------
    type
        The single-row solver class, or its ``MULTIROW_SOLVER`` when the
        bearing has a ``rows`` attribute. A class, not an instance.

    Raises
    ------
    SolverDispatchError
        If any of the three checks fails.

    Notes
    -----
    The multi-row solver classes are in the registry too; check (3) leaves
    them out because their ``MULTIROW_SOLVER`` is None.
    """
    family = bearing.family

    candidates = [c for c in _REGISTRY if family_matches_capability(family, c.CAPABILITY)]
    if not candidates:
        raise SolverDispatchError(
            f"'{label}': nenhuma capability registada corresponde à family "
            f"'{family.name}' ({type(family).__name__})."
        )

    if not getattr(bearing, "iso16281_analysis", False):
        raise SolverDispatchError(
            f"'{label}': iso16281_analysis nao esta True nesta bearing -- "
            f"a family nunca completou a montagem de rigidez ISO/TS 16281."
        )

    single_row = [c for c in candidates if c.MULTIROW_SOLVER is not None]
    if len(single_row) != 1:
        raise SolverDispatchError(
            f"'{label}': esperava exactamente 1 solver single-row registado "
            f"para esta capability, encontrei {[c.__name__ for c in single_row]}."
        )

    return single_row[0].MULTIROW_SOLVER if hasattr(bearing, "rows") else single_row[0]