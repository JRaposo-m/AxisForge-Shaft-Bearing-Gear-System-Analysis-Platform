"""
core/machine_elements/Bearings/family.py

BearingFamily -- contract every family/subtype must implement to be
pluggable into Bearing.assemble(). Passed as a plain instance, no
registry.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from axisforge.core.machine_elements.bearings.catalog import BearingCatalog


class BearingFamily(ABC):
    #: analyses this family can support, e.g. frozenset({"iso16281_point_contact"})
    CAPABILITIES: frozenset[str] = frozenset()

    #: analysis name -> geometry attribute names required after assemble_geometry()
    REQUIRED_FOR: dict[str, frozenset[str]] = {}

    #: mirrored onto Bearing.bearing_type -- kept for solver-side dispatch
    #: tables still keyed by the enum (see bearing_types.py)
    BEARING_TYPE: Any = None

    #: "radial" | "thrust" -- mirrored onto Bearing.duty. Pure label, no
    #: branching logic in core/ reads it; for a downstream ISO 281 solver
    #: (or reporting/GUI) to pick the right equation set.
    DUTY: str | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'deep_groove_ball'."""

    @abstractmethod
    def assemble_geometry(self, catalog: "BearingCatalog", **geometry_kwargs: Any) -> dict[str, Any]:
        """
        Pure function: raw geometry inputs -> flat attribute dict mirrored
        onto the Bearing. No side effects, no cached state on self.

        catalog : read-only, for families that need to cross-check
                  catalogue fields (e.g. reject arrangement="locating")
        Returns : must include every field referenced in this family's
                  own REQUIRED_FOR.
        """