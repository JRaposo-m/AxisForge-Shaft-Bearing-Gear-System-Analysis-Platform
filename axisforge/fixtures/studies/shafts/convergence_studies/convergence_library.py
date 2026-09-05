"""
axisforge/fixtures/studies/shafts/convergence_studies/convergence_library.py

Per-shaft registry of MeshRefinementResult -- same role
RigidBearingFEMResultsLibrary plays for ShaftResults over in the sibling
fem_studies/ package: a mesh convergence study runs once per ShaftSystem
(MeshConvergenceStudy.run(shaft_system, intervals) -> one
MeshRefinementResult), and a SpurHelicalGearSystem has several shafts,
so a study run across the whole system needs somewhere to keep "which
shaft got which convergence result" -- this is that somewhere.

Lives in its own convergence_studies/ package, a sibling of fem_studies/
under axisforge/fixtures/studies/shafts/ -- not nested inside
fem_studies/ itself. Mesh convergence is its own study, with its own
fixture-level "does the work" module and its own output/ report module
still to come, the same shape fem_studies/ already has for shaft_fem
resolution and comparison.

Same store()/get_or_none()/names()/__len__/__repr__ shape as
RigidBearingFEMResultsLibrary -- keyed by MeshRefinementResult.shaft_name
(added to that dataclass specifically so this library has something to
key on; see convergence_results.py's own top docstring). store()
overwrites an existing entry for the same shaft name -- "fresh solve
replaces stale", same convention RigidBearingFEMResultsLibrary already
documents for itself.

This module does not run anything -- it only stores and retrieves
MeshRefinementResult objects a caller already produced (e.g. via
MeshConvergenceStudy.run(), one call per shaft in system.shafts). A
future convergence_study.py fixture (mirroring comparison_study.py's
own "does the work" role, but living here in convergence_studies/
rather than in fem_studies/) would be the natural place to loop over a
SpurHelicalGearSystem's shafts, call MeshConvergenceStudy.run() on
each, and store() every result into one of these -- not yet built;
this library only needed to exist first so that future module has
somewhere to put its output.

Dependency (results/ only, read-only access -- no core modification):
  axisforge.results.fem_results.convergence_results
      MeshRefinementResult -- the result SHAPE this library stores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.results.fem_results.convergence_results import MeshRefinementResult

__all__ = ["ConvergenceResultsLibrary"]


class ConvergenceResultsLibrary:
    """Registry of MeshRefinementResult, one per shaft name."""

    def __init__(self) -> None:
        self._by_name: dict[str, "MeshRefinementResult"] = {}

    def store(self, result: "MeshRefinementResult") -> None:
        """
        Store `result` keyed by result.shaft_name. Overwrites any
        existing entry for the same name -- a fresh solve replaces a
        stale one, never merges with it.

        Raises
        ------
        ValueError
            If result.shaft_name is empty -- an unnamed result has no
            key to store under, and silently keying on "" would let a
            second unnamed result silently overwrite the first.
        """
        if not result.shaft_name:
            raise ValueError(
                "ConvergenceResultsLibrary.store(): result.shaft_name is "
                "empty -- set it before storing (MeshConvergenceStudy.run() "
                "sets it from shaft_system.name automatically; a hand-built "
                "MeshRefinementResult must set it explicitly)."
            )
        self._by_name[result.shaft_name] = result

    def get_or_none(self, name: str) -> "MeshRefinementResult | None":
        """The stored result for `name`, or None if nothing was stored
        under that name (not yet run, or run into a different library)."""
        return self._by_name.get(name)

    def names(self) -> list[str]:
        """Shaft names currently stored, in insertion order."""
        return list(self._by_name.keys())

    def __len__(self) -> int:
        return len(self._by_name)

    def __iter__(self) -> Iterator["MeshRefinementResult"]:
        return iter(self._by_name.values())

    def __repr__(self) -> str:
        return f"ConvergenceResultsLibrary({len(self._by_name)} shafts: {self.names()})"