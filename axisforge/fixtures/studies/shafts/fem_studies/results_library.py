"""axisforge/fixtures/studies/shaft/results_library.py"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from axisforge.results.fem_results.shaft_results import ShaftResults


@dataclass
class RigidSupportFEMResultsLibrary:
    """Registry of ShaftResults from the RigidSupportFEMSolver pipeline, keyed by shaft name."""

    _store: dict[str, ShaftResults] = field(default_factory=dict, init=False, repr=False)

    def store(self, results: ShaftResults) -> None:
        if not results.name:
            raise ValueError(
                "ShaftResults.name must be non-empty to use as library key. "
                "Ensure ShaftSystem.name is set before solving."
            )
        self._store[results.name] = results

    def remove(self, name: str) -> None:
        self._store.pop(name, None)

    def clear(self) -> None:
        self._store.clear()

    def get(self, name: str) -> ShaftResults:
        try:
            return self._store[name]
        except KeyError:
            raise KeyError(
                f"No RigidBearingFEM results stored for shaft '{name}'. "
                f"Available: {self.names()}"
            ) from None

    def get_or_none(self, name: str) -> ShaftResults | None:
        return self._store.get(name)

    def names(self) -> list[str]:
        return list(self._store)

    def iter(self) -> Iterator[tuple[str, ShaftResults]]:
        return iter(self._store.items())

    def all_results(self) -> list[ShaftResults]:
        return list(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __repr__(self) -> str:
        return f"RigidBearingFEMResultsLibrary(shafts={self.names()})"