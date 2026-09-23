"""
axisforge/results/bearings/life/basic_life_results.py

Basic REFERENCE rating life, ISO/TS 16281:2008 -- bearing-level container.

The per-row life objects and the Pref object are YOUR classes from
contact_postprocessing.py, stored as-is (no parallel copy here):
  rows : BallBasicReferenceRatingLife / RollerBasicReferenceRatingLife  -- eq.(29) / eq.(65)
  pref : BallDynamicEquivalentReferenceLoad /
         RollerDynamicEquivalentReferenceLoad                           -- eq.(30)-(31) / (66)-(67)
This class only adds what is bearing-level: the combined L10r (row 0, or
Zaretsky eq.(49a) for n >= 2 -- computed by combine_row_L10r()) and a
uniform read access (L10r, Pref_r, Pref_a) for ball and roller alike --
which is all modified life (a_ISO) will need.

Data only: every equation stays in contact_postprocessing.py.

Future sibling in this file: ISO 281 basic rating life L10 (catalogue
method, P = X*Fr + Y*Fa) -- TODO.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Union

import numpy as np

from axisforge.results._base import DC

if TYPE_CHECKING:
    # Type-only: no runtime dependency on solvers/.
    from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_postprocessing import (
        BallBasicReferenceRatingLife, RollerBasicReferenceRatingLife,
        DynamicEquivalentReferenceLoadBase,
    )
    RowLife = Union[BallBasicReferenceRatingLife, RollerBasicReferenceRatingLife]


@dataclass(**DC)
class BasicReferenceRatingLifeResult:
    METHOD: ClassVar[str]   = "ISO/TS 16281:2008"
    _RTOL : ClassVar[float] = 1e-9

    label : str
    rows  : tuple[RowLife, ...]                  # one per bearing row
    L10r  : float                                # bearing [10^6 rev]
    pref  : DynamicEquivalentReferenceLoadBase   # Pref_r / Pref_a from bearing L10r

    def __post_init__(self):
        owner = f"{type(self).__name__}({self.label!r})"
        if not isinstance(self.rows, tuple) or len(self.rows) == 0:
            raise ValueError(f"{owner}: rows must be a non-empty tuple.")
        row_t = type(self.rows[0])
        if any(type(r) is not row_t for r in self.rows):
            raise TypeError(f"{owner}: mixed row types.")
        if not (np.isfinite(self.L10r) and self.L10r > 0.0):
            raise ValueError(f"{owner}: L10r must be finite and > 0; got {self.L10r}.")
        if self.is_single and not np.isclose(self.L10r, self.rows[0].L10r, rtol=self._RTOL):
            raise ValueError(f"{owner}: single-row L10r must equal rows[0].L10r.")
        if not self.is_single and self.L10r > min(r.L10r for r in self.rows) * (1 + self._RTOL):
            raise ValueError(f"{owner}: combined L10r cannot exceed the weakest row (series system).")
        if self.pref.label != self.label:
            raise ValueError(f"{owner}: pref.label {self.pref.label!r} != label.")

    @property
    def Pref_r(self) -> float | None:
        return self.pref.Pref_r

    @property
    def Pref_a(self) -> float | None:
        return self.pref.Pref_a

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def is_single(self) -> bool:
        return len(self.rows) == 1