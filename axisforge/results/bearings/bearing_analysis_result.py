"""
axisforge/results/bearings/bearing_analysis_result.py

Aggregate per bearing: load distribution (always) + life (optional, grows).
The ONLY module that knows both load_distribution/ and life/ -- neither of
those imports the other. solve() returns dict[label, BearingAnalysisResult].
"""
from __future__ import annotations

from dataclasses import dataclass

from axisforge.results._base import DC
from axisforge.results.bearings.load_distribution.load_distribution_results import BearingResult
from axisforge.results.bearings.life.basic_life_results import BasicReferenceRatingLifeResult


@dataclass(**DC)
class BearingAnalysisResult:
    label             : str
    load_distribution : BearingResult
    basic_life        : BasicReferenceRatingLifeResult | None = None
    # modified_life   : ModifiedReferenceRatingLifeResult | None = None   -- next step

    def __post_init__(self):
        owner = f"BearingAnalysisResult({self.label!r})"
        ld = self.load_distribution
        if ld.label != self.label:
            raise ValueError(f"{owner}: load_distribution.label {ld.label!r} != label.")
        if self.basic_life is not None:
            bl = self.basic_life
            if bl.label != self.label:
                raise ValueError(f"{owner}: basic_life.label {bl.label!r} != label.")
            if bl.n_rows != ld.n_rows:
                raise ValueError(f"{owner}: basic_life has {bl.n_rows} rows, "
                                 f"load_distribution has {ld.n_rows}.")