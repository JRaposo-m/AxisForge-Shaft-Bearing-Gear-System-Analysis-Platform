"""
Container that joins, for one bearing, the ISO/TS 16281 load distribution (always present)
and the basic reference rating life (optional). It is the only module that knows both
``load_distribution/`` and ``life/``; neither of those imports the other. The bearing solvers
return ``dict[label, BearingAnalysisResult]``. Data only: nothing is computed here.
"""
from __future__ import annotations

from dataclasses import dataclass

from axisforge.results._base import DC
from axisforge.results.bearings.load_distribution.load_distribution_results import BearingResult
from axisforge.results.bearings.life.basic_life_results import BasicReferenceRatingLifeResult


@dataclass(**DC)
class BearingAnalysisResult:
    """Result of the ISO/TS 16281 analysis of one bearing.

    Holds the load distribution of the bearing and, when the solver was run with
    post-processing, its basic reference rating life. The class computes nothing; it only
    checks that the two parts belong to the same bearing.

    Parameters
    ----------
    label : str
        Bearing label; must equal the label of both parts [-].
    load_distribution : BearingResult
        Load distribution of the bearing (forces [N], displacements [mm], angles [rad]).
    basic_life : BasicReferenceRatingLifeResult or None, optional
        Basic reference rating life L10r [10^6 rev] and reference loads Pref [N]; None when the
        life was not computed. Its number of rows must equal ``load_distribution.n_rows``.

    Raises
    ------
    ValueError
        If a label differs from ``label`` or the number of rows of the life and of the load
        distribution differ.

    Notes
    -----
    Only the single-row ball and roller solvers fill ``basic_life``; the multi-row solvers do
    not implement the post-processing (contact_solver.py, ``_attach_postprocessing``).

    The modified reference rating life L_nmr (ISO_TS_16281_2008, 4.3.5, Eq. (32)-(33) and
    5.3.7, Eq. (68), (70)) is not implemented; it needs a_1, a_ISO and e_C, for which clause 7
    refers to ISO 281:2007 (not in the library).

    References
    ----------
    .. [1] ISO_TS_16281_2008, 4.3.3, Eq. (29) and 5.3.5, Eq. (65) (basic reference rating life).
    """
    label             : str
    load_distribution : BearingResult
    basic_life        : BasicReferenceRatingLifeResult | None = None
    # modified_life   : ModifiedReferenceRatingLifeResult | None = None   -- next step

    def __post_init__(self):
        """Validate the fields; see the class docstring for what is checked."""
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