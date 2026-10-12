"""
Basic REFERENCE rating life, ISO/TS 16281:2008 -- bearing-level container.

The per-row life objects and the reference-load object are the classes of
``contact_postprocessing.py``, stored as they are (no parallel copy here). This module holds
the bearing-level L10r and a uniform read access (L10r, Pref_r, Pref_a) for ball and roller
bearings. Data only: every equation stays in ``contact_postprocessing.py``. The ISO 281
catalogue life and the modified reference rating life are not implemented.
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
    """Basic reference rating life L10r of one bearing and its reference loads.

    Bearing-level container of the life of ISO/TS 16281: the life of each row, the life of the
    bearing and the dynamic equivalent reference load that gives that life with the basic
    dynamic load rating. It only validates; the numbers come from the solver.

    Parameters
    ----------
    label : str
        Bearing label [-].
    rows : tuple
        One life object per row (``BallBasicReferenceRatingLife`` or
        ``RollerBasicReferenceRatingLife``, all of the same type); each has its own ``L10r``
        [10^6 rev].
    L10r : float
        Basic reference rating life of the bearing, in millions of revolutions [10^6 rev].
        Finite and positive.
    pref : DynamicEquivalentReferenceLoadBase
        Dynamic equivalent reference loads Pref_r and Pref_a [N] computed from ``L10r``; its
        ``label`` must equal ``label``.

    Raises
    ------
    ValueError
        If ``rows`` is empty or not a tuple, ``L10r`` is not finite and positive, a single-row
        ``L10r`` differs from ``rows[0].L10r`` (relative tolerance 1e-9), a multi-row ``L10r``
        exceeds ``min(row L10r)`` by more than 1e-9 (relative), or the labels differ.
    TypeError
        If the row objects are of different types.

    Warnings
    --------
    The solvers build only ``rows=(life,)`` today (single row); the multi-row branch of the
    validation is never reached by a solver. The check ``L10r <= min(row L10r)`` is the series
    system statement (the life of the system is shorter than that of its shortest-lived
    component, Zaretsky2016, Eq. (47), (49a)), not an equation of ISO/TS 16281.

    Notes
    -----
    The container does not combine rows. The combination of row lives is done by
    ``combine_row_L10r`` in ``contact_postprocessing.py`` (strict series, Weibull slope common
    to all rows); it is not part of ISO/TS 16281. The row life is Eq. (29) (ball) or Eq. (65)
    (roller); ``Pref = C / L10r^w`` with w = 1/3 (ball) or 3/10 (roller), Eq. (30)-(31) and
    (66)-(67). ISO/TS 16281 prints P_ref,r for radial and P_ref,a for thrust bearings.

    ``L10r`` is a life in revolutions, not in hours: no speed is stored. It is
    the reference life, before the reliability factor a_1, the life modification factor a_ISO
    and the contamination factor e_C. The modified reference rating life L_nmr
    (ISO_TS_16281_2008, 4.3.5, Eq. (32)-(33) and 5.3.7, Eq. (68), (70)) is not implemented; it
    needs a_1, a_ISO and e_C, for which clause 7 refers to ISO 281:2007 (not in the library).
    The catalogue life L10 = (C/P)^p is not implemented either. The form is printed in
    ISO_TR_1281, Part 1, clause 8: Formula (87) (p = 3, point contact) and Formula (89)
    (p = 10/3, roller bearings); Formula (88) is the line-contact exponent 4. The equivalent
    load P = X Fr + Y Fa belongs to ISO 281:2007, not in the library (source to find).

    References
    ----------
    .. [1] ISO_TS_16281_2008, clause 3 (L_10r "in million revolutions"); 4.3.3, Eq. (29);
       4.3.4, Eq. (30)-(31); 5.3.5, Eq. (65); 5.3.6, Eq. (66)-(67).
    .. [2] Zaretsky2016, Eq. (47), (49a), Appendix C, Eq. (C11) (series system).
    """
    METHOD: ClassVar[str]   = "ISO/TS 16281:2008"
    _RTOL : ClassVar[float] = 1e-9

    label : str
    rows  : tuple[RowLife, ...]                  # one per bearing row
    L10r  : float                                # bearing [10^6 rev]
    pref  : DynamicEquivalentReferenceLoadBase   # Pref_r / Pref_a from bearing L10r

    def __post_init__(self):
        """Validate the fields; see the class docstring for what is checked."""
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
        """Dynamic equivalent reference radial load Pref_r [N]; None without a radial rating."""
        return self.pref.Pref_r

    @property
    def Pref_a(self) -> float | None:
        """Dynamic equivalent reference axial load Pref_a [N]; None without an axial rating."""
        return self.pref.Pref_a

    @property
    def n_rows(self) -> int:
        """Number of bearing rows [-]."""
        return len(self.rows)

    @property
    def is_single(self) -> bool:
        """True for a single-row bearing."""
        return len(self.rows) == 1