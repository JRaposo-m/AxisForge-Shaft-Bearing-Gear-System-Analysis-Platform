"""
axisforge/solvers/machine_elements/bearings/ISO_16281/validation.py

Pre-condition and sanity checks run on a bearing before / around an
ISO/TS 16281 solve.

These are contact-agnostic on purpose: they check that the type-specific
setup has been performed and that the declared axial arrangement is
consistent with the applied load, without knowing anything about point vs
line contact. The contact physics itself lives in each type's own package
(Ball_Bearing/, Roller_Bearing/).

Import direction
----------------
    Ball_Bearing/**, Roller_Bearing/**  ---->  validation.py

Never the reverse, and never towards library.py: a solver must be able to
run without the results registry existing. `Bearing` is imported under
TYPE_CHECKING only -- real objects arrive as arguments.

Console / warning text is ASCII only (Windows PowerShell, cp1252).
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axisforge.core.machine_elements.bearings.bearing import Bearing


__all__ = ["FA_FLOATING_EPS", "check_bearing_ready", "warn_if_floating_loaded"]


FA_FLOATING_EPS = 1e-6   # [N] Fa below this on a floating bearing is treated as zero


def check_bearing_ready(bearing: "Bearing", label: str,
                        required_attrs: tuple[str, ...]) -> None:
    """
    Raise if the bearing has not had its type-specific setup called yet.

    Each per-type solver declares which attributes its own setup chain is
    expected to have populated (e.g. internal geometry, Hertz contact
    constants) and passes them in as `required_attrs`. An attribute that is
    absent and one that is present but still None are treated the same way:
    both mean the setup step did not run.

    Parameters
    ----------
    bearing : Bearing
        The bearing object about to be solved.
    label : str
        The bearing's label in the shaft system, used in the error message.
    required_attrs : tuple of str
        Attribute names the type-specific setup is expected to have filled.

    Raises
    ------
    RuntimeError
        If any required attribute is missing or still None. The message
        lists every missing attribute at once, so a caller fixes the whole
        setup chain in one pass rather than one attribute per run.
    """
    missing = [a for a in required_attrs if getattr(bearing, a, None) is None]
    if missing:
        raise RuntimeError(
            f"Bearing '{label}': missing {missing} -- call the type-specific "
            f"geometry/contact setup before solving."
        )


def warn_if_floating_loaded(bearing: "Bearing", label: str, Fa: float,
                            eps: float = FA_FLOATING_EPS) -> None:
    """
    Warn if a bearing marked arrangement='floating' carries a non-zero axial load.

    A floating bearing is by definition not part of the shaft's axial load
    path, so Fa != 0 on one points at a modelling error upstream -- usually
    an axial load applied to the wrong support, or an arrangement label that
    was never updated after the support scheme changed.

    This is a warning, not an error: the solve is still well posed and the
    result is still meaningful, so the caller decides whether it matters.
    A bearing with no `arrangement` attribute is left alone.

    Parameters
    ----------
    bearing : Bearing
        The bearing object being solved.
    label : str
        The bearing's label in the shaft system, used in the warning message.
    Fa : float
        Applied axial load [N].
    eps : float, optional
        Magnitude below which Fa is treated as zero. Defaults to
        FA_FLOATING_EPS.
    """
    if getattr(bearing, "arrangement", None) == "floating" and abs(Fa) > eps:
        warnings.warn(
            f"Bearing '{label}' is floating but Fa = {Fa:.3f} N != 0 -- "
            f"check the shaft's axial load path.",
            stacklevel=2,
        )