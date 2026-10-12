"""
Beam model choice for a shaft analysis: beam theory, shear correction theory and shear integration method, with the allowed values.
"""

from __future__ import annotations

from dataclasses import dataclass

# Allowed values. Shear theory and integration method apply to "timoshenko" only.
# "single_point": one-point (reduced) integration of the shear term; "exact": full integration.
VALID_BEAM_THEORIES       = ("euler_bernoulli", "timoshenko")
VALID_SHEAR_THEORIES      = ("cowper", "hutchinson")
VALID_INTEGRATION_METHODS = ("single_point", "exact")


@dataclass(frozen=True)
class BeamModelSettings:
    """Beam model choice for the whole shaft, made once per analysis and not per element.

    There are no defaults on purpose: the caller must state the choice.
    Elem copies these three fields into every element it builds.

    Parameters
    ----------
    beam_theory : str
        ``"euler_bernoulli"`` or ``"timoshenko"`` [-].
    shear_theory : str or None
        Shear correction factor expression, ``"cowper"`` or ``"hutchinson"``
        (see ShearFactor). Must be None for ``"euler_bernoulli"`` and one of
        the two names for ``"timoshenko"``.
    integration_method : str or None
        Integration of the shear stiffness: ``"single_point"`` (reduced, one
        Gauss point; no shear locking, converges with mesh refinement) or
        ``"exact"`` (exact integration; too stiff unless the elements are much
        shorter than the shaft diameter).
        See TimoshenkoBeam. Must be None for
        ``"euler_bernoulli"`` (closed-form matrix) and one of the two names
        for ``"timoshenko"``.

    Raises
    ------
    ValueError
        At construction, if a value is not allowed or the combination is
        inconsistent with ``beam_theory``.

    Notes
    -----
    There is no ``element_order`` field; Elem reads it with
    ``getattr(settings, "element_order", "linear")``, so only the linear
    2-node element is reachable.

    References
    ----------
    .. [1] Onate2013_vol2, 2.4, shear locking and reduced integration of the
       shear term (reason for ``"single_point"``).
    """
    beam_theory: str
    shear_theory: str | None
    integration_method: str | None

    def __post_init__(self) -> None:
        """Validate the fields; see the class docstring for what is checked."""
        if self.beam_theory not in VALID_BEAM_THEORIES:
            raise ValueError(
                f"BeamModelSettings: beam_theory must be one of "
                f"{VALID_BEAM_THEORIES}, got '{self.beam_theory}'"
            )

        if self.beam_theory == "euler_bernoulli":
            if self.shear_theory is not None:
                raise ValueError(
                    "BeamModelSettings: beam_theory='euler_bernoulli' does not "
                    f"use shear_theory (got '{self.shear_theory}'); pass None."
                )
            if self.integration_method is not None:
                raise ValueError(
                    "BeamModelSettings: beam_theory='euler_bernoulli' uses a "
                    "closed-form stiffness matrix and does not use "
                    f"integration_method (got '{self.integration_method}'); "
                    "pass None."
                )

        elif self.beam_theory == "timoshenko":
            if self.shear_theory not in VALID_SHEAR_THEORIES:
                raise ValueError(
                    f"BeamModelSettings: shear_theory must be one of "
                    f"{VALID_SHEAR_THEORIES}, got '{self.shear_theory}'"
                )
            if self.integration_method not in VALID_INTEGRATION_METHODS:
                raise ValueError(
                    f"BeamModelSettings: integration_method must be one of "
                    f"{VALID_INTEGRATION_METHODS}, got "
                    f"'{self.integration_method}'"
                )