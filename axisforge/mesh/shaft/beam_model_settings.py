"""
axisforge/mesh/shaft/beam_model_settings.py
"""

from __future__ import annotations

from dataclasses import dataclass

VALID_BEAM_THEORIES       = ("euler_bernoulli", "timoshenko")
VALID_SHEAR_THEORIES      = ("cowper", "hutchinson")
VALID_INTEGRATION_METHODS = ("single_point", "exact")


@dataclass(frozen=True)
class BeamModelSettings:
    """
    System/analysis-level beam model choice — decided once in the script,
    not per element. No defaults on purpose: whoever builds this is
    forced to write the choice explicitly.
    """
    beam_theory: str
    shear_theory: str | None
    integration_method: str | None

    def __post_init__(self) -> None:
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