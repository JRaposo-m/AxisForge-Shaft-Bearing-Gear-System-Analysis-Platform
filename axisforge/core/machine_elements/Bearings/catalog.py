"""
core/machine_elements/Bearings/catalog.py

BearingCatalog -- generic, family-agnostic catalogue data.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingCatalog:
    """
    d, D, b       : bore / outer diameter / width [mm]
    C, C0         : dynamic (ISO 281) / static (ISO 76) load rating [N]
    designation   : manufacturer designation, e.g. "6208"
    label         : identifier for reporting/traceability
    position      : axial coordinate along the shaft [mm]
    arrangement   : "locating" | "floating" | "non-locating"
    """
    d: float
    D: float
    b: float = 0.0
    C: float = 0.0
    C0: float = 0.0
    designation: str = ""
    label: str = ""
    position: float = 0.0
    arrangement: str = "locating"

    def validate(self) -> list[str]:
        errors: list[str] = []
        tag = self.label or self.designation or "BearingCatalog"

        if self.position < 0:
            errors.append(f"{tag}: position must be >= 0, got {self.position}")
        if self.C < 0:
            errors.append(f"{tag}: C must be >= 0, got {self.C}")
        if self.C0 < 0:
            errors.append(f"{tag}: C0 must be >= 0, got {self.C0}")
        if self.arrangement not in ("locating", "floating", "non-locating"):
            errors.append(
                f"{tag}: arrangement must be 'locating', 'floating', or "
                f"'non-locating', got '{self.arrangement}'"
            )
        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))