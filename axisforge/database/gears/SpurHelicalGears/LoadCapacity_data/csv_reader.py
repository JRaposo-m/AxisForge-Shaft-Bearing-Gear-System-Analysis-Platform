"""
axisforge/database/gears/SpurHelicalGears/LoadCapacity_data/csv_reader.py
"""

from pathlib import Path
import csv

_CSV_PATH_KA_B = Path(__file__).parent / "KA_methodB.csv"

_VALID_KA_B = {"uniform", "light_shocks", "moderate_shocks", "heavy_shocks"}

def lookup_KA(driving: str, driven: str) -> float:
    """
    Application factor K_A from ISO 6336-1 Table 4 (Method B).

    Parameters
    ----------
    driving : working characteristic of the driving machine
    driven  : working characteristic of the driven machine
    Both    : "uniform" | "light_shocks" | "moderate_shocks" | "heavy_shocks"

    Returns
    -------
    K_A : float  — minimum value for heavy/heavy (2.25); confirm with designer.

    Reference: ISO 6336-1:2019 §5.3.2, Table 4.
    """
    if driving not in _VALID_KA_B:
        raise ValueError(f"driving={driving!r} not in {sorted(_VALID_KA_B)}")
    if driven not in _VALID_KA_B:
        raise ValueError(f"driven={driven!r} not in {sorted(_VALID_KA_B)}")

    with _CSV_PATH_KA_B.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["driving_characteristic"] == driving:
                return float(row[driven])

    raise KeyError(f"driving={driving!r} not found in {_CSV_PATH_KA_B.name}")