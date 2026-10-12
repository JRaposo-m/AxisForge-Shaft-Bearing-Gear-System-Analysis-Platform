"""
Reads the application factor K_A of Method B from ``KA_methodB.csv``
(ISO_6336-1_2019, 5.3.2, Table 4).
"""

from pathlib import Path
import csv

_CSV_PATH_KA_B = Path(__file__).parent / "KA_methodB.csv"

_VALID_KA_B = {"uniform", "light_shocks", "moderate_shocks", "heavy_shocks"}

def lookup_KA(driving: str, driven: str) -> float:
    """
    Application factor K_A of Method B for a pair of working characteristics [-].

    Looks up the row of the driving machine and the column of the driven
    machine in the 4 x 4 table stored in ``KA_methodB.csv`` (dimensionless).

    Parameters
    ----------
    driving : str
        Working characteristic of the driving machine (row of Table 4): one of
        ``"uniform"``, ``"light_shocks"``, ``"moderate_shocks"``,
        ``"heavy_shocks"``.
    driven : str
        Working characteristic of the driven machine (column of Table 4): same
        four values.

    Returns
    -------
    K_A : float
        Application factor K_A-B [-].

    Raises
    ------
    ValueError
        If ``driving`` or ``driven`` is not one of the four values.
    KeyError
        If the row of ``driving`` is not found in the CSV file.

    Notes
    -----
    K_A-B is applied to the nominal torque of the machine under consideration
    and applies to transmissions operating outside the resonance speed range
    under relatively steady loading (text after Table 4).

    The last cell (heavy shocks, heavy shocks) is printed in the standard as
    ">= 2,25", a lower bound; the file stores and the function returns 2.25.

    References
    ----------
    .. [1] ISO_6336-1_2019, 5.3.2, Table 4.
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