"""
core/machine_elements/Bearings/families/roller/radial/functions/contact_stiffness.py

Line-contact ("roller") math -- ISO/TS 16281 Sec 5.2 eq.(34)-(37),
lamina positions Sec 5.2.2, gamma Sec 5.3.1.2. Pure functions, no state.
P_xk (reference roller profile, Sec 6.2) is NOT here -- subtype-specific,
lives in families/roller/radial/subtypes/.
"""
from __future__ import annotations
import numpy as np

LOAD_DEFLECTION_EXPONENT = 10.0 / 9.0


def lamina_positions(Lwe: float, n_s: int) -> np.ndarray:
    """x_k -- lamina midpoints, strictly inside (-Lwe/2, Lwe/2). Sec 5.2.2."""
    lamina_length = Lwe / n_s
    return lamina_length * (np.arange(n_s) + 0.5) - Lwe / 2.0


def gamma(Dwe: float, Dpw: float, alpha_0: float) -> float:
    """
    Dwe*cos(alpha_0)/Dpw -- Sec 5.3.1.2, eq.(47)-(52). cos term dropped at
    alpha_0=90deg (thrust) -- cos(pi/2) doesn't round to a clean 0.0 in
    floating point.

    Raises
    ------
    ValueError : if gamma is outside (0, 1) -- checked once here instead
                 of at every capacity call site.
    """
    if np.isclose(alpha_0, np.pi / 2, atol=1e-9):
        g = Dwe / Dpw
    else:
        g = Dwe * np.cos(alpha_0) / Dpw
    if not (0.0 < g < 1.0):
        raise ValueError(
            f"gamma = Dwe*cos(alpha_0)/Dpw = {g:.6f} is outside (0, 1) -- "
            f"check Dwe/Dpw/alpha_0."
        )
    return g


def line_contact_spring_constant(Lwe: float, n_s: int) -> tuple[float, float]:
    """(cL, cs) [N/mm^(10/9)] -- eq.(35), (37)."""
    cL = 35948.0 * Lwe ** (8.0 / 9.0)
    return cL, cL / n_s