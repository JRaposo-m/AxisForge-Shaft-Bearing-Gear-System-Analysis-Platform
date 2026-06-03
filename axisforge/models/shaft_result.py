# shaft_results.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from models.elem import Elem
from enum import Enum
# ── 1. Statics ───────────────────────────────────────────────────────────────

def _array_field() -> np.ndarray: ...

@dataclass
class StaticsResult:
    x_nodes: list[float]
    elements: list[Elem]
    d_total_xz: np.ndarray
    d_total_xy: np.ndarray
    f_xz_ext: np.ndarray
    f_xy_ext: np.ndarray
    d_contributions: list[dict]
    # cada dict: {'label': str, 'type': 'gear'|'external', 'd_xz': np.ndarray, 'd_xy': np.ndarray}
# ── 2. Stress (fatigue) ──────────────────────────────────────────────────────

#class StressRaiserType(Enum):
#    SHOULDER, KEYWAY, PRESS_FIT, GROOVE, OTHER

#@dataclass
#class CriticalSection:
#    x, diameter, raiser_type, Kt, Kts, q, qs, Kf, Kfs
#    Ma, Mm, Ta, Tm
#    sigma_a, sigma_m, Se_prime, ka, kb, ke
#    nf_goodman, nf_asme, ny, langer_ok
    # properties: governing_nf, is_safe

@dataclass
class StressResult:
    sigma: list[tuple[float, Elem, float, float, float]]        # (zeta, elem, σ_xz, σ_xy)
    internal_forces: list[tuple[float, Elem, np.ndarray, np.ndarray]]  # (zeta, elem, f_xz[3], f_xy[3])
    sigma_contributions: list[dict] | None = None
    tau: list[tuple[float, float]] | None = None

# ── 3. Static failure ────────────────────────────────────────────────────────

#@dataclass
#class StaticFailureSection:
#    x, diameter, sigma_x, tau_xy, sigma_prime, sigma_eff_mss
#    n_DE, n_MSS, governing_n, governing_theory, yielded, label
    # properties: risk_label

#@dataclass
#class StaticFailureResult:
#    sections, material_id, Sy
    # properties: critical_section, min_governing_n, any_yielded