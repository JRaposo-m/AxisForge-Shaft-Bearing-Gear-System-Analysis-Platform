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

@dataclass
class StressRaiser:
    label: str                          # ex: "shoulder@100.0", "keyway@175.0"
    x: float                            # posição [mm]
    raiser_type: str = "other"          # "shoulder", "keyway", "press_fit", "groove", "other"
    Kf: float = 1.0                     # concentração flexão
    Kfs: float = 1.0                    # concentração torção

@dataclass
class StressResult:
    sigma: list[tuple[float, Elem, float, float, float]]        # (zeta, elem, σ_xz, σ_xy)
    internal_forces: list[tuple[float, Elem, np.ndarray, np.ndarray]]  # (zeta, elem, f_xz[3], f_xy[3])
    sigma_contributions: list[dict] | None = None
    tau: list[dict] | None = None

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