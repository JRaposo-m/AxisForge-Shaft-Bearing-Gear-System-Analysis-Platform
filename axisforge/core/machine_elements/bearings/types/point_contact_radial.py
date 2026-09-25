# axisforge/core/machine_elements/bearings/types/point_contact_radial.py
"""
core/machine_elements/bearings/types/point_contact_radial.py

Rolamentos de esferas radiais, contacto pontual: DeepGrooveBallFamily,
AngularContactFamily, SelfAligningBallFamily. Portado de
families/family.py (antigo) -- a fisica (formulas de ri/re, ISO 281 via
capacity.py) NAO mudou. O que mudou:

  - Ja nao chama PointContactStiffness/SelfAligningPointContactStiffness
    (ISO/TS 16281 -- isso moveu para
    solvers/machine_elements/bearings/load_distribution/iso_16281/).
    gamma e a curvatura das vias vem de bearing_properties.py (pura
    geometria, sem material).
  - Ja nao ha assemble_geometry()/.assemble() -- tudo e construido e
    validado em __init__.
  - E/nu deixam de ser kwargs escalares -- vem de `surfaces: SurfacePair`
    (BearingFamily.__init__); core nao os usa aqui, ficam disponiveis
    para o solver que precisar deles.
"""
from __future__ import annotations

import numpy as np

from axisforge.core.machine_elements.bearings.base import BearingType
from axisforge.core.machine_elements.bearings.family import BearingFamily, register_family
from axisforge.core.machine_elements.bearings.bearing_properties import (
    SurfacePair, gamma_point_contact, point_contact_radii,
)
import axisforge.core.machine_elements.bearings.capacity as bcap


# =====================================================================
# ---- deep groove ball -------------------------------------------------
# =====================================================================

@register_family
class DeepGrooveBallFamily(BearingFamily):
    BEARING_TYPE = BearingType.DEEP_GROOVE_BALL
    DUTY = "radial"
    RI_OVER_DW = 0.52
    RE_OVER_DW = 0.52
    REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.90}

    def __init__(self, *, position: float, arrangement, d: float, D: float, b: float,
                 surfaces: SurfacePair, Dw: float, Dpw: float, Z: int, s: float,
                 i: int = 1, C: float | None = None, C0: float | None = None,
                 label: str = "", designation: str = ""):
        super().__init__(position=position, arrangement=arrangement, d=d, D=D, b=b,
                          surfaces=surfaces, C=C, C0=C0, label=label, designation=designation)

        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(f"DeepGrooveBallFamily: i must be one of "
                              f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}")
        ri, re = self.RI_OVER_DW * Dw, self.RE_OVER_DW * Dw
        if ri <= Dw / 2.0 or re <= Dw / 2.0:
            raise ValueError(f"DeepGrooveBallFamily: invalid groove geometry for Dw={Dw}")
        if s < 0:
            raise ValueError(f"DeepGrooveBallFamily: s must be >= 0, got {s}")

        A = ri + re - Dw
        alpha_0 = np.arccos(1.0 - s / (2.0 * A))

        self.Dw, self.Dpw, self.Z, self.i = Dw, Dpw, Z, i
        self.ri, self.re, self.s, self.A, self.alpha_0 = ri, re, s, A, alpha_0
        self.gamma = gamma_point_contact(Dw, alpha_0, Dpw)
        self.reduction_factor = self.REDUCTION_FACTOR_BY_ROWS[i]
        self.phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

    @property
    def name(self) -> str:
        return "deep_groove_ball"

    def curvature_rolling_element(self) -> tuple[float, float]:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r1"]

    def curvature_inner_raceway(self) -> tuple[float, float]:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r2_inner"]

    def curvature_outer_raceway(self) -> tuple[float, float] | None:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r2_outer"]

    def _capacity_calculator(self) -> bcap.PointContactCapacityRadial:
        return bcap.PointContactCapacityRadial(
            Z=self.Z, Dw=self.Dw, alpha_0=self.alpha_0, ri=self.ri, re=self.re,
            gamma=self.gamma, reduction_factor=self.reduction_factor, i=self.i)

    def dynamic_capacity(self) -> float:
        return self._capacity_calculator().Cr

    def static_capacity(self) -> float:
        return self._capacity_calculator().C0

    def per_element_dynamic_capacity(self) -> tuple[float, float]:
        return self._capacity_calculator().Q_elements


# =====================================================================
# ---- angular contact ---------------------------------------------------
# =====================================================================

@register_family
class AngularContactFamily(BearingFamily):
    BEARING_TYPE = BearingType.ANGULAR_CONTACT
    DUTY = "radial"
    RI_OVER_DW = 0.52
    RE_OVER_DW = 0.52
    REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.95}

    def __init__(self, *, position: float, arrangement, d: float, D: float, b: float,
                 surfaces: SurfacePair, Dw: float, Dpw: float, Z: int, alpha_0_deg: float,
                 i: int = 1, C: float | None = None, C0: float | None = None,
                 label: str = "", designation: str = ""):
        super().__init__(position=position, arrangement=arrangement, d=d, D=D, b=b,
                          surfaces=surfaces, C=C, C0=C0, label=label, designation=designation)

        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(f"AngularContactFamily: i must be one of "
                              f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}")
        ri, re = self.RI_OVER_DW * Dw, self.RE_OVER_DW * Dw
        if ri <= Dw / 2.0 or re <= Dw / 2.0:
            raise ValueError(f"AngularContactFamily: invalid groove geometry for Dw={Dw}")
        if not (0.0 < alpha_0_deg <= 45.0):
            raise ValueError(f"AngularContactFamily: alpha_0_deg must be in (0, 45], got {alpha_0_deg}")

        A = ri + re - Dw
        alpha_0 = np.radians(alpha_0_deg)

        self.Dw, self.Dpw, self.Z, self.i = Dw, Dpw, Z, i
        self.ri, self.re, self.A, self.alpha_0 = ri, re, A, alpha_0
        self.gamma = gamma_point_contact(Dw, alpha_0, Dpw)
        self.reduction_factor = self.REDUCTION_FACTOR_BY_ROWS[i]
        self.phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

    @property
    def name(self) -> str:
        return "angular_contact"

    def curvature_rolling_element(self) -> tuple[float, float]:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r1"]

    def curvature_inner_raceway(self) -> tuple[float, float]:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r2_inner"]

    def curvature_outer_raceway(self) -> tuple[float, float] | None:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r2_outer"]

    def _capacity_calculator(self) -> bcap.PointContactCapacityRadial:
        return bcap.PointContactCapacityRadial(
            Z=self.Z, Dw=self.Dw, alpha_0=self.alpha_0, ri=self.ri, re=self.re,
            gamma=self.gamma, reduction_factor=self.reduction_factor, i=self.i)

    def dynamic_capacity(self) -> float:
        return self._capacity_calculator().Cr

    def static_capacity(self) -> float:
        return self._capacity_calculator().C0

    def per_element_dynamic_capacity(self) -> tuple[float, float]:
        return self._capacity_calculator().Q_elements


# =====================================================================
# ---- self-aligning ball -------------------------------------------------
# =====================================================================

@register_family
class SelfAligningBallFamily(BearingFamily):
    """NOTE: este tipo nao tem o modulo de contacto ISO 16281 completo
    (ver _outer_term em NotImplementedError no antigo iso16281_contact.py
    -- SelfAligningPointContactStiffness); o valor de cp havera de vir de
    slippy/Hertz ou de uma formula fechada derivada fora da norma. Isso
    e problema do solver, nao deste ficheiro -- aqui so ha geometria."""
    BEARING_TYPE = BearingType.SELF_ALIGNING_BALL
    DUTY = "radial"
    RI_OVER_DW = 0.53
    REDUCTION_FACTOR_BY_ROWS = {1: 1.0, 2: 1.0}

    def __init__(self, *, position: float, arrangement, d: float, D: float, b: float,
                 surfaces: SurfacePair, Dw: float, Dpw: float, Z: int, alpha_0_deg: float,
                 i: int = 1, C: float | None = None, C0: float | None = None,
                 label: str = "", designation: str = ""):
        super().__init__(position=position, arrangement=arrangement, d=d, D=D, b=b,
                          surfaces=surfaces, C=C, C0=C0, label=label, designation=designation)

        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(f"SelfAligningBallFamily: i must be one of "
                              f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}")
        if not (0.0 < alpha_0_deg <= 45.0):
            raise ValueError(f"SelfAligningBallFamily: alpha_0_deg must be in (0, 45], got {alpha_0_deg}")

        alpha_0 = np.radians(alpha_0_deg)
        ri = self.RI_OVER_DW * Dw
        gamma = gamma_point_contact(Dw, alpha_0, Dpw)
        if gamma <= 0.0:
            raise ValueError(f"SelfAligningBallFamily: gamma must be > 0, got {gamma}.")
        re = 0.5 * (1.0 / gamma + 1.0) * Dw
        if ri <= Dw / 2.0 or re <= Dw / 2.0:
            raise ValueError(f"SelfAligningBallFamily: invalid groove geometry for Dw={Dw}")

        A = ri + re - Dw
        self.Dw, self.Dpw, self.Z, self.i = Dw, Dpw, Z, i
        self.ri, self.re, self.A, self.alpha_0 = ri, re, A, alpha_0
        self.gamma = gamma
        self.reduction_factor = self.REDUCTION_FACTOR_BY_ROWS[i]
        self.phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

    @property
    def name(self) -> str:
        return "self_aligning"

    def curvature_rolling_element(self) -> tuple[float, float]:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r1"]

    def curvature_inner_raceway(self) -> tuple[float, float]:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r2_inner"]

    def curvature_outer_raceway(self) -> tuple[float, float] | None:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r2_outer"]

    def _capacity_calculator(self) -> bcap.PointContactCapacityRadial:
        return bcap.PointContactCapacityRadial(
            Z=self.Z, Dw=self.Dw, alpha_0=self.alpha_0, ri=self.ri, re=self.re,
            gamma=self.gamma, reduction_factor=self.reduction_factor, i=self.i)

    def dynamic_capacity(self) -> float:
        return self._capacity_calculator().Cr

    def static_capacity(self) -> float:
        return self._capacity_calculator().C0

    def per_element_dynamic_capacity(self) -> tuple[float, float]:
        return self._capacity_calculator().Q_elements