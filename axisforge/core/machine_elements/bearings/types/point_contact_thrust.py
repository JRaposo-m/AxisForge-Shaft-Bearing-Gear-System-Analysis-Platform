# axisforge/core/machine_elements/bearings/types/point_contact_thrust.py
"""
core/machine_elements/bearings/types/point_contact_thrust.py

Rolamentos de esferas de encosto (thrust), contacto pontual:
ThrustBallSingleRowFamily, ThrustBallMultiRowFamily. Portado de
families/family.py (antigo). ThrustBallMultiRowFamily usa COMPOSICAO
(rows: list[ThrustBallSingleRowFamily]), nao subclassing -- as filas
podem genuinamente diferir entre si (Dw, alpha_0, Z proprios por fila).
"""
from __future__ import annotations

import numpy as np

from axisforge.core.machine_elements.bearings.base import BearingType
from axisforge.core.machine_elements.bearings.family import BearingFamily, register_family
from axisforge.core.machine_elements.bearings.bearing_properties import (
    SurfacePair, gamma_point_contact, point_contact_radii,
)
import axisforge.core.machine_elements.bearings.capacity as bcap


@register_family
class ThrustBallSingleRowFamily(BearingFamily):
    BEARING_TYPE = BearingType.THRUST_BALL
    DUTY = "thrust"
    RI_OVER_DW = 0.535
    RE_OVER_DW = 0.535
    LAM = 0.90  # thrust single-row

    def __init__(self, *, position: float, arrangement, d: float, D: float, b: float,
                 surfaces: SurfacePair, Dw: float, Dpw: float, Z: int,
                 alpha_0_deg: float | None = None,
                 C: float | None = None, C0: float | None = None,
                 label: str = "", designation: str = ""):
        super().__init__(position=position, arrangement=arrangement, d=d, D=D, b=b,
                          surfaces=surfaces, C=C, C0=C0, label=label, designation=designation)

        if not isinstance(Z, (int, np.integer)):
            raise TypeError(f"ThrustBallSingleRowFamily: Z must be a scalar int, got {type(Z).__name__}")

        ri, re = self.RI_OVER_DW * Dw, self.RE_OVER_DW * Dw
        if ri <= Dw / 2.0 or re <= Dw / 2.0:
            raise ValueError(f"ThrustBallSingleRowFamily: invalid groove geometry for Dw={Dw}")

        if alpha_0_deg is None:
            alpha_0 = np.pi / 2
        else:
            if not (45.0 < alpha_0_deg <= 90.0):
                raise ValueError(
                    f"ThrustBallSingleRowFamily: alpha_0_deg must be in (45, 90], got {alpha_0_deg}")
            alpha_0 = np.radians(alpha_0_deg)

        self.Dw, self.Dpw, self.Z = Dw, Dpw, Z
        self.ri, self.re, self.alpha_0 = ri, re, alpha_0
        self.gamma = gamma_point_contact(Dw, alpha_0, Dpw)
        self.eta = self._eta(alpha_0)
        self.lam = self.LAM
        self.phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

    @staticmethod
    def _eta(alpha_0: float) -> float:
        """Reduction factor eta -- Formula (18)-(20)/(23)-(25). alpha_0 em radianos."""
        return 1.0 - np.sin(alpha_0) / 3.0

    @property
    def name(self) -> str:
        return "thrust_ball_single_row"

    def curvature_rolling_element(self) -> tuple[float, float]:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r1"]

    def curvature_inner_raceway(self) -> tuple[float, float]:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r2_inner"]

    def curvature_outer_raceway(self) -> tuple[float, float] | None:
        return point_contact_radii(self.Dw, self.ri, self.re, self.alpha_0, self.Dpw)["r2_outer"]

    def _capacity_class(self) -> type[bcap.ThrustCapacityCalculator]:
        if np.isclose(self.alpha_0, np.pi / 2):
            return bcap.PointContactCapacityThrust_90deg
        return bcap.PointContactCapacityThrust_Non_90deg

    def _capacity_calculator(self, i: int = 1) -> bcap.ThrustCapacityCalculator:
        cls_ = self._capacity_class()
        return cls_(Z=self.Z, Dw=self.Dw, alpha_0=self.alpha_0, ri=self.ri, re=self.re,
                     gamma=self.gamma, lam=self.lam, eta=self.eta, i=i)

    def dynamic_capacity(self, i: int = 1) -> float:
        return self._capacity_calculator(i=i).Ca

    def static_capacity(self) -> float:
        return self._capacity_calculator().C0

    def per_element_dynamic_capacity(self, i: int = 1) -> tuple[float, float]:
        return self._capacity_calculator(i=i).Q_elements

    def _capacity_kwargs(self) -> dict:
        """Dict de kwargs para instanciar diretamente um calculador de
        capacity.py -- usado por ThrustBallMultiRowFamily.combine_multirow()."""
        return dict(Z=self.Z, Dw=self.Dw, alpha_0=self.alpha_0, ri=self.ri, re=self.re,
                    gamma=self.gamma, lam=self.lam, eta=self.eta)


@register_family
class ThrustBallMultiRowFamily(BearingFamily):
    """Thrust multi-fila onde as filas podem ser genuinamente diferentes
    entre si -- COMPOSICAO (rows: list[ThrustBallSingleRowFamily]), nao
    subclassing. Combina via combine_multirow() (Formula 29, Sec 6.4).
    Quando as filas sao identicas, usar ThrustBallSingleRowFamily
    diretamente (N vezes) ja chega."""
    BEARING_TYPE = BearingType.THRUST_BALL
    DUTY = "thrust"

    def __init__(self, *, position: float, arrangement, d: float, D: float, b: float,
                 surfaces: SurfacePair, rows: list[dict],
                 C: float | None = None, C0: float | None = None,
                 label: str = "", designation: str = ""):
        super().__init__(position=position, arrangement=arrangement, d=d, D=D, b=b,
                          surfaces=surfaces, C=C, C0=C0, label=label, designation=designation)

        if len(rows) < 2:
            raise ValueError(f"ThrustBallMultiRowFamily: needs >= 2 rows, got {len(rows)}")

        self.rows: list[ThrustBallSingleRowFamily] = [
            ThrustBallSingleRowFamily(position=position, arrangement=arrangement,
                                       d=d, D=D, b=b, surfaces=surfaces, **row_kwargs)
            for row_kwargs in rows
        ]

        cls_ = self.rows[0]._capacity_class()
        if any(row._capacity_class() is not cls_ for row in self.rows):
            raise ValueError(
                "ThrustBallMultiRowFamily: cannot combine rows with mixed "
                "90 deg / non-90 deg contact angle via combine_multirow().")
        self._combine_cls = cls_

    @property
    def name(self) -> str:
        return "thrust_ball_multirow"

    def curvature_rolling_element(self) -> tuple[float, float]:
        raise NotImplementedError(
            "ThrustBallMultiRowFamily has no single curvature -- read "
            "bearing.rows[i].curvature_rolling_element() per row.")

    def curvature_inner_raceway(self) -> tuple[float, float]:
        raise NotImplementedError(
            "ThrustBallMultiRowFamily has no single curvature -- read "
            "bearing.rows[i].curvature_inner_raceway() per row.")

    def curvature_outer_raceway(self) -> tuple[float, float] | None:
        raise NotImplementedError(
            "ThrustBallMultiRowFamily has no single curvature -- read "
            "bearing.rows[i].curvature_outer_raceway() per row.")

    def dynamic_capacity(self) -> float:
        rows_kwargs = [row._capacity_kwargs() for row in self.rows]
        return self._combine_cls.combine_multirow(rows_kwargs)

    def static_capacity(self) -> float:
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    def per_element_dynamic_capacity(self) -> list[tuple[float, float]]:
        return [row.per_element_dynamic_capacity() for row in self.rows]