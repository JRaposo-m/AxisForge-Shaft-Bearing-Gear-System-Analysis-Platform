# axisforge/core/machine_elements/bearings/types/line_contact_thrust.py
"""
core/machine_elements/bearings/types/line_contact_thrust.py

ThrustCylindricalRollerFamily (single row) + RollerThrustMultiRowFamily
(composicao, filas potencialmente diferentes). Portado de
families/family.py (antigo). Mesma logica de CylindricalRollerFamily,
duty=thrust.
"""
from __future__ import annotations

import numpy as np

from axisforge.core.machine_elements.bearings.base import BearingType
from axisforge.core.machine_elements.bearings.family import BearingFamily, register_family
from axisforge.core.machine_elements.bearings.bearing_properties import (
    SurfacePair, gamma_line_contact, line_contact_radii,
    lamina_positions, reference_roller_profile,
)
import axisforge.core.machine_elements.bearings.capacity as bcap


@register_family
class ThrustCylindricalRollerFamily(BearingFamily):
    BEARING_TYPE = BearingType.THRUST_CYLINDRICAL_ROLLER
    DUTY = "thrust"
    LAMBDA_V = 0.73

    def __init__(self, *, position: float, arrangement, d: float, D: float, b: float,
                 surfaces: SurfacePair, Dwe: float, Lwe: float, Dpw: float, Z: int,
                 s: float, n_s: int, alpha_0_deg: float, i: int = 1,
                 C: float | None = None, C0: float | None = None,
                 label: str = "", designation: str = ""):
        super().__init__(position=position, arrangement=arrangement, d=d, D=D, b=b,
                          surfaces=surfaces, C=C, C0=C0, label=label, designation=designation)

        if not isinstance(Z, (int, np.integer)):
            raise TypeError(f"ThrustCylindricalRollerFamily: Z must be a scalar int, got {type(Z).__name__}")
        if n_s < 30:
            raise ValueError(f"ThrustCylindricalRollerFamily: n_s must be >= 30, got {n_s}")
        if s < 0:
            raise ValueError(f"ThrustCylindricalRollerFamily: s must be >= 0, got {s}")
        if not (0.0 < alpha_0_deg <= 90.0):
            raise ValueError(f"ThrustCylindricalRollerFamily: alpha_0_deg must be in (0, 90], got {alpha_0_deg}")

        alpha_0 = np.radians(alpha_0_deg)

        self.Dwe, self.Lwe, self.Dpw, self.Z = Dwe, Lwe, Dpw, Z
        self.s, self.n_s, self.alpha_0, self.i = s, n_s, alpha_0, i
        self.eta = self._eta(alpha_0)
        self.gamma = gamma_line_contact(Dwe, alpha_0, Dpw)
        self.x_k = lamina_positions(Lwe, n_s)
        self.P_xk = reference_roller_profile(self.x_k, Dwe, Lwe)
        self.lambda_v = self.LAMBDA_V
        self.phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

    @staticmethod
    def _eta(alpha_0: float) -> float:
        return 1.0 - 0.5 * np.sin(alpha_0)

    @property
    def name(self) -> str:
        return "thrust_cylindrical_roller"

    def curvature_rolling_element(self) -> tuple[float, float]:
        return line_contact_radii(self.Dwe, self.Dpw, self.alpha_0)["r1"]

    def curvature_inner_raceway(self) -> tuple[float, float]:
        return line_contact_radii(self.Dwe, self.Dpw, self.alpha_0)["r2_inner"]

    def curvature_outer_raceway(self) -> tuple[float, float] | None:
        return line_contact_radii(self.Dwe, self.Dpw, self.alpha_0)["r2_outer"]

    def _capacity_class(self) -> type[bcap.ThrustCapacityCalculator]:
        if np.isclose(self.alpha_0, np.pi / 2):
            return bcap.LineContactCapacityThrust_90deg
        return bcap.LineContactCapacityThrust_Non_90deg

    def _capacity_calculator(self, i: int = 1) -> bcap.ThrustCapacityCalculator:
        cls_ = self._capacity_class()
        return cls_(Z=self.Z, Dwe=self.Dwe, Lwe=self.Lwe, alpha_0=self.alpha_0,
                     gamma=self.gamma, lambda_v=self.lambda_v, eta=self.eta, i=i)

    def dynamic_capacity(self, i: int = 1) -> float:
        return self._capacity_calculator(i=i).Ca

    def static_capacity(self) -> float:
        return self._capacity_calculator().C0

    def per_element_dynamic_capacity(self, i: int = 1) -> tuple[float, float]:
        return self._capacity_calculator(i=i).Q_elements

    def per_lamina_dynamic_capacity(self, i: int = 1) -> tuple[float, float]:
        return self._capacity_calculator(i=i).per_lamina

    def _capacity_kwargs(self) -> dict:
        """Dict de kwargs para instanciar diretamente um calculador de
        capacity.py -- usado por RollerThrustMultiRowFamily.combine_multirow()."""
        return dict(Z=self.Z, Dwe=self.Dwe, Lwe=self.Lwe, alpha_0=self.alpha_0,
                    gamma=self.gamma, lambda_v=self.lambda_v, eta=self.eta)


@register_family
class RollerThrustMultiRowFamily(BearingFamily):
    """Espelha ThrustBallMultiRowFamily: filas podem diferir entre si
    (Dwe, alpha_0, Z, eta proprios por fila) -- COMPOSICAO
    (rows: list[ThrustCylindricalRollerFamily]). combine_multirow() para
    contacto linear ainda precisa de ser derivada/confirmada (ver nota
    em capacity.py, _MultirowCombinableLineContact)."""
    BEARING_TYPE = BearingType.THRUST_CYLINDRICAL_ROLLER
    DUTY = "thrust"

    def __init__(self, *, position: float, arrangement, d: float, D: float, b: float,
                 surfaces: SurfacePair, rows: list[dict],
                 C: float | None = None, C0: float | None = None,
                 label: str = "", designation: str = ""):
        super().__init__(position=position, arrangement=arrangement, d=d, D=D, b=b,
                          surfaces=surfaces, C=C, C0=C0, label=label, designation=designation)

        if len(rows) < 2:
            raise ValueError(f"RollerThrustMultiRowFamily: needs >= 2 rows, got {len(rows)}")

        self.rows: list[ThrustCylindricalRollerFamily] = [
            ThrustCylindricalRollerFamily(position=position, arrangement=arrangement,
                                           d=d, D=D, b=b, surfaces=surfaces, **row_kwargs)
            for row_kwargs in rows
        ]

        cls_ = self.rows[0]._capacity_class()
        if any(row._capacity_class() is not cls_ for row in self.rows):
            raise ValueError(
                "RollerThrustMultiRowFamily: cannot combine rows with mixed "
                "90 deg / non-90 deg contact angle via combine_multirow().")
        self._combine_cls = cls_

    @property
    def name(self) -> str:
        return "thrust_cylindrical_roller_multirow"

    def curvature_rolling_element(self) -> tuple[float, float]:
        raise NotImplementedError(
            "RollerThrustMultiRowFamily has no single curvature -- read "
            "bearing.rows[i].curvature_rolling_element() per row.")

    def curvature_inner_raceway(self) -> tuple[float, float]:
        raise NotImplementedError(
            "RollerThrustMultiRowFamily has no single curvature -- read "
            "bearing.rows[i].curvature_inner_raceway() per row.")

    def curvature_outer_raceway(self) -> tuple[float, float] | None:
        raise NotImplementedError(
            "RollerThrustMultiRowFamily has no single curvature -- read "
            "bearing.rows[i].curvature_outer_raceway() per row.")

    def dynamic_capacity(self) -> float:
        rows_kwargs = [row._capacity_kwargs() for row in self.rows]
        return self._combine_cls.combine_multirow(rows_kwargs)

    def static_capacity(self) -> float:
        raise NotImplementedError("Static rating (f_0) not provided yet -- see ISO 76:2006 Sec 5")

    def per_element_dynamic_capacity(self) -> list[tuple[float, float]]:
        return [row.per_element_dynamic_capacity() for row in self.rows]