# axisforge/core/machine_elements/bearings/types/line_contact_radial.py
"""
core/machine_elements/bearings/types/line_contact_radial.py

CylindricalRollerFamily -- rolamento de rolos cilindricos, contacto
linear, radial. Portado de families/family.py (antigo). gamma, x_k
(lamina_positions) e o perfil do rolo vem de bearing_properties.py
(pura geometria); cL/cs (ISO 16281) ja NAO sao calculados aqui -- isso
e o solver (solvers/.../iso_16281/).
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
class CylindricalRollerFamily(BearingFamily):
    BEARING_TYPE = BearingType.CYLINDRICAL_ROLLER
    DUTY = "radial"
    LAMBDA_V = 0.83

    def __init__(self, *, position: float, arrangement, d: float, D: float, b: float,
                 surfaces: SurfacePair, Dwe: float, Lwe: float, Dpw: float, Z: int,
                 s: float, n_s: int, alpha_0_deg: float, i: int = 1,
                 C: float | None = None, C0: float | None = None,
                 label: str = "", designation: str = ""):
        super().__init__(position=position, arrangement=arrangement, d=d, D=D, b=b,
                          surfaces=surfaces, C=C, C0=C0, label=label, designation=designation)

        if arrangement not in ("floating", "non-locating"):
            raise ValueError(
                f"CylindricalRollerFamily(label={label or designation!r}): "
                f"arrangement={arrangement!r} is not valid -- an NU/N-type "
                f"bearing has no flange to react axial load, so it cannot be "
                f"'locating'. Use 'floating' or 'non-locating'.")
        if n_s < 30:
            raise ValueError(f"CylindricalRollerFamily: n_s must be >= 30, got {n_s}")
        if s < 0:
            raise ValueError(f"CylindricalRollerFamily: s must be >= 0, got {s}")
        if i < 1:
            raise ValueError(f"CylindricalRollerFamily: i (number of rows) must be >= 1, got {i}")

        alpha_0 = np.radians(alpha_0_deg)

        self.Dwe, self.Lwe, self.Dpw, self.Z = Dwe, Lwe, Dpw, Z
        self.s, self.n_s, self.alpha_0, self.i = s, n_s, alpha_0, i
        self.gamma = gamma_line_contact(Dwe, alpha_0, Dpw)
        self.x_k = lamina_positions(Lwe, n_s)
        self.P_xk = reference_roller_profile(self.x_k, Dwe, Lwe)
        self.lambda_v = self.LAMBDA_V
        self.phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

    @property
    def name(self) -> str:
        return "cylindrical_roller"

    def curvature_rolling_element(self) -> tuple[float, float]:
        return line_contact_radii(self.Dwe, self.Dpw, self.alpha_0)["r1"]

    def curvature_inner_raceway(self) -> tuple[float, float]:
        return line_contact_radii(self.Dwe, self.Dpw, self.alpha_0)["r2_inner"]

    def curvature_outer_raceway(self) -> tuple[float, float] | None:
        return line_contact_radii(self.Dwe, self.Dpw, self.alpha_0)["r2_outer"]

    def _capacity_calculator(self) -> bcap.LineContactCapacityRadial:
        return bcap.LineContactCapacityRadial(
            Z=self.Z, Dwe=self.Dwe, Lwe=self.Lwe, alpha_0=self.alpha_0,
            gamma=self.gamma, lambda_v=self.lambda_v, n_s=self.n_s, i=self.i)

    def dynamic_capacity(self) -> float:
        return self._capacity_calculator().Cr

    def static_capacity(self) -> float:
        return self._capacity_calculator().C0

    def per_element_dynamic_capacity(self) -> tuple[float, float]:
        return self._capacity_calculator().Q_elements

    def per_lamina_dynamic_capacity(self) -> tuple[float, float]:
        return self._capacity_calculator().per_lamina