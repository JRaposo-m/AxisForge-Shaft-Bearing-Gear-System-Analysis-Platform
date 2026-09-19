# =====================================================================
# families/family.py
# =====================================================================
"""
core/machine_elements/bearings/families/family.py

The contract (BearingFamily), the auto-registration mechanism, and
every concrete family. Math lives in the sibling files ./contact.py
and ./capacity.py, imported below.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from axisforge.core.machine_elements.bearings.catalog import BearingCatalog

from axisforge.core.machine_elements.bearings.bearing_types import BearingType
from . import iso16281_contact as bc
from . import capacity as bcap


# =====================================================================
# ---- registration -----------------------------------------------------
# =====================================================================

_FAMILY_REGISTRY: dict[str, type["BearingFamily"]] = {}

def register_family(cls: type["BearingFamily"]) -> type["BearingFamily"]:
    _FAMILY_REGISTRY[cls.__name__] = cls
    return cls


# =====================================================================
# ---- contract -----------------------------------------------------
# =====================================================================

class BearingFamily(ABC):
    CAPABILITIES: frozenset[str] = frozenset()
    REQUIRED_FOR: dict[str, frozenset[str]] = {}
    BEARING_TYPE: Any = None
    DUTY: str | None = None

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def assemble_geometry(self, catalog: "BearingCatalog", **geometry_kwargs: Any) -> dict[str, Any]: ...

    @staticmethod
    @abstractmethod
    def dynamic_capacity(bearing) -> float: ...

    @staticmethod
    @abstractmethod
    def per_element_dynamic_capacity(bearing, Cr: float | None = None) -> tuple[float, float]: ...



# =====================================================================
# ---- ball bearing / radial, point contact --------------------------
# =====================================================================

@register_family
class DeepGrooveBallFamily(BearingFamily):
    BEARING_TYPE = BearingType.DEEP_GROOVE_BALL
    DUTY = "radial"
    CAPABILITIES = frozenset({"point_contact"})
    REQUIRED_FOR = {
        "point_contact": frozenset({
            "ri", "re", "Dw", "Dpw", "Z", "E", "nu",
            "A", "alpha_0", "Ri", "phi_j", "gamma", "cp",
        }),
    }
    RI_OVER_DW = 0.52
    RE_OVER_DW = 0.52
    REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.90}

    @property
    def name(self) -> str:
        return "deep_groove_ball"

    @classmethod
    def reference_raceway_radii(cls, Dw: float) -> tuple[float, float]:
        return cls.RI_OVER_DW * Dw, cls.RE_OVER_DW * Dw

    def assemble_geometry(self, catalog, Dw, Dpw, Z, E, s, nu=0.3, i=1) -> dict[str, Any]:
        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(f"DeepGrooveBallFamily: i must be one of "
                              f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}")
        ri, re = self.reference_raceway_radii(Dw)
        if ri <= Dw / 2.0 or re <= Dw / 2.0:
            raise ValueError(f"DeepGrooveBallFamily: invalid groove geometry for Dw={Dw}")
        if s < 0:
            raise ValueError(f"DeepGrooveBallFamily: s must be >= 0, got {s}")

        A = ri + re - Dw
        alpha_0, s = bc.contact_angle_and_clearance(A, s=s)

        stiff = bc.PointContactStiffness(Dw, ri, re, E, nu, alpha_0, Dpw)
        g = stiff.gamma
        Ri = stiff.raceway_contact_radius
        cp = stiff.stiffness

        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        return dict(bearing_type=self.BEARING_TYPE, ri=ri, re=re, Dw=Dw, Dpw=Dpw,
                    Z=Z, s=s, E=E, nu=nu, A=A, alpha_0=alpha_0, Ri=Ri, phi_j=phi_j,
                    gamma=g, cp=cp, raceway_radii_from_reference=True,
                    i=i, reduction_factor=self.REDUCTION_FACTOR_BY_ROWS[i])

    @staticmethod
    def per_element_dynamic_capacity(bearing, Cr=None):
        calc = bcap.PointContactCapacityRadial(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i)
        return calc.Q_elements

    @staticmethod
    def dynamic_capacity(bearing):
        calc = bcap.PointContactCapacityRadial(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i)
        return calc.Cr

@register_family
class AngularContactFamily(BearingFamily):
    BEARING_TYPE = BearingType.ANGULAR_CONTACT
    DUTY = "radial"
    CAPABILITIES = frozenset({"point_contact"})
    REQUIRED_FOR = {
        "point_contact": frozenset({
            "ri", "re", "Dw", "Dpw", "Z", "E", "nu",
            "A", "alpha_0", "Ri", "phi_j", "gamma", "cp",
        }),
    }
    RI_OVER_DW = 0.52
    RE_OVER_DW = 0.52
    REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.95}

    @property
    def name(self) -> str:
        return "angular_contact"

    @classmethod
    def reference_raceway_radii(cls, Dw: float) -> tuple[float, float]:
        return cls.RI_OVER_DW * Dw, cls.RE_OVER_DW * Dw

    def assemble_geometry(self, catalog, Dw, Dpw, Z, E, alpha_0_deg, nu=0.3, i=1) -> dict[str, Any]:
        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(f"AngularContactFamily: i must be one of "
                              f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}")
        ri, re = self.reference_raceway_radii(Dw)
        if ri <= Dw / 2.0 or re <= Dw / 2.0:
            raise ValueError(f"AngularContactFamily: invalid groove geometry for Dw={Dw}")
        if not (0.0 < alpha_0_deg <= 45.0):
            raise ValueError(
                f"AngularContactFamily: alpha_0_deg must be in (0, 45], got {alpha_0_deg}")
         
        A = ri + re - Dw
        alpha_0, s = bc.contact_angle_and_clearance(A, alpha_0_deg=alpha_0_deg)

        stiff = bc.PointContactStiffness(Dw, ri, re, E, nu, alpha_0, Dpw)
        g = stiff.gamma
        Ri = stiff.raceway_contact_radius
        cp = stiff.stiffness

        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        return dict(bearing_type=self.BEARING_TYPE, ri=ri, re=re, Dw=Dw, Dpw=Dpw,
                    Z=Z, s=s, E=E, nu=nu, A=A, alpha_0=alpha_0, Ri=Ri, phi_j=phi_j,
                    gamma=g, cp=cp, raceway_radii_from_reference=True,
                    i=i, reduction_factor=self.REDUCTION_FACTOR_BY_ROWS[i])

    @staticmethod
    def per_element_dynamic_capacity(bearing, Cr=None):
        calc = bcap.PointContactCapacityRadial(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i)
        return calc.Q_elements

    @staticmethod
    def dynamic_capacity(bearing):
        calc = bcap.PointContactCapacityRadial(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i)
        return calc.Cr

@register_family
class SelfAligningBallFamily(BearingFamily):
    """
    NOTE: This type of bearing does not have the right contact module
    elaborated in iso16281_contact for this contact model the value will
    be either be done taken into account slippy or elaborated outside the norm
    """
    BEARING_TYPE = BearingType.SELF_ALIGNING_BALL
    DUTY = "radial"
    CAPABILITIES = frozenset({"point_contact"})
    REQUIRED_FOR = {
        "point_contact": frozenset({
            "ri", "re", "Dw", "Dpw", "Z", "E", "nu",
            "A", "alpha_0", "Ri", "phi_j", "gamma", "cp",
        }),
    }
    RI_OVER_DW = 0.53
    REDUCTION_FACTOR_BY_ROWS = {1: 1.0, 2: 1.0}

    @property
    def name(self) -> str:
        return "self_aligning"

    @classmethod
    def reference_raceway_radii(cls, Dw: float, gamma: float) -> tuple[float, float]:
        ri = cls.RI_OVER_DW * Dw
        if gamma <= 0.0:
            raise ValueError(f"SelfAligningBallFamily: gamma must be > 0, got {gamma}.")
        re = 0.5 * (1.0 / gamma + 1.0) * Dw
        return ri, re

    def assemble_geometry(self, catalog, Dw, Dpw, Z, E, alpha_0_deg, nu=0.3, i=1) -> dict[str, Any]:
        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(f"AngularContactFamily: i must be one of "
                              f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}")
        ri, re = self.reference_raceway_radii(Dw)
        if ri <= Dw / 2.0 or re <= Dw / 2.0:
            raise ValueError(f"AngularContactFamily: invalid groove geometry for Dw={Dw}")
        if not (0.0 < alpha_0_deg <= 45.0):
            raise ValueError(
                f"AngularContactFamily: alpha_0_deg must be in (0, 45], got {alpha_0_deg}")
         
        A = ri + re - Dw
        alpha_0, s = bc.contact_angle_and_clearance(A, alpha_0_deg=alpha_0_deg)

        stiff = bc.SelfAligningPointContactStiffness(Dw, ri, re, E, nu, alpha_0, Dpw)
        g = stiff.gamma
        Ri = stiff.raceway_contact_radius
        cp = stiff.stiffness

        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        return dict(bearing_type=self.BEARING_TYPE, ri=ri, re=re, Dw=Dw, Dpw=Dpw,
                    Z=Z, s=s, E=E, nu=nu, A=A, alpha_0=alpha_0, Ri=Ri, phi_j=phi_j,
                    gamma=g, cp=cp, raceway_radii_from_reference=True,
                    i=i, reduction_factor=self.REDUCTION_FACTOR_BY_ROWS[i])

    @staticmethod
    def per_element_dynamic_capacity(bearing, Cr=None):
        calc = bcap.PointContactCapacityRadial(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i)
        return calc.Q_elements

    @staticmethod
    def dynamic_capacity(bearing):
        calc = bcap.PointContactCapacityRadial(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i)
        return calc.Cr

# =====================================================================
# ---- ball bearing / thrust, point contact --------------------------
# =====================================================================

@register_family
class ThrustBallSingleRowFamily(BearingFamily):
    BEARING_TYPE = BearingType.THRUST_BALL
    DUTY = "thrust"
    CAPABILITIES = frozenset({"point_contact"})
    REQUIRED_FOR = {
        "point_contact": frozenset({
            "ri", "re", "Dw", "Dpw", "Z", "E", "nu",
            "A", "alpha_0", "Ri", "phi_j", "gamma", "cp",
        }),
    }
    RI_OVER_DW = 0.535
    RE_OVER_DW = 0.535
    REDUCTION_FACTOR = 0.90   # thrust single-row: sem redução (ver nota acima -- confirmar valor com a norma)

    @property
    def name(self) -> str:
        return "thrust_ball"

    @classmethod
    def reference_raceway_radii(cls, Dw: float) -> tuple[float, float]:
        return cls.RI_OVER_DW * Dw, cls.RE_OVER_DW * Dw

    def assemble_geometry(self, catalog, Dw, Dpw, Z, E, alpha_0_deg=None, nu=0.3) -> dict[str, Any]:
        if not isinstance(Z, (int, np.integer)):
            raise TypeError(f"ThrustBallFamily: Z must be a scalar int, got {type(Z).__name__}")

        ri, re = self.reference_raceway_radii(Dw)
        if ri <= Dw / 2.0 or re <= Dw / 2.0:
            raise ValueError(f"ThrustBallFamily: invalid groove geometry for Dw={Dw}")

        A = ri + re - Dw
        if alpha_0_deg is None:
            alpha_0, s = np.pi / 2, 0.0
        else:
            if not (45.0 < alpha_0_deg <= 90.0):
                raise ValueError(
                    f"ThrustBallFamily: alpha_0_deg must be in (45, 90], got {alpha_0_deg}")
            alpha_0, s = bc.contact_angle_and_clearance(A, alpha_0_deg=alpha_0_deg)

        stiff = bc.PointContactStiffness(Dw, ri, re, E, nu, alpha_0, Dpw)

        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        return dict(bearing_type=self.BEARING_TYPE, ri=ri, re=re, Dw=Dw, Dpw=Dpw,
                    Z=Z, s=s, E=E, nu=nu, A=A, alpha_0=alpha_0, Ri=stiff.raceway_contact_radius,
                    phi_j=phi_j, gamma=stiff.gamma, cp=stiff.stiffness,
                    raceway_radii_from_reference=True, reduction_factor=self.REDUCTION_FACTOR)

    @staticmethod
    def _capacity_class(alpha_0: float) -> type[bcap.CapacityCalculator]:
        if np.isclose(alpha_0, np.pi / 2):
            return bcap.PointContactCapacityThrust_90deg
        return bcap.PointContactCapacityThrust_Non_90deg

    @staticmethod
    def _capacity_calculator(row_or_bearing) -> bcap.CapacityCalculator:
        """row_or_bearing: um dict de assemble_geometry() OU um Bearing
        já montado -- ambos têm os mesmos campos por nome."""
        get = row_or_bearing.__getitem__ if isinstance(row_or_bearing, dict) else \
              (lambda k: getattr(row_or_bearing, k))
        cls_ = ThrustBallSingleRowFamily._capacity_class(get("alpha_0"))
        return cls_(Z=get("Z"), Dw=get("Dw"), alpha_0=get("alpha_0"),
                     ri=get("ri"), re=get("re"), gamma=get("gamma"),
                     reduction_factor=get("reduction_factor"))

    @staticmethod
    def per_element_dynamic_capacity(bearing, Cr=None):
        return ThrustBallSingleRowFamily._capacity_calculator(bearing).Q_elements

    @staticmethod
    def dynamic_capacity(bearing):
        return ThrustBallSingleRowFamily._capacity_calculator(bearing).Cr


@register_family
class ThrustBallMultiRowFamily(BearingFamily):
    """Thrust multi-row onde as filas podem ser genuinely diferentes
    entre si (Dw, alpha_0, Z próprios por fila) -- reutiliza
    ThrustBallFamily.assemble_geometry() fila a fila (composição, não
    reimplementa a física) e combina via combine_multirow() (Formula 29,
    Sec 6.4). É esta a classe a usar quando as filas NÃO são idênticas;
    quando são, ThrustBallFamily sozinha (chamada N vezes) já chega."""

    BEARING_TYPE = BearingType.THRUST_BALL_MULTIROW
    DUTY = "thrust"
    CAPABILITIES = frozenset({"point_contact"})
    REQUIRED_FOR = {"point_contact": frozenset({"rows", "Cr", "Q_elements"})}

    _single_row_family = ThrustBallSingleRowFamily()

    @property
    def name(self) -> str:
        return "thrust_ball_multirow"

    def assemble_geometry(self, catalog, rows: list[dict]) -> dict[str, Any]:
        """rows = [{"Dw":.., "Dpw":.., "Z":.., "E":.., "alpha_0_deg":.., "nu":..}, ...],
        um dict de kwargs por fila -- livre de variar entre filas."""
        assembled_rows = [
            self._single_row_family.assemble_geometry(catalog, **row_kwargs)
            for row_kwargs in rows
        ]

        calculators = [ThrustBallSingleRowFamily._capacity_calculator(row) for row in assembled_rows]

        cls_ = type(calculators[0])
        if any(type(c) is not cls_ for c in calculators):
            raise ValueError(
                "ThrustBallMultiRowFamily: cannot combine rows with mixed "
                "90 deg / non-90 deg contact angle via combine_multirow().")

        capacity_kwargs_rows = [
            dict(Z=row["Z"], Dw=row["Dw"], alpha_0=row["alpha_0"],
                 ri=row["ri"], re=row["re"], gamma=row["gamma"],
                 reduction_factor=row["reduction_factor"])
            for row in assembled_rows
        ]
        Cr_total = cls_.combine_multirow(capacity_kwargs_rows)

        return dict(bearing_type=self.BEARING_TYPE, rows=assembled_rows,
                    Cr=Cr_total, Q_elements=[c.Q_elements for c in calculators])

    @staticmethod
    def dynamic_capacity(bearing):
        return bearing.Cr

    @staticmethod
    def per_element_dynamic_capacity(bearing, Cr=None):
        return bearing.Q_elements