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
    from axisforge.core.machine_elements.bearings.base import BearingCatalog

from axisforge.core.machine_elements.bearings.base import BearingType
import axisforge.core.machine_elements.bearings.families.iso16281_contact as bc
import axisforge.core.machine_elements.bearings.families.capacity as bcap


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
    def per_element_dynamic_capacity(bearing, capacity: float | None = None) -> tuple[float, float]: ...



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
    LAM = 0.90   # thrust single-row

    @property
    def name(self) -> str:
        return "thrust_ball_single_row"

    @staticmethod
    def eta(alpha_0: float) -> float:
        """Reduction factor eta for Formula (18)-(20)/(23)-(25). alpha_0 in radians."""
        return 1.0 - np.sin(alpha_0) / 3.0

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
        eta = self.eta(alpha_0)
        g = stiff.gamma
        Ri = stiff.raceway_contact_radius
        cp = stiff.stiffness

        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        return dict(bearing_type=self.BEARING_TYPE, ri=ri, re=re, Dw=Dw, Dpw=Dpw,
                    Z=Z, s=s, E=E, nu=nu, A=A, alpha_0=alpha_0, eta=eta, Ri=Ri,
                    phi_j=phi_j, gamma=g, cp=cp,
                    raceway_radii_from_reference=True, lam=self.LAM)

    @staticmethod
    def _capacity_class(alpha_0: float) -> type[bcap.ThrustCapacityCalculator]:
        if np.isclose(alpha_0, np.pi / 2):
            return bcap.PointContactCapacityThrust_90deg
        return bcap.PointContactCapacityThrust_Non_90deg

    @staticmethod
    def _capacity_calculator(row_or_bearing, i: int = 1) -> bcap.ThrustCapacityCalculator:
        """row_or_bearing: um dict de assemble_geometry() OU um Bearing
        já montado. i não vem da geometria (não afeta ri/re/alpha_0/...),
        por isso entra aqui como argumento próprio, não lido do row/bearing."""
        get = row_or_bearing.__getitem__ if isinstance(row_or_bearing, dict) else \
              (lambda k: getattr(row_or_bearing, k))
        cls_ = ThrustBallSingleRowFamily._capacity_class(get("alpha_0"))
        return cls_(Z=get("Z"), Dw=get("Dw"), alpha_0=get("alpha_0"),
                     ri=get("ri"), re=get("re"), gamma=get("gamma"),
                     lam=get("lam"), eta=get("eta"), i=i)

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca=None, i: int = 1):
        return ThrustBallSingleRowFamily._capacity_calculator(bearing, i=i).Q_elements

    @staticmethod
    def dynamic_capacity(bearing, i: int = 1):
        return ThrustBallSingleRowFamily._capacity_calculator(bearing, i=i).Ca


@register_family
class ThrustBallMultiRowFamily(BearingFamily):
    """Thrust multi-row onde as filas podem ser genuinely diferentes
    entre si (Dw, alpha_0, Z próprios por fila) -- reutiliza
    ThrustBallFamily.assemble_geometry() fila a fila (composição, não
    reimplementa a física) e combina via combine_multirow() (Formula 29,
    Sec 6.4). É esta a classe a usar quando as filas NÃO são idênticas;
    quando são, ThrustBallFamily sozinha (chamada N vezes) já chega."""

    BEARING_TYPE = BearingType.THRUST_BALL
    DUTY = "thrust"
    CAPABILITIES = frozenset({"point_contact"})
    REQUIRED_FOR = {"point_contact": frozenset({"rows", "Ca", "Q_elements"})}

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
                 lam=row["lam"], eta=row["eta"])
            for row in assembled_rows
        ]
        Ca_total = cls_.combine_multirow(capacity_kwargs_rows)

        return dict(bearing_type=self.BEARING_TYPE, rows=assembled_rows,
                    Ca=Ca_total, Q_elements=[c.Q_elements for c in calculators])

    @staticmethod
    def dynamic_capacity(bearing):
        return bearing.Ca

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca=None):
        return bearing.Q_elements


# =====================================================================
# ---- roller bearing / thrust, point contact --------------------------
# =====================================================================

@register_family
class CylindricalRollerFamily(BearingFamily):
    BEARING_TYPE = BearingType.CYLINDRICAL_ROLLER
    DUTY = "radial"
    CAPABILITIES = frozenset({"line_contact"})
    REQUIRED_FOR = {
        "line_contact": frozenset({
            "Dwe", "Lwe", "Dpw", "Z", "s", "n_s", "alpha_0",
            "x_k", "phi_j", "gamma", "cL", "cs", "P_xk", "i",
        }),
    }
    LAMBDA_V = 0.83
    _LOG_ARG_EPS = 1e-12  # floor for the log() argument in the profile function
    # in te future this shall be passed to config.py

    @property
    def name(self) -> str:
        return "cylindrical_roller"

    @classmethod
    def _reference_roller_profile(cls, x_k: np.ndarray, Dwe: float, Lwe: float) -> np.ndarray:
        """P(x_k) [mm] -- ISO/TS 16281 Sec 6.2 eq.(42)-(44)."""
        P = np.zeros_like(x_k)
        if Lwe <= 2.5 * Dwe:
            arg = 1.0 - (2.0 * x_k / Lwe) ** 2
            arg = np.maximum(arg, cls._LOG_ARG_EPS)
            P = 0.000350 * Dwe * np.log(1.0 / arg)
        else:
            half_flat = (Lwe - 2.5 * Dwe) / 2.0
            edge = np.abs(x_k) > half_flat
            if np.any(edge):
                xe = x_k[edge]
                arg = 1.0 - ((2.0 * np.abs(xe) - (Lwe - 2.5 * Dwe)) / (2.5 * Dwe)) ** 2
                arg = np.maximum(arg, cls._LOG_ARG_EPS)
                P[edge] = 0.000500 * Dwe * np.log(1.0 / arg)
        return P

    def assemble_geometry(self, catalog, Dwe, Lwe, Dpw, Z, s, n_s, alpha_0_deg, i=1) -> dict[str, Any]:

        if catalog.arrangement not in ("floating", "non-locating"):
            raise ValueError(
                f"CylindricalRollerFamily(label={catalog.label or catalog.designation!r}): "
                f"arrangement={catalog.arrangement!r} is not valid -- an NU/N-type "
                f"bearing has no flange to react axial load, so it cannot be "
                f"'locating'. Use 'floating' or 'non-locating' on the BearingCatalog."
            )
        if n_s < 30:
            raise ValueError(f"CylindricalRollerFamily: n_s must be >= 30, got {n_s}")
        if s < 0:
            raise ValueError(f"CylindricalRollerFamily: s must be >= 0, got {s}")
        if i < 1:
            raise ValueError(f"CylindricalRollerFamily: i (number of rows) must be >= 1, got {i}")

        alpha_0 = np.radians(alpha_0_deg)

        stiff = bc.LineContactStiffness(Dwe, Dpw, alpha_0, Lwe, n_s)

        g   = stiff.gamma
        x_k = stiff.lamina_positions
        cL  = stiff.stiffness
        cs  = stiff.lamina_stiffness
        P_xk = self._reference_roller_profile(x_k, Dwe, Lwe)

        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        return dict(bearing_type=self.BEARING_TYPE, Dwe=Dwe, Lwe=Lwe, Dpw=Dpw, 
                    Z=Z, s=s, n_s=n_s, alpha_0=alpha_0, x_k=x_k, phi_j=phi_j, 
                    gamma=g, cL=cL, lambda_v = self.LAMBDA_V, cs=cs, P_xk=P_xk, i=i)

    @staticmethod
    def per_element_dynamic_capacity(bearing, Cr=None):
        calc = bcap.LineContactCapacityRadial(
            Z=bearing.Z, Dwe=bearing.Dwe, Lwe=bearing.Lwe, alpha_0=bearing.alpha_0,
            gamma=bearing.gamma, lambda_v=bearing.lambda_v, n_s=bearing.n_s, i=bearing.i)
        return calc.Q_elements

    @staticmethod
    def dynamic_capacity(bearing):
        calc = bcap.LineContactCapacityRadial(
            Z=bearing.Z, Dwe=bearing.Dwe, Lwe=bearing.Lwe, alpha_0=bearing.alpha_0,
            gamma=bearing.gamma, lambda_v=bearing.lambda_v, n_s=bearing.n_s, i=bearing.i)
        return calc.Cr

    @staticmethod
    def per_lamina_dynamic_capacity(bearing):
        calc = bcap.LineContactCapacityRadial(
            Z=bearing.Z, Dwe=bearing.Dwe, Lwe=bearing.Lwe, alpha_0=bearing.alpha_0,
            gamma=bearing.gamma, lambda_v=bearing.lambda_v, n_s=bearing.n_s, i=bearing.i)
        return calc.per_lamina

# =====================================================================
# ---- roller bearing / thrust, line contact --------------------------
# =====================================================================

@register_family
class ThrustCylindricalRollerFamily(BearingFamily):
    BEARING_TYPE = BearingType.THRUST_CYLINDRICAL_ROLLER
    DUTY = "thrust"
    CAPABILITIES = frozenset({"line_contact"})
    REQUIRED_FOR = {
        "line_contact": frozenset({
            "Dwe", "Lwe", "Dpw", "Z", "s", "n_s", "alpha_0",
            "x_k", "phi_j", "gamma", "cL", "cs", "P_xk",
        }),
    }
    LAMBDA_V = 0.73
    _LOG_ARG_EPS = 1e-12 

    @property
    def name(self) -> str:
        return "thrust_cylindrical_roller"

    @staticmethod
    def eta(alpha_0: float) -> float:
        return 1.0 - 0.5 * np.sin(alpha_0)

    @classmethod
    def _reference_roller_profile(cls, x_k: np.ndarray, Dwe: float, Lwe: float) -> np.ndarray:
        """P(x_k) [mm] -- ISO/TS 16281 Sec 6.2 eq.(42)-(44). Identical to
        CylindricalRollerFamily's (radial) -- same roller, same profile."""
        P = np.zeros_like(x_k)
        if Lwe <= 2.5 * Dwe:
            arg = 1.0 - (2.0 * x_k / Lwe) ** 2
            arg = np.maximum(arg, cls._LOG_ARG_EPS)
            P = 0.000350 * Dwe * np.log(1.0 / arg)
        else:
            half_flat = (Lwe - 2.5 * Dwe) / 2.0
            edge = np.abs(x_k) > half_flat
            if np.any(edge):
                xe = x_k[edge]
                arg = 1.0 - ((2.0 * np.abs(xe) - (Lwe - 2.5 * Dwe)) / (2.5 * Dwe)) ** 2
                arg = np.maximum(arg, cls._LOG_ARG_EPS)
                P[edge] = 0.000500 * Dwe * np.log(1.0 / arg)
        return P

    def assemble_geometry(self, catalog, Dwe, Lwe, Dpw, Z, s, n_s,
                           alpha_0_deg, i: int = 1) -> dict[str, Any]:
        if not isinstance(Z, (int, np.integer)):
            raise TypeError(f"ThrustCylindricalRollerFamily: Z must be a scalar int, got {type(Z).__name__}")
        if n_s < 30:
            raise ValueError(f"ThrustCylindricalRollerFamily: n_s must be >= 30, got {n_s}")
        if s < 0:
            raise ValueError(f"ThrustCylindricalRollerFamily: s must be >= 0, got {s}")
        if not (0.0 < alpha_0_deg <= 90.0):
            raise ValueError(f"ThrustCylindricalRollerFamily: alpha_0_deg must be in (0, 90], got {alpha_0_deg}")

        alpha_0 = np.radians(alpha_0_deg)
        eta     = self.eta(alpha_0)

        stiff = bc.LineContactStiffness(Dwe, Dpw, alpha_0, Lwe, n_s)

        g    = stiff.gamma
        x_k  = stiff.lamina_positions
        cL   = stiff.stiffness
        cs   = stiff.lamina_stiffness
        P_xk = self._reference_roller_profile(x_k, Dwe, Lwe)

        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        return dict(bearing_type=self.BEARING_TYPE, Dwe=Dwe, Lwe=Lwe, Dpw=Dpw,
                    Z=Z, s=s, n_s=n_s, alpha_0=alpha_0, x_k=x_k, phi_j=phi_j,
                    gamma=g, cL=cL, cs=cs, P_xk=P_xk, eta=eta,
                    lambda_v=self.LAMBDA_V, i=i)

    @staticmethod
    def _capacity_class(alpha_0: float) -> type[bcap.ThrustCapacityCalculator]:
        if np.isclose(alpha_0, np.pi / 2):
            return bcap.LineContactCapacityThrust_90deg
        return bcap.LineContactCapacityThrust_Non_90deg

    @staticmethod
    def _capacity_calculator(row_or_bearing, i: int = 1) -> bcap.ThrustCapacityCalculator:
        get = row_or_bearing.__getitem__ if isinstance(row_or_bearing, dict) else \
              (lambda k: getattr(row_or_bearing, k))
        cls_ = ThrustCylindricalRollerFamily._capacity_class(get("alpha_0"))
        return cls_(Z=get("Z"), Dwe=get("Dwe"), Lwe=get("Lwe"), alpha_0=get("alpha_0"),
                     gamma=get("gamma"), lambda_v=get("lambda_v"), eta=get("eta"), i=i)

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca=None, i: int = 1):
        return ThrustCylindricalRollerFamily._capacity_calculator(bearing, i=i).Q_elements

    @staticmethod
    def dynamic_capacity(bearing, i: int = 1):
        return ThrustCylindricalRollerFamily._capacity_calculator(bearing, i=i).Ca

    @staticmethod
    def per_lamina_dynamic_capacity(bearing, i: int = 1):
        return ThrustCylindricalRollerFamily._capacity_calculator(bearing, i=i).per_lamina


@register_family
class RollerThrustMultiRowFamily(BearingFamily):
    """Espelha ThrustBallMultiRowFamily: filas podem diferir entre si
    (Dwe, alpha_0, Z, eta próprios por fila); reutiliza
    RollerThrustSingleRowFamily.assemble_geometry() fila a fila e combina
    via combine_multirow() herdado de _MultirowCombinableLineContact --
    cuja fórmula, note-se, ainda precisa de ser derivada/confirmada (ver
    nota acima sobre os expoentes 9/2 e 2/9 por analogia)."""

    BEARING_TYPE = BearingType.THRUST_CYLINDRICAL_ROLLER
    DUTY = "thrust"
    CAPABILITIES = frozenset({"multirow_capacity"})
    REQUIRED_FOR = {
        "multirow_capacity": frozenset({"rows", "i"}),
    }

    _single_row_family = ThrustCylindricalRollerFamily()

    @property
    def name(self) -> str:
        return "thrust_cylindrical_roller_multirow"

    def assemble_geometry(self, catalog, rows: list[dict]) -> dict[str, Any]:
        """rows = [{"Dwe":.., "Lwe":.., "Dpw":.., "Z":.., "s":.., "n_s":..,
        "alpha_0_deg":.., "eta":..}, ...]"""
        assembled_rows = [
            self._single_row_family.assemble_geometry(catalog, **row_kwargs)
            for row_kwargs in rows
        ]

        calculators = [ThrustCylindricalRollerFamily._capacity_calculator(row) for row in assembled_rows]

        cls_ = type(calculators[0])
        if any(type(c) is not cls_ for c in calculators):
            raise ValueError(
                "RollerThrustMultiRowFamily: cannot combine rows with mixed "
                "90 deg / non-90 deg contact angle via combine_multirow().")

        capacity_kwargs_rows = [
            dict(Z=row["Z"], Dwe=row["Dwe"], Lwe=row["Lwe"], alpha_0=row["alpha_0"],
                 gamma=row["gamma"], lambda_v=row["lambda_v"], eta=row["eta"])
            for row in assembled_rows
        ]
        Ca_total = cls_.combine_multirow(capacity_kwargs_rows)  # depende do fix pendente

        return dict(bearing_type=self.BEARING_TYPE, rows=assembled_rows,
                    Ca=Ca_total, Q_elements=[c.Q_elements for c in calculators])

    @staticmethod
    def dynamic_capacity(bearing):
        return bearing.Ca

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca=None):
        return bearing.Q_elements