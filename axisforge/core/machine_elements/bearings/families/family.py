# =====================================================================
# families/family.py
# =====================================================================
"""
core/machine_elements/bearings/families/family.py

"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np
import typing

if TYPE_CHECKING:
    from axisforge.core.machine_elements.bearings.base import BearingCatalog

from axisforge.core.machine_elements.bearings.base import BearingFamily
import axisforge.core.machine_elements.bearings.contact_models.iso16281 as iso

import axisforge.core.machine_elements.bearings.families.geometry as _geo

from axisforge.core.machine_elements.bearings.contact_models.analysis import (
    ContactAnalysis, _verify_contact_inputs, _resolve_pair,
)
import axisforge.core.machine_elements.bearings.families.capacity as bcap


# ====================================================================
# ---- decorators-----------------------------------------------------
# ====================================================================

_FAMILY_REGISTRY: dict[str, type["BearingFamily"]] = {}
_POINT_CONTACT: dict[str, type["BearingFamily"]] = {}
_LINE_CONTACT: dict[str, type["BearingFamily"]] = {}
_THRUST_BEARING: dict[str, type["BearingFamily"]] = {}

def register_family(cls: type["BearingFamily"]) -> type["BearingFamily"]:
    _FAMILY_REGISTRY[cls.__name__] = cls
    return cls

def point_contact(cls: type["BearingFamily"]) -> type["BearingFamily"]:
    _POINT_CONTACT[cls.__name__] = cls
    return cls

def line_contact(cls: type["BearingFamily"]) -> type["BearingFamily"]:
    _LINE_CONTACT[cls.__name__] = cls
    return cls

def thrust(cls: type["BearingFamily"]) -> type["BearingFamily"]:
    _THRUST_BEARING[cls.__name__] = cls
    return cls


# ====================================================================
# ---- helpers -------------------------------------------------------
# ====================================================================

def is_point_contact_family(family: "BearingFamily") -> bool:
    return type(family).__name__ in _POINT_CONTACT

def is_line_contact_family(family: "BearingFamily") -> bool:
    return type(family).__name__ in _LINE_CONTACT

def is_thrust_family(family: "BearingFamily") -> bool:
    return type(family).__name__ in _THRUST_BEARING


# =====================================================================
# ---- ball bearing / radial, point contact --------------------------
# =====================================================================

@register_family
@point_contact
class DeepGrooveBallFamily(BearingFamily):

    RI_OVER_DW = 0.52
    RE_OVER_DW = 0.52
    REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.90}

    @property
    def name(self) -> str:
        return "deep_groove_ball"

    @classmethod
    def reference_raceway_radii(cls, Dw: float) -> tuple[float, float]:
        return cls.RI_OVER_DW * Dw, cls.RE_OVER_DW * Dw

    def assemble_geometry(self, catalog, Dw, Dpw, Z, s, i=1, *, 
                           contact: ContactAnalysis = ContactAnalysis.NONE,
                           e1=None, e2=None, nu1=None, nu2=None,) -> dict[str, Any]:
        """
        The materials for e1 and nu1 are refering to the rolling element while 
        e2 and nu2 refer to the raceway material, both inner and outer raceways
        """

        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(f"DeepGrooveBallFamily: i must be one of "
                              f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}")
        ri, re = self.reference_raceway_radii(Dw)
        if ri <= Dw / 2.0 or re <= Dw / 2.0:
            raise ValueError(f"DeepGrooveBallFamily: invalid groove geometry for Dw={Dw}")
        if s < 0:
            raise ValueError(f"DeepGrooveBallFamily: s must be >= 0, got {s}")

        A            = ri + re - Dw
        alpha_0, s   = _geo.contact_angle_and_clearance(A, s=s)
        ball_geo     = _geo.BallBearingGeometry(Dw=Dw, Dpw=Dpw, A=A, s=s, ri=ri, re=re, alpha_0=alpha_0)
        gamma        = ball_geo.gamma 
        P_e          = ball_geo.free_end_play
        f_i          = ball_geo.f_i
        f_o          = ball_geo.f_o
        r_rolling_el = [ball_geo.r_ax, ball_geo.r_ay]
        r_inner      = [ball_geo.r_bx_inner, ball_geo.r_by_inner]
        r_outer      = [ball_geo.r_bx_outer, ball_geo.r_by_outer]

        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        result = dict( ri=ri, re=re, Dw=Dw, Dpw=Dpw, gamma=gamma,
                       Z=Z, s=s, A=A, alpha_0=alpha_0, phi_j=phi_j,
                       P_e=P_e, f_i=f_i, f_o=f_o, r_rolling_el=r_rolling_el,
                       r_inner=r_inner, r_outer=r_outer, i=i,
                       reduction_factor=self.REDUCTION_FACTOR_BY_ROWS[i])

        if contact is ContactAnalysis.NONE:
            return result

        e1, e2 = _resolve_pair(e1, e2, "e1", "e2")
        nu1, nu2 = _resolve_pair(nu1, nu2, "nu1", "nu2")
        result.update(e1=e1, e2=e2, nu1=nu1, nu2=nu2)

        if contact in (ContactAnalysis.ISO16281):
            point_stiff = iso.PointContactStiffness(Dw, Dpw, gamma, ri, re, alpha_0, e1, e2, nu1, nu2)
            cp = point_stiff.cp
            Ri = point_stiff.raceway_contact_radius
            result.update(iso16281_analysis=True, cp=cp, Ri=Ri)

        return result

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
@point_contact
class AngularContactFamily(BearingFamily):

    RI_OVER_DW = 0.52
    RE_OVER_DW = 0.52
    REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.95}

    @property
    def name(self) -> str:
        return "angular_contact"

    @classmethod
    def reference_raceway_radii(cls, Dw: float) -> tuple[float, float]:
        return cls.RI_OVER_DW * Dw, cls.RE_OVER_DW * Dw

    def assemble_geometry(self, catalog, Dw, Dpw, Z, alpha_0_deg, i=1,
                           *, contact: ContactAnalysis = ContactAnalysis.NONE,
                           e1=None, e2=None, nu1=None, nu2=None,) -> dict[str, Any]:
        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(f"AngularContactFamily: i must be one of "
                              f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}")
        ri, re = self.reference_raceway_radii(Dw)
        if ri <= Dw / 2.0 or re <= Dw / 2.0:
            raise ValueError(f"AngularContactFamily: invalid groove geometry for Dw={Dw}")
        if not (0.0 < alpha_0_deg <= 45.0):
            raise ValueError(f"AngularContactFamily: alpha_0_deg must be in (0, 45], got {alpha_0_deg}")

        A            = ri + re - Dw
        alpha_0, s   = _geo.contact_angle_and_clearance(A, alpha_0_deg=alpha_0_deg)
        ball_geo     = _geo.BallBearingGeometry(Dw=Dw, Dpw=Dpw, A=A, s=s, ri=ri, re=re, alpha_0=alpha_0)
        gamma        = ball_geo.gamma
        P_e          = ball_geo.free_end_play
        f_i          = ball_geo.f_i
        f_o          = ball_geo.f_o
        r_rolling_el = [ball_geo.r_ax, ball_geo.r_ay]
        r_inner      = [ball_geo.r_bx_inner, ball_geo.r_by_inner]
        r_outer      = [ball_geo.r_bx_outer, ball_geo.r_by_outer]

        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        result = dict( ri=ri, re=re, Dw=Dw, Dpw=Dpw, gamma=gamma,
                       Z=Z, s=s, A=A, alpha_0=alpha_0, phi_j=phi_j,
                       P_e=P_e, f_i=f_i, f_o=f_o, r_rolling_el=r_rolling_el,
                       r_inner=r_inner, r_outer=r_outer, i=i,
                       reduction_factor=self.REDUCTION_FACTOR_BY_ROWS[i])

        if contact is ContactAnalysis.NONE:
            return result

        e1, e2 = _resolve_pair(e1, e2, "e1", "e2")
        nu1, nu2 = _resolve_pair(nu1, nu2, "nu1", "nu2")
        result.update(e1=e1, e2=e2, nu1=nu1, nu2=nu2)

        if contact is ContactAnalysis.ISO16281:
            point_stiff = iso.PointContactStiffness(Dw, Dpw, gamma, ri, re, alpha_0, e1, e2, nu1, nu2)
            cp = point_stiff.cp
            Ri = point_stiff.raceway_contact_radius
            result.update(iso16281_analysis=True, cp=cp, Ri=Ri)

        return result

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
@point_contact
class SelfAligningBallFamily(BearingFamily):
    """iso16281's SelfAligningPointContactStiffness._outer_term is not
    implemented yet -- ISO16281/ISO16281_AND_HERTZ analysis will raise
    NotImplementedError when .cp is actually read for this family."""

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

    def assemble_geometry(self, catalog, Dw, Dpw, Z, alpha_0_deg, i=1,
                           *, contact: ContactAnalysis = ContactAnalysis.NONE,
                           e1=None, e2=None, nu1=None, nu2=None,) -> dict[str, Any]:
        if i not in self.REDUCTION_FACTOR_BY_ROWS:
            raise ValueError(f"SelfAligningBallFamily: i must be one of "
                              f"{sorted(self.REDUCTION_FACTOR_BY_ROWS)}, got {i}")

        A            = ri + re - Dw
        alpha_0, s   = _geo.contact_angle_and_clearance(A, alpha_0_deg=alpha_0_deg)
        gamma_ref = Dw * np.cos(alpha_0) / Dpw

        ri, re = self.reference_raceway_radii(Dw, gamma_ref)
        if ri <= Dw / 2.0 or re <= Dw / 2.0:
            raise ValueError(f"SelfAligningBallFamily: invalid groove geometry for Dw={Dw}")
        if not (0.0 < alpha_0_deg <= 45.0):
            raise ValueError(f"SelfAligningBallFamily: alpha_0_deg must be in (0, 45], got {alpha_0_deg}")

        ball_geo     = _geo.BallBearingGeometry(Dw=Dw, Dpw=Dpw, A=A, s=s, ri=ri, re=re, alpha_0=alpha_0)
        gamma        = ball_geo.gamma
        P_e          = ball_geo.free_end_play
        f_i          = ball_geo.f_i
        f_o          = ball_geo.f_o
        r_rolling_el = [ball_geo.r_ax, ball_geo.r_ay]
        r_inner      = [ball_geo.r_bx_inner, ball_geo.r_by_inner]
        r_outer      = [ball_geo.r_bx_outer, ball_geo.r_by_outer]

        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        result = dict( ri=ri, re=re, Dw=Dw, Dpw=Dpw, gamma=gamma,
                       Z=Z, s=s, A=A, alpha_0=alpha_0, phi_j=phi_j,
                       P_e=P_e, f_i=f_i, f_o=f_o, r_rolling_el=r_rolling_el,
                       r_inner=r_inner, r_outer=r_outer, i=i,
                       reduction_factor=self.REDUCTION_FACTOR_BY_ROWS[i])

        if contact is ContactAnalysis.NONE:
            return result

        e1, e2 = _resolve_pair(e1, e2, "e1", "e2")
        nu1, nu2 = _resolve_pair(nu1, nu2, "nu1", "nu2")
        result.update(e1=e1, e2=e2, nu1=nu1, nu2=nu2)

        if contact in (ContactAnalysis.ISO16281):
            point_stiff = iso.PointContactStiffness(Dw, Dpw, gamma, ri, re, alpha_0, e1, e2, nu1, nu2)
            cp = point_stiff.cp
            Ri = point_stiff.raceway_contact_radius
            result.update(iso16281_analysis=True, cp=cp, Ri=Ri)

        return result

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
@point_contact
@thrust
class ThrustBallSingleRowFamily(BearingFamily):

    RI_OVER_DW = 0.535
    RE_OVER_DW = 0.535
    LAM = 0.90

    @property
    def name(self) -> str:
        return "thrust_ball_single_row"

    @staticmethod
    def eta(alpha_0: float) -> float:
        return 1.0 - np.sin(alpha_0) / 3.0

    @classmethod
    def reference_raceway_radii(cls, Dw: float) -> tuple[float, float]:
        return cls.RI_OVER_DW * Dw, cls.RE_OVER_DW * Dw

    def assemble_geometry(self, catalog, Dw, Dpw, Z, alpha_0_deg=None, i=1,
                           *, contact: ContactAnalysis = ContactAnalysis.NONE,
                           e1=None, e2=None, nu1=None, nu2=None,) -> dict[str, Any]:
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
                raise ValueError(f"ThrustBallFamily: alpha_0_deg must be in (45, 90], got {alpha_0_deg}")
            alpha_0, s = _geo.contact_angle_and_clearance(A, alpha_0_deg=alpha_0_deg)


        ball_geo = _geo.BallBearingGeometry(Dw=Dw, Dpw=Dpw, A=A, s=s, ri=ri, re=re, alpha_0=alpha_0)
        gamma        = ball_geo.gamma
        P_e          = ball_geo.free_end_play
        f_i          = ball_geo.f_i
        f_o          = ball_geo.f_o
        r_rolling_el = [ball_geo.r_ax, ball_geo.r_ay]
        r_inner      = [float("inf"), ball_geo.r_by_inner]
        r_outer      = [float("inf") , ball_geo.r_by_outer]

        eta = self.eta(alpha_0)
        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)
        

        result = dict( ri=ri, re=re, Dw=Dw, Dpw=Dpw, gamma=gamma,
                       Z=Z, s=s, A=A, alpha_0=alpha_0, eta=eta, phi_j=phi_j,
                       P_e=P_e, f_i=f_i, f_o=f_o, r_rolling_el=r_rolling_el,
                       r_inner=r_inner, r_outer=r_outer, i=i,
                       lam=self.LAM)

        if contact is ContactAnalysis.NONE:
            return result

        e1, e2 = _resolve_pair(e1, e2, "e1", "e2")
        nu1, nu2 = _resolve_pair(nu1, nu2, "nu1", "nu2")
        result.update(e1=e1, e2=e2, nu1=nu1, nu2=nu2)

        if contact in (ContactAnalysis.ISO16281):
            point_stiff = iso.PointContactStiffness(Dw, Dpw, gamma, ri, re, alpha_0, e1, e2, nu1, nu2)
            cp = point_stiff.cp
            Ri = point_stiff.raceway_contact_radius
            result.update(iso16281_analysis=True, cp=cp, Ri=Ri)

        return result

    @staticmethod
    def _capacity_class(alpha_0: float) -> type[bcap.ThrustCapacityCalculator]:
        if np.isclose(alpha_0, np.pi / 2):
            return bcap.PointContactCapacityThrust_90deg
        return bcap.PointContactCapacityThrust_Non_90deg

    @staticmethod
    def _capacity_calculator(row_or_bearing, i: int = 1) -> bcap.ThrustCapacityCalculator:
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
@point_contact
@thrust
class ThrustBallMultiRowFamily(BearingFamily):

    _single_row_family = ThrustBallSingleRowFamily()

    @property
    def name(self) -> str:
        return "thrust_ball_multirow"

    def assemble_geometry(self, catalog, rows: list[dict]) -> dict[str, Any]:
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
        return dict(rows=assembled_rows,
                    Ca=Ca_total, Q_elements=[c.Q_elements for c in calculators])

    @staticmethod
    def dynamic_capacity(bearing):
        return bearing.Ca

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca=None):
        return bearing.Q_elements


# =====================================================================
# ---- roller bearing / radial, line contact -----------------
# =====================================================================

@register_family
@line_contact
class CylindricalRollerFamily(BearingFamily):

    LAMBDA_V = 0.83
    _LOG_ARG_EPS = 1e-12
    CROWNING = 100          # this is the value for r_ay/d

    @property
    def name(self) -> str:
        return "cylindrical_roller"

    def assemble_geometry(self, catalog, Dwe, Lwe, Dpw, Z, s, n_s, i=1, *,
                           contact: ContactAnalysis = ContactAnalysis.NONE,
                           e1=None, e2=None, nu1=None, nu2=None,) -> dict[str, Any]:
        """
        Cylindrical roller bearings have a contact angle of zero 

        there is the input for the material properties, however, if those are not 
        both steel then iso 16281 is not applicable.
            It will be implemented other load distribution solver in the future
            and verifications here will be done after improving materials/ as well
        """
        if catalog.arrangement not in ("floating", "non-locating"):
            raise ValueError(
                f"CylindricalRollerFamily(label={catalog.label or catalog.designation!r}): "
                f"arrangement={catalog.arrangement!r} is not valid -- an NU/N-type "
                f"bearing has no flange to react axial load.")
        if n_s < 30:
            raise ValueError(f"CylindricalRollerFamily: n_s must be >= 30, got {n_s}")
        if s < 0:
            raise ValueError(f"CylindricalRollerFamily: s must be >= 0, got {s}")
        if i < 1:
            raise ValueError(f"CylindricalRollerFamily: i (number of rows) must be >= 1, got {i}")

        roller_geo   = _geo.CylindricalRollerBearingGeometry(Dwe=Dwe, Dpw=Dpw, Lwe=Lwe, n_s=n_s)
        gamma        = roller_geo.gamma
        x_k          = roller_geo.lamina_positions
        r_rolling_el = [roller_geo.r_ax, roller_geo.r_ay]
        r_inner      = [roller_geo.r_bx_inner, roller_geo.r_by_inner]
        r_outer      = [roller_geo.r_bx_outer, roller_geo.r_by_outer]
        phi_j        = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        result = dict(Dwe=Dwe, Lwe=Lwe, Dpw=Dpw,
                    Z=Z, s=s, n_s=n_s, x_k=x_k, phi_j=phi_j, r_rolling_el=r_rolling_el, alpha_0=0.0,
                    r_inner=r_inner, r_outer=r_outer, gamma=gamma, lambda_v=self.LAMBDA_V, i=i)

        if contact is ContactAnalysis.NONE:
            return result

        e1, e2 = _resolve_pair(e1, e2, "e1", "e2")
        nu1, nu2 = _resolve_pair(nu1, nu2, "nu1", "nu2")
        result.update(e1=e1, e2=e2, nu1=nu1, nu2=nu2)

        if contact in (ContactAnalysis.ISO16281):
            line_stiff = iso.LineContactStiffness(Lwe=Lwe, n_s=n_s, Dwe=Dwe, x_k=x_k, _LOG_ARG_EPS=self._LOG_ARG_EPS)
            cL = line_stiff.cl
            cs = line_stiff.cs
            P_xk = line_stiff.reference_roller_profile
            result.update(iso16281_analysis=True, cL=cL, cs=cs, P_xk=P_xk)

        return result
    
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
# ---- roller bearing / thrust, Line contact --------------------------
# =====================================================================

@register_family
@line_contact
@thrust
class ThrustCylindricalRollerFamily(BearingFamily):

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
        eta = self.eta(alpha_0)
        stiff = iso.LineContactStiffness(Dwe, Dpw, alpha_0, Lwe, n_s)
        g = stiff.gamma
        x_k = stiff.lamina_positions
        cL = stiff.cl
        cs = stiff.cs
        P_xk = self._reference_roller_profile(x_k, Dwe, Lwe)
        phi_j = np.linspace(0, 2 * np.pi, Z, endpoint=False)

        return dict(Dwe=Dwe, Lwe=Lwe, Dpw=Dpw,
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
@line_contact
@thrust
class RollerThrustMultiRowFamily(BearingFamily):

    _single_row_family = ThrustCylindricalRollerFamily()

    @property
    def name(self) -> str:
        return "thrust_cylindrical_roller_multirow"

    def assemble_geometry(self, catalog, rows: list[dict]) -> dict[str, Any]:
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
        Ca_total = cls_.combine_multirow(capacity_kwargs_rows)
        return dict(rows=assembled_rows,
                    Ca=Ca_total, Q_elements=[c.Q_elements for c in calculators])

    @staticmethod
    def dynamic_capacity(bearing):
        return bearing.Ca

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca=None):
        return bearing.Q_elements