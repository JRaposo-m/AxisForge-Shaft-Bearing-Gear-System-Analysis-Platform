# =====================================================================
# families/family.py
# =====================================================================
"""Bearing families: geometry, contact constants and load ratings of each bearing type.

The concrete ``BearingFamily`` classes (deep groove, angular contact,
self-aligning, thrust ball single and multi row, cylindrical roller, thrust
cylindrical roller single and multi row), the registries and class decorators
that classify them, and the predicates the solvers use to read the
classification. A family turns the principal dimensions of a bearing into the
attributes stored on the ``Bearing`` (geometry from ``geometry.py``, ISO/TS 16281
contact constants from ``contact_models``) and evaluates the load ratings with
the calculators of ``capacity.py``.
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

# class name -> class; filled by the decorators below at class definition
_FAMILY_REGISTRY: dict[str, type["BearingFamily"]] = {}
_POINT_CONTACT: dict[str, type["BearingFamily"]] = {}
_LINE_CONTACT: dict[str, type["BearingFamily"]] = {}
_THRUST_BEARING: dict[str, type["BearingFamily"]] = {}

def register_family(cls: type["BearingFamily"]) -> type["BearingFamily"]:
    """Class decorator: add the family to the registry of all families (by class name)."""
    _FAMILY_REGISTRY[cls.__name__] = cls
    return cls

def point_contact(cls: type["BearingFamily"]) -> type["BearingFamily"]:
    """Class decorator: classify the family as point contact (ball bearing).

    The ISO/TS 16281 contact-solver dispatch selects the solver by this
    classification; ThrustBallSingleRowFamily carries it too and stores
    ``iso16281_analysis = True``, so it is matched with the point-contact
    solver, whose equilibrium ISO_TS_16281_2008, 4.2.2 prints for a radial
    ball bearing.
    """
    _POINT_CONTACT[cls.__name__] = cls
    return cls

def line_contact(cls: type["BearingFamily"]) -> type["BearingFamily"]:
    """Class decorator: classify the family as line contact (roller bearing).

    The ISO/TS 16281 contact-solver dispatch selects the solver by this
    classification; the thrust roller families carry it too, so they are
    matched with the line-contact (radial roller) solver.
    """
    _LINE_CONTACT[cls.__name__] = cls
    return cls

def thrust(cls: type["BearingFamily"]) -> type["BearingFamily"]:
    """Class decorator: classify the family as thrust (axial duty)."""
    _THRUST_BEARING[cls.__name__] = cls
    return cls


# ====================================================================
# ---- helpers -------------------------------------------------------
# ====================================================================

def is_point_contact_family(family: "BearingFamily") -> bool:
    """Tell whether the family class is classified as point contact (ball).

    Returns
    -------
    bool
        True if ``type(family).__name__`` is in the point-contact registry.
    """
    return type(family).__name__ in _POINT_CONTACT

def is_line_contact_family(family: "BearingFamily") -> bool:
    """Tell whether the family class is classified as line contact (roller).

    Returns
    -------
    bool
        True if ``type(family).__name__`` is in the line-contact registry.
    """
    return type(family).__name__ in _LINE_CONTACT

def is_thrust_family(family: "BearingFamily") -> bool:
    """Tell whether the family class is classified as thrust (axial duty).

    Returns
    -------
    bool
        True if ``type(family).__name__`` is in the thrust registry.
    """
    return type(family).__name__ in _THRUST_BEARING


# =====================================================================
# ---- ball bearing / radial, point contact --------------------------
# =====================================================================

@register_family
@point_contact
class DeepGrooveBallFamily(BearingFamily):
    """Single- and double-row radial deep groove ball bearing.

    The groove radii are the reference values, the contact angle follows from
    the diametral clearance ``s``, and the ratings come from
    PointContactCapacityRadial. Classified as point contact, radial duty.

    Attributes
    ----------
    RI_OVER_DW : float
        Inner groove radius over ball diameter, ri/Dw [-]: 0.52.
    RE_OVER_DW : float
        Outer groove radius over ball diameter, re/Dw [-]: 0.52.
    REDUCTION_FACTOR_BY_ROWS : dict[int, float]
        Reduction factor lambda by number of rows i [-]: {1: 0.95, 2: 0.90}.

    Notes
    -----
    Distance between the groove curvature centres A = ri + re - Dw [mm]
    (ISO_TS_16281_2008, 4.2.2, Eq. (13); Harris2007_vol1, 2.2.2, Eq. 2.6).
    The remaining geometry equations are those of ``contact_angle_and_clearance``
    and ``BallBearingGeometry``.

    Warnings
    --------
    Differs from ISO_TS_16281_2008, 6.3, Eq. (72)-(73), which give
    r_e = 0,53 Dw and r_i = 0,52 Dw for deep groove ball bearings. The code uses
    ri = re = 0.52 Dw and lambda = 0.95 (single row) / 0.90 (double row), the
    values of ISO_TR_1281, Table 1, i.e. the set of the capacity derivation.

    References
    ----------
    .. [1] ISO_TR_1281, Table 1 (groove radii and lambda).
    .. [2] ISO_TS_16281_2008, 6.3, Eq. (72)-(73).
    """

    RI_OVER_DW = 0.52
    RE_OVER_DW = 0.52
    REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.90}

    @property
    def name(self) -> str:
        """Family identifier, ``'deep_groove_ball'``."""
        return "deep_groove_ball"

    @classmethod
    def reference_raceway_radii(cls, Dw: float) -> tuple[float, float]:
        """Reference groove radii (ri, re) for a ball diameter Dw [mm].

        Returns
        -------
        tuple[float, float]
            (0.52 Dw, 0.52 Dw) [mm].
        """
        return cls.RI_OVER_DW * Dw, cls.RE_OVER_DW * Dw

    def assemble_geometry(self, catalog, Dw, Dpw, Z, s, i=1, *,
                           contact: ContactAnalysis = ContactAnalysis.NONE,
                           e1=None, e2=None, nu1=None, nu2=None,) -> dict[str, Any]:
        """Geometry, ISO/TS 16281 point-contact constants and rating inputs of a deep groove ball bearing.

        Parameters
        ----------
        catalog : BearingCatalog
            Not used by this family.
        Dw : float
            Ball diameter [mm].
        Dpw : float
            Pitch diameter of the ball set [mm].
        Z : int
            Number of balls per row [-].
        s : float
            Diametral internal clearance [mm], >= 0.
        i : int, optional
            Number of rows [-], 1 or 2.
        contact : ContactAnalysis, optional
            ``NONE`` stores geometry only; any other value requires the
            materials; ``ISO16281`` also stores the stiffness constants.
        e1, e2 : float, optional
            Modulus of elasticity of the rolling element (e1) and of the
            raceways, both rings (e2) [MPa]; not checked. If only one is given
            it is used for both.
        nu1, nu2 : float, optional
            Poisson's ratio of the rolling element (nu1) and of the raceways
            (nu2) [-]. If only one is given it is used for both.

        Returns
        -------
        dict
            Attributes mirrored onto the Bearing: ``ri``, ``re``, ``Dw``,
            ``Dpw``, ``A`` [mm]; ``gamma`` [-]; ``Z`` [-]; ``s`` [mm];
            ``alpha_0`` [rad]; ``phi_j`` angular positions of the balls,
            ``linspace(0, 2 pi, Z, endpoint=False)`` [rad]; ``P_e`` free end play
            [mm]; ``f_i``, ``f_o`` [-]; ``r_rolling_el`` = [r_ax, r_ay],
            ``r_inner`` = [r_bx_inner, r_by_inner], ``r_outer`` = [r_bx_outer,
            r_by_outer] [mm], signs as in BallBearingGeometry; ``i`` [-];
            ``reduction_factor`` lambda [-]. If ``contact`` is not ``NONE``
            also ``e1``, ``e2``, ``nu1``, ``nu2``; if ``ISO16281`` also
            ``iso16281_analysis = True``, ``cp`` spring constant of the ball
            contact [N/mm^1.5] and ``Ri`` distance between the centre of
            curvature of the inner raceway groove and the bearing axis [mm]
            (ISO_TS_16281_2008, 4.2.2, Eq. (14)).

        Raises
        ------
        ValueError
            If ``i`` is not 1 or 2, if a groove radius is not larger than
            Dw/2, if ``s`` < 0, or if neither of e1/e2 (or nu1/nu2) is given
            when ``contact`` is not ``NONE``.

        Notes
        -----
        ``e1``, ``nu1`` refer to the rolling element and ``e2``, ``nu2`` to
        the raceway material, inner and outer raceway alike. The test
        ``contact is ContactAnalysis.ISO16281`` selects the stiffness
        calculation.
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

        if contact is ContactAnalysis.ISO16281:
            point_stiff = iso.PointContactStiffness(Dw, Dpw, gamma, ri, re, alpha_0, e1, e2, nu1, nu2)
            cp = point_stiff.cp
            Ri = point_stiff.raceway_contact_radius
            result.update(iso16281_analysis=True, cp=cp, Ri=Ri)

        return result

    @staticmethod
    def per_element_dynamic_capacity(bearing, Cr=None):
        """Load capacities (Q_ci, Q_ce) of one ball at the inner and outer ring contact [N].

        Parameters
        ----------
        bearing : Bearing
            Assembled bearing of this family.
        Cr : float, optional
            Accepted and ignored: PointContactCapacityRadial computes its own
            Cr from the geometry.

        Returns
        -------
        tuple[float, float]
            (Q_ci, Q_ce) [N]; equations in PointContactCapacityRadial.
        """
        calc = bcap.PointContactCapacityRadial(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i)
        return calc.Q_elements

    @staticmethod
    def dynamic_capacity(bearing):
        """Basic dynamic radial load rating Cr of the bearing [N].

        ISO_TR_1281, 6.2, Formulae (13)-(15) via PointContactCapacityRadial;
        no rating factor b_m (see that class).
        """
        calc = bcap.PointContactCapacityRadial(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i)
        return calc.Cr


@register_family
@point_contact
class AngularContactFamily(BearingFamily):
    """Single- and double-row angular contact ball bearing.

    Same groove radii and ratings as DeepGrooveBallFamily; the contact angle
    is given and the diametral clearance follows from it. Classified as point
    contact, radial duty.

    Attributes
    ----------
    RI_OVER_DW : float
        Inner groove radius over ball diameter, ri/Dw [-]: 0.52.
    RE_OVER_DW : float
        Outer groove radius over ball diameter, re/Dw [-]: 0.52.
    REDUCTION_FACTOR_BY_ROWS : dict[int, float]
        Reduction factor lambda by number of rows i [-]: {1: 0.95, 2: 0.95}.

    Notes
    -----
    A = ri + re - Dw [mm] (ISO_TS_16281_2008, 4.2.2, Eq. (13)); s =
    2 A (1 - cos(alpha_0)) [mm] (``contact_angle_and_clearance``).

    Warnings
    --------
    Differs from ISO_TS_16281_2008, 6.3, Eq. (72)-(73) (r_e = 0,53 Dw,
    r_i = 0,52 Dw): ri = re = 0.52 Dw and lambda = 0.95 are the values of
    ISO_TR_1281, Table 1 (angular contact: 0,95 for single and double row).

    References
    ----------
    .. [1] ISO_TR_1281, Table 1.
    .. [2] ISO_TS_16281_2008, 6.3, Eq. (72)-(73).
    """

    RI_OVER_DW = 0.52
    RE_OVER_DW = 0.52
    REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.95}

    @property
    def name(self) -> str:
        """Family identifier, ``'angular_contact'``."""
        return "angular_contact"

    @classmethod
    def reference_raceway_radii(cls, Dw: float) -> tuple[float, float]:
        """Reference groove radii (ri, re) for a ball diameter Dw [mm]: (0.52 Dw, 0.52 Dw)."""
        return cls.RI_OVER_DW * Dw, cls.RE_OVER_DW * Dw

    def assemble_geometry(self, catalog, Dw, Dpw, Z, alpha_0_deg, i=1,
                           *, contact: ContactAnalysis = ContactAnalysis.NONE,
                           e1=None, e2=None, nu1=None, nu2=None,) -> dict[str, Any]:
        """Geometry, ISO/TS 16281 point-contact constants and rating inputs of an angular contact ball bearing.

        Same as DeepGrooveBallFamily.assemble_geometry, except that the
        initial contact angle is the input and the diametral clearance ``s``
        [mm] is computed from it.

        Parameters
        ----------
        catalog : BearingCatalog
            Not used by this family.
        Dw : float
            Ball diameter [mm].
        Dpw : float
            Pitch diameter of the ball set [mm].
        Z : int
            Number of balls per row [-].
        alpha_0_deg : float
            Initial contact angle [deg], in (0, 45].
        i : int, optional
            Number of rows [-], 1 or 2.
        contact : ContactAnalysis, optional
            As in DeepGrooveBallFamily.
        e1, e2, nu1, nu2 : float, optional
            Materials of the rolling element (1) and raceways (2), as in
            DeepGrooveBallFamily (e1, e2 in [MPa], not checked).

        Returns
        -------
        dict
            The same keys as DeepGrooveBallFamily.assemble_geometry.

        Raises
        ------
        ValueError
            If ``i`` is not 1 or 2, if a groove radius is not larger than
            Dw/2, if ``alpha_0_deg`` is outside (0, 45], or if e1/e2 (nu1/nu2)
            are both missing when ``contact`` is not ``NONE``.
        """
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
        """Load capacities (Q_ci, Q_ce) of one ball at the inner and outer ring contact [N].

        ``Cr`` is accepted and ignored; see DeepGrooveBallFamily. Returns
        (Q_ci, Q_ce) [N] from PointContactCapacityRadial.
        """
        calc = bcap.PointContactCapacityRadial(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i)
        return calc.Q_elements

    @staticmethod
    def dynamic_capacity(bearing):
        """Basic dynamic radial load rating Cr of the bearing [N].

        ISO_TR_1281, 6.2, Formulae (13)-(15) via PointContactCapacityRadial;
        no rating factor b_m (see that class).
        """
        calc = bcap.PointContactCapacityRadial(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i)
        return calc.Cr


@register_family
@point_contact
class SelfAligningBallFamily(BearingFamily):
    """Single- and double-row self-aligning ball bearing (spherical outer raceway).

    Inner groove radius 0.53 Dw and outer raceway radius 0.5 (1/gamma + 1) Dw
    with gamma = Dw cos(alpha_0)/Dpw. Classified as point contact, radial
    duty.

    Attributes
    ----------
    RI_OVER_DW : float
        Inner groove radius over ball diameter, ri/Dw [-]: 0.53.
    REDUCTION_FACTOR_BY_ROWS : dict[int, float]
        Reduction factor lambda by number of rows i [-]: {1: 1.0, 2: 1.0}.

    Notes
    -----
    Equations as coded::

        ri = 0.53 Dw                        [mm]
        re = 0.5 (1/gamma + 1) Dw           [mm],  gamma = Dw cos(alpha_0) / Dpw

    The groove radii equal ISO_TS_16281_2008, 6.6, Eq. (78)-(79) and
    ISO_TR_1281, Table 1 (lambda = 1).

    Warnings
    --------
    This family cannot assemble: ``assemble_geometry`` computes
    ``A = ri + re - Dw`` before ``ri`` and ``re`` are assigned, so it raises
    UnboundLocalError (xfail test in tests/test_family.py). The inputs do
    allow the order alpha_0 = radians(alpha_0_deg), gamma, (ri, re), A, s; the
    code does not follow it.

    gamma in ``re`` is computed with the initial angle alpha_0;
    ISO_TS_16281_2008, clause 3 defines gamma with the nominal contact angle
    alpha.

    For ISO16281 the code calls the plain ``PointContactStiffness`` (not
    reached); ``SelfAligningPointContactStiffness`` in
    contact_models/iso16281.py is a stub that no family builds.

    References
    ----------
    .. [1] ISO_TS_16281_2008, 6.6, Eq. (78)-(79).
    .. [2] ISO_TR_1281, Table 1.
    """

    RI_OVER_DW = 0.53
    REDUCTION_FACTOR_BY_ROWS = {1: 1.0, 2: 1.0}

    @property
    def name(self) -> str:
        """Family identifier, ``'self_aligning'``."""
        return "self_aligning"

    @classmethod
    def reference_raceway_radii(cls, Dw: float, gamma: float) -> tuple[float, float]:
        """Reference radii (ri, re) of the inner groove and of the spherical outer raceway [mm].

        Parameters
        ----------
        Dw : float
            Ball diameter [mm].
        gamma : float
            Dw cos(alpha_0)/Dpw [-], > 0.

        Returns
        -------
        tuple[float, float]
            (0.53 Dw, 0.5 (1/gamma + 1) Dw) [mm].

        Raises
        ------
        ValueError
            If ``gamma`` <= 0.
        """
        ri = cls.RI_OVER_DW * Dw
        if gamma <= 0.0:
            raise ValueError(f"SelfAligningBallFamily: gamma must be > 0, got {gamma}.")
        re = 0.5 * (1.0 / gamma + 1.0) * Dw
        return ri, re

    def assemble_geometry(self, catalog, Dw, Dpw, Z, alpha_0_deg, i=1,
                           *, contact: ContactAnalysis = ContactAnalysis.NONE,
                           e1=None, e2=None, nu1=None, nu2=None,) -> dict[str, Any]:
        """Geometry and rating inputs of a self-aligning ball bearing (cannot assemble, see the class Warnings).

        Intended to work as AngularContactFamily.assemble_geometry, with the
        groove radii of ``reference_raceway_radii``. As written it raises
        UnboundLocalError at ``A = ri + re - Dw``, before any result is built.

        Parameters
        ----------
        catalog : BearingCatalog
            Not used by this family.
        Dw : float
            Ball diameter [mm].
        Dpw : float
            Pitch diameter of the ball set [mm].
        Z : int
            Number of balls per row [-].
        alpha_0_deg : float
            Initial contact angle [deg], in (0, 45].
        i : int, optional
            Number of rows [-], 1 or 2.
        contact : ContactAnalysis, optional
            As in DeepGrooveBallFamily.
        e1, e2, nu1, nu2 : float, optional
            Materials, as in DeepGrooveBallFamily (e1, e2 in [MPa], not
            checked).

        Returns
        -------
        dict
            Same keys as DeepGrooveBallFamily.assemble_geometry (not reached,
            see above).
        """
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
        """Load capacities (Q_ci, Q_ce) of one ball at the inner and outer ring contact [N].

        ``Cr`` is accepted and ignored; see DeepGrooveBallFamily. Returns
        (Q_ci, Q_ce) [N] from PointContactCapacityRadial.
        """
        calc = bcap.PointContactCapacityRadial(
            Z=bearing.Z, Dw=bearing.Dw, alpha_0=bearing.alpha_0,
            ri=bearing.ri, re=bearing.re, gamma=bearing.gamma,
            reduction_factor=bearing.reduction_factor, i=bearing.i)
        return calc.Q_elements

    @staticmethod
    def dynamic_capacity(bearing):
        """Basic dynamic radial load rating Cr of the bearing [N].

        ISO_TR_1281, 6.2, Formulae (13)-(15) via PointContactCapacityRadial;
        no rating factor b_m (see that class).
        """
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
    """Single-row thrust ball bearing (axial duty), contact angle in (45, 90] deg.

    Ratings from PointContactCapacityThrust_90deg or _Non_90deg, chosen by the
    contact angle. Without a contact angle the bearing is taken at 90 deg with
    zero clearance. Classified as point contact and thrust.

    Attributes
    ----------
    RI_OVER_DW : float
        Inner (shaft washer) groove radius over ball diameter [-]: 0.535.
    RE_OVER_DW : float
        Outer (housing washer) groove radius over ball diameter [-]: 0.535.
    LAM : float
        Reduction factor lambda [-]: 0.90.

    Notes
    -----
    Equations as coded::

        A       = ri + re - Dw                    [mm]
        eta     = 1 - sin(alpha_0) / 3            [-]
        alpha_0 = pi/2, s = 0                     if alpha_0_deg is None

    Code names and symbols of ISO_TR_1281: LAM = lambda, eta = eta. The
    reduction factor eta is the printed 1 - sin(alpha)/3 evaluated with
    alpha_0 (Harris2007_vol1, Eq. 11.136 prints 1 - 0.33 sin(alpha)).

    Warnings
    --------
    Differs from ISO_TS_16281_2008, 6.8, Eq. (80)-(81), which give
    r_e = r_i = 0,54 Dw: ri = re = 0.535 Dw are the values of ISO_TR_1281,
    Table 1.

    ``r_inner`` and ``r_outer`` are stored with ``inf`` as the rolling-direction
    radius: exact only at alpha_0 = 90 deg.

    The ISO/TS 16281 contact-solver dispatch selects the solver by the
    point-contact classification; this family carries it too and stores
    ``iso16281_analysis = True``, so it is matched with the point-contact
    solver, whose equilibrium ISO_TS_16281_2008, 4.2.2 prints for a radial
    ball bearing.

    The number of rows ``i`` is stored, but ``dynamic_capacity`` and
    ``per_element_dynamic_capacity`` take their own ``i`` argument (default 1)
    and do not read ``bearing.i``; no thrust ball equation contains ``i``.

    References
    ----------
    .. [1] ISO_TR_1281, Table 1 (groove radii, lambda, eta).
    .. [2] ISO_TS_16281_2008, 6.8, Eq. (80)-(81).
    """

    RI_OVER_DW = 0.535
    RE_OVER_DW = 0.535
    LAM = 0.90

    @property
    def name(self) -> str:
        """Family identifier, ``'thrust_ball_single_row'``."""
        return "thrust_ball_single_row"

    @staticmethod
    def eta(alpha_0: float) -> float:
        """Reduction factor eta for the unequal load distribution among the balls [-].

        ``1 - sin(alpha_0) / 3`` with alpha_0 the initial contact angle [rad];
        ISO_TR_1281, Table 1.
        """
        return 1.0 - np.sin(alpha_0) / 3.0

    @classmethod
    def reference_raceway_radii(cls, Dw: float) -> tuple[float, float]:
        """Reference groove radii (ri, re) for a ball diameter Dw [mm]: (0.535 Dw, 0.535 Dw)."""
        return cls.RI_OVER_DW * Dw, cls.RE_OVER_DW * Dw

    def assemble_geometry(self, catalog, Dw, Dpw, Z, alpha_0_deg=None, i=1,
                           *, contact: ContactAnalysis = ContactAnalysis.NONE,
                           e1=None, e2=None, nu1=None, nu2=None,) -> dict[str, Any]:
        """Geometry, ISO/TS 16281 point-contact constants and rating inputs of a thrust ball bearing.

        Parameters
        ----------
        catalog : BearingCatalog
            Not used by this family.
        Dw : float
            Ball diameter [mm].
        Dpw : float
            Pitch diameter of the ball set [mm].
        Z : int
            Number of balls [-]; must be an integer.
        alpha_0_deg : float, optional
            Initial contact angle [deg], in (45, 90]. If None the bearing is
            taken at alpha_0 = 90 deg with s = 0.
        i : int, optional
            Number of rows [-]; stored, not used.
        contact : ContactAnalysis, optional
            As in DeepGrooveBallFamily.
        e1, e2, nu1, nu2 : float, optional
            Materials, as in DeepGrooveBallFamily (e1, e2 in [MPa], not
            checked).

        Returns
        -------
        dict
            The keys of DeepGrooveBallFamily.assemble_geometry, except that
            ``reduction_factor`` is replaced by ``eta`` [-] and ``lam`` [-];
            ``r_inner`` = [inf, r_by_inner] and ``r_outer`` = [inf,
            r_by_outer] [mm]. ``s`` = 2 A (1 - cos(alpha_0)) and ``P_e`` =
            sqrt(4 A s - s^2) [mm] are the formal values of the radial-bearing
            relations, not a clearance of the thrust bearing; they are 0 when
            ``alpha_0_deg`` is None and 2 A when ``alpha_0_deg`` = 90.

        Raises
        ------
        TypeError
            If ``Z`` is not an integer.
        ValueError
            If a groove radius is not larger than Dw/2, if ``alpha_0_deg`` is
            outside (45, 90], or if e1/e2 (nu1/nu2) are both missing when
            ``contact`` is not ``NONE``.
        """
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
        """Calculator class for a contact angle [rad]: the 90 deg one if alpha_0 is close to pi/2 (np.isclose), else the non-90 deg one."""
        if np.isclose(alpha_0, np.pi / 2):
            return bcap.PointContactCapacityThrust_90deg
        return bcap.PointContactCapacityThrust_Non_90deg

    @staticmethod
    def _capacity_calculator(row_or_bearing, i: int = 1) -> bcap.ThrustCapacityCalculator:
        """Capacity calculator built from an assembled row (dict) or a Bearing (attributes Z, Dw, alpha_0, ri, re, gamma, lam, eta)."""
        get = row_or_bearing.__getitem__ if isinstance(row_or_bearing, dict) else \
              (lambda k: getattr(row_or_bearing, k))
        cls_ = ThrustBallSingleRowFamily._capacity_class(get("alpha_0"))
        return cls_(Z=get("Z"), Dw=get("Dw"), alpha_0=get("alpha_0"),
                     ri=get("ri"), re=get("re"), gamma=get("gamma"),
                     lam=get("lam"), eta=get("eta"), i=i)

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca=None, i: int = 1):
        """Load capacities (Q_ci, Q_ce) of one ball at the inner and outer ring contact [N].

        Parameters
        ----------
        bearing : Bearing
            Assembled bearing of this family.
        Ca : float, optional
            Accepted and ignored: the calculator computes its own Ca.
        i : int, optional
            Number of rows [-]; passed to the calculator, which does not use it.

        Returns
        -------
        tuple[float, float]
            (Q_ci, Q_ce) [N]; ISO_TS_16281_2008, 4.3.1.3-4.3.1.4, Eq. (21)-(24)
            via PointContactCapacityThrust_*.
        """
        return ThrustBallSingleRowFamily._capacity_calculator(bearing, i=i).Q_elements

    @staticmethod
    def dynamic_capacity(bearing, i: int = 1):
        """Basic dynamic axial load rating Ca of the bearing [N].

        ISO_TR_1281, 6.3, Formulae (18)-(25) via PointContactCapacityThrust_*;
        no rating factor b_m (see those classes). ``i`` as in
        ``per_element_dynamic_capacity``.
        """
        return ThrustBallSingleRowFamily._capacity_calculator(bearing, i=i).Ca


@register_family
@point_contact
@thrust
class ThrustBallMultiRowFamily(BearingFamily):
    """Thrust ball bearing with several rows of balls.

    Each row is a ThrustBallSingleRowFamily; the dynamic axial rating of the
    bearing combines the single-row ratings. Classified as point contact and
    thrust.

    Notes
    -----
    Ca = (sum_k Z_k) [sum_k (Z_k / Ca_k)^(10/3)]^(-3/10) [N]; see
    ``_MultirowCombinablePointContact`` in capacity.py.

    References
    ----------
    .. [1] ISO_TR_1281, 6.4, Formula (29) (2021 edition numbering).
    """

    _single_row_family = ThrustBallSingleRowFamily()

    @property
    def name(self) -> str:
        """Family identifier, ``'thrust_ball_multirow'``."""
        return "thrust_ball_multirow"

    def assemble_geometry(self, catalog, rows: list[dict]) -> dict[str, Any]:
        """Per-row geometry and the combined dynamic axial rating of a multi-row thrust ball bearing.

        Parameters
        ----------
        catalog : BearingCatalog
            Passed to each single-row assembly.
        rows : list[dict]
            One dict of keyword arguments of
            ThrustBallSingleRowFamily.assemble_geometry per row; at least 2.

        Returns
        -------
        dict
            ``rows``: the assembled dict of each row; ``Ca``: combined dynamic
            axial rating [N]; ``Q_elements``: list of (Q_ci, Q_ce) [N], one per
            row.

        Raises
        ------
        ValueError
            If rows mix 90 deg and non-90 deg contact angles, or exactly one
            row is given.
        IndexError
            If ``rows`` is empty.
        """
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
        """Combined basic dynamic axial load rating Ca of the bearing [N], as stored at assembly."""
        return bearing.Ca

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca=None):
        """List of (Q_ci, Q_ce) [N], one per row, as stored at assembly. ``Ca`` is accepted and ignored."""
        return bearing.Q_elements


# =====================================================================
# ---- roller bearing / radial, line contact -----------------
# =====================================================================

@register_family
@line_contact
class CylindricalRollerFamily(BearingFamily):
    """Radial cylindrical roller bearing without flanges to react axial load (floating or non-locating).

    Straight rollers at zero contact angle; the ISO/TS 16281 line-contact
    constants and the lamina positions are stored for the load distribution,
    and the ratings come from LineContactCapacityRadial. Classified as line
    contact, radial duty.

    Attributes
    ----------
    LAMBDA_V : float
        Product lambda*nu of the radial roller bearing [-]: 0.83.
    CROWNING : int
        Value of r_ay/d noted in the source comment (100). Never read.

    Notes
    -----
    The laminae and the profile follow ISO_TS_16281_2008, 5.2.2 (at least
    ``n_s`` = 30 laminae) and 5.2.3. The contact angle stored is zero.

    References
    ----------
    .. [1] ISO_TR_1281, Table 2 (lambda*nu = 0,83 for radial roller bearings).
    .. [2] ISO_TS_16281_2008, 5.2.2, 5.2.3, Eq. (49).
    """

    LAMBDA_V = 0.83
    _LOG_ARG_EPS = 1e-12    # lower bound of the logarithm argument of the roller profile (not printed in the standard)
    CROWNING = 100          # this is the value for r_ay/d

    @property
    def name(self) -> str:
        """Family identifier, ``'cylindrical_roller'``."""
        return "cylindrical_roller"

    def assemble_geometry(self, catalog, Dwe, Lwe, Dpw, Z, s, n_s, i=1, *,
                           contact: ContactAnalysis = ContactAnalysis.NONE,
                           e1=None, e2=None, nu1=None, nu2=None,) -> dict[str, Any]:
        """Geometry, ISO/TS 16281 line-contact constants and rating inputs of a cylindrical roller bearing.

        The contact angle is zero. The ISO/TS 16281 line-contact constants hold
        for steel only (ISO_TS_16281_2008, 5.2.1, Eq. (35): LineContactStiffness
        takes no material); e1, e2, nu1, nu2 are stored on the bearing and not
        used.

        Parameters
        ----------
        catalog : BearingCatalog
            Its ``arrangement`` must be ``"floating"`` or ``"non-locating"``.
        Dwe : float
            Mean roller diameter [mm].
        Lwe : float
            Effective contact length of the roller [mm].
        Dpw : float
            Pitch diameter of the roller set [mm].
        Z : int
            Number of rollers per row [-].
        s : float
            Radial operating clearance measured on the diameter [mm], >= 0
            (the ISO/TS 16281 roller solver uses s/2). Stored; not used by the
            geometry or the ratings.
        n_s : int
            Number of laminae per roller [-], >= 30.
        i : int, optional
            Number of rows [-], >= 1.
        contact : ContactAnalysis, optional
            ``NONE`` stores geometry only; any other value requires the
            materials; ``ISO16281`` also stores the line-contact constants.
        e1, e2 : float, optional
            Modulus of elasticity [MPa]; stored on the bearing, not used by
            cL, cs, P_xk. If only one is given it is used for both.
        nu1, nu2 : float, optional
            Poisson's ratio of the rolling element and of the raceways [-].

        Returns
        -------
        dict
            ``Dwe``, ``Lwe``, ``Dpw`` [mm]; ``Z`` [-]; ``s`` [mm]; ``n_s`` [-];
            ``x_k`` lamina positions from the roller mid-length [mm];
            ``phi_j`` angular positions of the rollers,
            ``linspace(0, 2 pi, Z, endpoint=False)`` [rad]; ``r_rolling_el``,
            ``r_inner``, ``r_outer`` as in CylindricalRollerBearingGeometry
            [mm]; ``alpha_0`` = 0.0 [rad]; ``gamma`` [-]; ``lambda_v`` [-];
            ``i`` [-]. If ``contact`` is not ``NONE`` also ``e1``, ``e2``,
            ``nu1``, ``nu2``; if ``ISO16281`` also ``iso16281_analysis =
            True``, ``cL`` and ``cs`` [N/mm^(10/9)] and ``P_xk``, the
            reference roller profile at ``x_k`` [mm].

        Raises
        ------
        ValueError
            If the arrangement is not floating or non-locating, if ``n_s`` <
            30, ``s`` < 0 or ``i`` < 1, or if e1/e2 (nu1/nu2) are both missing
            when ``contact`` is not ``NONE``.

        Notes
        -----
        The test ``contact in (ContactAnalysis.ISO16281)`` is a substring test
        on the enum string, not a tuple membership test; for the two members
        it selects the ISO/TS 16281 calculation as ``contact is
        ContactAnalysis.ISO16281`` would.
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
        """Load capacities (Q_ci, Q_ce) of one roller at the inner and outer ring contact [N].

        ``Cr`` is accepted and ignored: LineContactCapacityRadial computes its
        own Cr. ISO_TS_16281_2008, 5.3.1.2, Eq. (47)-(48).
        """
        calc = bcap.LineContactCapacityRadial(
            Z=bearing.Z, Dwe=bearing.Dwe, Lwe=bearing.Lwe, alpha_0=bearing.alpha_0,
            gamma=bearing.gamma, lambda_v=bearing.lambda_v, n_s=bearing.n_s, i=bearing.i)
        return calc.Q_elements

    @staticmethod
    def dynamic_capacity(bearing):
        """Basic dynamic radial load rating Cr of the bearing [N].

        ISO_TR_1281, 6.5, Formulae (33)-(34) via LineContactCapacityRadial; no
        rating factor b_m (see that class).
        """
        calc = bcap.LineContactCapacityRadial(
            Z=bearing.Z, Dwe=bearing.Dwe, Lwe=bearing.Lwe, alpha_0=bearing.alpha_0,
            gamma=bearing.gamma, lambda_v=bearing.lambda_v, n_s=bearing.n_s, i=bearing.i)
        return calc.Cr

    @staticmethod
    def per_lamina_dynamic_capacity(bearing):
        """Load capacities (q_ci, q_ce) of one lamina of a roller [N].

        ISO_TS_16281_2008, 5.3.2, Eq. (56)-(57): Q * (1/n_s)^(7/9).
        """
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
    """Single-row thrust cylindrical roller bearing (axial duty), contact angle in (0, 90] deg.

    Ratings from LineContactCapacityThrust_90deg or _Non_90deg, chosen by the
    contact angle. Classified as line contact and thrust.

    Attributes
    ----------
    LAMBDA_V : float
        Product lambda*nu of the thrust roller bearing [-]: 0.73.

    Warnings
    --------
    ``assemble_geometry`` cannot complete: it builds ``iso.LineContactStiffness``
    with the positional values (Dwe, Dpw, alpha_0, Lwe, n_s), which that class
    takes as (Lwe, n_s, Dwe, x_k, _LOG_ARG_EPS), and then reads ``gamma`` and
    ``lamina_positions``, which do not exist: AttributeError (xfail test).

    The reference roller profile is coded again here
    (``_reference_roller_profile``) and duplicates
    ``LineContactStiffness.reference_roller_profile`` of iso16281.py.

    The family is classified as line contact, so the ISO/TS 16281
    contact-solver dispatch matches it with the line-contact (radial roller)
    solver.

    The reduction factor eta differs from the printed one (see ``eta``).

    References
    ----------
    .. [1] ISO_TR_1281, Table 2 (lambda*nu = 0,73, eta).
    """

    LAMBDA_V = 0.73
    _LOG_ARG_EPS = 1e-12

    @property
    def name(self) -> str:
        """Family identifier, ``'thrust_cylindrical_roller'``."""
        return "thrust_cylindrical_roller"

    @staticmethod
    def eta(alpha_0: float) -> float:
        """Reduction factor eta for the unequal load distribution among the rollers [-].

        ``1 - 0.5 sin(alpha_0)`` with alpha_0 the initial contact angle [rad].
        It feeds f_c and Ca of the thrust roller calculators.

        Warnings
        --------
        Differs from ISO_TR_1281, Table 2 and 7.2.7.1, which print
        eta = 1 - 0,15 sin(alpha); Harris2007_vol1, Eq. 11.161 prints
        1 - 0.15 sin(alpha). The code coefficient is 0.5.
        """
        return 1.0 - 0.5 * np.sin(alpha_0)

    @classmethod
    def _reference_roller_profile(cls, x_k: np.ndarray, Dwe: float, Lwe: float) -> np.ndarray:
        """Reference roller profile P(x_k): relief of the roller surface at the lamina positions [mm].

        Parameters
        ----------
        x_k : np.ndarray
            Lamina positions from the roller mid-length [mm].
        Dwe : float
            Mean roller diameter [mm].
        Lwe : float
            Effective contact length of the roller [mm].

        Returns
        -------
        np.ndarray
            P(x_k) [mm], same shape as ``x_k``.

        Notes
        -----
        Equations as coded (arg is bounded below by ``_LOG_ARG_EPS`` = 1e-12,
        a bound the standard does not print)::

            Lwe <= 2.5 Dwe:  P = 0.000350 Dwe ln(1 / (1 - (2 x_k / Lwe)^2))
            Lwe >  2.5 Dwe:  P = 0                                  for |x_k| <= (Lwe - 2.5 Dwe)/2
                             P = 0.000500 Dwe ln(1 / (1 - ((2 |x_k| - (Lwe - 2.5 Dwe)) / (2.5 Dwe))^2))
                                                                    otherwise

        References
        ----------
        .. [1] ISO_TS_16281_2008, 5.2.3, Eq. (42)-(44); 6.7.
        """
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
        """Geometry, line-contact constants and rating inputs of a thrust cylindrical roller bearing (cannot complete, see the class Warnings).

        Parameters
        ----------
        catalog : BearingCatalog
            Not used by this family.
        Dwe : float
            Mean roller diameter [mm].
        Lwe : float
            Effective contact length of the roller [mm].
        Dpw : float
            Pitch diameter of the roller set [mm].
        Z : int
            Number of rollers [-]; must be an integer.
        s : float
            Radial operating clearance measured on the diameter [mm], >= 0
            (the ISO/TS 16281 roller solver uses s/2). Stored; not used.
        n_s : int
            Number of laminae per roller [-], >= 30.
        alpha_0_deg : float
            Initial contact angle [deg], in (0, 90]; converted to [rad] here.
        i : int, optional
            Number of rows [-]; stored.

        Returns
        -------
        dict
            As intended: ``Dwe``, ``Lwe``, ``Dpw`` [mm]; ``Z`` [-]; ``s`` [mm];
            ``n_s`` [-]; ``alpha_0`` [rad]; ``x_k`` [mm]; ``phi_j`` [rad];
            ``gamma`` [-]; ``cL``, ``cs`` [N/mm^(10/9)]; ``P_xk`` [mm]; ``eta``
            [-]; ``lambda_v`` [-]; ``i`` [-]. Not reached, see the class
            Warnings.

        Raises
        ------
        TypeError
            If ``Z`` is not an integer.
        ValueError
            If ``n_s`` < 30, ``s`` < 0 or ``alpha_0_deg`` is outside (0, 90].
        AttributeError
            Always, after the input checks (see the class Warnings).
        """
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
        """Calculator class for a contact angle [rad]: the 90 deg one if alpha_0 is close to pi/2 (np.isclose), else the non-90 deg one."""
        if np.isclose(alpha_0, np.pi / 2):
            return bcap.LineContactCapacityThrust_90deg
        return bcap.LineContactCapacityThrust_Non_90deg

    @staticmethod
    def _capacity_calculator(row_or_bearing, i: int = 1) -> bcap.ThrustCapacityCalculator:
        """Capacity calculator built from an assembled row (dict) or a Bearing (attributes Z, Dwe, Lwe, alpha_0, gamma, lambda_v, eta)."""
        get = row_or_bearing.__getitem__ if isinstance(row_or_bearing, dict) else \
              (lambda k: getattr(row_or_bearing, k))
        cls_ = ThrustCylindricalRollerFamily._capacity_class(get("alpha_0"))
        return cls_(Z=get("Z"), Dwe=get("Dwe"), Lwe=get("Lwe"), alpha_0=get("alpha_0"),
                     gamma=get("gamma"), lambda_v=get("lambda_v"), eta=get("eta"), i=i)

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca=None, i: int = 1):
        """Load capacities (Q_ci, Q_ce) of one roller at the inner and outer ring contact [N].

        ``Ca`` is accepted and ignored; ``i`` is passed to the calculator,
        which does not use it. ISO_TS_16281_2008, 5.3.1.3-5.3.1.4, Eq. (50)-(55) via
        LineContactCapacityThrust_*.
        """
        return ThrustCylindricalRollerFamily._capacity_calculator(bearing, i=i).Q_elements

    @staticmethod
    def dynamic_capacity(bearing, i: int = 1):
        """Basic dynamic axial load rating Ca of the bearing [N].

        ISO_TR_1281, 6.6, Formulae (37)-(42) via LineContactCapacityThrust_*;
        no rating factor b_m (see those classes).
        """
        return ThrustCylindricalRollerFamily._capacity_calculator(bearing, i=i).Ca

    @staticmethod
    def per_lamina_dynamic_capacity(bearing, i: int = 1):
        """Load capacities (q_ci, q_ce) of one lamina of a roller [N].

        Raises AttributeError: the LineContactCapacityThrust_* calculators
        read ``n_s``, which they never receive.
        """
        return ThrustCylindricalRollerFamily._capacity_calculator(bearing, i=i).per_lamina


@register_family
@line_contact
@thrust
class RollerThrustMultiRowFamily(BearingFamily):
    """Thrust cylindrical roller bearing with several rows of rollers.

    Each row is a ThrustCylindricalRollerFamily; the dynamic axial rating
    combines the single-row ratings. Classified as line contact and thrust.

    Notes
    -----
    Ca = (sum_k Z_k Lwe_k) [sum_k (Z_k Lwe_k / Ca_k)^(9/2)]^(-2/9) [N]; see
    ``_MultirowCombinableLineContact`` in capacity.py.

    Warnings
    --------
    Each row is assembled with ThrustCylindricalRollerFamily.assemble_geometry,
    which cannot complete (see that class).

    References
    ----------
    .. [1] ISO_TR_1281, 6.7, Formula (46).
    """

    _single_row_family = ThrustCylindricalRollerFamily()

    @property
    def name(self) -> str:
        """Family identifier, ``'thrust_cylindrical_roller_multirow'``."""
        return "thrust_cylindrical_roller_multirow"

    def assemble_geometry(self, catalog, rows: list[dict]) -> dict[str, Any]:
        """Per-row geometry and the combined dynamic axial rating of a multi-row thrust roller bearing.

        Parameters
        ----------
        catalog : BearingCatalog
            Passed to each single-row assembly.
        rows : list[dict]
            One dict of keyword arguments of
            ThrustCylindricalRollerFamily.assemble_geometry per row; at least 2.

        Returns
        -------
        dict
            ``rows``: the assembled dict of each row; ``Ca``: combined dynamic
            axial rating [N]; ``Q_elements``: list of (Q_ci, Q_ce) [N], one per
            row.

        Raises
        ------
        ValueError
            If rows mix 90 deg and non-90 deg contact angles, or exactly one
            row is given.
        IndexError
            If ``rows`` is empty.
        """
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
        """Combined basic dynamic axial load rating Ca of the bearing [N], as stored at assembly."""
        return bearing.Ca

    @staticmethod
    def per_element_dynamic_capacity(bearing, Ca=None):
        """List of (Q_ci, Q_ce) [N], one per row, as stored at assembly. ``Ca`` is accepted and ignored."""
        return bearing.Q_elements