"""
solvers/gear_solver.py
GearSolver — cylindrical gear geometry and force calculation.

Produces GearGeometryResult and GearForceResult, then assembles a
GearElement for injection into the AxisForge pipeline (StaticsSolver).

Formulation:
  MAAG Gear Book / ISO 21771:2007.
  Force equations: Shigley §13-7; GEARpie FORCES_SPEEDS.py (C. Fernandes, MIT).

Phase 1 scope:
  - External cylindrical gears: spur (β=0) and helical (β>0).
  - Profile shift: x1 = x2 = 0 (no addendum modification).
  - al supplied by user → αtw derived directly (no iterative solver needed
    for x1=x2=0 case; brentq reserved for Phase 2 profile shift support).
  - Face width b optional (used only for εβ; if not supplied, εβ = 0).
  - haP = 1.0, hfP = 1.25 (ISO standard rack proportions).

Unit convention (AxisForge SI):
  Input/output angles : degrees
  Internal angles     : radians (suffix _rad)
  Lengths             : mm
  Forces              : N
  Torques             : N·mm  (T1_Nm input × 1000)

Restrictions:
  - No GUI imports. No database calls. No global state.
  - All methods pure — no side effects.
  - validate_or_raise() equivalent: ValueError on invalid inputs.

References:
  MAAG Gear Book, 2nd ed.
  ISO 21771:2007 — Gears — Cylindrical involute gears and gear pairs.
  Shigley 10th ed. §13-7.
  GEARpie CALC_GEOMETRY.py / FORCES_SPEEDS.py (C. Fernandes, MIT License).
"""
from __future__ import annotations

import math
import warnings

from core.components import GearElement
from models.gear_result import GearGeometryResult, GearForceResult

# Standard rack proportions (ISO 53 / DIN 867)
_HAP: float = 1.0    # addendum coefficient
_HFP: float = 1.25   # dedendum coefficient

# Undercut threshold for α=20°: zmin = 2/sin²(α) ≈ 17.1
_Z_UNDERCUT_THRESHOLD: int = 17


class GearSolver:
    """
    Computes cylindrical gear geometry and mesh forces.

    Usage:
        solver = GearSolver()
        geo    = solver.compute_geometry(mn, z1, z2, alpha_n_deg, beta_deg, al)
        forces = solver.compute_forces(T1_Nm, geo)
        ge     = solver.to_gear_element(position, forces, geo, label)

    All methods are pure functions — no internal state.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_geometry(
        self,
        mn: float,
        z1: int,
        z2: int,
        alpha_n_deg: float,
        beta_deg: float,
        al: float,
        b: float = 0.0,
    ) -> GearGeometryResult:
        """
        Compute full gear pair geometry.

        Parameters
        ----------
        mn : float
            Normal module [mm]. Must be > 0.
        z1, z2 : int
            Number of teeth on pinion and wheel. Both must be ≥ 1.
        alpha_n_deg : float
            Normal pressure angle [°]. Typically 20°.
        beta_deg : float
            Helix angle [°]. 0° for spur gears.
        al : float
            Working centre distance [mm]. Must be > 0.
        b : float
            Face width [mm]. Used only for εβ calculation. 0 = not supplied.

        Returns
        -------
        GearGeometryResult

        Raises
        ------
        ValueError
            On invalid inputs (mn ≤ 0, z ≤ 0, al ≤ 0, angle out of range).
        """
        self._validate_geometry_inputs(mn, z1, z2, alpha_n_deg, beta_deg, al)

        # Warn for undercut risk
        z_min = self._undercut_z_min(alpha_n_deg, beta_deg)
        if z1 < z_min:
            warnings.warn(
                f"z1={z1} < z_min={z_min} (α={alpha_n_deg}°, β={beta_deg}°) — "
                f"undercut risk. Profile shift (x > 0) recommended.",
                UserWarning,
                stacklevel=2,
            )

        # --- Convert angles to radians for internal use ---
        alpha_n = math.radians(alpha_n_deg)
        beta    = math.radians(beta_deg)

        # --- Derived angles (ISO 21771 §5) ---
        mt       = mn / math.cos(beta)
        alpha_t  = math.atan(math.tan(alpha_n) / math.cos(beta))
        # βb = arcsin(sin(β) * cos(αn))  — base helix angle
        beta_b   = math.asin(math.sin(beta) * math.cos(alpha_n))

        u = z2 / z1

        # --- Reference diameters ---
        d1 = mt * z1
        d2 = mt * z2

        # --- Standard centre distance ---
        a = (d1 + d2) / 2.0

        # --- Working pressure angle from supplied al ---
        # αtw = arccos( a * cos(αt) / al )
        cos_alpha_t = math.cos(alpha_t)
        cos_alpha_tw = a * cos_alpha_t / al
        if cos_alpha_tw > 1.0 or cos_alpha_tw < 0.0:
            raise ValueError(
                f"Working centre distance al={al:.3f} mm is geometrically "
                f"inconsistent: cos(αtw)={cos_alpha_tw:.6f} out of (0, 1]. "
                f"Standard centre distance a={a:.3f} mm. Check al."
            )
        alpha_tw = math.acos(cos_alpha_tw)

        # --- Working pitch diameters ---
        dl1 = 2.0 * al / (u + 1.0)
        dl2 = 2.0 * al * u / (u + 1.0)

        # --- Base diameters ---
        db1 = d1 * math.cos(alpha_t)
        db2 = d2 * math.cos(alpha_t)

        # --- Tip and root diameters (x1=x2=0, k=0 for no centre modification) ---
        # da = d + 2 * mn * haP
        # df = d - 2 * mn * hfP
        da1 = d1 + 2.0 * mn * _HAP
        da2 = d2 + 2.0 * mn * _HAP
        df1 = d1 - 2.0 * mn * _HFP
        df2 = d2 - 2.0 * mn * _HFP

        # --- Transverse base pitch ---
        p_bt = math.pi * mt * math.cos(alpha_t)

        # --- Contact ratios ---
        eps_alpha = self._contact_ratio_alpha(
            z1, z2, da1, da2, db1, db2, al, alpha_tw, p_bt
        )
        eps_beta  = self._contact_ratio_beta(b, beta_b, p_bt)
        eps_gamma = eps_alpha + eps_beta

        return GearGeometryResult(
            mn=mn,
            z1=z1,
            z2=z2,
            alpha_n_deg=alpha_n_deg,
            beta_deg=beta_deg,
            x1=0.0,
            x2=0.0,
            mt=mt,
            alpha_t_deg=math.degrees(alpha_t),
            beta_b_deg=math.degrees(beta_b),
            u=u,
            a=a,
            al=al,
            alpha_tw_deg=math.degrees(alpha_tw),
            d1=d1,
            d2=d2,
            db1=db1,
            db2=db2,
            da1=da1,
            da2=da2,
            df1=df1,
            df2=df2,
            dl1=dl1,
            dl2=dl2,
            p_bt=p_bt,
            eps_alpha=eps_alpha,
            eps_beta=eps_beta,
            eps_gamma=eps_gamma,
            b=b,
        )

    def compute_forces(
        self,
        T1_Nm: float,
        geometry: GearGeometryResult,
    ) -> GearForceResult:
        """
        Compute gear mesh forces from input torque and geometry.

        Parameters
        ----------
        T1_Nm : float
            Input torque on pinion [N·m]. Must be ≥ 0.
        geometry : GearGeometryResult
            Output of compute_geometry().

        Returns
        -------
        GearForceResult
            All forces in N, torques in N·mm.

        Raises
        ------
        ValueError
            If T1_Nm < 0.
        """
        if T1_Nm < 0.0:
            raise ValueError(
                f"T1_Nm must be ≥ 0, got {T1_Nm}. "
                f"Torque magnitude is always positive; direction is handled by sign convention."
            )

        # --- Unit conversion: N·m → N·mm ---
        T1_Nmm = T1_Nm * 1000.0
        T2_Nmm = T1_Nmm * geometry.u

        if T1_Nm == 0.0:
            return GearForceResult(
                Ft=0.0, Fr=0.0, Fa=0.0,
                Fn=0.0, Fbt=0.0, Fbn=0.0,
                T1_Nmm=0.0, T2_Nmm=0.0,
                gear_ratio=geometry.u,
            )

        # --- Internal angles in radians ---
        alpha_tw = math.radians(geometry.alpha_tw_deg)
        beta_b   = math.radians(geometry.beta_b_deg)

        # --- Tangential force at working pitch circle ---
        # Ft = T1 [N·mm] / rl1 [mm]  →  N
        rl1 = geometry.rl1
        Ft = T1_Nmm / rl1

        # --- Tangential force at base circle ---
        # Fbt = T1 / rb1
        rb1 = geometry.rb1
        Fbt = T1_Nmm / rb1

        # --- Radial force (ISO 21771 / GEARpie FORCES_SPEEDS) ---
        # Fr = Fbt * sin(αtw)
        Fr = Fbt * math.sin(alpha_tw)

        # --- Axial force ---
        # Fa = Fbt * tan(βb)   (= 0 exactly for spur, βb=0)
        if geometry.is_spur:
            Fa = 0.0
        else:
            Fa = Fbt * math.tan(beta_b)

        # --- Normal force at base plane ---
        # Fbn = Fbt / cos(βb)
        cos_beta_b = math.cos(beta_b)
        Fbn = Fbt / cos_beta_b if cos_beta_b > 1e-9 else Fbt

        # --- Resultant normal force at tooth ---
        # Fn = Ft / cos(βb)  (alternative: sqrt(Ft²+Fr²+Fa²))
        Fn = math.sqrt(Ft**2 + Fr**2 + Fa**2)

        return GearForceResult(
            Ft=Ft,
            Fr=Fr,
            Fa=Fa,
            Fn=Fn,
            Fbt=Fbt,
            Fbn=Fbn,
            T1_Nmm=T1_Nmm,
            T2_Nmm=T2_Nmm,
            gear_ratio=geometry.u,
        )

    def to_gear_element(
        self,
        position: float,
        forces: GearForceResult,
        geometry: GearGeometryResult,
        label: str = "",
    ) -> GearElement:
        """
        Assemble a GearElement for injection into MechanicalSystem.

        pitch_diameter = dl1 (working pitch diameter of pinion) so that
        GearElement.__post_init__ torque consistency check passes:
            T = Ft × dl1/2 = Ft × rl1 ✓

        Angles passed in degrees (GearElement convention).

        Parameters
        ----------
        position : float
            Axial coordinate of gear centre on shaft [mm].
        forces : GearForceResult
            Output of compute_forces().
        geometry : GearGeometryResult
            Output of compute_geometry().
        label : str
            Optional identifier.

        Returns
        -------
        GearElement
        """
        return GearElement(
            position=position,
            tangential_force=forces.Ft,
            radial_force=forces.Fr,
            axial_force=forces.Fa,
            pitch_diameter=geometry.dl1,   # working pitch diameter [mm]
            torque=forces.T1_Nmm,          # N·mm
            pressure_angle=geometry.alpha_n_deg,
            helix_angle=geometry.beta_deg,
            label=label,
        )

    # ------------------------------------------------------------------
    # Private: geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _contact_ratio_alpha(
        z1: int,
        z2: int,
        da1: float,
        da2: float,
        db1: float,
        db2: float,
        al: float,
        alpha_tw: float,
        p_bt: float,
    ) -> float:
        """
        Transverse contact ratio εα (ISO 21771 Eq. 62).

        εα = (εa1 + εa2)
        εa1 = z1 * (tan(αa1) - tan(αtw)) / (2π)
        εa2 = z2 * (tan(αa2) - tan(αtw)) / (2π)

        αa1 = arccos(db1/da1),  αa2 = arccos(db2/da2)
        """
        # Tip pressure angles
        cos_aa1 = db1 / da1
        cos_aa2 = db2 / da2
        # Clamp to valid arccos domain (guard for numerical noise)
        cos_aa1 = min(max(cos_aa1, -1.0), 1.0)
        cos_aa2 = min(max(cos_aa2, -1.0), 1.0)
        alpha_a1 = math.acos(cos_aa1)
        alpha_a2 = math.acos(cos_aa2)

        tan_atw = math.tan(alpha_tw)
        eps_a1 = z1 * (math.tan(alpha_a1) - tan_atw) / (2.0 * math.pi)
        eps_a2 = z2 * (math.tan(alpha_a2) - tan_atw) / (2.0 * math.pi)
        return eps_a1 + eps_a2

    @staticmethod
    def _contact_ratio_beta(b: float, beta_b: float, p_bt: float) -> float:
        """
        Overlap contact ratio εβ = b * tan(βb) / p_bt.
        Returns 0.0 if b=0 (not supplied) or β=0.
        """
        if b <= 0.0 or p_bt <= 0.0:
            return 0.0
        return b * math.tan(beta_b) / p_bt

    @staticmethod
    def _undercut_z_min(alpha_n_deg: float, beta_deg: float) -> int:
        """
        Minimum teeth to avoid undercut (x=0).
        zmin = 2 / sin²(αt)   (ISO 21771; Shigley §13-7)
        """
        alpha_n = math.radians(alpha_n_deg)
        beta    = math.radians(beta_deg)
        alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))
        sin_at  = math.sin(alpha_t)
        if sin_at < 1e-9:
            return 999  # degenerate — treat as always undercut
        # int() = floor: zmin=17.097 → 17, so z1=17 is safe (Shigley §13-10,
        # Tab. 13-11: minimum N_p = 17 for φ=20°).
        return int(2.0 / sin_at ** 2)

    # ------------------------------------------------------------------
    # Private: input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_geometry_inputs(
        mn: float,
        z1: int,
        z2: int,
        alpha_n_deg: float,
        beta_deg: float,
        al: float,
    ) -> None:
        if mn <= 0.0:
            raise ValueError(f"Normal module mn must be > 0, got {mn}")
        if z1 <= 0:
            raise ValueError(f"z1 must be ≥ 1, got {z1}")
        if z2 <= 0:
            raise ValueError(f"z2 must be ≥ 1, got {z2}")
        if not (0.0 < alpha_n_deg < 90.0):
            raise ValueError(
                f"alpha_n_deg must be in (0, 90), got {alpha_n_deg}"
            )
        if not (0.0 <= beta_deg < 90.0):
            raise ValueError(
                f"beta_deg must be in [0, 90), got {beta_deg}"
            )
        if al <= 0.0:
            raise ValueError(f"Working centre distance al must be > 0, got {al}")
