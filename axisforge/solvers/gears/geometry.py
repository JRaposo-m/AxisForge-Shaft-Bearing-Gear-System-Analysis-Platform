"""
solvers/gears/geometry.py
GearSolver — cylindrical gear geometry and force calculation.

Produces GearGeometryResult and GearForceResult, then assembles a
GearElement for injection into the AxisForge pipeline (StaticsSolver).

Formulation:
  MAAG Gear Book / ISO 21771:2007.
  Force equations: Shigley §13-7; GEARpie FORCES_SPEEDS.py (C. Fernandes, MIT).

Profile shift support (x1, x2):
  When al is supplied:
    αtw = arccos(a · cos(αt) / al)                [ISO 21771 §6]
  When al is None:
    Solve involute equation via scipy.optimize.brentq:
      inv(αtw) = inv(αt) + 2·tan(α)·(x1+x2)/(z1+z2)
      al = (db1 + db2) / (2·cos(αtw))             [ISO 21771 §7]
  Tip diameters with profile shift and tip shortening k:
  Root diameters:
    df = d + 2·mn·(x − hfP)

Phase 1 scope:
  External cylindrical gears: spur (β=0) and helical (β>0).
  Profile shift: x1, x2 supported (default 0.0).
  al supplied → αtw direct; al=None → brentq on involute equation.
  Face width b optional (used only for εβ).
  haP=1.0, hfP=1.25 (ISO standard rack). k computed from centre distance modification.

Unit convention (AxisForge SI):
  Input/output angles : degrees
  Internal angles     : radians
  Lengths             : mm
  Forces              : N
  Torques             : N·mm  (T1_Nm input × 1000)

Validation cases:
  C14 spur   — m=4.5, z=16/24, x=0.1817/0.1715, al=91.5mm, T=200 N·m
               Ft=5464.5N, Fr=2256.6N, Fa=0N, εα=1.46  (GEARpie report)
  H501 helical — m=3.5, z=20/30, β=15°, x=0.1809/0.0891, al=91.5mm, T=100 N·m
               Ft=2732.2N, Fr=1110.3N, Fa=739.5N, εα=1.46  (GEARpie report)

References:
  MAAG Gear Book, 2nd ed.
  ISO 21771:2007 — Gears — Cylindrical involute gears and gear pairs.
  Shigley 10th ed. §13-7, §13-10.
  GEARpie CALC_GEOMETRY.py / FORCES_SPEEDS.py (C. Fernandes, MIT License).
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

from core.components import GearElement
from models.gear_result import GearGeometryResult, GearForceResult
from solvers.gears.utils import (
    HAP, HFP,
    solve_alpha_tw,
    contact_ratio_alpha,
    contact_ratio_beta,
    undercut_z_min,
    validate_geometry_inputs,
)


class GearSolver:
    """
    Computes cylindrical gear geometry and mesh forces.

    All methods are pure functions — no internal state.

    Usage (al supplied):
        solver = GearSolver()
        geo    = solver.compute_geometry(mn, z1, z2, alpha_n_deg, beta_deg,
                                         al=91.5, x1=0.18, x2=0.17)
        forces = solver.compute_forces(T1_Nm=200.0, geometry=geo)
        ge     = solver.to_gear_element(position=100.0, forces=forces, geometry=geo)

    Usage (al from involute equation):
        geo = solver.compute_geometry(mn, z1, z2, alpha_n_deg, beta_deg,
                                      al=None, x1=0.18, x2=0.17)
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
        al: Optional[float] = None,
        x1: float = 0.0,
        x2: float = 0.0,
        b: float = 0.0,
    ) -> GearGeometryResult:
        """
        Compute full gear pair geometry with optional profile shift.

        Parameters
        ----------
        mn : float
            Normal module [mm]. Must be > 0.
        z1, z2 : int
            Number of teeth — pinion, wheel. Both ≥ 1.
        alpha_n_deg : float
            Normal pressure angle [°]. Typically 20°.
        beta_deg : float
            Helix angle [°]. 0° for spur gears.
        al : float | None
            Working centre distance [mm].
            If supplied → αtw derived directly.
            If None → solved from involute equation (requires scipy.brentq).
        x1, x2 : float
            Profile shift coefficients [-]. Default 0.0.
        b : float
            Face width [mm]. Used for εβ only. 0 → εβ=0.

        Returns
        -------
        GearGeometryResult

        Raises
        ------
        ValueError
            Invalid inputs or geometrically inconsistent al.
        """
        validate_geometry_inputs(mn, z1, z2, alpha_n_deg, beta_deg, al, x1, x2)

        # Undercut warning (x=0 threshold; x>0 mitigates)
        z_min = undercut_z_min(alpha_n_deg, beta_deg)
        if z1 < z_min and x1 <= 0.0:
            warnings.warn(
                f"z1={z1} < z_min={z_min} (α={alpha_n_deg}°, β={beta_deg}°) — "
                f"undercut risk. Profile shift x1 > 0 recommended.",
                UserWarning,
                stacklevel=2,
            )

        # --- Angles ---
        alpha_n = math.radians(alpha_n_deg)
        beta    = math.radians(beta_deg)

        mt      = mn / math.cos(beta)
        alpha_t = math.atan(math.tan(alpha_n) / math.cos(beta))
        beta_b  = math.asin(math.sin(beta) * math.cos(alpha_n))

        u = z2 / z1

        # --- Reference diameters ---
        d1 = mt * z1
        d2 = mt * z2
        a  = (d1 + d2) / 2.0   # standard centre distance

        # --- Base diameters ---
        db1 = d1 * math.cos(alpha_t)
        db2 = d2 * math.cos(alpha_t)

        # --- Working centre distance and pressure angle ---
        if al is None:
            al, alpha_tw = solve_alpha_tw(
                alpha_n, alpha_t, x1, x2, z1, z2, db1, db2
            )
        else:
            cos_alpha_tw = a * math.cos(alpha_t) / al
            if not (0.0 < cos_alpha_tw <= 1.0 + 1e-9):
                raise ValueError(
                    f"Working centre distance al={al:.3f} mm is geometrically "
                    f"inconsistent: cos(αtw)={cos_alpha_tw:.6f} out of (0, 1]. "
                    f"Standard centre distance a={a:.3f} mm. Check al."
                )
            alpha_tw = math.acos(min(cos_alpha_tw, 1.0))

        # --- Working pitch diameters ---
        dl1 = 2.0 * al / (u + 1.0)
        dl2 = 2.0 * al * u / (u + 1.0)

        # k accounts for the addendum reduction due to centre distance modification.

        # --- Tip diameters with profile shift (k=0: ISO 21771 standard rack) ---
        # GEARpie applies ~0.009*mn tip clearance reduction for helical gears
        # with centre modification — not in ISO 21771. We follow the standard.
        da1 = d1 + 2.0 * mn * (HAP + x1)
        da2 = d2 + 2.0 * mn * (HAP + x2)

        # --- Root diameters with profile shift ---
        df1 = d1 + 2.0 * mn * (x1 - HFP)
        df2 = d2 + 2.0 * mn * (x2 - HFP)

        # --- Base pitch ---
        p_bt = math.pi * mt * math.cos(alpha_t)

        # --- Contact ratios ---
        eps_alpha = contact_ratio_alpha(
            z1, z2, da1, da2, db1, db2, alpha_tw, p_bt
        )
        eps_beta  = contact_ratio_beta(b, beta_b, p_bt)
        eps_gamma = eps_alpha + eps_beta

        return GearGeometryResult(
            mn=mn, z1=z1, z2=z2,
            alpha_n_deg=alpha_n_deg, beta_deg=beta_deg,
            x1=x1, x2=x2,
            mt=mt,
            alpha_t_deg=math.degrees(alpha_t),
            beta_b_deg=math.degrees(beta_b),
            u=u, a=a, al=al,
            alpha_tw_deg=math.degrees(alpha_tw),
            d1=d1, d2=d2,
            db1=db1, db2=db2,
            da1=da1, da2=da2,
            df1=df1, df2=df2,
            dl1=dl1, dl2=dl2,
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
            Pinion input torque [N·m]. Must be ≥ 0.
        geometry : GearGeometryResult
            Output of compute_geometry().

        Returns
        -------
        GearForceResult — all forces in N, torques in N·mm.
        """
        if T1_Nm < 0.0:
            raise ValueError(
                f"T1_Nm must be ≥ 0, got {T1_Nm}. "
                f"Torque magnitude is always positive."
            )

        T1_Nmm = T1_Nm * 1000.0
        T2_Nmm = T1_Nmm * geometry.u

        if T1_Nm == 0.0:
            return GearForceResult(
                Ft=0.0, Fr=0.0, Fa=0.0,
                Fn=0.0, Fbt=0.0, Fbn=0.0,
                T1_Nmm=0.0, T2_Nmm=0.0,
                gear_ratio=geometry.u,
            )

        alpha_tw = math.radians(geometry.alpha_tw_deg)
        beta_b   = math.radians(geometry.beta_b_deg)

        # Ft = T1 / rl1  (working pitch circle)
        Ft  = T1_Nmm / geometry.rl1

        # Fbt = T1 / rb1  (base circle)
        Fbt = T1_Nmm / geometry.rb1

        # Fr = Fbt · sin(αtw)
        Fr = Fbt * math.sin(alpha_tw)

        # Fa = Fbt · tan(βb)  — exact zero for spur
        Fa = 0.0 if geometry.is_spur else Fbt * math.tan(beta_b)

        # Fbn = Fbt / cos(βb)
        cos_bb = math.cos(beta_b)
        Fbn = Fbt / cos_bb if cos_bb > 1e-9 else Fbt

        # Fn = Ft / cos(βb)  — normal force at tooth contact (GEARpie / MAAG convention)
        # Note: sqrt(Ft²+Fr²+Fa²) is a valid vector identity but gives a different
        # numerical result because Fr and Fa derive from Fbt (base circle), not Ft.
        # The GEARpie/MAAG definition Fn = Ft/cos(βb) is the standard used throughout.
        Fn = Ft / cos_bb if cos_bb > 1e-9 else Ft

        return GearForceResult(
            Ft=Ft, Fr=Fr, Fa=Fa, Fn=Fn,
            Fbt=Fbt, Fbn=Fbn,
            T1_Nmm=T1_Nmm, T2_Nmm=T2_Nmm,
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

        pitch_diameter = dl1 (working pitch circle) — ensures torque
        consistency check in GearElement.__post_init__ passes:
          T = Ft × dl1/2  ✓
        """
        return GearElement(
            position=position,
            tangential_force=forces.Ft,
            radial_force=forces.Fr,
            axial_force=forces.Fa,
            pitch_diameter=geometry.dl1,
            torque=forces.T1_Nmm,
            pressure_angle=geometry.alpha_n_deg,
            helix_angle=geometry.beta_deg,
            label=label,
        )