"""
solvers/bearing_life.py
BearingLifeSolver — ISO 281 rolling bearing life and static safety.

Formulation:
  Equivalent dynamic load (ISO 281 §7):
    P = X × Fr + Y × Fa
    Phase 1: X = 1.0, Y = 0.0  → P = Fr   (see L4.1)

  Basic rating life (ISO 281 §6):
    L10 = (C / P)^p    [10⁶ rev]
      p = 3.0    for ball bearings (DGBB, ACB)
      p = 10/3   for roller bearings (CR, TR, SR)

  Life in hours:
    L10h = L10 × 10⁶ / (60 × n_rpm)    [h]

  Equivalent static load (ISO 76):
    P0 = X0 × Fr + Y0 × Fa
    Phase 1: X0 = 0.6, Y0 = 0.5  (DGBB values — see L4.3)

  Static safety factor:
    S0 = C0 / P0

Axial force distribution (extract_bearing_forces):
  1 fixed, 1 floating:  fixed gets Fa_total, floating gets 0.
  2 fixed:              each gets Fa_total / 2  (equilibrium, same axis).
  0 fixed:              Fa = 0 for all, warnings.warn raised.

Phase 1 simplifications (Phase1_Limitations.md):
  L4.1: X=1, Y=0 — axial load ignored in dynamic equivalent load P.
  L4.2: No aISO modification factor — basic L10 only.
  L4.3: X0=0.6, Y0=0.5 fixed — DGBB values applied to all bearing types.
  L4.4: No load distribution between rolling elements.
  L4.5: No temperature or contamination correction.

Restrictions:
  - No GUI imports. No database calls. No global state.
  - All methods are pure functions of their arguments.
  - warnings.warn() for all engineering edge cases (never print).
  - system.validate_or_raise() called at entry of any method receiving system.

References:
  ISO 281:2007 — Rolling bearings — Dynamic load ratings and rating life.
  ISO 76:2006  — Rolling bearings — Static load ratings.
  SKF General Catalogue pub. 10000 EN, §2.1
"""
from __future__ import annotations

import math
import warnings

from core.components import Bearing, BearingType
from core.system import MechanicalSystem
from models.bearing_result import BearingLifeResult
from models.statics_result import StaticsResult
from config import DEFAULT_DESIGN_LIFE_HOURS, STATIC_SAFETY_FACTOR_MIN


# Phase 1 fixed load factors (documented limitation L4.1, L4.3)
_X_DYNAMIC: float = 1.0    # radial factor — dynamic equivalent load
_Y_DYNAMIC: float = 0.0    # axial factor — dynamic (axial ignored in Phase 1)
_X0_STATIC: float = 0.6    # radial factor — static equivalent load (DGBB)
_Y0_STATIC: float = 0.5    # axial factor — static equivalent load (DGBB)

# Guard: minimum equivalent load to prevent division by zero
_P_MIN_N: float = 1.0


class BearingLifeSolver:
    """
    Computes ISO 281 L10 life and static safety factor for rolling bearings.

    Usage:
        solver = BearingLifeSolver()

        # Single bearing:
        result = solver.solve_bearing(bearing, Fr=5000, Fa=0, speed_rpm=1450)

        # Full pipeline (reactions from StaticsSolver):
        forces = solver.extract_bearing_forces(statics_result, system)
        for label, (Fr, Fa) in forces.items():
            result = solver.solve_bearing(bearings[label], Fr, Fa, speed_rpm)

    The solver is stateless — all data flows through arguments and return values.
    """

    def solve_bearing(
        self,
        bearing: Bearing,
        Fr: float,
        Fa: float,
        speed_rpm: float,
        design_life_hours: float = DEFAULT_DESIGN_LIFE_HOURS,
    ) -> BearingLifeResult:
        """
        Compute L10 life and static safety factor for one bearing.

        Phase 1 simplifications (see module docstring):
          L4.1: P = Fr  (X=1, Y=0 — axial load not included in P)
          L4.2: Basic L10 only — no aISO factor
          L4.3: P0 = 0.6*Fr + 0.5*Fa — DGBB X0/Y0 for all types

        Args:
            bearing: Bearing with C, C0, bearing_type, label, position.
            Fr: Resultant radial force [N]. Must be ≥ 0.
            Fa: Axial force [N]. Must be ≥ 0.
            speed_rpm: Operating rotational speed [rpm]. 0 = static.
            design_life_hours: Target life for meets_life_target flag [h].

        Returns:
            BearingLifeResult with all life and safety fields populated.

        Raises:
            ValueError: If bearing has no catalogue data (C = 0 or C0 = 0).
        """
        # Guard: catalogue data required
        if not bearing.has_catalogue_data:
            raise ValueError(
                f"Bearing '{bearing.label}' has no catalogue data "
                f"(C={bearing.C}, C0={bearing.C0}). "
                f"Set C and C0 before calling solve_bearing()."
            )

        p = bearing.life_exponent

        # Equivalent dynamic load — Phase 1: P = Fr
        P_raw = _X_DYNAMIC * Fr + _Y_DYNAMIC * Fa
        if P_raw < _P_MIN_N:
            warnings.warn(
                f"Bearing '{bearing.label}': equivalent load P={P_raw:.3f} N is "
                f"below minimum guard ({_P_MIN_N} N). Clamping to {_P_MIN_N} N. "
                f"Check that Fr and Fa are correctly specified.",
                stacklevel=2,
            )
            P_raw = _P_MIN_N
        P = P_raw

        # C/P ratio and life
        C_over_P = bearing.C / P
        if C_over_P < 1.0:
            warnings.warn(
                f"Bearing '{bearing.label}': C/P = {C_over_P:.3f} < 1.0. "
                f"L10 < 1×10⁶ rev. Bearing is severely overloaded. "
                f"Consider a larger bearing or reduced load.",
                stacklevel=2,
            )

        L10 = C_over_P ** p  # [10⁶ rev]

        # L10h — life in hours
        if speed_rpm <= 0.0:
            warnings.warn(
                f"Bearing '{bearing.label}': speed_rpm={speed_rpm}. "
                f"Cannot compute L10h for zero or negative speed. "
                f"Returning L10h = inf (static application).",
                stacklevel=2,
            )
            L10h = float("inf")
        else:
            L10h = L10 * 1.0e6 / (60.0 * speed_rpm)

        # Equivalent static load — Phase 1: X0=0.6, Y0=0.5
        P0 = _X0_STATIC * Fr + _Y0_STATIC * Fa
        if P0 < _P_MIN_N:
            P0 = _P_MIN_N  # guard: avoid div/zero for S0

        # Static safety factor
        S0 = bearing.C0 / P0
        meets_static = S0 >= STATIC_SAFETY_FACTOR_MIN
        if not meets_static:
            warnings.warn(
                f"Bearing '{bearing.label}': S0 = {S0:.3f} < "
                f"{STATIC_SAFETY_FACTOR_MIN} (STATIC_SAFETY_FACTOR_MIN). "
                f"Static yield of raceways possible under peak load.",
                stacklevel=2,
            )

        meets_life = math.isfinite(L10h) and L10h >= design_life_hours

        return BearingLifeResult(
            bearing_label=bearing.label,
            bearing_position=bearing.position,
            Fr=Fr,
            Fa=Fa,
            P=P,
            P0=P0,
            L10=L10,
            L10h=L10h,
            S0=S0,
            C_over_P=C_over_P,
            life_target_hours=design_life_hours,
            meets_life_target=meets_life,
            meets_static_safety=meets_static,
            X=_X_DYNAMIC,
            Y=_Y_DYNAMIC,
            X0=_X0_STATIC,
            Y0=_Y0_STATIC,
            life_exponent=p,
            speed_rpm=speed_rpm,
        )

    def extract_bearing_forces(
        self,
        statics_result: StaticsResult,
        system: MechanicalSystem,
    ) -> dict[str, tuple[float, float]]:
        """
        Extract (Fr, Fa) for each bearing from StaticsResult reactions.

        Fr is computed from the planar reactions stored in statics_result:
          Fr = sqrt(R_xz² + R_xy²)

        Axial force distribution (Phase 1 — L4.4):
          1 fixed + 1 floating: fixed bearing carries full Fa_total.
          2 fixed:              each bearing carries Fa_total / 2.
          0 fixed:              Fa = 0 for all; warnings.warn raised.

        Fa_total = |reactions["axial"]|  (magnitude; sign is equilibrium convention)

        Args:
            statics_result: Output of StaticsSolver.solve() — must contain reactions.
            system: MechanicalSystem with bearings sorted by position.

        Returns:
            dict mapping bearing_label → (Fr [N], Fa [N]).

        Raises:
            ValueError: Via system.validate_or_raise() if system is invalid.
        """
        system.validate_or_raise()

        bearings = system.bearings  # sorted left → right
        reactions = statics_result.reactions

        # Identify reaction keys — bearings are sorted, so A=first, B=second
        # Reaction keys are positional: A_xz, A_xy, B_xz, B_xy
        labels_positional = ["A", "B"]

        forces: dict[str, tuple[float, float]] = {}

        for i, bearing in enumerate(bearings):
            pos_key = labels_positional[i]   # "A" or "B" positional key
            R_xz = reactions.get(f"{pos_key}_xz", 0.0)
            R_xy = reactions.get(f"{pos_key}_xy", 0.0)
            Fr = math.sqrt(R_xz ** 2 + R_xy ** 2)
            forces[bearing.label] = (Fr, 0.0)  # Fa to be set below

        # Axial load distribution
        Fa_total = abs(reactions.get("axial", 0.0))
        fixed_bearings = [b for b in bearings if b.arrangement == "fixed"]
        n_fixed = len(fixed_bearings)

        if Fa_total > 0.0:
            if n_fixed == 0:
                warnings.warn(
                    f"System has axial load ({Fa_total:.1f} N) but no fixed bearing. "
                    f"Axial load cannot be reacted. Setting Fa = 0 for all bearings. "
                    f"Add at least one bearing with arrangement='fixed'.",
                    stacklevel=2,
                )
                # Fa stays 0 for all — already set above
            elif n_fixed == 1:
                fixed = fixed_bearings[0]
                Fr_fixed, _ = forces[fixed.label]
                forces[fixed.label] = (Fr_fixed, Fa_total)
                # floating bearing: Fa = 0 (already set)
            else:
                # 2 fixed: split equally — equilibrium on collinear axis
                Fa_each = Fa_total / n_fixed
                for b in fixed_bearings:
                    Fr_b, _ = forces[b.label]
                    forces[b.label] = (Fr_b, Fa_each)

        return forces
