"""PLACEHOLDER - not reviewed by José.

Slot for the load-capacity data of a cylindrical gear pair that depend on the element alone (the
pair, its materials, its tool): the factors of ISO 6336 that need no torque, no speed, no lubricant
state and no number of cycles. Nothing is implemented: every function raises
``NotImplementedError``. No formula, table or constant of a standard is in this file.

Clause map (clause level only, unchecked: no equation was transcribed; vault note
``standards_by_quantity.md``, section C2-1):

- ISO_6336-1_2019: 5.3, Table 4 (guide values of K_A; checked); 6.6, Table 11 (Method C); clause 9
  (tooth and mesh stiffness c', c_gamma; clause level); Table 2, footnote a (a, d, d_a, z_2
  negative for internal gearing, whereas ``CylindricalFamily`` keeps them positive: finding I9).
- ISO_6336-2_2006: 6.1 Eq. (16) Z_H; 6.2 Eq. (17)-(18) Z_B, Z_D; clause 7 Eq. (19)-(22), Table 1
  Z_E; 8.1.2 Eq. (24)-(26) Z_eps; clause 9 Eq. (36) Z_beta; 12.3.1.3 Eq. (44)-(51) Z_R.
- ISO_6336-3_2006: clause 6 Eq. (9)-(31) Y_F (one fixed-point iteration, Eq. (14); needs the basic
  rack or tool, and the contact ratio of the PAIR); 7.2 Eq. (36)-(38) Y_S; 8.2 Eq. (40) Y_beta;
  9.2 Eq. (41)-(44) Y_B; 10.2 Eq. (45)-(47) Y_DT.
- ISO_6336-5_2016: 5.5, Formula (2) and Table 1 (sigma_Hlim, sigma_Flim by material family, quality
  grade and surface hardness): a material record, not this file.

Units of the standards: forces [N], lengths [mm], stresses [N/mm2], torque [N m], stiffness
[N/(mm um)], roughness [um]. No worked example is printed in ISO 6336-2 or -3.
"""
from __future__ import annotations


def application_factor_guide(driving_machine: str, driven_machine: str) -> float:
    """Guide value of the application factor K_A [-] for the working characteristics of the two machines.

    Inputs: the class of the driving and of the driven machine (uniform, light, moderate or heavy
    shocks), given by the user for the transmission. Source: ISO_6336-1_2019, 5.3, Table 4.
    """
    raise NotImplementedError("placeholder: not reviewed")


def mesh_stiffness(pair, material_1, material_2) -> tuple[float, float]:
    """Single tooth stiffness c' and mesh stiffness c_gamma_alpha [N/(mm um)] of a pair.

    Inputs: ``pair`` (``GearPair`` of ``CylindricalFamily``: tooth numbers, profile shifts, helix
    angle, contact ratio, basic rack), the two material records (E [N/mm2], nu [-]) and the gear
    blank data (not held yet). Source: ISO_6336-1_2019, clause 9.
    """
    raise NotImplementedError("placeholder: not reviewed")


def pitting_geometry_factors(pair, material_1, material_2) -> dict[str, float]:
    """Factors of the contact stress that depend on the pair alone.

    Returns Z_H [-], Z_E [(N/mm2)^0.5], Z_eps [-], Z_beta [-], Z_B [-], Z_D [-]. Inputs: ``pair``
    (``GearPair``: alpha_t, alpha_wt, beta_b, tip and base diameters, contact ratios) and the two
    material records (E [N/mm2], nu [-]). Source: ISO_6336-2_2006, clauses 6 to 9.
    """
    raise NotImplementedError("placeholder: not reviewed")


def roughness_factor(pair, sigma_Hlim_MPa: float) -> float:
    """Roughness factor Z_R [-] of the contact stress.

    Inputs: ``pair`` (``GearPair``: Rz of both flanks [um], reduced radius of curvature at the pitch
    point [mm]) and sigma_Hlim of the softer material [N/mm2] (material record). Source:
    ISO_6336-2_2006, 12.3.1.3.
    """
    raise NotImplementedError("placeholder: not reviewed")


def bending_geometry_factors(pair, tool_1, tool_2) -> dict[str, tuple[float, float]]:
    """Factors of the tooth root stress that depend on the pair and on the tools, one value per gear.

    Returns Y_F, Y_S, Y_beta, Y_B, Y_DT [-] as (gear 1, gear 2). Inputs: ``pair`` (``GearPair``:
    geometry of both gears, transverse contact ratio, overlap ratio), the tool or basic rack of
    each gear (dedendum, root fillet radius, protuberance; a pinion cutter for an internal gear;
    not held yet), rim thickness [mm] and accuracy grade. Source: ISO_6336-3_2006, clauses 6 to 10.
    """
    raise NotImplementedError("placeholder: not reviewed")


def iso6336_signs(pair) -> dict[str, float]:
    """Centre distance, reference and tip diameters, tooth number of gear 2 and gear ratio with the signs of ISO 6336-1.

    Returns a [mm], d1, d2, da1, da2 [mm], z2 [-], u [-], negative for internal gearing where
    ISO_6336-1_2019, Table 2, footnote a says so. Input: ``pair`` (``GearPair``, where only z is
    signed).
    """
    raise NotImplementedError("placeholder: not reviewed")
