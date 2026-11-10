"""PLACEHOLDER - not reviewed by José.

Slot for the load-capacity calculation of a cylindrical gear pair at an operating point, ISO 6336:
the load-dependent factors, the contact and tooth root stresses with their safety factors, and the
damage sum under a load spectrum. Nothing is implemented: every function raises
``NotImplementedError``. No formula, table or constant of a standard is in this file.

Clause map (clause level only, unchecked except where stated; vault note
``standards_by_quantity.md``, sections 3 and C2-1):

- ISO_6336-1_2019, 4.2.2, Formulae (1)-(4): nominal tangential load at the REFERENCE cylinder,
  torque, power, speed (checked; Formula (5) is misprinted). 4.1.18: the factors are calculated in
  succession, K_A, K_gamma, K_v, K_Hbeta (K_Fbeta), K_Halpha (K_Falpha), each at a load that holds
  the previous ones. 4.2.1: K_gamma (no formula). Clause 6: K_v, Methods B (6.5) and C (6.6)
  (formulae checked). Clause 7: K_Hbeta, K_Fbeta (needs the shaft, bearing and housing deflections
  at the mesh). Clause 8: K_Halpha, K_Falpha.
- ISO_6336-2_2006: 5.2 Eq. (1)-(2) S_H; 5.3 Eq. (3)-(5) sigma_H0, sigma_H; 5.4 Eq. (6)-(15)
  sigma_HP; clause 11 Z_NT; 12.3.1 Z_L, Z_v; clause 13 Z_W.
- ISO_6336-3_2006: 5.1 Eq. (1)-(2) S_F; 5.2.2 Eq. (3)-(4) sigma_F0, sigma_F; 5.3 Eq. (5)-(8)
  sigma_FP; clause 12 Y_NT; clauses 13 to 15 Y_delta_relT, Y_R_relT, Y_X.
- ISO_6336-6_2006: clause 5 (stress per bin, damage parts, Miner sum, safety factor by iteration
  on the stress level); Annex A (K_A from a spectrum). Worked examples: A.4 (complete) and Annex C
  (safety factors and per-bin stresses printed).

Units of the standards: forces [N], lengths [mm], stresses [N/mm2], torque [N m], speed [min-1],
pitch line velocity [m/s], viscosity [mm2/s]. Layer: this module may read ``core`` objects (the
pair and ``cylindrical_capacity``); ``core`` imports nothing from here.
"""
from __future__ import annotations


def dynamic_factor(pair, F_t_N: float, n1_rpm: float, K_A: float, K_gamma: float, method: str) -> float:
    """Internal dynamic factor K_v [-] of a pair at one load and speed.

    Inputs: ``pair`` (``GearPair`` of ``CylindricalFamily``) with its mesh stiffness and accuracy
    data (``cylindrical_capacity``); nominal tangential load F_t [N] at the reference cylinder and
    pinion speed n1 [min-1] (from the resolved transmission); K_A, K_gamma [-]; the method of
    the standard to use. Source: ISO_6336-1_2019, clause 6.
    """
    raise NotImplementedError("placeholder: not reviewed")


def face_load_factors(pair, F_t_N: float, K_A: float, K_gamma: float, K_v: float, shaft_results=None) -> tuple[float, float]:
    """Face load factors K_Hbeta and K_Fbeta [-].

    Inputs: ``pair`` (``GearPair``: face width, helix data, mesh stiffness); F_t [N] and the factors
    already calculated [-]; the deflections of the two shafts at the mesh (``ShaftResults`` of the
    FEM solver, Method B) or the shaft geometry (Method C). Source: ISO_6336-1_2019, clause 7.
    """
    raise NotImplementedError("placeholder: not reviewed")


def transverse_load_factors(pair, F_tH_N: float) -> tuple[float, float]:
    """Transverse load factors K_Halpha and K_Falpha [-].

    Inputs: ``pair`` (``GearPair``: contact ratios, mesh stiffness, pitch deviation) and the load
    F_tH [N] (ISO_6336-1_2019, Table 2). Source: ISO_6336-1_2019, clause 8.
    """
    raise NotImplementedError("placeholder: not reviewed")


def contact_safety(pair, F_t_N: float, K_factors: dict, material_1, material_2, lubricant, N_L: tuple[float, float]) -> dict:
    """Contact stress sigma_H [N/mm2] and safety factor against pitting S_H [-], per gear.

    Inputs: ``pair`` and its element factors (``cylindrical_capacity``); F_t [N]; K_A, K_gamma, K_v,
    K_Hbeta, K_Halpha [-]; material records (sigma_Hlim [N/mm2]); nominal oil viscosity [mm2/s] and
    pitch line velocity [m/s]; number of load cycles of each gear [-]. Source: ISO_6336-2_2006.
    """
    raise NotImplementedError("placeholder: not reviewed")


def bending_safety(pair, F_t_N: float, K_factors: dict, material_1, material_2, N_L: tuple[float, float]) -> dict:
    """Tooth root stress sigma_F [N/mm2] and safety factor against tooth breakage S_F [-], per gear.

    Inputs: ``pair`` and its element factors (``cylindrical_capacity``); F_t [N]; K_A, K_gamma, K_v,
    K_Fbeta, K_Falpha [-]; material records (sigma_Flim [N/mm2]); number of load cycles of each gear
    [-]. Source: ISO_6336-3_2006.
    """
    raise NotImplementedError("placeholder: not reviewed")


def damage_sum(stresses_MPa, cycles, sn_curve) -> float:
    """Miner damage sum U [-] of one gear and one failure mode over the bins of a load spectrum.

    Inputs: stress of each bin [N/mm2] (from ``contact_safety`` or ``bending_safety`` with K_A = 1),
    number of load cycles of each bin [-], S-N curve of the gear (life factor against cycles).
    Source: ISO_6336-6_2006, clause 5.
    """
    raise NotImplementedError("placeholder: not reviewed")
