"""PLACEHOLDER - not reviewed by José.

Slot for the power loss and the thermal balance of a gear unit, ISO/TR 14179-2 (the published
"German" method; ISO/TR 14179-1 is held as a proof copy only). Nothing is implemented: every
function raises ``NotImplementedError``. No formula, table or constant of a report is in this file.

Clause map of ISO_TR_14179-2_2001 (clause level only, unchecked: no equation was transcribed;
vault note ``standards_by_quantity.md``, section C2-3):

- Clause 4, Eq. (1)-(2): equivalent power for variable load.
- 5.1, Eq. (3)-(4): total power loss (gear no-load and load, bearing no-load and load, seals,
  other) and efficiency.
- 5.2.2: gear no-load loss, splash (Eq. (5)-(8); needs the oil level and the housing section) or
  injection (Eq. (9)-(10)).
- 5.2.3, Eq. (11)-(12): mean coefficient of friction of the mesh (inputs: line load, sum velocity,
  reduced radius of curvature at the pitch point, dynamic viscosity at the operating temperature,
  mean Ra, lubricant factor).
- 5.2.4, Eq. (13)-(14): load-dependent gear loss from the input power, the coefficient of friction
  and the loss factor H_v (``GearPair.HV`` of ``CylindricalFamily``; question Q53).
- 5.3.1, Eq. (25)-(30), Tables 4-7: rolling bearing losses. 5.4, Eq. (31): shaft seals.
- 6.2-6.5, Eq. (32)-(70): heat dissipation through the housing (convection, radiation, fins), the
  foundation, shafts and couplings, an external cooler.
- 7.1, Eq. (71)-(73): equilibrium, by iteration on the oil temperature.
- Clause 8: worked example (total loss and oil temperature printed; it has printed slips).

Units of the report: power [W], torque [N m], speed [min-1], velocities [m/s], lengths [mm] (housing
[m]), viscosity [mm2/s] and [mPa s], temperatures [degC] and [K]. Layer: this module may read
``core`` objects; ``core`` imports nothing from here.
"""
from __future__ import annotations


def mesh_friction_coefficient(pair, F_bt_N: float, v_t_m_s: float, eta_oil_mPas: float, lubricant_factor: float) -> float:
    """Mean coefficient of friction of the mesh mu_mz [-].

    Inputs: ``pair`` (``GearPair`` of ``CylindricalFamily``: face width, working pressure angle,
    base helix angle, reduced radius at the pitch point [mm], mean roughness Ra [um]); transverse
    base force F_bt [N] and pitch line velocity [m/s] (resolved transmission); dynamic viscosity
    of the oil at the operating temperature [mPa s]; lubricant factor [-]. Source:
    ISO_TR_14179-2_2001, 5.2.3.
    """
    raise NotImplementedError("placeholder: not reviewed")


def gear_load_loss(pair, P_A_W: float, mu_mz: float) -> float:
    """Load-dependent power loss of one mesh P_VZP [W].

    Inputs: ``pair`` (``GearPair``: loss factor ``HV`` [-]); input power of the stage [W] (resolved
    transmission); mean coefficient of friction [-]. Source: ISO_TR_14179-2_2001, 5.2.4.
    """
    raise NotImplementedError("placeholder: not reviewed")


def gear_no_load_loss(pair, n2_rpm: float, immersion_depths_mm: tuple[float, float], housing) -> float:
    """No-load (churning) power loss of one splash-lubricated stage P_VZ0 [W].

    Inputs: ``pair`` (``GearPair``: face width [mm], tip diameters [mm]); speed of the wheel [min-1];
    immersion depth of each gear [mm] and the housing section (gear-unit data, not held yet).
    Source: ISO_TR_14179-2_2001, 5.2.2.2.
    """
    raise NotImplementedError("placeholder: not reviewed")


def bearing_loss(bearing, F_r_N: float, F_a_N: float, n_rpm: float, nu_oil_mm2_s: float, lubrication: str) -> float:
    """Power loss of one rolling bearing P_VL [W], no-load and load-dependent.

    Inputs: ``bearing`` (``Bearing``: mean diameter [mm], type, static rating C0 [N]); radial and
    axial loads [N] (bearing reactions of the shaft solution); speed [min-1]; kinematic viscosity at
    the operating temperature [mm2/s]; kind of lubrication. Source: ISO_TR_14179-2_2001, 5.3.1.
    """
    raise NotImplementedError("placeholder: not reviewed")


def seal_loss(d_sh_mm: float, n_rpm: float) -> float:
    """Power loss of one contacting shaft seal P_VD [W].

    Inputs: shaft diameter at the seal [mm] (``Shaft`` section) and speed [min-1]. Source:
    ISO_TR_14179-2_2001, 5.4.
    """
    raise NotImplementedError("placeholder: not reviewed")


def heat_dissipation(gear_unit, theta_oil_degC: float, theta_ambient_degC: float, v_air_m_s: float) -> float:
    """Heat flow Q [W] leaving the gear unit through housing, foundation, shafts and cooler.

    Inputs: gear-unit data (housing areas [m2], wall, fins, foundation, shaft ends, cooler; not
    held yet); oil and ambient temperatures [degC]; air velocity [m/s]. Source:
    ISO_TR_14179-2_2001, clause 6.
    """
    raise NotImplementedError("placeholder: not reviewed")


def oil_temperature(transmission, gear_unit, lubricant, theta_ambient_degC: float) -> dict:
    """Quasi-stationary oil temperature [degC], total power loss [W] and efficiency [-] of a gear unit.

    Inputs: the resolved transmission (meshes, torques, speeds, bearing reactions), the gear-unit
    data and the lubricant (viscosity against temperature); ambient temperature [degC]. Iteration
    on the oil temperature until the loss equals the dissipated heat. Source: ISO_TR_14179-2_2001,
    7.1.
    """
    raise NotImplementedError("placeholder: not reviewed")
