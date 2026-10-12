"""Pure helper functions for the fatigue assessment of shafts: Marin endurance-limit modifying
factors, stress concentration factors of shoulders (Peterson charts, D/d fixed at 1.5), Neuber
notch sensitivity and the fatigue stress concentration factor Kf.

Units: strengths [MPa], lengths [mm]; factors dimensionless. The formulas and the table or figure
numbers quoted in the functions (Shigley 10th ed., Peterson 3rd ed.) are named in the code and are
not in the library; they have not been checked against the sources (source to find). The module is
not re-exported by any package ``__init__``.
"""
from __future__ import annotations

import math

from scipy.interpolate import interp1d


# ===========================================================================
# Marin surface finish factor -- ka
# Shigley Tab. 6-2: ka = a x Sut^b  (Sut in MPa)
# ===========================================================================

_KA_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "ground":     (1.58,  -0.085),
    "machined":   (4.51,  -0.265),
    "cold_drawn": (4.51,  -0.265),
    "hot_rolled": (57.7,  -0.718),
    "as_forged":  (272.0, -0.995),
}


def ka_surface_finish(Sut_MPa: float, finish: str = "machined") -> float:
    """Marin surface finish factor ka.

    Parameters
    ----------
    Sut_MPa : float
        Ultimate tensile strength [MPa].
    finish : str, optional
        ``"ground"``, ``"machined"``, ``"cold_drawn"``, ``"hot_rolled"`` or ``"as_forged"``.

    Returns
    -------
    float
        ka [-], in (0, 1].

    Raises
    ------
    ValueError
        If ``finish`` is unknown.

    Notes
    -----
    ka = min(1, a * Sut^b) with Sut in MPa and (a, b) = ground (1.58, -0.085); machined and cold drawn
    (4.51, -0.265); hot rolled (57.7, -0.718); as forged (272, -0.995). The cap at 1.0 is applied
    because ground finish on low-Sut material can nominally exceed 1.

    References
    ----------
    Shigley 10th ed., Eq. 6-19 and Tab. 6-2, as named in the code (source to find).
    """
    if finish not in _KA_COEFFICIENTS:
        available = ", ".join(_KA_COEFFICIENTS.keys())
        raise ValueError(f"Unknown finish '{finish}'. Available: {available}")
    a, b = _KA_COEFFICIENTS[finish]
    return float(min(1.0, a * (Sut_MPa ** b)))


# ===========================================================================
# Marin size factor -- kb
# Shigley Eq. 6-20 (rotating round shaft in bending)
# ===========================================================================

def kb_size(diameter_mm: float) -> float:
    """Marin size factor kb for rotating round shafts in bending.

    Parameters
    ----------
    diameter_mm : float
        Shaft diameter at the section [mm].

    Returns
    -------
    float
        kb [-]. Not capped: 1.11 at d = 2.79 mm, above 1 up to d = 7.5 mm (kb = 1.24 d^-0.107 exceeds
        1 for d below about 7.5 mm), 0.633 at d = 254 mm, then 0.70 (jump) above 254 mm.

    Raises
    ------
    ValueError
        If ``diameter_mm <= 0``.

    Notes
    -----
    d < 2.79 mm: kb = 1.0 (the correlation does not extend below 2.79 mm); 2.79 <= d <= 51 mm:
    kb = 1.24 d^-0.107; 51 < d <= 254 mm: kb = 1.51 d^-0.157; d > 254 mm: kb = 0.70. The function does
    not depend on the load type (no adjustment for axial loading).

    References
    ----------
    Shigley 10th ed., Eq. 6-20 (SI form), as named in the code (source to find).
    """
    if diameter_mm <= 0:
        raise ValueError(f"diameter_mm must be > 0, got {diameter_mm}")
    if diameter_mm < 2.79:
        return 1.0
    elif diameter_mm <= 51.0:
        return float(1.24 * diameter_mm ** (-0.107))
    elif diameter_mm <= 254.0:
        return float(1.51 * diameter_mm ** (-0.157))
    else:
        return 0.70


# ===========================================================================
# Marin load factor -- kc
# Shigley Tab. 6-3
# ===========================================================================

def kc_load(load_type: str = "bending") -> float:
    """Marin load type factor kc.

    Parameters
    ----------
    load_type : str, optional
        ``"bending"`` (kc = 1.0), ``"axial"`` (0.85) or ``"torsion"`` (0.59).

    Returns
    -------
    float
        kc [-].

    Raises
    ------
    ValueError
        If ``load_type`` is unknown.

    Notes
    -----
    For combined bending and torsion of a rotating shaft use ``"bending"`` (kc = 1.0) and combine the
    stresses with the von Mises equations (Shigley 10th ed., 6-14, as named in the code).

    References
    ----------
    Shigley 10th ed., Eq. 6-26 (the section banner names Tab. 6-3), as named in the code (source to
    find).
    """
    mapping = {"bending": 1.0, "axial": 0.85, "torsion": 0.59}
    if load_type not in mapping:
        raise ValueError(
            f"load_type must be one of {list(mapping.keys())}, got '{load_type}'"
        )
    return mapping[load_type]


# ===========================================================================
# Marin temperature factor -- kd
# Shigley sec. 6-6
# ===========================================================================

def kd_temperature(T_celsius: float = 20.0) -> float:
    """Marin temperature factor kd.

    Parameters
    ----------
    T_celsius : float, optional
        Operating temperature [deg C]. Not used.

    Returns
    -------
    float
        Always 1.0 [-]: the temperature is not used (no temperature effect is modelled).

    References
    ----------
    Shigley 10th ed., 6-6, as named in the code (source to find).
    """
    return 1.0


# ===========================================================================
# Marin reliability factor -- ke
# Shigley Tab. 6-6
# ===========================================================================

_KE_TABLE: dict[float, float] = {
    50.0:     1.000,
    90.0:     0.897,
    95.0:     0.868,
    99.0:     0.814,
    99.9:     0.753,
    99.99:    0.702,
    99.999:   0.659,
    99.9999:  0.620,
}


def ke_reliability(reliability_percent: float = 99.0) -> float:
    """Marin reliability factor ke from the tabulated values.

    Parameters
    ----------
    reliability_percent : float, optional
        Target reliability [%]; one of 50, 90, 95, 99, 99.9, 99.99, 99.999, 99.9999 (exact match).

    Returns
    -------
    float
        ke [-], from 1.000 (50 %) to 0.620 (99.9999 %).

    Raises
    ------
    ValueError
        If the value is not in the table; use ``ke_reliability_from_z`` for other values.

    References
    ----------
    Shigley 10th ed., Tab. 6-6 and Eq. 6-29, as named in the code (source to find).
    """
    if reliability_percent not in _KE_TABLE:
        available = sorted(_KE_TABLE.keys())
        raise ValueError(
            f"reliability_percent must be one of {available}, "
            f"got {reliability_percent}. "
            f"Use ke_reliability_from_z() for arbitrary values."
        )
    return _KE_TABLE[reliability_percent]


def ke_reliability_from_z(z_score: float) -> float:
    """Marin reliability factor from the standard normal deviate z.

    Parameters
    ----------
    z_score : float
        Standard normal deviate [-]; for example 1.288 for 90 %, 1.645 for 95 %, 2.326 for 99 %.

    Returns
    -------
    float
        ke = 1 - 0.08 z [-].

    References
    ----------
    Shigley 10th ed., 6-6, as named in the code (source to find).
    """
    return float(1.0 - 0.08 * z_score)


# ===========================================================================
# Corrected endurance limit Se'
# Se' = Se x ka x kb x kc x kd x ke  [Shigley sec. 6-7]
# ===========================================================================

def endurance_limit_corrected(
    Se_base: float,
    Sut_MPa: float,
    diameter_mm: float,
    finish: str = "machined",
    reliability_percent: float = 99.0,
    load_type: str = "bending",
    T_celsius: float = 20.0,
) -> dict[str, float]:
    """Corrected endurance limit Se' with all Marin factors.

    Parameters
    ----------
    Se_base : float
        Specimen endurance limit [MPa] (input; the function does not derive it from Sut).
    Sut_MPa : float
        Ultimate tensile strength [MPa].
    diameter_mm : float
        Section diameter [mm].
    finish : str, optional
        Surface finish key (see ``ka_surface_finish``).
    reliability_percent : float, optional
        Target reliability [%] (see ``ke_reliability``).
    load_type : str, optional
        ``"bending"``, ``"axial"`` or ``"torsion"``.
    T_celsius : float, optional
        Operating temperature [deg C] (does not change kd).

    Returns
    -------
    dict
        ``"ka"``, ``"kb"``, ``"kc"``, ``"kd"``, ``"ke"`` [-] and ``"Se_prime"`` [MPa].

    Notes
    -----
    Se' = Se * ka * kb * kc * kd * ke.

    References
    ----------
    Shigley 10th ed., 6-7, as named in the code (source to find).
    """
    ka = ka_surface_finish(Sut_MPa, finish)
    kb = kb_size(diameter_mm)
    kc = kc_load(load_type)
    kd = kd_temperature(T_celsius)
    ke = ke_reliability(reliability_percent)
    Se_prime = Se_base * ka * kb * kc * kd * ke
    return {
        "ka":       ka,
        "kb":       kb,
        "kc":       kc,
        "kd":       kd,
        "ke":       ke,
        "Se_prime": Se_prime,
    }


# ===========================================================================
# Peterson stress concentration factors -- Kt (shoulders)
# Peterson 3rd ed.: Fig. A-15-9 (bending), A-15-10 (torsion)
# Phase 1: D/d ~ 1.5 fixed (one-parameter interpolation on r/d)
# Phase 2+: RegularGridInterpolator over (r/d, D/d) grid
# ===========================================================================

_KT_SHOULDER_BENDING_R_OVER_D = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
_KT_SHOULDER_BENDING_KT       = [3.00,  2.70, 2.30, 2.10, 1.85, 1.60, 1.40]

_kt_shoulder_bending_interp = interp1d(
    _KT_SHOULDER_BENDING_R_OVER_D,
    _KT_SHOULDER_BENDING_KT,
    kind="linear",
    bounds_error=False,
    fill_value=(_KT_SHOULDER_BENDING_KT[0], _KT_SHOULDER_BENDING_KT[-1]),
)

_KT_SHOULDER_TORSION_R_OVER_D = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]
_KT_SHOULDER_TORSION_KT       = [2.00,  1.85, 1.65, 1.55, 1.40, 1.25, 1.15]

_kt_shoulder_torsion_interp = interp1d(
    _KT_SHOULDER_TORSION_R_OVER_D,
    _KT_SHOULDER_TORSION_KT,
    kind="linear",
    bounds_error=False,
    fill_value=(_KT_SHOULDER_TORSION_KT[0], _KT_SHOULDER_TORSION_KT[-1]),
)


def kt_shoulder_bending(r_over_d: float, D_over_d: float = 1.5) -> float:
    """Theoretical stress concentration factor Kt of a stepped-shaft shoulder in bending.

    Parameters
    ----------
    r_over_d : float
        Fillet radius divided by the small diameter [-].
    D_over_d : float, optional
        Large diameter divided by the small diameter [-]. Not used: the curve is for D/d = 1.5 only.

    Returns
    -------
    float
        Kt [-]: linear interpolation of a 7-point table (r/d = 0.005 ... 0.20 gives Kt = 3.00 ... 1.40),
        clamped to the end values outside that range.

    References
    ----------
    Peterson 3rd ed., Fig. A-15-9, as named in the code (source to find; the table values were not
    checked against the figure).
    """
    return float(_kt_shoulder_bending_interp(r_over_d))


def kt_shoulder_torsion(r_over_d: float, D_over_d: float = 1.5) -> float:
    """Theoretical stress concentration factor Kts of a stepped-shaft shoulder in torsion.

    Parameters
    ----------
    r_over_d : float
        Fillet radius divided by the small diameter [-].
    D_over_d : float, optional
        Large diameter divided by the small diameter [-]. Not used: the curve is for D/d = 1.5 only.

    Returns
    -------
    float
        Kts [-]: linear interpolation of a 7-point table (r/d = 0.005 ... 0.20 gives Kts = 2.00 ...
        1.15), clamped to the end values outside that range.

    References
    ----------
    Peterson 3rd ed., Fig. A-15-10, as named in the code (source to find; the table values were not
    checked against the figure).
    """
    return float(_kt_shoulder_torsion_interp(r_over_d))


# ===========================================================================
# Neuber notch sensitivity and fatigue stress concentration factor
# Shigley sec. 6-10, Eq. 6-33 to 6-35
# ===========================================================================

def neuber_constant_sqrt_a(Sut_MPa: float, loading: str = "bending") -> float:
    """Neuber material constant sqrt(a) for steels.

    Parameters
    ----------
    Sut_MPa : float
        Ultimate tensile strength [MPa].
    loading : str, optional
        ``"torsion"`` selects the torsion polynomial; any other value uses the bending/axial one.

    Returns
    -------
    float
        sqrt(a) [mm^0.5]; decreases as Sut grows. The polynomial becomes negative above about
        1750 MPa (bending) and 1600 MPa (torsion); its range of validity is not checked.

    Notes
    -----
    The polynomials take Sut in kpsi (``Sut_kpsi = Sut_MPa / 6.895``) and give sqrt(a) in sqrt(in):

    - bending or axial: 0.246 - 3.08e-3 S + 1.51e-5 S^2 - 2.67e-8 S^3
    - torsion: 0.190 - 2.51e-3 S + 1.35e-5 S^2 - 2.67e-8 S^3

    with S = Sut in kpsi. The result is multiplied by sqrt(25.4) to give mm^0.5.

    References
    ----------
    Shigley 10th ed., Eq. 6-35a (bending/axial) and 6-35b (torsion), as named in the code (source to
    find).
    """
    Sut_kpsi = Sut_MPa / 6.895
    if loading == "torsion":
        sqrt_a_in = (
            0.190
            - 2.51e-3 * Sut_kpsi
            + 1.35e-5 * Sut_kpsi ** 2
            - 2.67e-8 * Sut_kpsi ** 3
        )
    else:
        sqrt_a_in = (
            0.246
            - 3.08e-3 * Sut_kpsi
            + 1.51e-5 * Sut_kpsi ** 2
            - 2.67e-8 * Sut_kpsi ** 3
        )
    return float(sqrt_a_in * math.sqrt(25.4))


def notch_sensitivity(r_mm: float, Sut_MPa: float, loading: str = "bending") -> float:
    """Notch sensitivity q of a fillet.

    Parameters
    ----------
    r_mm : float
        Fillet radius [mm]; clamped below to ``MIN_FILLET_RADIUS_mm`` of the configuration.
    Sut_MPa : float
        Ultimate tensile strength [MPa].
    loading : str, optional
        ``"bending"`` or ``"torsion"`` (see ``neuber_constant_sqrt_a``).

    Returns
    -------
    float
        q [-]; between 0 and 1 while sqrt(a) >= 0 (0 is notch-insensitive, 1 fully sensitive). The
        polynomial of ``neuber_constant_sqrt_a`` becomes negative above about 1750 MPa (bending) and
        1600 MPa (torsion), which gives q > 1; the range of validity of the polynomial is not checked
        (source to find).

    Raises
    ------
    ModuleNotFoundError
        See Warnings.

    Warnings
    --------
    The function imports ``MIN_FILLET_RADIUS_mm`` with ``from config import ...`` (a top-level
    ``config`` module, not ``axisforge.config``). With ``axisforge`` imported as a package the call
    raises ``ModuleNotFoundError``; it runs only if the ``axisforge`` folder
    itself is on ``sys.path``.

    Notes
    -----
    q = 1 / (1 + sqrt(a) / sqrt(r)), with sqrt(a) [mm^0.5] from ``neuber_constant_sqrt_a`` and r [mm].

    References
    ----------
    Shigley 10th ed., Eq. 6-34, as named in the code (source to find).
    """
    from config import MIN_FILLET_RADIUS_mm
    r_mm = max(r_mm, MIN_FILLET_RADIUS_mm)
    sqrt_a = neuber_constant_sqrt_a(Sut_MPa, loading)
    return float(1.0 / (1.0 + sqrt_a / math.sqrt(r_mm)))


def kf_from_kt(Kt: float, q: float) -> float:
    """Fatigue stress concentration factor Kf from the theoretical factor and the notch sensitivity.

    Parameters
    ----------
    Kt : float
        Theoretical (geometric) stress concentration factor [-].
    q : float
        Notch sensitivity [-], in [0, 1].

    Returns
    -------
    float
        Kf = 1 + q (Kt - 1) [-], in [1, Kt].

    References
    ----------
    Shigley 10th ed., Eq. 6-32, as named in the code (source to find).
    """
    return float(1.0 + q * (Kt - 1.0))