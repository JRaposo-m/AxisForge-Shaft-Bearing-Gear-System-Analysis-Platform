"""
Contact inputs along the path of contact of a cylindrical gear pair: load sharing between the tooth
pairs and one record per step of the contact point (radii, speeds, load per unit length).

These functions take a force or a time, so they belong to the solvers layer (D-0026 item 7; Q31,
Q37, José 2026-10-15); the geometry and kinematics of the path that they use stay on the family:
``CylindricalFamily.path_points``, ``curvature_radii``, ``surface_speeds`` (ISO 21771, 5.4.3-5.4.8
and 5.6). Pure functions of the pair: they store nothing.

Units: positions along the path g_y [mm] from the pitch point C, positive toward the tip of gear 1;
lengths [mm], speeds [m/s], forces [N], load per unit length [N/mm], time [s].
"""
from __future__ import annotations

import numpy as np

from axisforge.core.machine_elements.gears.cylindrical import CylindricalFamily

__all__ = ["load_sharing", "contact_inputs"]


def load_sharing(pair, g_y) -> dict[str, np.ndarray]:
    """Share of the normal tooth force carried by one tooth pair at positions g_y of the path.

    Several tooth pairs, one transverse base pitch apart along the line of action, are in
    contact at the same time. The model: the teeth are rigid and equal, so the normal force
    is spread with ONE load per unit length over all the contact lines present. A tooth pair
    then carries the fraction (length of its contact line) / (sum of the lengths). For a
    spur pair this is 1 in the zone of single contact and 1/2 in the two zones of double
    contact (1/2 and 1/3 for 2 <= eps_alpha < 3).

    Parameters
    ----------
    pair : GearPair
        Pair of the cylindrical family.
    g_y : float or array_like
        Position [mm] of the contact point of the tooth pair in the transverse section at
        MID FACE WIDTH, from C, positive toward the tip of gear 1.

    Returns
    -------
    dict
        share : fraction of the normal force on this tooth pair [-]; zero where it has no
        contact.
        l_mm : length of the contact line of this tooth pair [mm].
        l_sum_mm : sum of the lengths of all contact lines present at that instant [mm].
        n_pairs : number of tooth pairs in contact at that instant [-].

    Notes
    -----
    Contact lines: in the plane of action (length g_alpha along the path, width b) the
    contact lines are straight, inclined at beta_b to the gear axes. With s the distance from
    the stored point A of the mid-face point of a line, the line covers the path interval
    [s - b tan(beta_b)/2, s + b tan(beta_b)/2]; the part of it inside [0, g_alpha], divided by
    sin(beta_b), is its length. A spur pair has l = b for 0 <= s <= g_alpha. The largest value
    is the l_max of ISO_21771_2007, Eq. (98)-(99). The neighbouring tooth pairs are at
    s + k p_bt. With the mid-face section as the reference the result depends neither on the
    driver nor on the hand of helix.

    A helical tooth pair is in contact before its mid-face point reaches A and after it leaves
    E (by b tan(beta_b)/2): ``share`` is not zero just outside the path. The sum of ``share``
    over the tooth pairs k is 1 at every position.

    Assumptions: teeth of equal, constant stiffness per unit length (rigid limit), no
    modifications, no misalignment, no friction; common face width b = min(b1, b2) of aligned gears.

    References
    ----------
    .. [1] ISO_21771_2007, 5.4.8, p. 39-40, Eq. (98)-(99): contact lines at beta_b, their
       largest length; the sum of the lengths is defined in words only (checked).
    .. [2] Uniform load per unit length and the lengths along the path: derived, no library
       source. GEARpie uses the same model (CLASSES/LOAD_SHARING.py:62-90,
       CLASSES/FORCES_SPEEDS.py:72), with the section at the face end as its first column.
    """
    # Q30 (José, 2026-10-15): keep - rigid load sharing (one load per unit length on all contact lines, as GEARpie); FUTURE: mesh stiffness model (ISO_6336-1_2019, clause 9) for the contact studies.
    g_y = np.asarray(g_y, dtype=float)
    pbt = pair.gear1.pbt
    betab = pair.gear1.betab
    s = g_y + pair.AC                          # distance from the stored A (C is at AC from A)
    # Q36 (José, 2026-10-15): keep - the line whose MID-FACE point is at g_y (a 2D slice); FUTURE: one record per point of the contact line (path x face width) when helical EHL needs it.
    half = 0.5 * pair.b * np.tan(betab)        # half of the path interval covered by one contact line [mm]
    tol = 1e-9 * pbt                           # the ends A and E count as in contact: sums of stored segments round at 1e-15 relative

    def line_length(s):
        if betab == 0.0:
            return np.where((s >= -tol) & (s <= pair.AE + tol), pair.b, 0.0)   # spur: the whole face width
        inside = np.clip(np.minimum(s + half, pair.AE) - np.maximum(s - half, 0.0), 0.0, None)
        return inside / np.sin(betab)

    # every tooth pair that can touch while this one is within reach of the path
    n_max = int(np.ceil((pair.AE + 2.0 * half) / pbt))
    lengths = np.array([line_length(s + k * pbt) for k in range(-n_max, n_max + 1)])
    l_own, l_sum = lengths[n_max], lengths.sum(axis=0)
    share = np.divide(l_own, l_sum, out=np.zeros_like(l_sum), where=l_sum > 0.0)
    return {"share": share, "l_mm": l_own, "l_sum_mm": l_sum, "n_pairs": (lengths > 0.0).sum(axis=0)}


def contact_inputs(pair, driver: str, n1_rpm: float, F_bn_N: float, *, n_points: int | None = None,
                   t_s=None) -> dict[str, np.ndarray]:
    """Everything a contact calculation needs at each step of a contact point along the path.

    One record per step, from the start of meshing A to its end E for the given driver, of
    the contact point in ONE transverse section: where it is and when, the two equivalent
    cylinders, how fast the surfaces move, and the load per unit length of contact line.
    These are the inputs of a line contact (Hertz, EHL) that changes along the path. For a
    spur pair this is the whole life of a tooth pair; for a helical pair it is the section at
    mid face width only (Notes).

    Parameters
    ----------
    pair : GearPair
        Pair of the cylindrical family.
    driver : str
        "gear1" or "gear2": the driving gear (fixes A, E and the direction of travel).
    n1_rpm : float
        Rotational speed of GEAR 1 [rpm], > 0 (a magnitude, as in ``surface_speeds``),
        whichever gear drives.
    F_bn_N : float
        Total force normal to the flanks transmitted by the mesh [N], >= 0 (F_bn = F_bt /
        cos(beta_b)). An argument: no force is derived from a torque here (``MeshForces.F_bn_N`` of
        ``mesh_forces`` is this force).
    n_points : int, optional
        Number of equally spaced steps from A to E, both included (>= 2).
    t_s : array_like, optional
        Times of the steps [s], t = 0 at A; finite, at least one, all within the time of
        travel from A to E.
        Exactly one of ``n_points`` and ``t_s`` is given.

    Returns
    -------
    dict
        Arrays of equal length, one entry per step:

        s_mm : distance travelled from A along the path [mm].
        t_s : time since A [s].
        g_y_mm : position from the pitch point C, positive toward the tip of gear 1 [mm].
        rho1_t_mm, rho2_t_mm, rho_red_t_mm, rho1_n_mm, rho2_n_mm, rho_red_n_mm : radii of
        curvature, signed (``curvature_radii``) [mm].
        u1_m_per_s, u2_m_per_s, vg_m_per_s, ue_m_per_s : surface, sliding and entrainment speeds
        (``surface_speeds``) [m/s].
        share, l_mm, l_sum_mm, n_pairs : load share and contact-line lengths
        (``load_sharing``) [-], [mm], [mm], [-].
        w_N_per_mm : load per unit length of contact line, share F_bn / l [N/mm].

    Raises
    ------
    ValueError
        If ``driver`` is not "gear1" or "gear2", if n1_rpm <= 0, if F_bn_N < 0, if not
        exactly one of ``n_points`` and ``t_s`` is given, if n_points < 2, or if ``t_s`` is
        empty, not finite or outside the travel from A to E.

    Notes
    -----
    Time and position: the contact point moves along the line of action at the normal speed
    of the flanks, v_n = omega1 r_b1 (ISO_21771_2007, Eq. (105)), so s = omega1 r_b1 t. The
    standard gives v_n; the relation s(t) is derived from it (no library sentence).

    Helical pair: a step is the contact point in the transverse section at MID face width.
    A tooth pair is in contact over g_alpha + b tan(beta_b) of travel (case 2: 21.44 mm); the
    record covers g_alpha (15.68 mm, 73 %). ``share`` and ``l_mm`` describe the whole contact
    line, the radii and speeds only its mid point, and the load per unit length is uniform by
    the model. Along the line the path position spans +-b tan(beta_b)/2: in case 2 at C,
    +-2.88 mm, rho1 from 10.90 to 16.66 mm (+-21 %) and a sliding speed of +-0.73 m/s at the
    line ends against 0 recorded; at A and E the recorded point is the END of a half-length
    line. The record is fair for a two-dimensional slice at mid face, not for the line (Q36).

    Every step is on the two involutes for a pair that passed ``validate_pair`` (assembled
    with ``check=True``): T1A >= 0 and s2 T2E >= 0 keep A..E between the base-tangent points
    (Q39), so the radii are those of the flanks and no warning is needed. A pair assembled with
    ``check=False`` (an equivalence record such as old case 4) may give rho <= 0 at the ends;
    the numbers are returned as they are.

    Assumptions: those of ``CylindricalFamily.path_points``, ``curvature_radii``, ``surface_speeds``
    and ``load_sharing``; steady speed and constant F_bn.

    References
    ----------
    .. [1] ISO_21771_2007, 5.6.1, Eq. (105), p. 42 (checked).
    .. [2] Load per unit length: no library source.
    """
    if (n_points is None) == (t_s is None):
        raise ValueError("give exactly one of n_points and t_s.")
    if n1_rpm <= 0:
        raise ValueError(f"n1_rpm must be > 0, got {n1_rpm}.")
    if F_bn_N < 0:
        raise ValueError(f"F_bn_N must be >= 0, got {F_bn_N}.")
    points = CylindricalFamily.path_points(pair, driver)
    v_n = n1_rpm * 2.0 * np.pi / 60.0 * pair.gear1.rb   # [rad/s] [mm] = [mm/s], speed of the contact point along the path
    if t_s is None:
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}.")
        s = np.linspace(0.0, pair.AE, n_points)
    else:
        s = v_n * np.atleast_1d(np.asarray(t_s, dtype=float))
        if s.size == 0 or not np.all(np.isfinite(s)):
            raise ValueError("t_s must hold at least one time, all finite.")
        if s.min() < 0.0 or s.max() > pair.AE * (1.0 + 1e-12):   # 1e-12: rounding of t = AE / v_n given back as a time
            raise ValueError(f"t_s must lie in [0, {pair.AE / v_n:.6g}] s, the travel from A to E.")
    # from A toward E: increasing g_y when gear 1 drives (root to tip of the driving flank), decreasing when gear 2 drives
    g_y = points["A"] + np.sign(points["E"] - points["A"]) * s

    radii = CylindricalFamily.curvature_radii(pair, g_y)
    sharing = load_sharing(pair, g_y)
    # Q32 (José, 2026-10-15): keep - the Hertz half-width, p0 and the grid belong to slippy_bridge (D-0019), one Hertz code for bearings and gears.
    # Q33 (José, 2026-10-15): keep - local equivalent cylinders (the radii); FUTURE: involute flank surfaces when tip relief or edge contact is studied.
    return {"s_mm": s, "t_s": s / v_n, "g_y_mm": g_y,
            **radii,
            **CylindricalFamily.surface_speeds(pair, g_y, n1_rpm),
            **sharing,
            # share F_bn / l with share = l / l_sum: written without l, which is zero at the ends of a helical line
            "w_N_per_mm": F_bn_N / sharing["l_sum_mm"]}
