"""
Internal dynamic factor K_v of a spur or helical gear pair by Method C
(ISO_6336-1_2019, clause 6.6: calculation of 6.6.3, Formulae (36)-(38),
Table 11, and the interpolation of 6.6.2 c), Formula (35)).

Warnings
--------
This module cannot be imported: its two import statements name
``axisforge.core.mechanical_system.Parallel_Axis_systems``, which does not
exist (``ModuleNotFoundError``). The docstrings below describe the code as
written.
"""

import numpy as np

from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import SpurHelicalGearMeshing
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import SpurHelicalMeshLink


# Table 11 — K1 and K2 values (ISO 6336-1 Table 11)
_K1_SPUR    = {3: 2.1, 4: 3.9, 5: 7.5, 6: 14.9, 7: 26.8, 8: 39.1, 9: 52.8, 10: 76.6, 11: 102.6}
_K1_HELICAL = {3: 1.9, 4: 3.5, 5: 6.7, 6: 13.3, 7: 23.9, 8: 34.8, 9: 47.0, 10: 68.2, 11:  91.4}
_K2_SPUR    = 0.0193
_K2_HELICAL = 0.0087


class DynamicFactorC:
    """
    Internal dynamic factor K_v of one gear mesh by Method C.

    Evaluates K_v = 1 + (K_1/w + K_2) (v z_1/100) K_3 sqrt(u^2/(1+u^2)) with
    K_1, K_2 of Table 11 and K_3 of Formulae (37)-(38), for the subcritical
    range, with w = max(F_t K_A K_gamma/b, 100) [N/mm]. The result is stored
    in ``self.Kv`` [-].

    Attributes
    ----------
    link : SpurHelicalMeshLink
        Mesh link given to the constructor.
    meshing : SpurHelicalGearMeshing
        ``link.meshing``.
    forces : dict
        Result of ``meshing.forces`` (F_t, F_r, F_a, F_n in [N]).
    gear_tolerance_class : int
        Tolerance class, key of the K_1 tables [-].
    rpm_driver, KA
        The constructor arguments of the same name, stored unchanged.
    Kv : float
        Dynamic factor K_v [-].
    u_iso : float
        Gear ratio max(z)/min(z) [-]; the pinion is gear1 when z1 <= z2.
    pinion, pinion_rpm : str, float
        Pinion name and speed [min^-1]; ``rpm_driver * u_iso`` if the driver
        is not the pinion.
    v : float
        Circumferential speed [m/s], see Warnings.
    specific_loading : float
        F_t K_A K_gamma / b [N/mm] (K_gamma = ``link.meshing_load_factor``).
    is_helical : bool
        True if ``gear1_w.beta_deg > 0``.

    Warnings
    --------
    This class cannot be imported (see the module docstring).

    Differs from ISO_6336-1_2019 (as coded; none is fixed here):

    - v: the code uses ``dl`` of the ``<pinion>_w`` copy
      (v = pi dl n_1/60 000). ``SpurHelicalGearMeshing`` sets
      ``gear1_w.dl = 2 al/(u + 1)`` and ``gear2_w.dl = 2 u al/(u + 1)`` with
      u = max(z)/min(z), so ``dl`` is the working pitch diameter of the
      pinion only when z1 <= z2; when z1 > z2 the value read is u times
      larger. 4.2.2, Formula (4) and Table 2 define v at the reference circle
      (v = d n/19 098).
    - F_t: ``forces`` returns F_t = T1/(rl1/1000) with rl1 = ``gear1_w.rl``;
      4.2.2, Formula (1) defines F_t = 2000 T/d at the reference cylinder.
      As for ``dl`` above, ``gear1_w.rl`` is the working pitch radius of gear1
      only when z1 <= z2; when z1 > z2 it is the radius of the smaller gear
      (gear2) and F_t, hence the specific load, is u times larger. The face
      width in the specific load is ``gear1_w.b``.
    - Helical gear with eps_beta >= 1: the code returns the value computed
      with the SPUR row of Table 11 (K_v alpha); 6.6.3 a) prints Formula (36)
      with the helical row there. For eps_beta < 1 the interpolation is
      Formula (35) as printed.
    - ``is_helical`` reads ``gear1_w.beta_deg``; the gear class
      (``spur_helical_gear.py``) defines ``beta_n_deg`` [deg] and ``beta``
      [rad], and no ``beta_deg``. The standard prints no test "beta > 0".
    - ``_K1_K2`` range-checks the tolerance class (3 to 11) but ``_Kv`` does
      not call it; an unlisted class gives ``KeyError`` there.
    - ``applicability`` tests K_3 < 0; the fields of application of 6.6.1 are
      (v z_1/100) sqrt(u^2/(1+u^2)) < 10 m/s (and < 3 m/s for all
      cylindrical gears), beta <= 30 deg, z_1 < 50. None of these is tested.
    - The graphical method with f_F (Formula (34), Tables 9 and 10) is not
      implemented.

    Notes
    -----
    The torque T1 acts on gear1 (``SpurHelicalGearMeshing.forces``) whether
    or not gear1 is the pinion. Assumptions of Method C (steel solid disc
    wheels, alpha_t = 20 deg, C_a = 0 um, see 6.4.3) are not checked.

    References
    ----------
    .. [1] ISO_6336-1_2019, 6.6.3, Formulae (36)-(38), Table 11.
    .. [2] ISO_6336-1_2019, 6.6.2 c), Formula (35).
    """

    def __init__(self,
                 link: SpurHelicalMeshLink,
                 T1: float,
                 rpm_driver: float,
                 KA: float,
                 gear_tolerance_class: int,
                 rotation_dir: float = 1):
        """
        Compute K_v at construction.

        Parameters
        ----------
        link : SpurHelicalMeshLink
            Mesh link; provides ``meshing``, ``phi_deg`` [deg] and
            ``meshing_load_factor`` (K_gamma) [-].
        T1 : float
            Driving torque on gear1 [N m] (unit of ``forces``).
        rpm_driver : float
            Speed of the driving gear [min^-1].
        KA : float
            Application factor K_A [-].
        gear_tolerance_class : int
            Tolerance class, 3 to 11 [-], key of the K_1 tables. Table 11
            heads the columns "Tolerance class as specified in ISO
            1328-1:2013" (that standard was not read) and notes "Consider the
            worst tolerance class between pinion and gear".
        rotation_dir : float, optional
            +1 or -1, sign of gear1 rotation, passed to ``forces`` [-].

        Raises
        ------
        AttributeError
            At ``gear1_w.beta_deg`` for a ``SpurHelicalGear`` (the class
            defines ``beta_n_deg`` and ``beta`` only), i.e. for every
            construction with the current gear class; read statically, the
            module cannot be run.
        """
        self.link                            = link
        self.meshing: SpurHelicalGearMeshing = link.meshing
        self.rpm_driver                      = rpm_driver
        self.KA                              = KA
        self.gear_tolerance_class            = gear_tolerance_class

        # --- gear ratio ---
        z1, z2 = self.meshing.gear1_w.z, self.meshing.gear2_w.z
        self.u_iso = max(z1, z2) / min(z1, z2)

        pinion = "gear1" if z1 <= z2 else "gear2"
        if self.meshing.driver != pinion:
            pinion_rpm = rpm_driver * self.u_iso
        else:
            pinion_rpm = rpm_driver

        self.pinion     = pinion
        self.pinion_rpm = pinion_rpm

        # --- pitch line velocity at pinion [m/s] ---
        self.v = (np.pi * getattr(self.meshing, f"{pinion}_w").dl / 1000) * pinion_rpm / 60.0

        # --- forces & specific loading ---
        self.forces = self.meshing.forces(T1, self.link.phi_deg, rotation_dir=rotation_dir)
        Ft = abs(self.forces["Ft"])
        self.specific_loading = Ft * KA * self.link.meshing_load_factor / self.meshing.gear1_w.b

        # --- is helical? ---
        self.is_helical = self.meshing.gear1_w.beta_deg > 0.0

        # --- compute Kv ---
        self.Kv = self._Kv()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _K1_K2(self) -> tuple[float, float]:
        """
        K_1 and K_2 of Table 11 for the tolerance class and gear type
        (numerical values; no unit printed).

        Returns
        -------
        K1, K2 : float
            Helical row if ``is_helical`` else spur row. Formula (36) is a
            numerical-value equation: w in [N/mm], v in [m/s].

        Raises
        ------
        ValueError
            If the tolerance class is not in 3..11.

        Notes
        -----
        ``_Kv`` does not call this method (see class Warnings).

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.6.3, Table 11.
        """
        tc = self.gear_tolerance_class
        if tc not in range(3, 12):
            raise ValueError(f"Tolerance class {tc} out of range 3–11.")
        if self.is_helical:
            return _K1_HELICAL[tc], _K2_HELICAL
        return _K1_SPUR[tc], _K2_SPUR

    def _speed_term(self) -> float:
        """
        Speed term (v z_1/100) sqrt(u^2/(1+u^2)) [m/s].

        v in [m/s] (see class Warnings), z_1 the pinion tooth number [-].

        Returns
        -------
        speed_term : float
            (v z_1/100) sqrt(u^2/(1+u^2)) [m/s].

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.6.3, Formula (36).
        """
        z1 = getattr(self.meshing, f"{self.pinion}_w").z
        u  = self.u_iso
        return (self.v * z1 / 100.0) * np.sqrt(u**2 / (1.0 + u**2))

    def _K3(self, speed_term: float) -> float:
        """
        Factor K_3 of Formulae (37) and (38) (numerical value; speed term in
        [m/s]).

        Parameters
        ----------
        speed_term : float
            (v z_1/100) sqrt(u^2/(1+u^2)) [m/s].

        Returns
        -------
        K3 : float
            2,0 if speed_term <= 0,2, else -0,357 speed_term + 2,071.

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.6.3, Formulae (37)-(38).
        """
        if speed_term <= 0.2:
            return 2.0
        return -0.357 * speed_term + 2.071

    # ------------------------------------------------------------------
    # Kv (Method C, eq. 36)
    # ------------------------------------------------------------------

    def _Kv(self) -> float:
        """
        Dynamic factor K_v by Method C [-].

        Returns
        -------
        Kv : float
            Kv_alpha = 1 + (K_1/w + K_2) speed_term K_3, with the SPUR row of
            Table 11, w = max(specific_loading, 100) [N/mm]. Returned for a
            spur gear, and for a helical gear with eps_beta >= 1. For a
            helical gear with eps_beta < 1: Kv_alpha - eps_beta (Kv_alpha -
            Kv_beta), Kv_beta from the helical row.

        Raises
        ------
        KeyError
            If ``gear_tolerance_class`` is not a key 3..11 of the tables.

        Warnings
        --------
        Differs from ISO_6336-1_2019, 6.6.3 a), Formula (36): for a helical
        gear with eps_beta >= 1 the standard takes K_1, K_2 from the helical
        row; the code returns the spur-row value. Formula (35) for
        eps_beta < 1 is as printed. No limit to [1,0; 2,0] is applied
        (ISO_6336-1_2019, 6.2.6 and 6.3).

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.6.3, Formulae (36)-(38), Table 11.
        .. [2] ISO_6336-1_2019, 6.6.2 c), Formula (35).
        """
        speed_term = self._speed_term()
        K3         = self._K3(speed_term)
        w_t        = max(self.specific_loading, 100.0)

        eps_beta = self.meshing.epslon_beta

        # spur case (a) — always needed
        K1_spur, K2_spur = _K1_SPUR[self.gear_tolerance_class], _K2_SPUR
        Kv_alpha = 1.0 + (K1_spur / w_t + K2_spur) * speed_term * K3

        if not self.is_helical or eps_beta >= 1.0:
            # spur gear OR helical with ε_β ≥ 1 → use eq. 36 directly
            return Kv_alpha

        # helical with ε_β < 1 → interpolate with eq. 35
        K1_hel, K2_hel = _K1_HELICAL[self.gear_tolerance_class], _K2_HELICAL
        Kv_beta = 1.0 + (K1_hel / w_t + K2_hel) * speed_term * K3

        return Kv_alpha - eps_beta * (Kv_alpha - Kv_beta)   # eq. 35

    # ------------------------------------------------------------------
    # Validity check
    # ------------------------------------------------------------------

    def applicability(self) -> list[str]:
        """
        Check whether K_3 of Formula (38) is still positive.

        Returns
        -------
        errors : list of str
            One message if speed_term > 0,2 and K_3 < 0, else empty.

        Warnings
        --------
        Differs from ISO_6336-1_2019, 6.6.1: the fields of application there
        are (v z_1/100) sqrt(u^2/(1+u^2)) < 10 m/s (a), < 3 m/s for all
        cylindrical gears (g), beta <= 30 deg (d), z_1 < 50 (e). The code
        tests "K_3 < 0" only, which the standard does not print.

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.6.1.
        """
        errors = []
        speed_term = self._speed_term()
        if speed_term > 0.2 and self._K3(speed_term) < 0:
            errors.append(
                f"K3 = {self._K3(speed_term):.3f} < 0 — speed term too high for Method C; "
                f"consider Method B."
            )
        return errors