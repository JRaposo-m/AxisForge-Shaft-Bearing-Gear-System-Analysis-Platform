"""
Internal dynamic factor K_v of a spur or helical gear pair by Method B
(ISO_6336-1_2019, clause 6.5), including the single tooth stiffness c'
(ISO_6336-1_2019, clause 9.3.2).

Warnings
--------
This module cannot be imported: its first two import statements name
``axisforge.core.mechanical_system.Parallel_Axis_systems``, which does not
exist (``ModuleNotFoundError``). The third imports ``get_gear_material`` from
``axisforge.core.materials``, which is defined nowhere in the repository. The
docstrings below describe the code as written.
"""
import numpy as np

from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing import SpurHelicalGearMeshing
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system import SpurHelicalMeshLink
from axisforge.core.materials import get_gear_material 


class DynamicFactor:
    """
    Internal dynamic factor K_v of one gear mesh by Method B.

    From the pinion speed, the reduced mass and the stiffness it forms the
    resonance ratio N, picks the running range (subcritical, main resonance,
    intermediate, supercritical) and evaluates K_v of that range from the
    parameters B_p, B_f, B_k and the factors C_v1..C_v7. The single tooth
    stiffness c' is computed in ``single_stiffness``.

    Attributes
    ----------
    link : SpurHelicalMeshLink
        Mesh link given to the constructor.
    meshing : SpurHelicalGearMeshing
        ``link.meshing``.
    forces : dict
        Result of ``meshing.forces`` (F_t, F_r, F_a, F_n in [N]).
    material_pinion, material_wheel : object
        Result of ``get_gear_material``; ``sigma_Hlim`` and ``E`` are read,
        [MPa] required by Table 8 and Formula (90) (not verifiable: the
        function does not exist).
    rpm_driver, moment_inertia_1, moment_inertia_2, stiffness, KA, f_pb_eff, f_fa_eff
        The constructor arguments of the same name, stored unchanged.
    Kv : float
        K_v [-]. The constructor ends with ``self.Kv = self.Kv()``, so after
        construction the instance attribute (a float) shadows the method
        ``Kv``; the method is then reachable only as ``DynamicFactor.Kv``.
    u_iso : float
        Gear ratio max(z)/min(z) [-].
    pinion, wheel : str
        ``"gear1"`` or ``"gear2"``; the pinion is gear1 when z1 <= z2.
    pinion_rpm : float
        Pinion speed [min^-1].
    v : float
        Circumferential speed [m/s], see Notes.
    m_red : float
        Reduced mass per unit face width [kg/mm] if the moments of inertia
        are in [kg mm^2/mm] (units not stated in the code).
    n_E_pinion : float
        Resonance speed of the pinion n_E1 [min^-1].
    N_pinion, Ns : float
        Resonance ratio N and its lower limit N_S of the main resonance
        range [-].
    range : str
        ``"Subcritical"``, ``"Main Resonance"``, ``"Intermediate"`` or
        ``"Supercritical"``.
    specific_loading : float
        F_t K_A K_gamma / b [N/mm].
    c_prime : float
        Single tooth stiffness c' [N/(mm um)] (set by ``single_stiffness``).
    Cv1 .. Cv7, Cay : float
        Factors of Table 8 [-] (C_ay in [um]).

    Warnings
    --------
    This class cannot be imported (see the module docstring).

    Differs from ISO_6336-1_2019 (as coded; none is fixed here), see also
    the Warnings of ``_C_R``, ``_C_B``, ``single_stiffness`` and
    ``_compute_Bp_Bf_Bk``:

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
    - m_red: Formulae (7)-(8) print m_red = J1* J2*/(J1* r_b2^2 + J2* r_b1^2);
      the code computes I1 I2/(I1 rb1^2 + I2 rb2^2) and reads both rb1 and
      rb2 from ``gear1_w``, so rb2 = rb1.
    - n_E1: Formula (6) uses c_gamma_alpha; the code uses the untyped
      constructor argument ``stiffness``. c_gamma_alpha = c'(0,75 eps_alpha +
      0,25), ISO_6336-1_2019, 9.3.3.1, Formula (93), is not computed anywhere
      in this module.
    - N_S: Formula (11) prints 0,5 + 0,35 sqrt(F_t K_A K_gamma/(100 b)); the
      code's line load is already per unit width [N/mm] and is divided by b
      again, using ``meshing.gear_w.b``, an attribute that does not exist
      (``gear1_w`` and ``gear2_w`` do). The constructor therefore stops with
      ``AttributeError`` for every specific load below 100 N/mm.
    - Ranges: the first test is N < N_S (printed N <= N_S), so N == N_S falls
      through to the last branch (Supercritical). N = 1,5 is classed as
      Intermediate, as in the list of 6.5.3; the headings 6.5.6 and 6.5.7
      print N >= 1,5 as supercritical.
    - K_v limits of 6.2.6 and 6.3 (K_v set to 2 when above 2 outside
      resonance; 1,0 used when below 1,0) are not applied.
    - ``applicability`` evaluates the criterion with ``gear1_w.z``; the
      standard's z_1 is the pinion (6.4.2).

    Notes
    -----
    Sequence (names of the standard in brackets):

    - u = max(z1, z2)/min(z1, z2) (Table 2: u = z2/z1 >= 1).
    - v = pi dl n_1/60000 [m/s], dl = ``<pinion>_w.dl`` [mm] (see Warnings).
    - m_red, n_E1, N: ISO_6336-1_2019, 6.5.3, Formulae (6)-(9).
    - N_S: ISO_6336-1_2019, 6.5.3, Formulae (10)-(12).
    - C_v1..C_v7, C_ay: ISO_6336-1_2019, 6.5.4, Table 8.
    - B_p, B_f, B_k: ISO_6336-1_2019, 6.5.4, Formulae (15)-(17).
    - K_v: Formulae (13)-(14) subcritical, (20) main resonance,
      (21) supercritical, (22) intermediate.

    Assumptions and conventions: the torque T1 acts on gear1 whether or not
    gear1 is the pinion; the mesh load factor is ``link.meshing_load_factor``
    (K_gamma); K_v is not limited to the interval [1,0; 2,0] (see
    Warnings).

    References
    ----------
    .. [1] ISO_6336-1_2019, 6.5 (Method B), Formulae (6)-(22), Table 8.
    .. [2] ISO_6336-1_2019, 9.3, Formulae (81)-(92), Table 13.
    """

    def __init__(self,
                 link: SpurHelicalMeshLink, 
                 T1: float,
                 rpm_driver: float,
                 moment_inertia_1: float,
                 moment_inertia_2: float,
                 stiffness: float,
                 KA: float,
                 f_pb_eff: float,
                 f_fa_eff: float,
                 rotation_dir: float = 1):
        """
        Compute K_v at construction (resonance ratio, range, c', K_v).

        Parameters
        ----------
        link : SpurHelicalMeshLink
            Mesh link; provides ``meshing``, ``phi_deg`` [deg] and
            ``meshing_load_factor`` (K_gamma) [-].
        T1 : float
            Driving torque on gear1 [N m] (the unit of
            ``SpurHelicalGearMeshing.forces``, which divides by rl1/1000).
        rpm_driver : float
            Speed of the driving gear [min^-1].
        moment_inertia_1, moment_inertia_2 : float
            Moments of inertia of gear1 and gear2. Unit not stated in the
            code; for m_red in [kg/mm] with r_b in [mm] they must be
            J* in [kg mm^2/mm] (per unit face width, Table 2).
        stiffness : float
            Stiffness used in n_E1. Unit not stated in the code; Formula (6)
            needs c_gamma_alpha in [N/(mm um)]. Note that the code does not
            compute c_gamma_alpha from c'.
        KA : float
            Application factor K_A [-].
        f_pb_eff : float
            Effective base pitch deviation [um].
        f_fa_eff : float
            Effective profile form deviation f_f_alpha eff [um].
        rotation_dir : float, optional
            +1 or -1, sign of gear1 rotation, passed to ``forces`` [-].

        Raises
        ------
        ValueError
            If the total contact ratio epslon_gamma <= 1,0 (``_C_constants``).
        AttributeError
            If ``specific_loading`` < 100 N/mm: the line computing N_S reads
            ``meshing.gear_w``, which does not exist (read statically, the
            module cannot be run).
        """
        self.link   = link
        self.meshing: SpurHelicalGearMeshing = link.meshing
        self.rpm_driver                      = rpm_driver
        self.moment_inertia_1                = moment_inertia_1
        self.moment_inertia_2                = moment_inertia_2
        self.stiffness                       = stiffness
        self.KA                              = KA
        self.f_pb_eff                        = f_pb_eff
        self.f_fa_eff                        = f_fa_eff

        driver = self.meshing.driver

        z1, z2 = self.meshing.gear1_w.z, self.meshing.gear2_w.z
        self.u_iso = max(z1, z2) / min(z1, z2)

        pinion = "gear1" if z1 <= z2 else "gear2"
        self.wheel  = "gear1" if z1 > z2 else "gear2"

        if driver != pinion:
            pinion_rpm = rpm_driver * self.u_iso
        else:
            pinion_rpm = rpm_driver

        self.pinion = pinion
        self.pinion_rpm = pinion_rpm
        self.v = (np.pi * getattr(self.meshing, f"{pinion}_w").dl / 1000) * pinion_rpm / 60.0

        rb1 = self.meshing.gear1_w.rb
        rb2 = self.meshing.gear1_w.rb
        self.m_red = self.moment_inertia_1 * self.moment_inertia_2 / (
            self.moment_inertia_1 * rb1**2 + self.moment_inertia_2 * rb2**2)
        
        self.n_E_pinion = 30000 * np.sqrt(stiffness/self.m_red) / (np.pi * getattr(self.meshing, f"{pinion}_w").z)
        self.N_pinion = self.pinion_rpm / self.n_E_pinion

        self.forces = self.meshing.forces(T1, self.link.phi_deg, rotation_dir=rotation_dir)
        Ft = abs(self.forces["Ft"])
        self.specific_loading = Ft * KA * self.link.meshing_load_factor / self.meshing.gear1_w.b

        if self.specific_loading < 100:
            self.Ns = 0.5 + 0.35 * np.sqrt(self.specific_loading/(100 * self.meshing.gear_w.b))
        else:
            self.Ns = 0.85

        if self.N_pinion < self.Ns:
            self.range = "Subcritical"
        elif self.N_pinion > self.Ns and self.N_pinion <= 1.15:
            self.range = "Main Resonance"
        elif self.N_pinion > 1.15 and self.N_pinion <= 1.5:
            self.range = "Intermediate"
        else:
            self.range = "Supercritical"

        self.material_pinion = get_gear_material(
            getattr(self.meshing, f"{self.pinion}").material_id
        )

        self.material_wheel = get_gear_material(
            getattr(self.meshing, f"{self.wheel}").material_id
        )

        self._C_constants()
        self.single_stiffness()   # popula self.c_prime
        self.Kv = self.Kv()


    # ------------------------------------------------------------------
    # Functions for diferent ranges
    # ------------------------------------------------------------------

    # NOTE: Here it shall be divided with the helpers and principal functions

    def _C_constants(self) -> None:
        """
        Factors C_v1..C_v7 and running-in tip relief C_ay for Method B.

        Called once at construction; the results are stored as the
        attributes ``Cv1`` .. ``Cv7`` [-] and ``Cay`` [um].

        Raises
        ------
        ValueError
            If the total contact ratio ``meshing.epslon_gamma`` <= 1,0.

        Notes
        -----
        C_v1..C_v6 use the column 1 < eps_gamma <= 2 or eps_gamma > 2 of
        Table 8; C_v7 uses the three columns split at 1,5 and 2,5.
        ``Cay = 0.5 (Cay_1 + Cay_2)`` with
        ``Cay_i = (1/18) (sigma_Hlim_i/97 - 18,45)^2 + 1,5``, sigma_Hlim in
        [MPa] and C_ay in [um] (Table 8, NOTE 1; units from the key of
        Figure 3). The mean of NOTE 2 is taken always; the standard prints it
        for different materials. C_ay is computed here but not used by the
        rest of the module. Table 8 has no column for eps_gamma <= 1; the
        code raises there.

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.5.4, Table 8.
        """
        eg = self.meshing.epslon_gamma

        if eg <= 1.0:
            raise ValueError(
                f"epslon_gamma={eg:.4f} <= 1.0 — contact ratio out of range for Method B."
            )

        # --- Cv1 to Cv6 ---
        if eg <= 2.0:
            Cv1 = 0.32
            Cv2 = 0.34
            Cv3 = 0.23
            Cv4 = 0.90
            Cv5 = 0.47
            Cv6 = 0.47
        else:
            Cv1 = 0.32
            Cv2 = 0.57  / (eg - 0.3)
            Cv3 = 0.096 / (eg - 1.56)
            Cv4 = (0.57 - 0.05 * eg) / (eg - 1.44)
            Cv5 = 0.47
            Cv6 = 0.12  / (eg - 1.74)

        # --- Cv7 ---
        if eg <= 1.5:
            Cv7 = 0.75
        elif eg <= 2.5:
            Cv7 = 0.125 * np.sin(np.pi * (eg - 2.0)) + 0.875
        else:
            Cv7 = 1.0

        # --- Cay ---
        Cay_1 = (1/18) * (self.material_pinion.sigma_Hlim / 97 - 18.45) ** 2 + 1.5
        Cay_2 = (1/18) * (self.material_wheel.sigma_Hlim  / 97 - 18.45) ** 2 + 1.5
        Cay   = 0.5 * (Cay_1 + Cay_2)

        self.Cv1 = Cv1
        self.Cv2 = Cv2
        self.Cv3 = Cv3
        self.Cv4 = Cv4
        self.Cv5 = Cv5
        self.Cv6 = Cv6
        self.Cv7 = Cv7
        self.Cay = Cay


    # ------------------------------------------------------------------
    # Single tooth stiffness c' (ISO 6336-1 Clause 9)
    # ------------------------------------------------------------------

    @staticmethod
    def _q_prime(zn1: float, zn2: float, x1: float, x2: float) -> float:
        """
        Minimum flexibility q' of a pair of teeth [mm um/N].

        Used by ``single_stiffness`` as c'_th = 1/q'.

        Parameters
        ----------
        zn1, zn2 : float
            Virtual (equivalent) numbers of teeth of gear 1 and gear 2 [-].
        x1, x2 : float
            Profile shift coefficients [-].

        Returns
        -------
        q_prime : float
            q' = C1 + C2/zn1 + C3/zn2 + C4 x1 + C5 x1/zn1 + C6 x2 + C7 x2/zn2
            + C8 x1^2 + C9 x2^2 [mm um/N].

        Notes
        -----
        ``single_stiffness`` passes gear1 as index 1 and gear2 as index 2; in
        the standard index 1 is the pinion (the two coincide when z1 <= z2).

        Coefficients C1..C9 are those of Table 13. Basic rack assumed:
        alpha_P = 20 deg, h_aP = m_n, h_fP = 1,2 m_n, rho_fP = 0,2 m_n.
        The standard states the formula for x_1 >= x_2 and
        -0,5 <= x_1 + x_2 <= 2,0; the code does not check this range.

        References
        ----------
        .. [1] ISO_6336-1_2019, 9.3.2.2, Formula (84), Table 13.
        """
        C1 = 0.04723
        C2 = 0.15551
        C3 = 0.25791
        C4 = -0.00635
        C5 = -0.11654
        C6 = -0.00193
        C7 = -0.24188
        C8 =  0.00529
        C9 =  0.00182

        return (C1
                + C2 / zn1 + C3 / zn2
                + C4 * x1  + C5 * x1 / zn1
                + C6 * x2  + C7 * x2 / zn2
                + C8 * x1**2 + C9 * x2**2)

    @staticmethod
    def _C_R(bs: float | None, b: float, s_R: float, mn: float) -> float:
        """
        Gear blank factor C_R of the single tooth stiffness [-].

        Parameters
        ----------
        bs : float or None
            Central web thickness b_s [mm]; None means solid disc.
        b : float
            Face width [mm].
        s_R : float
            Gear rim thickness [mm].
        mn : float
            Normal module [mm].

        Returns
        -------
        C_R : float
            1,0 if ``bs`` is None or ``bs >= b``; otherwise
            1 + ln(bs/b)/(5 exp(s_R/(5 mn))) [-].

        Warnings
        --------
        Differs from ISO_6336-1_2019, Formula (87): the substitutions printed
        in 9.3.2.4 b), after Formula (87) (b_s/b within [0,2; 1,2], s_R/m_n >= 1) are not
        applied; the code returns 1,0 when b_s >= b (the printed upper clip
        is b_s/b = 1,2).
        In ``single_stiffness`` the call passes ``meshing.b`` as b.

        References
        ----------
        .. [1] ISO_6336-1_2019, 9.3.2.4, Formulae (86)-(87), Figure 19.
        """
        if bs is None or bs >= b:
            return 1.0
        return 1.0 + np.log(bs / b) / (5.0 * np.exp(s_R / (5.0 * mn)))

    @staticmethod
    def _C_B(hfp: float, mn: float, alpha_pn_deg: float) -> float:
        """
        Basic rack factor C_B of one gear [-].

        Parameters
        ----------
        hfp : float
            Dedendum of the basic rack h_fP [mm]. The formula uses hfp/mn, so
            hfp is in [mm] only if the caller passes a length (see
            ``_C_B_pair``).
        mn : float
            Normal module [mm].
        alpha_pn_deg : float
            Normal pressure angle of the basic rack [deg].

        Returns
        -------
        C_B : float
            (1,0 + 0,5 (1,2 - hfp/mn)) (1,0 - 0,02 (20 - alpha_pn_deg)) [-].

        Notes
        -----
        By Formula (88), as coded and as printed, h_fP = 1,2 m_n with
        alpha_Pn = 20 deg gives 1,0 and h_fP = 1,25 m_n gives 0,975 (the
        default of the gear class is hfP = haP + cP = 1,25).

        References
        ----------
        .. [1] ISO_6336-1_2019, 9.3.2.5, Formula (88).
        """
        return (1.0 + 0.5 * (1.2 - hfp / mn)) * (1.0 - 0.02 * (20.0 - alpha_pn_deg))

    def _C_B_pair(self) -> float:
        """
        Basic rack factor C_B of the pair, 0,5 (C_B1 + C_B2) [-].

        Returns
        -------
        C_B : float
            Mean of ``_C_B`` of gear1 and gear2 [-].

        Notes
        -----
        The call passes ``g.hfP * g.mn`` as the dedendum, where ``g.hfP`` of
        the gear class is a dimensionless coefficient (``hfP = haP + cP``,
        multiples of m_n, as used in ``df = d + 2 mn (x - hfP)``). Inside
        ``_C_B`` the ratio hfp/mn is then ``g.hfP * g.mn / g.mn = g.hfP``, so
        the formula receives the coefficient h_fP/m_n as intended. The mean
        is applied always (the standard prints it for the case where the
        pinion basic rack dedendum differs from that of the wheel).

        References
        ----------
        .. [1] ISO_6336-1_2019, 9.3.2.5, Formula (89).
        """
        g1 = self.meshing.gear1
        g2 = self.meshing.gear2
        CB1 = self._C_B(g1.hfP * g1.mn, g1.mn, g1.alpha_n_deg)
        CB2 = self._C_B(g2.hfP * g2.mn, g2.mn, g2.alpha_n_deg)
        return 0.5 * (CB1 + CB2)

    def single_stiffness(self,
                         bs: float | None = None,
                         s_R: float = 0.0,
                         Ky: float = 1.0,
                         steel_only: bool = True) -> float:
        """
        Single tooth stiffness c' of the gear pair [N/(mm um)].

        Maximum stiffness of one pair of teeth per unit face width, used in
        B_p, B_f and B_k. The result is also stored in ``self.c_prime``.

        Parameters
        ----------
        bs : float or None, optional
            Central web thickness b_s [mm]; None for a solid disc (C_R = 1,0).
        s_R : float, optional
            Gear rim thickness [mm]; used only if ``bs`` is not None and
            ``bs`` < b. The default 0.0 gives s_R/m_n = 0 (see ``_C_R``
            Warnings).
        Ky : float, optional
            Not used by the code [-].
        steel_only : bool, optional
            If True (default) the steel/steel value is returned. If False the
            material correction of Formulae (90)-(91) is applied, with E from
            the pinion and wheel materials [MPa].

        Returns
        -------
        c_prime : float
            c' [N/(mm um)].

        Warnings
        --------
        The constant E_St = 206 000 [MPa] is not printed in the pages of
        ISO_6336-1_2019 read for this review (source to find).

        Notes
        -----
        .. math::

            c' = c'_{th} C_M C_R C_B \\cos\\beta, \\qquad c'_{th} = 1/q'

        with q' from ``_q_prime`` (zn and x of ``gear1_w``, ``gear2_w``),
        C_M = 0,8, C_R from ``_C_R`` (called with ``meshing.b`` and the
        normal module of gear1), C_B from ``_C_B_pair`` and beta the helix
        angle of ``meshing.gear1`` [rad].

        If ``specific_loading`` < 100 N/mm, c' is recomputed as above times
        ``(specific_loading/100) ** 0.25`` (Formula (92)). Not reached through
        the constructor, which fails earlier for this load (see ``__init__``
        Raises).

        If ``steel_only`` is False, c' is multiplied by E_pair/E_St, with
        E_pair = 2 E1 E2/(E1 + E2) and E_St = 206 000 [MPa] (Formulae (90),
        (91)). Formula (92) is evaluated before the E correction.

        c' is the stiffness of Formulae (15)-(17), as printed;
        c_gamma_alpha of Formula (93), which Formula (6) needs, is not
        computed in this module.

        References
        ----------
        .. [1] ISO_6336-1_2019, 9.3.2.1, Formula (82).
        .. [2] ISO_6336-1_2019, 9.3.2.2, Formulae (83)-(84), Table 13.
        .. [3] ISO_6336-1_2019, 9.3.2.3, Formula (85).
        .. [4] ISO_6336-1_2019, 9.3.2.6 c), Formulae (90)-(91).
        .. [5] ISO_6336-1_2019, 9.3.2.6 e), Formula (92).
        """
        g1_w = self.meshing.gear1_w
        g2_w = self.meshing.gear2_w

        # --- theoretical single stiffness c'_th (eq. 83) ---
        q_prime = self._q_prime(g1_w.zn, g2_w.zn, g1_w.x, g2_w.x)
        c_th = 1.0 / q_prime                          # [N/(mm·µm)]

        # --- correction factors ---
        C_M = 0.8                                      # eq. 85 — solid disc
        C_R = self._C_R(bs, self.meshing.b, s_R, g1_w.mn)
        C_B = self._C_B_pair()
        beta = self.meshing.gear1.beta                 # helix angle [rad]

        c_prime = c_th * C_M * C_R * C_B * np.cos(beta)   # eq. 82

        # --- low specific load correction (eq. 92) ---
        specific_load = self.specific_loading          # Ft·KA·Ky / b [N/mm]
        if specific_load < 100.0:
            c_prime = c_th * C_M * C_B * C_R * np.cos(beta) * (specific_load / 100.0) ** 0.25

        # --- material correction for non-steel pairs (eq. 90/91) ---
        if not steel_only:
            E1 = self.material_pinion.E
            E2 = self.material_wheel.E
            E_pair  = 2.0 * E1 * E2 / (E1 + E2)      # eq. 91
            E_St    = 206_000.0                        # [MPa] reference steel
            c_prime = c_prime * (E_pair / E_St)        # eq. 90

        self.c_prime = c_prime
        return c_prime


    def _compute_Bp_Bf_Bk(self) -> tuple[float, float, float]:
        """
        Non-dimensional parameters B_p, B_f, B_k of Method B.

        Returns
        -------
        Bp, Bf, Bk : float
            B_p = c' f_pb_eff / w, B_f = c' f_fa_eff / w,
            B_k = |1 - c' min(Ca1 + Cf2, Ca2 + Cf1) / w| [-], with
            w = ``specific_loading`` = F_t K_A K_gamma / b [N/mm],
            c' [N/(mm um)], f_pb_eff and f_fa_eff [um]. Ca and Cf are read
            from ``gear1_w`` and ``gear2_w``; their unit is not stated in the
            code (they must be [um] for B_k to be dimensionless).

        Warnings
        --------
        Differs from ISO_6336-1_2019, Formula (17): footnote 4 (C_a used only
        for tolerance classes 1 to 5; B_k = 1,0 for classes 6 to 11) and the
        text of 6.5.4 (C_ay in place of C_a for gears without a specified
        profile modification) are not applied; ``Cay`` is computed but not
        used.

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.5.4, Formulae (15), (16), (17).
        """
        Bp = self.c_prime * self.f_pb_eff / self.specific_loading
        Bf = self.c_prime * self.f_fa_eff / self.specific_loading
        Ca_1 = self.meshing.gear1_w.Ca
        Cf_1 = self.meshing.gear1_w.Cf
        Ca_2 = self.meshing.gear2_w.Ca
        Cf_2 = self.meshing.gear2_w.Cf
        Bk = abs(1 - (self.c_prime * min(Ca_1 + Cf_2, Ca_2 + Cf_1)) / self.specific_loading)
        return Bp, Bf, Bk

    # ------------------------------------------------------------------
    # Kv calculations
    # ------------------------------------------------------------------

    def Kv_subcritical(self, Bp: float, Bf: float, Bk: float) -> float:
        """
        K_v in the subcritical range, K_v = N K + 1 [-].

        K = Cv1 Bp + Cv2 Bf + Cv3 Bk; N is ``N_pinion``.

        Parameters
        ----------
        Bp, Bf, Bk : float
            B_p, B_f, B_k [-].

        Returns
        -------
        Kv : float
            K_v [-].

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.5.4, Formulae (13)-(14).
        """
        k = (self.Cv1 * Bp) + (self.Cv2 * Bf) + (self.Cv3 * Bk)
        return (self.N_pinion * k) + 1

    def Kv_main_resonance(self, Bp: float, Bf: float, Bk: float) -> float:
        """
        K_v in the main resonance range, Cv1 Bp + Cv2 Bf + Cv4 Bk + 1 [-].

        Parameters
        ----------
        Bp, Bf, Bk : float
            B_p, B_f, B_k [-].

        Returns
        -------
        Kv : float
            K_v [-].

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.5.5, Formula (20).
        """
        return (self.Cv1 * Bp) + (self.Cv2 * Bf) + (self.Cv4 * Bk) + 1

    def Kv_supercritical(self, Bp: float, Bf: float, Bk: float) -> float:
        """
        K_v in the supercritical range, Cv5 Bp + Cv6 Bf + Cv7 [-].

        ``Bk`` is accepted but not used.

        Parameters
        ----------
        Bp, Bf, Bk : float
            B_p, B_f, B_k [-].

        Returns
        -------
        Kv : float
            K_v [-].

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.5.6, Formula (21).
        """
        return (self.Cv5 * Bp) + (self.Cv6 * Bf) + self.Cv7
    
    def Kv_intermediate(self, Bp: float, Bf: float, Bk: float) -> float:
        """
        K_v in the intermediate range by linear interpolation [-].

        Between K_v(N = 1,15), taken from the main resonance formula, and
        K_v(N = 1,5), taken from the supercritical formula:
        K_v = K_v(1,5) + (K_v(1,15) - K_v(1,5)) (1,5 - N)/0,35.

        Parameters
        ----------
        Bp, Bf, Bk : float
            B_p, B_f, B_k [-].

        Returns
        -------
        Kv : float
            K_v [-].

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.5.7, Formula (22).
        """
        Kv_at_115 = self.Kv_main_resonance(Bp, Bf, Bk)   # N = 1.15
        Kv_at_150 = self.Kv_supercritical(Bp, Bf, Bk)    # N = 1.50
        # linear interpolation between 1.15 and 1.50
        Kv = Kv_at_150 + (Kv_at_115 - Kv_at_150) * (1.50 - self.N_pinion) / 0.35
        return Kv

    def Kv(self) -> float:
        """
        Dynamic factor K_v of the range stored in ``self.range`` [-].

        Returns
        -------
        Kv : float
            K_v [-].

        Notes
        -----
        Calls ``_compute_Bp_Bf_Bk`` and then the ``Kv_*`` method of the range.
        No limit to [1,0; 2,0] is applied (ISO_6336-1_2019, 6.2.6 and 6.3).
        The constructor replaces this method on the instance by its value
        (``self.Kv = self.Kv()``).
        """
        Bp, Bf, Bk = self._compute_Bp_Bf_Bk()
        if self.range == "Subcritical":
            return self.Kv_subcritical(Bp, Bf, Bk)
        elif self.range == "Main Resonance":
            return self.Kv_main_resonance(Bp, Bf, Bk)
        elif self.range == "Intermediate":
            return self.Kv_intermediate(Bp, Bf, Bk)
        else:
            return self.Kv_supercritical(Bp, Bf, Bk)

    # ------------------------------------------------------------------
    # Verification for validity
    # ------------------------------------------------------------------

    def applicability(self) -> list[str]:
        """
        Check the field in which Method B is recommended.

        Returns
        -------
        errors : list of str
            One message if (v z/100) sqrt(u^2/(1+u^2)) < 3 [m/s], else empty.

        Notes
        -----
        z is ``meshing.gear1_w.z``; the standard's z_1 is the pinion, which
        is gear2 when z1 > z2. v is the speed defined in the class Notes.
        ``applicability`` evaluates the criterion with ``gear1_w.z``; the
        standard's z_1 is the pinion (6.4.2).

        References
        ----------
        .. [1] ISO_6336-1_2019, 6.4.2.
        """
        errors = []

        u = self.u_iso
        condition = (self.v * self.meshing.gear1_w.z / 100) * np.sqrt(u**2 / (1 + u**2))
        if condition < 3:
            errors.append(
                f"KV Method B not recommended: "
                f"(v·z1/100)·√(u²/(1+u²)) = {condition:.3f} m/s < 3 m/s. "
                f"Use Method C instead."
            )
        return errors