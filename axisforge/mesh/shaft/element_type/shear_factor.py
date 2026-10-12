"""
Shear correction factor k of a circular shaft section, for the transverse shear stiffness k G A of the Timoshenko beam element.
"""

class ShearFactor:
    """Shear correction factor k [-] of a circular section (solid; hollow with the Cowper expression only).

    The transverse shear rigidity of a Timoshenko beam is ``D_s = k G A``
    with ``G = E / (2 (1 + nu))``; this class supplies ``k`` [-]. Stateless.

    Warnings
    --------
    Differs from Onate2013_vol2, Fig. 2.3: the code uses expressions that
    depend on Poisson ratio ``nu`` (named Cowper 1966 and Hutchinson 2001),
    whereas Onate2013_vol2 prints k = 6/7 for a solid circle and
    k = 6/(7 + 20 K^2), K = c/(1 + c^2), c = inner/outer radius, for a tube,
    with no Poisson term (equal to the Cowper expressions of this class at
    nu = 0). Cao2004 (Nomenclature) uses the constant k_s = 0.9 for circular
    sections. Neither Cowper nor Hutchinson is in the library.
    Elem.from_mesh and Elem.from_x_nodes do not pass the radius ratio, so
    for a hollow shaft section the solid-section factor is used with the
    tube A and I (review item). For a tube the solid expression (Cowper:
    ``6 (1 + nu) / (7 + 6 nu)``) and the hollow expression of this class
    (see :meth:`cowper_factor`, evaluated with ``m`` = inner/outer radius)
    give different k, hence different shear rigidity ``k G A``; the
    difference grows with ``m``. Evaluate both for the section at hand.

    References
    ----------
    .. [1] Cowper 1966, Hutchinson 2001 (not in the library; source to find).
    .. [2] Onate2013_vol2, 2.2.3.1, Eq. (2.7a)-(2.7b) (definition of k) and
       Fig. 2.3 (values).
    .. [3] Cao2004, Nomenclature (k_s = 0.9).
    """

    def cowper_factor(self, v: float, ratio: float = 0.0) -> float:
        """Shear correction factor of a circular section, expression named Cowper (1966).

        Parameters
        ----------
        v : float
            Poisson ratio ``nu`` [-].
        ratio : float, optional
            Inner radius / outer radius [-]; 0.0 (exact comparison) selects
            the solid section.

        Returns
        -------
        float
            Shear correction factor k [-].

        Warnings
        --------
        Neither form is printed in a library source. Onate2013_vol2, Fig. 2.3
        gives the same expressions at nu = 0 (6/7 and 6/(7 + 20 K^2)) and
        cites Cowper without printing the nu-dependent form.

        Notes
        -----
        Solid: ``k = 6 (1 + nu) / (7 + 6 nu)``. Hollow, with ``m = ratio``:
        ``k = 6 (1 + nu) (1 + m^2)^2 / ((7 + 6 nu) (1 + m^2)^2
        + (20 + 12 nu) m^2)``.

        References
        ----------
        .. [1] Cowper 1966 (not in the library; source to find).
        """
        if ratio == 0.0:
            # solid circular section: k = 6(1+nu)/(7+6nu)
            return 6 * (1 + v) / (7 + 6 * v)
        else:
            # hollow circular section (Cowper 1966)
            m2 = ratio**2
            num = 6 * (1 + v) * (1 + m2)**2
            den = (7 + 6 * v) * (1 + m2)**2 + (20 + 12 * v) * m2
            return num / den

    def hutchinson_factor(self, v: float, ratio: float = 0.0) -> float:
        """Shear correction factor of a solid circular section, expression named Hutchinson (2001).

        Parameters
        ----------
        v : float
            Poisson ratio ``nu`` [-].
        ratio : float, optional
            Inner radius / outer radius [-]; must be 0.0 (solid).

        Returns
        -------
        float
            Shear correction factor k [-].

        Raises
        ------
        NotImplementedError
            If ``ratio`` is not 0.0 (hollow section not implemented).

        Warnings
        --------
        The expression is not printed in any library source (unchecked).

        Notes
        -----
        Solid: ``k = 6 (1 + nu)^2 / (7 + 12 nu + 4 nu^2)``.

        References
        ----------
        .. [1] Hutchinson 2001 (not in the library; source to find).
        """
        if ratio == 0.0:
            # solid circular section: k = 6(1+nu)^2/(7+12nu+4nu^2); no library source
            return 6 * (1 + v)**2 / (7 + 12 * v + 4 * v**2)
        else:
            raise NotImplementedError(
                "Hutchinson (2001) hollow-section formula ainda não foi "
                "verificada contra a fonte original neste código. "
                "Usa theory='cowper' para secções ocas, ou confirma a "
                "expressão em Hutchinson, ASME J. Appl. Mech. 68 (2001) 87-92 "
                "antes de ativar este ramo."
            )

    def shear_correction_factor(self,
                                v: float,
                                ratio: float, 
                                E: float,
                                A: float, 
                                theory: str = "cowper", *,
                                kGA_override: float | None = None,
    ) -> float:
        """Shear correction factor k such that the shear rigidity is D_s = k G A.

        Parameters
        ----------
        v : float
            Poisson ratio ``nu`` [-].
        ratio : float
            Inner radius / outer radius [-]; 0.0 for a solid section.
        E : float
            Young modulus [MPa]; used only with ``kGA_override``.
        A : float
            Cross-section area [mm^2]; used only with ``kGA_override``.
        theory : str, optional
            ``"cowper"`` or ``"hutchinson"``; ignored if ``kGA_override`` is
            given.
        kGA_override : float or None, optional
            Target shear rigidity ``k G A`` [N]. If given, the effective
            factor ``k_eff = kGA_override / (G A)`` is returned, with
            ``G = E / (2 (1 + nu))`` [MPa], instead of the Cowper or Hutchinson
            expression. ``kGA_override`` is not validated (sign, k_eff > 1).

        Returns
        -------
        float
            Shear correction factor k [-] (``k_eff`` with the override).

        Raises
        ------
        ValueError
            If ``theory`` is not ``"cowper"`` or ``"hutchinson"``.
        NotImplementedError
            For ``"hutchinson"`` with ``ratio`` != 0.0.
        """
        if kGA_override is not None:
            G = E / (2 * (1 + v))
            return kGA_override / (G * A)

        if theory == "cowper":
            return self.cowper_factor(v, ratio)
        elif theory == "hutchinson":
            return self.hutchinson_factor(v, ratio)
        else:
            raise ValueError(f"Unknown shear correction theory: '{theory}'. "
                            f"Expected 'cowper' or 'hutchinson'.")