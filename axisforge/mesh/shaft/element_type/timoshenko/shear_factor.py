"""
mesh/shaft/element_type/timoshenko_selective_integration/timoshenko/shear_factor.py
"""

class ShearFactor:
    
    def cowper_factor(self, v: float, ratio: float = 0.0) -> float:
        """
        Cowper (1966) shear correction factor.
        ratio = inner_radius / outer_radius (0.0 for solid section).
        """
        if ratio == 0.0:
            # solid circular section
            return 6 * (1 + v) / (7 + 6 * v)
        else:
            # hollow circular section (Cowper 1966)
            m2 = ratio**2
            num = 6 * (1 + v) * (1 + m2)**2
            den = (7 + 6 * v) * (1 + m2)**2 + (20 + 12 * v) * m2
            return num / den

    def hutchinson_factor(self, v: float, ratio: float = 0.0) -> float:
        """
        Hutchinson (2001) shear correction factor.
        ratio = inner_radius / outer_radius (0.0 for solid section).
        """
        if ratio == 0.0:
            # solid circular section — confirmed formula
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
        """
        Returns the shear correction factor k such that the transverse
        shear stiffness term is K*G*A = k * G * A.

        kGA_override : if given, back-computes the EFFECTIVE shear
            factor implied by that TARGET K*G*A value --
            k_eff = kGA_override / (G*A) -- instead of looking one up
            from cowper/hutchinson; `theory` is ignored in that case.
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