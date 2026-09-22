# test_shear_factor.py
"""Testes de element_type/shear_factor.py -- ShearFactor.cowper_factor,
hutchinson_factor e o dispatcher shear_correction_factor. Fórmulas
verificadas contra as expressões escritas no próprio módulo (Cowper 1966,
Hutchinson 2001); o objetivo aqui é apanhar regressões, não re-derivar
a física."""
import pytest

from axisforge.mesh.shaft.element_type.shear_factor import ShearFactor

sf = ShearFactor()


class TestCowperFactor:
    def test_solid_section_matches_formula(self):
        v = 0.3
        expected = 6 * (1 + v) / (7 + 6 * v)
        assert sf.cowper_factor(v) == pytest.approx(expected)

    def test_hollow_section_matches_formula(self):
        v, ratio = 0.3, 0.5
        m2 = ratio ** 2
        num = 6 * (1 + v) * (1 + m2) ** 2
        den = (7 + 6 * v) * (1 + m2) ** 2 + (20 + 12 * v) * m2
        expected = num / den
        assert sf.cowper_factor(v, ratio) == pytest.approx(expected)

    def test_hollow_formula_reduces_to_solid_value_at_ratio_zero(self):
        """ratio=0.0 cai explicitamente no ramo 'solid' (if ratio ==
        0.0), mas a fórmula hollow também tem de dar o mesmo valor no
        limite -- caso contrário há uma descontinuidade artificial em
        ratio=0."""
        v = 0.3
        solid = sf.cowper_factor(v, 0.0)
        m2 = 0.0
        num = 6 * (1 + v) * (1 + m2) ** 2
        den = (7 + 6 * v) * (1 + m2) ** 2 + (20 + 12 * v) * m2
        assert solid == pytest.approx(num / den)

    def test_factor_between_zero_and_one_for_typical_v(self):
        assert 0.0 < sf.cowper_factor(0.3) < 1.0

    @pytest.mark.parametrize("ratio", [0.2, 0.5, 0.8])
    def test_hollow_factor_stays_positive(self, ratio):
        assert sf.cowper_factor(0.3, ratio) > 0.0


class TestHutchinsonFactor:
    def test_solid_section_matches_formula(self):
        v = 0.3
        expected = 6 * (1 + v) ** 2 / (7 + 12 * v + 4 * v ** 2)
        assert sf.hutchinson_factor(v) == pytest.approx(expected)

    def test_hollow_section_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Hutchinson"):
            sf.hutchinson_factor(0.3, ratio=0.5)


class TestShearCorrectionFactorDispatch:
    def test_cowper_theory_matches_direct_call(self):
        got = sf.shear_correction_factor(0.3, 0.0, 210_000.0, 500.0, theory="cowper")
        assert got == pytest.approx(sf.cowper_factor(0.3, 0.0))

    def test_hutchinson_theory_matches_direct_call(self):
        got = sf.shear_correction_factor(0.3, 0.0, 210_000.0, 500.0, theory="hutchinson")
        assert got == pytest.approx(sf.hutchinson_factor(0.3, 0.0))

    def test_unknown_theory_raises(self):
        with pytest.raises(ValueError, match="Unknown shear correction theory"):
            sf.shear_correction_factor(0.3, 0.0, 210_000.0, 500.0, theory="bogus")

    def test_kGA_override_ignores_theory(self):
        v, E, A = 0.3, 210_000.0, 500.0
        G = E / (2 * (1 + v))
        target_kGA = 123456.0
        k_eff = sf.shear_correction_factor(v, 0.0, E, A, theory="bogus", kGA_override=target_kGA)
        assert k_eff == pytest.approx(target_kGA / (G * A))

    def test_kGA_override_recovers_target_stiffness(self):
        v, E, A = 0.3, 210_000.0, 500.0
        G = E / (2 * (1 + v))
        target_kGA = 98765.0
        k_eff = sf.shear_correction_factor(v, 0.0, E, A, kGA_override=target_kGA)
        assert k_eff * G * A == pytest.approx(target_kGA)
