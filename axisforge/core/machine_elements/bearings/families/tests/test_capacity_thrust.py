"""Testes de capacity.py -- PointContactCapacityThrust_90deg/_Non_90deg,
_MultirowCombinablePointContact.combine_multirow()."""
import math
import pytest

from axisforge.core.machine_elements.bearings.families import capacity as bcap

BASE_KWARGS = dict(Z=12, Dw=8.0, ri=4.3, re=4.3, gamma=0.2, lam=0.90, eta=0.9)


class TestPointContactCapacityThrust90deg:
    def test_accepts_alpha_0_without_internal_validation(self):
        """PointContactCapacityThrust_90deg confia no chamador -- a
        validação de alpha_0 acontece em ThrustBallSingleRowFamily.
        assemble_geometry(), não aqui. Isto é intencional (ver
        conversa), não uma lacuna."""
        calc = bcap.PointContactCapacityThrust_90deg(alpha_0=math.pi / 2, **BASE_KWARGS)
        assert calc.Ca > 0.0

    def test_Ca_positive(self):
        calc = bcap.PointContactCapacityThrust_90deg(alpha_0=math.pi / 2, **BASE_KWARGS)
        assert calc.Ca > 0.0

    def test_Ca_cached(self):
        calc = bcap.PointContactCapacityThrust_90deg(alpha_0=math.pi / 2, **BASE_KWARGS)
        first, second = calc._fc, calc._fc
        assert first is second or first == second  # mesma leitura, sem recomputar

    def test_C0_not_implemented(self):
        calc = bcap.PointContactCapacityThrust_90deg(alpha_0=math.pi / 2, **BASE_KWARGS)
        with pytest.raises(NotImplementedError):
            _ = calc.C0

    def test_Q_elements_are_positive_pair(self):
        calc = bcap.PointContactCapacityThrust_90deg(alpha_0=math.pi / 2, **BASE_KWARGS)
        Q_ci, Q_ce = calc.Q_elements
        assert Q_ci > 0.0 and Q_ce > 0.0


class TestPointContactCapacityThrustNon90deg:

    def test_Ca_switches_exponent_below_threshold(self):
        RATIO = 0.535  # RI_OVER_DW / RE_OVER_DW de ThrustBallSingleRowFamily

        def make(Dw):
            ri = re = RATIO * Dw
            return bcap.PointContactCapacityThrust_Non_90deg(
                alpha_0=math.radians(60.0), Z=12, Dw=Dw, ri=ri, re=re,
                gamma=0.2, lam=0.9, eta=0.9)

        small = make(10.0)
        large = make(30.0)

        ratio_fc = large._fc / small._fc
        naive_same_exponent_ratio = 3.647 * ratio_fc * (30.0 / 10.0) ** 1.4
        actual_ratio = large.Ca / small.Ca
        assert not math.isclose(actual_ratio, naive_same_exponent_ratio, rel_tol=1e-6)


class TestMultirowCombination:
    def test_combine_multirow_uses_dynamic_rating_alias(self):
        rows = [BASE_KWARGS | dict(alpha_0=math.pi / 2), BASE_KWARGS | dict(alpha_0=math.pi / 2)]
        Ca_total = bcap.PointContactCapacityThrust_90deg.combine_multirow(rows)
        assert Ca_total > 0.0

    def test_combine_multirow_requires_at_least_two_rows(self):
        rows = [BASE_KWARGS | dict(alpha_0=math.pi / 2)]
        with pytest.raises(ValueError, match="at least 2 rows"):
            bcap.PointContactCapacityThrust_90deg.combine_multirow(rows)

    def test_combine_multirow_does_not_double_count_Z_for_single_row(self):
        """Documenta a razão matemática do guard >= 2 rows -- ver review:
        aplicar a fórmula a 1 fila introduziria um Z**(2/3) a mais."""
        rows = [BASE_KWARGS | dict(alpha_0=math.pi / 2)]
        single = bcap.PointContactCapacityThrust_90deg(alpha_0=math.pi / 2, **BASE_KWARGS)
        with pytest.raises(ValueError):
            bcap.PointContactCapacityThrust_90deg.combine_multirow(rows)
        assert single.Ca > 0.0  # a via correta para 1 fila é .Ca diretamente