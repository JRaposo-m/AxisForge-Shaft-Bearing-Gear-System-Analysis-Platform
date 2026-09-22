# test_contact_postprocessing.py
"""Testes de contact_postprocessing.py -- ContactBearingStiffness,
combine_row_L10r, _row_views, DynamicEquivalentReferenceLoadBase (secção
partilhada), depois ball_*/DynamicEquivalentRollingElementLoad/
BallBasicReferenceRatingLife (point contact) e roller_*/
LaminaDynamicEquivalentLoad/RollerBasicReferenceRatingLife (line
contact).

Usa objetos leves (SimpleNamespace) para bearing/result -- nunca as
classes *Result reais, cujo corpo interno não foi visto nesta
conversa; só o contrato de atributos que este módulo lê delas."""
from types import SimpleNamespace

import numpy as np
import pytest

from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281 import (
    contact_postprocessing as pp,
)


def make_ball_row(delta_j, alpha_j=None, phi_Fr=0.0, delta_r=0.0, delta_a=0.0):
    Z = len(delta_j)
    if alpha_j is None:
        alpha_j = np.zeros(Z)
    return SimpleNamespace(delta_j=np.asarray(delta_j, dtype=float), alpha_j=np.asarray(alpha_j, dtype=float),
                           phi_Fr=phi_Fr, delta_r=delta_r, delta_a=delta_a)


def make_roller_row(q_jk, phi_Fr=0.0, delta_r=0.0, delta_a=0.0, x_k=None):
    q_jk = np.asarray(q_jk, dtype=float)
    if x_k is None:
        x_k = np.linspace(-1.0, 1.0, q_jk.shape[1])
    return SimpleNamespace(q_jk=q_jk, phi_Fr=phi_Fr, delta_r=delta_r, delta_a=delta_a, x_k=x_k)


def make_result(*rows):
    return SimpleNamespace(rows=list(rows))


# =====================================================================
# ContactBearingStiffness
# =====================================================================

class TestContactBearingStiffness:
    def test_no_load_regime_when_Fa_is_zero(self):
        row = make_ball_row([1.0], phi_Fr=0.0, delta_r=0.1, delta_a=0.0)
        stiff = pp.ContactBearingStiffness.from_result("b1", row, Fr_xz=10.0, Fr_xy=0.0, Fa=0.0)
        assert stiff.Ka == float("inf")
        assert stiff.Ka_regime == "no_load"

    def test_engaged_regime_when_delta_a_positive(self):
        row = make_ball_row([1.0], delta_r=0.1, delta_a=0.05)
        stiff = pp.ContactBearingStiffness.from_result("b1", row, Fr_xz=10.0, Fr_xy=0.0, Fa=500.0)
        assert stiff.Ka == pytest.approx(500.0 / 0.05)
        assert stiff.Ka_regime == "engaged"

    def test_closing_clearance_regime_when_delta_a_negative(self):
        row = make_ball_row([1.0], delta_r=0.1, delta_a=-0.02)
        stiff = pp.ContactBearingStiffness.from_result("b1", row, Fr_xz=10.0, Fr_xy=0.0, Fa=500.0)
        assert stiff.Ka_regime == "closing_clearance"

    def test_infinite_Ka_when_delta_a_negligible_but_loaded(self):
        row = make_ball_row([1.0], delta_r=0.1, delta_a=0.0)
        stiff = pp.ContactBearingStiffness.from_result("b1", row, Fr_xz=10.0, Fr_xy=0.0, Fa=500.0)
        assert stiff.Ka == float("inf")
        assert stiff.Ka_regime == "engaged"

    def test_infinite_Kr_when_radial_displacement_negligible(self):
        row = make_ball_row([1.0], phi_Fr=0.0, delta_r=0.0, delta_a=0.0)
        stiff = pp.ContactBearingStiffness.from_result("b1", row, Fr_xz=10.0, Fr_xy=0.0, Fa=0.0)
        assert stiff.Kr_xz == float("inf")

    def test_finite_Kr_matches_Fr_over_displacement(self):
        row = make_ball_row([1.0], phi_Fr=0.0, delta_r=0.2, delta_a=0.0)
        stiff = pp.ContactBearingStiffness.from_result("b1", row, Fr_xz=100.0, Fr_xy=0.0, Fa=0.0)
        assert stiff.Kr_xz == pytest.approx(100.0 / 0.2)


class TestBearingStiffnessFunction:
    def test_uses_row_zero_of_the_result(self):
        bearing = SimpleNamespace(label="b1")
        row0 = make_ball_row([1.0], delta_r=0.2, delta_a=0.0)
        result = make_result(row0)
        stiff = pp.bearing_stiffness(bearing, result, Fr_xz=100.0, Fr_xy=0.0, Fa=0.0)
        assert stiff.label == "b1"
        assert stiff.Kr_xz == pytest.approx(500.0)


# =====================================================================
# combine_row_L10r
# =====================================================================

class TestCombineRowL10r:
    def test_requires_at_least_two_rows(self):
        with pytest.raises(ValueError, match=">= 2 rows"):
            pp.combine_row_L10r([10.0], e=2.0)

    def test_rejects_non_positive_values(self):
        with pytest.raises(ValueError, match="must be positive"):
            pp.combine_row_L10r([10.0, 0.0], e=2.0)

    def test_matches_weibull_combination_formula(self):
        e = 2.0
        rows = [10.0, 20.0]
        expected = sum(L ** (-e) for L in rows) ** (-1.0 / e)
        assert pp.combine_row_L10r(rows, e=e) == pytest.approx(expected)

    def test_two_identical_rows_below_single_row_value(self):
        """Combinar 2 filas iguais tem de dar menos vida que 1 fila
        sozinha (mais filas -> mais oportunidades de falhar)."""
        combined = pp.combine_row_L10r([10.0, 10.0], e=1.5)
        assert combined < 10.0


# =====================================================================
# _row_views
# =====================================================================

class TestRowViews:
    def test_single_row_bearing_without_rows_attribute(self):
        bearing = SimpleNamespace(label="b1")
        row = make_ball_row([1.0])
        result = make_result(row)
        views = pp._row_views(bearing, result)
        assert views == [(bearing, row)]

    def test_multirow_wraps_dict_rows_as_simplenamespace(self):
        bearing = SimpleNamespace(label="b1", rows=[{"cp": 1.0}, {"cp": 2.0}])
        rows = [make_ball_row([1.0]), make_ball_row([2.0])]
        result = make_result(*rows)
        views = pp._row_views(bearing, result)
        assert [rb.cp for rb, _ in views] == [1.0, 2.0]
        assert [rr for _, rr in views] == rows

    def test_multirow_leaves_object_rows_untouched(self):
        row_bearing = SimpleNamespace(cp=3.0)
        bearing = SimpleNamespace(label="b1", rows=[row_bearing])
        result = make_result(make_ball_row([1.0]))
        views = pp._row_views(bearing, result)
        assert views[0][0] is row_bearing


# =====================================================================
# DynamicEquivalentReferenceLoadBase -- via as subclasses concretas
# (a própria base tem _PREF_EXPONENT = 0.0, sempre NotImplementedError)
# =====================================================================

class TestDynamicEquivalentReferenceLoadBase:
    def test_base_class_exponent_not_set_raises(self):
        with pytest.raises(NotImplementedError):
            pp.DynamicEquivalentReferenceLoadBase.from_L10r("b1", 100.0, Cr=1000.0)

    def test_requires_Cr_or_Ca(self):
        with pytest.raises(ValueError, match="need Cr and/or Ca"):
            pp.BallDynamicEquivalentReferenceLoad.from_L10r("b1", 100.0)

    def test_rejects_non_positive_L10r(self):
        with pytest.raises(ValueError, match="must be positive"):
            pp.BallDynamicEquivalentReferenceLoad.from_L10r("b1", 0.0, Cr=1000.0)

    def test_computes_Pref_r_and_Pref_a(self):
        L10r = 8.0
        pref = pp.BallDynamicEquivalentReferenceLoad.from_L10r("b1", L10r, Cr=1000.0, Ca=2000.0)
        denom = L10r ** (1.0 / 3.0)
        assert pref.Pref_r == pytest.approx(1000.0 / denom)
        assert pref.Pref_a == pytest.approx(2000.0 / denom)

    def test_Pref_a_is_none_when_Ca_not_given(self):
        pref = pp.BallDynamicEquivalentReferenceLoad.from_L10r("b1", 8.0, Cr=1000.0)
        assert pref.Pref_a is None


# =====================================================================
# Point contact (ball)
# =====================================================================

class TestBallQJ:
    def test_single_row_returns_array_not_list(self):
        bearing = SimpleNamespace(label="b1", cp=100.0)
        result = make_result(make_ball_row([1.0, 2.0]))
        Q = pp.ball_Q_j(bearing, result)
        assert isinstance(Q, np.ndarray)
        assert Q == pytest.approx([100.0 * 1.0 ** 1.5, 100.0 * 2.0 ** 1.5])

    def test_clamps_negative_delta_j_to_zero(self):
        bearing = SimpleNamespace(label="b1", cp=100.0)
        result = make_result(make_ball_row([-5.0, 1.0]))
        Q = pp.ball_Q_j(bearing, result)
        assert Q[0] == 0.0

    def test_multirow_returns_list_of_arrays(self):
        bearing = SimpleNamespace(label="b1", rows=[{"cp": 10.0}, {"cp": 20.0}])
        result = make_result(make_ball_row([1.0]), make_ball_row([1.0]))
        Q = pp.ball_Q_j(bearing, result)
        assert isinstance(Q, list) and len(Q) == 2
        assert Q[0][0] == pytest.approx(10.0)
        assert Q[1][0] == pytest.approx(20.0)


class TestBallPhiJGlobal:
    def test_adds_phi_Fr_and_wraps_to_2pi(self):
        bearing = SimpleNamespace(label="b1", phi_j=np.array([0.0, np.pi]))
        result = make_result(make_ball_row([1.0, 1.0], phi_Fr=3.0 * np.pi / 2.0))
        phi = pp.ball_phi_j_global(bearing, result)
        assert (phi >= 0.0).all() and (phi < 2.0 * np.pi).all()
        assert phi[0] == pytest.approx(3.0 * np.pi / 2.0)


class TestBallContactDistribution:
    def test_shape_is_Z_by_2(self):
        bearing = SimpleNamespace(label="b1", cp=100.0, phi_j=np.array([0.0, np.pi]))
        result = make_result(make_ball_row([1.0, 2.0]))
        dist = pp.ball_contact_distribution(bearing, result)
        assert dist.shape == (2, 2)

    def test_local_frame_ignores_phi_Fr(self):
        bearing = SimpleNamespace(label="b1", cp=100.0, phi_j=np.array([0.5]))
        result = make_result(make_ball_row([1.0], phi_Fr=1.0))
        dist_local = pp.ball_contact_distribution(bearing, result, frame="local")
        assert dist_local[0, 0] == pytest.approx(0.5)

    def test_rejects_unknown_frame(self):
        bearing = SimpleNamespace(label="b1", cp=100.0, phi_j=np.array([0.0]))
        result = make_result(make_ball_row([1.0]))
        with pytest.raises(ValueError, match="frame must be"):
            pp.ball_contact_distribution(bearing, result, frame="bogus")


class TestDynamicEquivalentRollingElementLoad:
    def test_uniform_load_gives_equal_Q_ei_and_Q_ee(self):
        """Q_j uniforme -> a média generalizada de qualquer expoente p
        devolve o próprio valor -- independente de p_rotating/p_stationary,
        um bom teste de regressão que não depende dos expoentes exatos."""
        bearing = SimpleNamespace(label="b1", cp=1.0)
        result = make_ball_row([2.0 ** (2.0 / 3.0)] * 5)  # delta_j tal que Q_j = cp*delta_j^1.5 = 2.0
        dyn = pp.DynamicEquivalentRollingElementLoad.from_distribution(bearing, result)
        assert dyn.Q_ei == pytest.approx(2.0, rel=1e-6)
        assert dyn.Q_ee == pytest.approx(2.0, rel=1e-6)

    def test_from_bearing_result_single_row_no_suffix(self):
        bearing = SimpleNamespace(label="b1", cp=1.0)
        result = make_result(make_ball_row([1.0]))
        out = pp.DynamicEquivalentRollingElementLoad.from_bearing_result(bearing, result)
        assert len(out) == 1
        assert out[0].label == "b1"

    def test_from_bearing_result_multirow_suffixes_labels(self):
        bearing = SimpleNamespace(label="b1", rows=[{"cp": 1.0}, {"cp": 1.0}])
        result = make_result(make_ball_row([1.0]), make_ball_row([1.0]))
        out = pp.DynamicEquivalentRollingElementLoad.from_bearing_result(bearing, result)
        assert [o.label for o in out] == ["b1-row0", "b1-row1"]


class TestBallBasicReferenceRatingLife:
    def test_rejects_non_positive_loads(self):
        with pytest.raises(ValueError, match="must be positive"):
            pp.BallBasicReferenceRatingLife.from_loads("b1", Q_ci=0.0, Q_ei=1.0, Q_ce=1.0, Q_ee=1.0)

    def test_matches_formula(self):
        Q_ci, Q_ei, Q_ce, Q_ee = 100.0, 20.0, 90.0, 25.0
        expected = ((Q_ci / Q_ei) ** (-10.0 / 3.0) + (Q_ce / Q_ee) ** (-10.0 / 3.0)) ** (-9.0 / 10.0)
        life = pp.BallBasicReferenceRatingLife.from_loads("b1", Q_ci, Q_ei, Q_ce, Q_ee)
        assert life.L10r == pytest.approx(expected)


class TestBallBasicReferenceRatingLifeAggregate:
    def test_rejects_mismatched_row_list_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            pp.ball_basic_reference_rating_life("b1", [100.0], [10.0, 10.0], [90.0], [10.0])

    def test_rejects_zero_rows(self):
        with pytest.raises(ValueError, match="got 0 rows"):
            pp.ball_basic_reference_rating_life("b1", [], [], [], [])

    def test_single_row_bearing_L10r_equals_row_L10r(self):
        per_row, L10r_bearing = pp.ball_basic_reference_rating_life(
            "b1", [100.0], [20.0], [90.0], [25.0])
        assert len(per_row) == 1
        assert per_row[0].label == "b1"
        assert L10r_bearing == pytest.approx(per_row[0].L10r)

    def test_multirow_bearing_combines_via_combine_row_L10r(self):
        per_row, L10r_bearing = pp.ball_basic_reference_rating_life(
            "b1", [100.0, 100.0], [20.0, 20.0], [90.0, 90.0], [25.0, 25.0])
        assert [r.label for r in per_row] == ["b1-row0", "b1-row1"]
        expected = pp.combine_row_L10r([r.L10r for r in per_row], e=pp._E_BALL)
        assert L10r_bearing == pytest.approx(expected)


# =====================================================================
# Line contact (roller)
# =====================================================================

class TestRollerQJ:
    def test_single_row_sums_over_laminae(self):
        bearing = SimpleNamespace(label="b1")
        result = make_result(make_roller_row([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        Q = pp.roller_Q_j(bearing, result)
        assert Q == pytest.approx([6.0, 15.0])

    def test_multirow_returns_list(self):
        bearing = SimpleNamespace(label="b1", rows=[{}, {}])
        result = make_result(
            make_roller_row([[1.0, 1.0]]),
            make_roller_row([[2.0, 2.0]]),
        )
        Q = pp.roller_Q_j(bearing, result)
        assert isinstance(Q, list) and len(Q) == 2
        assert Q[0] == pytest.approx([2.0])
        assert Q[1] == pytest.approx([4.0])


class TestRollerContactDistribution:
    def test_shape_is_Z_by_2(self):
        bearing = SimpleNamespace(label="b1", phi_j=np.array([0.0, 1.0]))
        result = make_result(make_roller_row([[1.0, 1.0], [2.0, 2.0]]))
        dist = pp.roller_contact_distribution(bearing, result)
        assert dist.shape == (2, 2)

    def test_rejects_unknown_frame(self):
        bearing = SimpleNamespace(label="b1", phi_j=np.array([0.0]))
        result = make_result(make_roller_row([[1.0, 1.0]]))
        with pytest.raises(ValueError, match="frame must be"):
            pp.roller_contact_distribution(bearing, result, frame="bogus")


class TestRollerLaminaDistribution:
    def test_returns_x_k_and_q_row_for_roller_j(self):
        row = make_roller_row([[1.0, 2.0], [3.0, 4.0]], x_k=np.array([-0.5, 0.5]))
        dist = pp.roller_lamina_distribution(SimpleNamespace(), row, j=1)
        assert dist.shape == (2, 2)
        assert dist[:, 1] == pytest.approx([3.0, 4.0])

    def test_raises_index_error_out_of_range(self):
        row = make_roller_row([[1.0, 2.0]])
        with pytest.raises(IndexError, match="out of range"):
            pp.roller_lamina_distribution(SimpleNamespace(), row, j=5)


class TestStressRiserFactor:
    @pytest.mark.parametrize("n_s", [30, 31, 40])
    def test_shape_and_finiteness(self, n_s):
        f = pp.stress_riser_factor(n_s)
        assert f.shape == (n_s,)
        assert np.all(np.isfinite(f))

    def test_matches_formula(self):
        n_s = 30
        k = np.arange(1, n_s + 1)
        ratio = 1.985 * np.abs(2 * k - n_s - 1) / (2.0 * n_s - 2.0)
        ratio = np.maximum(ratio, 1e-12)
        expected = 1.0 - 0.01 / np.log(ratio)
        assert pp.stress_riser_factor(n_s) == pytest.approx(expected)

    def test_center_is_closer_to_one_than_the_edges(self):
        """k perto do meio -> ratio perto de 0 -> f perto de 1; k nas
        pontas -> ratio maior -> f afasta-se mais de 1."""
        f = pp.stress_riser_factor(31)  # ímpar -- há uma lamina exatamente no centro
        center = f[len(f) // 2]
        edge = f[0]
        assert abs(center - 1.0) < abs(edge - 1.0)


class TestLaminaDynamicEquivalentLoad:
    def test_uniform_load_gives_known_value_regardless_of_exponent(self):
        q_jk = np.full((4, 3), 2.0)  # Z=4 rollers, n_s=3 laminae, tudo igual
        row = make_roller_row(q_jk)
        bearing = SimpleNamespace(label="b1")
        dyn = pp.LaminaDynamicEquivalentLoad.from_distribution(bearing, row)
        f_k = pp.stress_riser_factor(3)
        assert dyn.q_kei == pytest.approx(f_k * 2.0)
        assert dyn.q_kee == pytest.approx(f_k * 2.0)

    def test_from_bearing_result_multirow_suffixes_labels(self):
        bearing = SimpleNamespace(label="b1", rows=[{}, {}])
        result = make_result(
            make_roller_row(np.full((4, 3), 1.0)),
            make_roller_row(np.full((4, 3), 1.0)),
        )
        out = pp.LaminaDynamicEquivalentLoad.from_bearing_result(bearing, result)
        assert [o.label for o in out] == ["b1-row0", "b1-row1"]


class TestRollerBasicReferenceRatingLife:
    def test_rejects_non_positive_qci_or_qce(self):
        with pytest.raises(ValueError, match="must be positive"):
            pp.RollerBasicReferenceRatingLife.from_loads(
                "b1", q_kci=np.array([0.0]), q_kei=np.array([1.0]),
                q_kce=np.array([1.0]), q_kee=np.array([1.0]))

    def test_rejects_negative_qei_or_qee(self):
        with pytest.raises(ValueError, match="non-negative"):
            pp.RollerBasicReferenceRatingLife.from_loads(
                "b1", q_kci=np.array([1.0]), q_kei=np.array([-1.0]),
                q_kce=np.array([1.0]), q_kee=np.array([1.0]))

    def test_matches_formula(self):
        q_kci = np.array([10.0, 10.0])
        q_kei = np.array([2.0, 3.0])
        q_kce = np.array([9.0, 9.0])
        q_kee = np.array([2.5, 3.5])
        terms = (q_kei / q_kci) ** 4.5 + (q_kee / q_kce) ** 4.5
        expected = float(np.sum(terms) ** (-8.0 / 9.0))
        life = pp.RollerBasicReferenceRatingLife.from_loads("b1", q_kci, q_kei, q_kce, q_kee)
        assert life.L10r == pytest.approx(expected)


class TestRollerBasicReferenceRatingLifeAggregate:
    def test_rejects_mismatched_row_list_lengths(self):
        arr = np.array([1.0])
        with pytest.raises(ValueError, match="same length"):
            pp.roller_basic_reference_rating_life("b1", [arr], [arr, arr], [arr], [arr])

    def test_rejects_zero_rows(self):
        with pytest.raises(ValueError, match="got 0 rows"):
            pp.roller_basic_reference_rating_life("b1", [], [], [], [])

    def test_single_row_bearing_L10r_equals_row_L10r(self):
        arr_ci = np.array([10.0])
        arr_ei = np.array([2.0])
        per_row, L10r_bearing = pp.roller_basic_reference_rating_life(
            "b1", [arr_ci], [arr_ei], [arr_ci], [arr_ei])
        assert L10r_bearing == pytest.approx(per_row[0].L10r)

    def test_multirow_bearing_combines_via_combine_row_L10r(self):
        arr_ci = np.array([10.0])
        arr_ei = np.array([2.0])
        per_row, L10r_bearing = pp.roller_basic_reference_rating_life(
            "b1", [arr_ci, arr_ci], [arr_ei, arr_ei], [arr_ci, arr_ci], [arr_ei, arr_ei])
        expected = pp.combine_row_L10r([r.L10r for r in per_row], e=pp._E_ROLLER)
        assert L10r_bearing == pytest.approx(expected)


class TestRollerDynamicEquivalentReferenceLoad:
    def test_uses_3_over_10_exponent(self):
        L10r = 8.0
        pref = pp.RollerDynamicEquivalentReferenceLoad.from_L10r("b1", L10r, Cr=1000.0)
        denom = L10r ** (3.0 / 10.0)
        assert pref.Pref_r == pytest.approx(1000.0 / denom)
