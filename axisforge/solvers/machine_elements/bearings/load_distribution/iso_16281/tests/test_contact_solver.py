# test_contact_solver.py
"""Testes de contact_solver.py -- SolverBase (batch loop, contrato,
testado com um solver-stub para não depender das classes *Result
reais nos testes genéricos de solve()), LineContactSolverBase,
MultiRowSolverBase, e os solvers concretos ISO16281BallSolver /
ISO16281RollerSolver / os dois multi-row.

Física: testes de equilíbrio/contrato (residual ~0, sinais plausíveis,
relações entre grandezas), não valores certificados contra a norma --
tal como os testes de capacity.py em bearings/families/tests.

Inclui também o caminho postprocess=True (SolverBase._Cr_Ca via
is_thrust_family, e _attach_postprocessing() de cada solver concreto)
-- contrato apenas (stiffness/basic_life ausentes com postprocess=False,
presentes e plausíveis com postprocess=True), não os valores em si, já
cobertos contract-a-contract em test_contact_postprocessing.py.

Mudanças face à versão anterior deste ficheiro, alinhadas ao
contact_solver.py atual:
- REQUIRED_ATTRS foi removido (readiness é só check_bearing_ready() +
  iso16281_analysis) -- os testes _StubSolver/_NeedsFoo baseados nisso
  foram substituídos por testes de capability mismatch.
- check_bearing_ready() agora corre para TODA bearing passada a
  solve(), e exige family + iso16281_analysis=True -- os bearings
  SimpleNamespace "nus" usados no batch loop genérico levam
  iso16281_analysis=True (CAPABILITY="" no _StubSolver base salta o
  check de family, ver family_matches_capability/_CAPABILITY_CHECK).
- SolverBase._Cr_Ca dispatch é via is_thrust_family(bearing.family),
  não bearing.duty (removido) -- testado com family real e com
  monkeypatch de is_thrust_family para simular o ramo thrust sem
  precisar de uma ThrustBallSingleRowFamily inteira montada.
- _wrap_result() é identidade para solvers single-row (o
  *BearingResult já vem completo de solve_contact()) -- não chama
  _result_cls.single() como a versão anterior assumia.
- _attach_postprocessing() nunca é chamado por _solve_one_bearing();
  é solve() que o chama quando postprocess=True. Os testes chamam
  _attach_postprocessing() diretamente sobre o resultado de
  _solve_one_bearing(), não confundem os dois.
- Q_j é sempre populado (parte do próprio *LoadDistributionResult,
  atribuído dentro de solve_contact()) -- não é None quando
  postprocess=False; só stiffness/basic_life dependem do flag.
- MultiRowSolverBase._solve_rows() devolve 7 valores
  (row_results, delta_r, delta_a, Fa, nfev, res, ok), não 5; e
  _fractions() não existe -- os testes correspondentes foram
  removidos/reescritos.
- warn_if_floating_loaded() já não é chamado por solve() -- o teste
  correspondente fica como xfail(strict=True), por indicação
  explícita: o fix correto é a montante, no solver de FEM (não
  deveria ser possível chegar aqui com uma bearing floating e Fa != 0),
  não um warning aqui."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from axisforge.core.machine_elements.bearings.families.family import (
    CylindricalRollerFamily, DeepGrooveBallFamily, ThrustBallSingleRowFamily,
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281 import (
    contact_solver as contact_solver_module,
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import (
    SolverBase, LineContactSolverBase, MultiRowSolverBase,
    ISO16281BallSolver, ISO16281RollerSolver,
    ISO16281MultiRowBallSolverSharedDisplacement,
    ISO16281MultiRowRollerSolverSharedDisplacement,
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.tests.conftest import (
    BALL_BEARING_KWARGS, ROLLER_BEARING_KWARGS, make_node, make_shaft_results,
)


# =====================================================================
# SolverBase -- batch loop, testado com um stub para isolar da física
# concreta e das classes *Result reais.
# =====================================================================

class _StubBearingResult:
    """Fake de *BearingResult só com o que o batch loop de solve() lê
    (.label) e o que os testes verificam (phi_Fr/psi passados)."""
    def __init__(self, label, phi_Fr, psi):
        self.label = label
        self.phi_Fr = phi_Fr
        self.psi = psi


class _StubSolver(SolverBase):
    """CAPABILITY fica "" (herdado de SolverBase) de propósito nos
    testes genéricos do batch loop -- family_matches_capability/
    _CAPABILITY_CHECK não têm entrada para "", por isso
    check_bearing_ready() salta o check de family e só exige
    iso16281_analysis=True na bearing."""
    _result_cls = _StubBearingResult

    def _solve_one_bearing(self, bearing, node, phi_Fr, psi):
        return _StubBearingResult(bearing.label, phi_Fr, psi)


class _StubSolverNoResultCls(SolverBase):
    def _solve_one_bearing(self, bearing, node, phi_Fr, psi):
        return None


class TestSolverBaseSolve:
    def test_requires_result_cls(self):
        solver = _StubSolverNoResultCls()
        with pytest.raises(NotImplementedError, match="_result_cls"):
            solver.solve(SimpleNamespace(), {"b1": SimpleNamespace()}, make_shaft_results())

    def test_wraps_each_bearing_and_returns_bearing_analysis_result(self):
        solver = _StubSolver()
        bearing = SimpleNamespace(label="b1", iso16281_analysis=True)
        node = make_node(Fr_xz=1.0, Fr_xy=0.0, Fa=0.0, psi_xz=0.0, psi_xy=0.0, label="b1")
        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))

        assert set(results) == {"b1"}
        analysis = results["b1"]
        assert analysis.label == "b1"
        assert analysis.load_distribution.label == "b1"
        assert analysis.basic_life is None  # postprocess=False por omissão

    def test_computes_psi_from_projection_when_not_overridden(self):
        solver = _StubSolver()
        bearing = SimpleNamespace(label="b1", iso16281_analysis=True)
        node = make_node(Fr_xz=3.0, Fr_xy=4.0, psi_xz=0.1, psi_xy=0.2, label="b1")
        phi_Fr_expected = math.atan2(4.0, 3.0)
        psi_expected = 0.1 * math.cos(phi_Fr_expected) + 0.2 * math.sin(phi_Fr_expected)

        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))
        row = results["b1"].load_distribution

        assert row.phi_Fr == pytest.approx(phi_Fr_expected)
        assert row.psi == pytest.approx(psi_expected)

    def test_psi_override_ignored_when_psi_input_false(self):
        solver = _StubSolver(psi_input=False)
        bearing = SimpleNamespace(label="b1", iso16281_analysis=True)
        node = make_node(Fr_xz=1.0, Fr_xy=0.0, psi_xz=0.5, psi_xy=0.0, label="b1")

        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node),
                                psi_override={"b1": 999.0})
        assert results["b1"].load_distribution.psi == pytest.approx(0.5)  # projeção normal, não o override

    def test_psi_override_honoured_when_psi_input_true(self):
        solver = _StubSolver(psi_input=True)
        bearing = SimpleNamespace(label="b1", iso16281_analysis=True)
        node = make_node(Fr_xz=1.0, Fr_xy=0.0, psi_xz=0.5, psi_xy=0.0, label="b1")

        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node),
                                psi_override={"b1": 999.0})
        assert results["b1"].load_distribution.psi == pytest.approx(999.0)

    def test_warns_on_psi_override_with_unknown_label(self):
        solver = _StubSolver(psi_input=True)
        bearing = SimpleNamespace(label="b1", iso16281_analysis=True)
        node = make_node(label="b1")

        with pytest.warns(UserWarning, match="has labels not in"):
            solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node),
                         psi_override={"typo_label": 1.0})

    def test_check_bearing_ready_enforced_via_capability_mismatch(self):
        """check_bearing_ready() corre antes de _solve_one_bearing() --
        uma family que não corresponde à CAPABILITY do solver rejeita o
        solve antes de sequer olhar para o node."""
        class _PointOnly(_StubSolver):
            CAPABILITY = "point_contact"

        solver = _PointOnly()
        bearing = SimpleNamespace(label="b1", family=CylindricalRollerFamily(),
                                   iso16281_analysis=True)
        node = make_node(label="b1")
        with pytest.raises(RuntimeError, match="is not a point_contact family"):
            solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))

    def test_check_bearing_ready_enforced_via_missing_iso16281_analysis(self):
        class _PointOnly(_StubSolver):
            CAPABILITY = "point_contact"

        solver = _PointOnly()
        bearing = SimpleNamespace(label="b1", family=DeepGrooveBallFamily())  # sem iso16281_analysis
        node = make_node(label="b1")
        with pytest.raises(RuntimeError, match="iso16281_analysis is not True"):
            solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))

    @pytest.mark.xfail(
        reason="warn_if_floating_loaded() ja nao e chamado (nem importado) em "
               "contact_solver.py -- ver NOTE no docstring do modulo. O gap "
               "'floating bearing carregada axialmente' deveria idealmente ser "
               "recusado a montante, pelo solver de FEM (nao deveria ser "
               "possivel chegar aqui nesse estado), nao avisado aqui.",
        strict=True,
    )
    def test_warns_when_floating_bearing_is_axially_loaded(self):
        solver = _StubSolver()
        bearing = SimpleNamespace(label="b1", arrangement="floating", iso16281_analysis=True)
        node = make_node(Fa=100.0, label="b1")
        with pytest.warns(UserWarning, match="floating but Fa"):
            solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))


class TestSolverBasePostprocessFlag:
    """postprocess é só mais um argumento guardado em self -- o que ele
    desencadeia é testado nos solvers concretos (TestISO16281*Postprocess
    abaixo) e o guard de multi-row (TestMultiRowSolverBasePostprocessGuard)."""

    def test_defaults_to_false(self):
        assert _StubSolver().postprocess is False

    def test_can_be_set_true(self):
        assert _StubSolver(postprocess=True).postprocess is True


class TestSolverBaseCrCaDispatch:
    """SolverBase._Cr_Ca(bearing) -- dispatch de Cr/Ca a partir de
    bearing.C, escolhendo o campo via is_thrust_family(bearing.family)
    (registo real @thrust em family.py), não mais um bearing.duty
    string -- usado por _attach_postprocessing() de ambos os solvers
    concretos para alimentar *DynamicEquivalentReferenceLoad."""

    def test_radial_family_gives_Cr_only(self):
        bearing = SimpleNamespace(label="b1", family=DeepGrooveBallFamily(), C=1234.0)
        Cr, Ca = SolverBase._Cr_Ca(bearing)
        assert Cr == 1234.0
        assert Ca is None

    def test_thrust_family_gives_Ca_only(self):
        bearing = SimpleNamespace(label="b1", family=ThrustBallSingleRowFamily(), C=999.0)
        Cr, Ca = SolverBase._Cr_Ca(bearing)
        assert Cr is None
        assert Ca == 999.0


# =====================================================================
# LineContactSolverBase -- check extra de laminae (Sec 5.2.2)
# =====================================================================

class TestLineContactSolverExtraReadyChecks:
    def test_rejects_fewer_than_min_laminae(self, roller_bearing):
        solver = ISO16281RollerSolver()
        roller_bearing.n_s = 20
        roller_bearing.x_k = np.linspace(-2.0, 2.0, 20)
        with pytest.raises(ValueError, match="Sec 5.2.2"):
            solver._extra_ready_checks(roller_bearing, "b2")

    def test_rejects_x_k_length_mismatch(self, roller_bearing):
        solver = ISO16281RollerSolver()
        roller_bearing.x_k = np.linspace(-2.0, 2.0, roller_bearing.n_s - 1)
        with pytest.raises(ValueError, match="x_k must have exactly one lamina"):
            solver._extra_ready_checks(roller_bearing, "b2")

    def test_passes_with_valid_laminae(self, roller_bearing):
        solver = ISO16281RollerSolver()
        solver._extra_ready_checks(roller_bearing, "b2")  # não deve levantar


# =====================================================================
# ISO16281BallSolver -- point contact
# =====================================================================

class TestISO16281BallSolverRegistration:
    def test_capability(self):
        assert ISO16281BallSolver.CAPABILITY == "point_contact"

    def test_linked_to_its_multirow_sibling(self):
        assert ISO16281BallSolver.MULTIROW_SOLVER is ISO16281MultiRowBallSolverSharedDisplacement


class TestISO16281BallSolverElements:
    def test_delta_j_never_negative(self, ball_bearing):
        delta_j, alpha_j, ca, sa, cp_j, d32 = ISO16281BallSolver.elements(
            ball_bearing, delta_r=-10.0, delta_a=-10.0, Vpsi=np.zeros(ball_bearing.Z),
        )
        assert (delta_j >= 0.0).all()

    def test_shapes_match_Z(self, ball_bearing):
        out = ISO16281BallSolver.elements(ball_bearing, delta_r=0.1, delta_a=0.0,
                                          Vpsi=np.zeros(ball_bearing.Z))
        for arr in out:
            assert np.shape(arr) == (ball_bearing.Z,)


class TestISO16281BallSolverSolveContact:
    def test_pure_radial_load_converges_and_balances(self, ball_bearing):
        solver = ISO16281BallSolver()
        Fr_xz, Fr_xy, Fa = 500.0, 0.0, 0.0
        result = solver.solve_contact(ball_bearing, Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3
        assert result.delta_r > 0.0
        assert (result.row.delta_j >= 0.0).all()

    def test_pure_axial_load_gives_positive_delta_a(self, ball_bearing):
        solver = ISO16281BallSolver()
        result = solver.solve_contact(ball_bearing, Fr_xz=0.0, Fr_xy=0.0, Fa=800.0,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3
        assert result.delta_a > 0.0

    def test_negative_axial_load_gives_negative_delta_a(self, ball_bearing):
        solver = ISO16281BallSolver()
        result = solver.solve_contact(ball_bearing, Fr_xz=0.0, Fr_xy=0.0, Fa=-800.0,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.delta_a < 0.0

    def test_zero_load_gives_near_zero_residual(self, ball_bearing):
        """A carga nula não tem raiz única em (delta_r, delta_a) --
        qualquer folga sem contacto (delta_j=0 em todo o lado) dá
        resíduo zero, por isso não se fixa um valor exato, só que o
        solver converge com resíduo desprezável."""
        solver = ISO16281BallSolver()
        result = solver.solve_contact(ball_bearing, Fr_xz=0.0, Fr_xy=0.0, Fa=0.0,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3

    def test_larger_radial_load_gives_larger_delta_r(self, ball_bearing):
        solver = ISO16281BallSolver()
        small = solver.solve_contact(ball_bearing, Fr_xz=200.0, Fr_xy=0.0, Fa=0.0,
                                     delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        large = solver.solve_contact(ball_bearing, Fr_xz=800.0, Fr_xy=0.0, Fa=0.0,
                                     delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert large.delta_r > small.delta_r


class TestISO16281BallSolverMinimumAxialLoad:
    def test_trivial_branch_when_already_engaged(self, ball_bearing):
        """Com alpha_0=0 e psi=0, delta_a=0 já satisfaz Fa=0 (V_j=0 em
        todo o lado, logo sin(alpha_j)=0) -- já está 'engaged' sem
        pré-carga nenhuma, então minimum_axial_load devolve 0.0
        diretamente, sem chamar brentq."""
        solver = ISO16281BallSolver()
        ball_bearing.alpha_0 = 0.0
        Fa_min, result = solver.minimum_axial_load(ball_bearing, Fr_xz=100.0, Fr_xy=0.0, psi=0.0)
        assert Fa_min == 0.0
        assert result.delta_a == pytest.approx(0.0, abs=1e-9)

    def test_finds_minimum_axial_load_that_closes_the_clearance(self, ball_bearing):
        """Com alpha_0=15deg a folga axial não está fechada a Fa=0 --
        minimum_axial_load tem de encontrar, via brentq, o Fa que traz
        delta_a para a fronteira de fecho (delta_a ~ 0)."""
        solver = ISO16281BallSolver()
        Fa_min, result = solver.minimum_axial_load(ball_bearing, Fr_xz=100.0, Fr_xy=0.0, psi=0.0)
        assert Fa_min > 0.0
        assert result.delta_a == pytest.approx(0.0, abs=1e-4)

    def test_raises_when_bracket_too_narrow(self, ball_bearing):
        """Forçar um alpha_0 muito raso para exigir mais pré-carga do
        que o bracket por omissão cobre."""
        solver = ISO16281BallSolver()
        ball_bearing.A = 5.0  # gap gigante -- precisa de muito mais Fa do que o bracket default
        with pytest.raises(ValueError, match="widen Fa_bracket"):
            solver.minimum_axial_load(ball_bearing, Fr_xz=100.0, Fr_xy=0.0, psi=0.0,
                                      Fa_bracket=(0.0, 1.0))


class _FakeBallFamily:
    """Stub de bearing.family só para exercitar a ligação em
    _attach_postprocessing() -- devolve (Q_ci, Q_ce) fixos, não uma
    capacidade real; a fórmula de per_element_dynamic_capacity() já tem
    os seus próprios testes em bearings/families/tests. Não é uma
    family registada (is_point_contact_family/is_thrust_family
    devolvem False para ela) -- inofensivo aqui porque estes testes
    chamam _solve_one_bearing()/_attach_postprocessing() diretamente,
    nunca solve() (que é o único caminho que passa por
    check_bearing_ready/is_*_family)."""
    @staticmethod
    def per_element_dynamic_capacity(bearing):
        return 500.0, 500.0


class TestISO16281BallSolverPostprocess:
    """_attach_postprocessing() é chamado por solve() quando
    postprocess=True -- nunca por _solve_one_bearing(), que devolve
    sempre o *BearingResult "cru" (Q_j já preenchido, mas sem
    stiffness). Aqui só se verifica a ligação (campos presentes e
    plausíveis com postprocess=True), os valores em si já estão
    cobertos em test_contact_postprocessing.py."""

    def test_raw_result_has_no_stiffness_before_postprocessing(self, ball_bearing):
        solver = ISO16281BallSolver(postprocess=False)
        node = make_node(Fr_xz=500.0, Fa=100.0, label="B1")
        result = solver._solve_one_bearing(ball_bearing, node, phi_Fr=0.0, psi=0.0)

        assert result.row.Q_j is not None
        assert result.row.Q_j.shape == (ball_bearing.Z,)
        assert (result.row.Q_j >= 0.0).all()
        assert result.stiffness is None

    def test_attach_postprocessing_fills_stiffness_and_life_for_radial_family(self, ball_bearing):
        ball_bearing.family = _FakeBallFamily()
        ball_bearing.C = 5000.0
        solver = ISO16281BallSolver(postprocess=True)
        node = make_node(Fr_xz=500.0, Fa=100.0, label="B1")
        result = solver._solve_one_bearing(ball_bearing, node, phi_Fr=0.0, psi=0.0)
        analysis = solver._attach_postprocessing(ball_bearing, result)

        assert analysis.label == "B1"
        assert analysis.load_distribution.stiffness is not None
        assert analysis.load_distribution.stiffness.label == "B1"
        assert analysis.basic_life is not None
        assert analysis.basic_life.L10r > 0.0
        assert analysis.basic_life.Pref_r is not None and analysis.basic_life.Pref_r > 0.0
        assert analysis.basic_life.Pref_a is None  # family não-thrust -- só Cr foi passado

    def test_attach_postprocessing_thrust_family_gives_axial_pref_only(self, ball_bearing, monkeypatch):
        ball_bearing.family = _FakeBallFamily()
        ball_bearing.C = 5000.0
        monkeypatch.setattr(contact_solver_module, "is_thrust_family", lambda family: True)
        solver = ISO16281BallSolver(postprocess=True)
        node = make_node(Fr_xz=500.0, Fa=100.0, label="B1")
        result = solver._solve_one_bearing(ball_bearing, node, phi_Fr=0.0, psi=0.0)
        analysis = solver._attach_postprocessing(ball_bearing, result)

        assert analysis.basic_life.Pref_r is None
        assert analysis.basic_life.Pref_a is not None and analysis.basic_life.Pref_a > 0.0


# =====================================================================
# ISO16281RollerSolver -- line contact
# =====================================================================

class TestISO16281RollerSolverRegistration:
    def test_capability(self):
        assert ISO16281RollerSolver.CAPABILITY == "line_contact"

    def test_linked_to_its_multirow_sibling(self):
        assert ISO16281RollerSolver.MULTIROW_SOLVER is ISO16281MultiRowRollerSolverSharedDisplacement


class TestISO16281RollerSolverElements:
    def test_delta_jk_never_negative(self, roller_bearing):
        _, _, delta_jk, q_jk, _ = ISO16281RollerSolver.elements(roller_bearing, delta_r=0.0, psi=0.0)
        assert (delta_jk >= 0.0).all()
        assert (q_jk >= 0.0).all()

    def test_shapes(self, roller_bearing):
        delta_j, psi_j, delta_jk, q_jk, cp_j = ISO16281RollerSolver.elements(
            roller_bearing, delta_r=0.05, psi=0.0)
        Z, n_s = roller_bearing.Z, roller_bearing.n_s
        assert delta_j.shape == (Z,)
        assert psi_j.shape == (Z,)
        assert delta_jk.shape == (Z, n_s)
        assert q_jk.shape == (Z, n_s)
        assert cp_j.shape == (Z,)


class TestISO16281RollerSolverSolveContact:
    def test_pure_radial_load_converges_and_balances(self, roller_bearing):
        solver = ISO16281RollerSolver()
        result = solver.solve_contact(roller_bearing, Fr_xz=1000.0, Fr_xy=0.0,
                                      delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-2
        assert result.delta_r > 0.0
        assert result.delta_a == 0.0  # roller radial não tem capacidade axial

    def test_zero_load_gives_near_zero_residual(self, roller_bearing):
        """Tal como no ball solver, carga nula não tem raiz única em
        delta_r -- qualquer delta_r que deixe todos os rolos fora de
        contacto (delta_jk=0 em todo o lado) dá resíduo zero. Só se
        verifica que o solver converge com resíduo desprezável."""
        solver = ISO16281RollerSolver()
        result = solver.solve_contact(roller_bearing, Fr_xz=0.0, Fr_xy=0.0,
                                      delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3

    def test_larger_radial_load_gives_larger_delta_r(self, roller_bearing):
        solver = ISO16281RollerSolver()
        small = solver.solve_contact(roller_bearing, Fr_xz=500.0, Fr_xy=0.0,
                                     delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        large = solver.solve_contact(roller_bearing, Fr_xz=2000.0, Fr_xy=0.0,
                                     delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert large.delta_r > small.delta_r

    def test_alpha_j_is_constant_equal_to_bearing_alpha_0(self, roller_bearing):
        solver = ISO16281RollerSolver()
        result = solver.solve_contact(roller_bearing, Fr_xz=500.0, Fr_xy=0.0,
                                      delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert np.all(result.row.alpha_j == roller_bearing.alpha_0)


class _FakeRollerFamily:
    """Stub de bearing.family, espelha _FakeBallFamily mas para
    per_lamina_dynamic_capacity() -- devolve (q_ci, q_ce) escalares
    fixos, como a família real (ver capacity.py: per_lamina é um par
    escalar, não arrays por lamina). Mesma ressalva de não-registo que
    _FakeBallFamily."""
    @staticmethod
    def per_lamina_dynamic_capacity(bearing):
        return 200.0, 200.0


class TestISO16281RollerSolverPostprocess:
    """Espelha TestISO16281BallSolverPostprocess para o lado roller --
    mesma ligação (_attach_postprocessing()), valores só verificados
    contract-level, física certificada em test_contact_postprocessing.py."""

    def test_raw_result_has_no_stiffness_before_postprocessing(self, roller_bearing):
        solver = ISO16281RollerSolver(postprocess=False)
        node = make_node(Fr_xz=1000.0, label="B2")
        result = solver._solve_one_bearing(roller_bearing, node, phi_Fr=0.0, psi=0.0)

        assert result.row.Q_j is not None
        assert result.row.Q_j.shape == (roller_bearing.Z,)
        assert (result.row.Q_j >= 0.0).all()
        assert result.stiffness is None

    def test_attach_postprocessing_fills_stiffness_and_life_for_radial_family(self, roller_bearing):
        roller_bearing.family = _FakeRollerFamily()
        roller_bearing.C = 8000.0
        solver = ISO16281RollerSolver(postprocess=True)
        node = make_node(Fr_xz=1000.0, label="B2")
        result = solver._solve_one_bearing(roller_bearing, node, phi_Fr=0.0, psi=0.0)
        analysis = solver._attach_postprocessing(roller_bearing, result)

        assert analysis.load_distribution.stiffness is not None
        assert analysis.basic_life is not None
        assert analysis.basic_life.L10r > 0.0
        assert analysis.basic_life.Pref_r is not None and analysis.basic_life.Pref_r > 0.0
        assert analysis.basic_life.Pref_a is None  # duty radial -- roller radial não leva axial de qualquer forma

    def test_attach_postprocessing_thrust_family_gives_axial_pref_only(self, roller_bearing, monkeypatch):
        roller_bearing.family = _FakeRollerFamily()
        roller_bearing.C = 8000.0
        monkeypatch.setattr(contact_solver_module, "is_thrust_family", lambda family: True)
        solver = ISO16281RollerSolver(postprocess=True)
        node = make_node(Fr_xz=1000.0, label="B2")
        result = solver._solve_one_bearing(roller_bearing, node, phi_Fr=0.0, psi=0.0)
        analysis = solver._attach_postprocessing(roller_bearing, result)

        assert analysis.basic_life.Pref_r is None
        assert analysis.basic_life.Pref_a is not None and analysis.basic_life.Pref_a > 0.0


# =====================================================================
# MultiRowSolverBase -- shared-displacement multi-row (não revisto para
# acoplamento fila-a-fila -- ver docstring do próprio módulo)
# =====================================================================

class TestMultiRowSolverBaseRowViews:
    def test_rejects_fewer_than_two_rows(self):
        bearing = SimpleNamespace(label="b1", rows=[{"cp": 1.0}])
        with pytest.raises(ValueError, match=">= 2 rows"):
            MultiRowSolverBase._row_views(bearing)

    def test_rejects_missing_rows_attribute(self):
        bearing = SimpleNamespace(label="b1")
        with pytest.raises(ValueError, match=">= 2 rows"):
            MultiRowSolverBase._row_views(bearing)

    def test_wraps_dict_rows_as_simplenamespace(self):
        bearing = SimpleNamespace(label="b1", rows=[{"cp": 1.0}, {"cp": 2.0}])
        views = MultiRowSolverBase._row_views(bearing)
        assert all(isinstance(v, SimpleNamespace) for v in views)
        assert [v.cp for v in views] == [1.0, 2.0]

    def test_leaves_object_rows_untouched(self):
        row_obj = SimpleNamespace(cp=3.0)
        bearing = SimpleNamespace(label="b1", rows=[row_obj, SimpleNamespace(cp=4.0)])
        views = MultiRowSolverBase._row_views(bearing)
        assert views[0] is row_obj


class TestISO16281MultiRowBallSolver:
    def test_check_row_ready_rejects_non_positive_cp(self):
        solver = ISO16281MultiRowBallSolverSharedDisplacement()
        row = SimpleNamespace(cp=0.0)
        with pytest.raises(ValueError, match="cp must be positive"):
            solver._check_row_ready(row, "b1[row0]")

    def test_check_row_ready_accepts_positive_cp(self):
        solver = ISO16281MultiRowBallSolverSharedDisplacement()
        solver._check_row_ready(SimpleNamespace(cp=1.0), "b1[row0]")  # não deve levantar

    def test_two_identical_rows_each_take_half_the_load(self):
        solver = ISO16281MultiRowBallSolverSharedDisplacement()
        row_views = [SimpleNamespace(**BALL_BEARING_KWARGS), SimpleNamespace(**BALL_BEARING_KWARGS)]
        row_results, delta_r, delta_a, Fa, nfev, res, ok = solver._solve_rows(
            row_views, Fr_xz=1000.0, Fr_xy=0.0, Fa=0.0, psi=0.0, phi_Fr=0.0,
        )
        assert ok
        assert len(row_results) == 2
        assert row_results[0].Fr_row == pytest.approx(row_results[1].Fr_row, rel=1e-6)


class TestMultiRowSolverBasePostprocessGuard:
    """postprocess=True nunca foi desenhado/revisto para o caso
    shared-displacement multi-row (Cr/Ca por fila, combinação de L10r
    entre filas) -- _extra_ready_checks() recusa de imediato, antes de
    sequer validar bearing.rows."""

    def test_ball_multirow_rejects_postprocess_true(self):
        solver = ISO16281MultiRowBallSolverSharedDisplacement(postprocess=True)
        bearing = SimpleNamespace(label="b1")  # nem sequer .rows -- não chega a ser lido
        with pytest.raises(NotImplementedError, match="postprocess=True is not supported"):
            solver._extra_ready_checks(bearing, "b1")

    def test_roller_multirow_rejects_postprocess_true(self):
        solver = ISO16281MultiRowRollerSolverSharedDisplacement(postprocess=True)
        bearing = SimpleNamespace(label="b2")
        with pytest.raises(NotImplementedError, match="postprocess=True is not supported"):
            solver._extra_ready_checks(bearing, "b2")

    def test_ball_multirow_postprocess_false_unaffected(self):
        """postprocess=False continua a validar as filas normalmente --
        o guard não interfere com o caminho já testado acima."""
        solver = ISO16281MultiRowBallSolverSharedDisplacement(postprocess=False)
        bearing = SimpleNamespace(label="b1", rows=[dict(BALL_BEARING_KWARGS), dict(BALL_BEARING_KWARGS)])
        solver._extra_ready_checks(bearing, "b1")  # não deve levantar


class TestISO16281MultiRowRollerSolver:
    def test_check_row_ready_rejects_too_few_laminae(self):
        solver = ISO16281MultiRowRollerSolverSharedDisplacement()
        row = SimpleNamespace(n_s=10, x_k=np.zeros(10))
        with pytest.raises(ValueError, match="Sec 5.2.2"):
            solver._check_row_ready(row, "b1[row0]")

    def test_check_row_ready_rejects_x_k_length_mismatch(self):
        solver = ISO16281MultiRowRollerSolverSharedDisplacement()
        row = SimpleNamespace(n_s=30, x_k=np.zeros(29))
        with pytest.raises(ValueError, match=r"len\(x_k\)"):
            solver._check_row_ready(row, "b1[row0]")

    def test_two_identical_rows_each_take_half_the_load(self):
        solver = ISO16281MultiRowRollerSolverSharedDisplacement()
        row_views = [SimpleNamespace(**ROLLER_BEARING_KWARGS), SimpleNamespace(**ROLLER_BEARING_KWARGS)]
        row_results, delta_r, delta_a, Fa, nfev, res, ok = solver._solve_rows(
            row_views, Fr_xz=1000.0, Fr_xy=0.0, Fa=0.0, psi=0.0, phi_Fr=0.0,
        )
        assert ok
        assert len(row_results) == 2
        assert row_results[0].Fr_row == pytest.approx(row_results[1].Fr_row, rel=1e-6)
        assert row_results[0].Fa_row == 0.0 and row_results[1].Fa_row == 0.0
        assert delta_a == 0.0
        assert Fa == 0.0# test_contact_solver.py
"""Testes de contact_solver.py -- SolverBase (batch loop, contrato,
testado com um solver-stub para não depender das classes *Result
reais nos testes genéricos de solve()), LineContactSolverBase,
MultiRowSolverBase, e os solvers concretos ISO16281BallSolver /
ISO16281RollerSolver / os dois multi-row.

Física: testes de equilíbrio/contrato (residual ~0, sinais plausíveis,
relações entre grandezas), não valores certificados contra a norma --
tal como os testes de capacity.py em bearings/families/tests.

Inclui também o caminho postprocess=True (SolverBase._Cr_Ca via
is_thrust_family, e _attach_postprocessing() de cada solver concreto)
-- contrato apenas (stiffness/basic_life ausentes com postprocess=False,
presentes e plausíveis com postprocess=True), não os valores em si, já
cobertos contract-a-contract em test_contact_postprocessing.py.

Mudanças face à versão anterior deste ficheiro, alinhadas ao
contact_solver.py atual:
- REQUIRED_ATTRS foi removido (readiness é só check_bearing_ready() +
  iso16281_analysis) -- os testes _StubSolver/_NeedsFoo baseados nisso
  foram substituídos por testes de capability mismatch.
- check_bearing_ready() agora corre para TODA bearing passada a
  solve(), e exige family + iso16281_analysis=True -- os bearings
  SimpleNamespace "nus" usados no batch loop genérico levam
  iso16281_analysis=True (CAPABILITY="" no _StubSolver base salta o
  check de family, ver family_matches_capability/_CAPABILITY_CHECK).
- SolverBase._Cr_Ca dispatch é via is_thrust_family(bearing.family),
  não bearing.duty (removido) -- testado com family real e com
  monkeypatch de is_thrust_family para simular o ramo thrust sem
  precisar de uma ThrustBallSingleRowFamily inteira montada.
- _wrap_result() é identidade para solvers single-row (o
  *BearingResult já vem completo de solve_contact()) -- não chama
  _result_cls.single() como a versão anterior assumia.
- _attach_postprocessing() nunca é chamado por _solve_one_bearing();
  é solve() que o chama quando postprocess=True. Os testes chamam
  _attach_postprocessing() diretamente sobre o resultado de
  _solve_one_bearing(), não confundem os dois.
- Q_j é sempre populado (parte do próprio *LoadDistributionResult,
  atribuído dentro de solve_contact()) -- não é None quando
  postprocess=False; só stiffness/basic_life dependem do flag.
- MultiRowSolverBase._solve_rows() devolve 7 valores
  (row_results, delta_r, delta_a, Fa, nfev, res, ok), não 5; e
  _fractions() não existe -- os testes correspondentes foram
  removidos/reescritos.
- warn_if_floating_loaded() já não é chamado por solve() -- o teste
  correspondente fica como xfail(strict=True), por indicação
  explícita: o fix correto é a montante, no solver de FEM (não
  deveria ser possível chegar aqui com uma bearing floating e Fa != 0),
  não um warning aqui."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from axisforge.core.machine_elements.bearings.families.family import (
    CylindricalRollerFamily, DeepGrooveBallFamily, ThrustBallSingleRowFamily,
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281 import (
    contact_solver as contact_solver_module,
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import (
    SolverBase, LineContactSolverBase, MultiRowSolverBase,
    ISO16281BallSolver, ISO16281RollerSolver,
    ISO16281MultiRowBallSolverSharedDisplacement,
    ISO16281MultiRowRollerSolverSharedDisplacement,
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.tests.conftest import (
    BALL_BEARING_KWARGS, ROLLER_BEARING_KWARGS, make_node, make_shaft_results,
)


# =====================================================================
# SolverBase -- batch loop, testado com um stub para isolar da física
# concreta e das classes *Result reais.
# =====================================================================

class _StubBearingResult:
    """Fake de *BearingResult só com o que o batch loop de solve() lê
    (.label) e o que os testes verificam (phi_Fr/psi passados)."""
    def __init__(self, label, phi_Fr, psi):
        self.label = label
        self.phi_Fr = phi_Fr
        self.psi = psi


class _StubSolver(SolverBase):
    """CAPABILITY fica "" (herdado de SolverBase) de propósito nos
    testes genéricos do batch loop -- family_matches_capability/
    _CAPABILITY_CHECK não têm entrada para "", por isso
    check_bearing_ready() salta o check de family e só exige
    iso16281_analysis=True na bearing."""
    _result_cls = _StubBearingResult

    def _solve_one_bearing(self, bearing, node, phi_Fr, psi):
        return _StubBearingResult(bearing.label, phi_Fr, psi)


class _StubSolverNoResultCls(SolverBase):
    def _solve_one_bearing(self, bearing, node, phi_Fr, psi):
        return None


class TestSolverBaseSolve:
    def test_requires_result_cls(self):
        solver = _StubSolverNoResultCls()
        with pytest.raises(NotImplementedError, match="_result_cls"):
            solver.solve(SimpleNamespace(), {"b1": SimpleNamespace()}, make_shaft_results())

    def test_wraps_each_bearing_and_returns_bearing_analysis_result(self):
        solver = _StubSolver()
        bearing = SimpleNamespace(label="b1", iso16281_analysis=True)
        node = make_node(Fr_xz=1.0, Fr_xy=0.0, Fa=0.0, psi_xz=0.0, psi_xy=0.0, label="b1")
        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))

        assert set(results) == {"b1"}
        analysis = results["b1"]
        assert analysis.label == "b1"
        assert analysis.load_distribution.label == "b1"
        assert analysis.basic_life is None  # postprocess=False por omissão

    def test_computes_psi_from_projection_when_not_overridden(self):
        solver = _StubSolver()
        bearing = SimpleNamespace(label="b1", iso16281_analysis=True)
        node = make_node(Fr_xz=3.0, Fr_xy=4.0, psi_xz=0.1, psi_xy=0.2, label="b1")
        phi_Fr_expected = math.atan2(4.0, 3.0)
        psi_expected = 0.1 * math.cos(phi_Fr_expected) + 0.2 * math.sin(phi_Fr_expected)

        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))
        row = results["b1"].load_distribution

        assert row.phi_Fr == pytest.approx(phi_Fr_expected)
        assert row.psi == pytest.approx(psi_expected)

    def test_psi_override_ignored_when_psi_input_false(self):
        solver = _StubSolver(psi_input=False)
        bearing = SimpleNamespace(label="b1", iso16281_analysis=True)
        node = make_node(Fr_xz=1.0, Fr_xy=0.0, psi_xz=0.5, psi_xy=0.0, label="b1")

        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node),
                                psi_override={"b1": 999.0})
        assert results["b1"].load_distribution.psi == pytest.approx(0.5)  # projeção normal, não o override

    def test_psi_override_honoured_when_psi_input_true(self):
        solver = _StubSolver(psi_input=True)
        bearing = SimpleNamespace(label="b1", iso16281_analysis=True)
        node = make_node(Fr_xz=1.0, Fr_xy=0.0, psi_xz=0.5, psi_xy=0.0, label="b1")

        results = solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node),
                                psi_override={"b1": 999.0})
        assert results["b1"].load_distribution.psi == pytest.approx(999.0)

    def test_warns_on_psi_override_with_unknown_label(self):
        solver = _StubSolver(psi_input=True)
        bearing = SimpleNamespace(label="b1", iso16281_analysis=True)
        node = make_node(label="b1")

        with pytest.warns(UserWarning, match="has labels not in"):
            solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node),
                         psi_override={"typo_label": 1.0})

    def test_check_bearing_ready_enforced_via_capability_mismatch(self):
        """check_bearing_ready() corre antes de _solve_one_bearing() --
        uma family que não corresponde à CAPABILITY do solver rejeita o
        solve antes de sequer olhar para o node."""
        class _PointOnly(_StubSolver):
            CAPABILITY = "point_contact"

        solver = _PointOnly()
        bearing = SimpleNamespace(label="b1", family=CylindricalRollerFamily(),
                                   iso16281_analysis=True)
        node = make_node(label="b1")
        with pytest.raises(RuntimeError, match="is not a point_contact family"):
            solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))

    def test_check_bearing_ready_enforced_via_missing_iso16281_analysis(self):
        class _PointOnly(_StubSolver):
            CAPABILITY = "point_contact"

        solver = _PointOnly()
        bearing = SimpleNamespace(label="b1", family=DeepGrooveBallFamily())  # sem iso16281_analysis
        node = make_node(label="b1")
        with pytest.raises(RuntimeError, match="iso16281_analysis is not True"):
            solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))

    @pytest.mark.xfail(
        reason="warn_if_floating_loaded() ja nao e chamado (nem importado) em "
               "contact_solver.py -- ver NOTE no docstring do modulo. O gap "
               "'floating bearing carregada axialmente' deveria idealmente ser "
               "recusado a montante, pelo solver de FEM (nao deveria ser "
               "possivel chegar aqui nesse estado), nao avisado aqui.",
        strict=True,
    )
    def test_warns_when_floating_bearing_is_axially_loaded(self):
        solver = _StubSolver()
        bearing = SimpleNamespace(label="b1", arrangement="floating", iso16281_analysis=True)
        node = make_node(Fa=100.0, label="b1")
        with pytest.warns(UserWarning, match="floating but Fa"):
            solver.solve(SimpleNamespace(), {"b1": bearing}, make_shaft_results(node))


class TestSolverBasePostprocessFlag:
    """postprocess é só mais um argumento guardado em self -- o que ele
    desencadeia é testado nos solvers concretos (TestISO16281*Postprocess
    abaixo) e o guard de multi-row (TestMultiRowSolverBasePostprocessGuard)."""

    def test_defaults_to_false(self):
        assert _StubSolver().postprocess is False

    def test_can_be_set_true(self):
        assert _StubSolver(postprocess=True).postprocess is True


class TestSolverBaseCrCaDispatch:
    """SolverBase._Cr_Ca(bearing) -- dispatch de Cr/Ca a partir de
    bearing.C, escolhendo o campo via is_thrust_family(bearing.family)
    (registo real @thrust em family.py), não mais um bearing.duty
    string -- usado por _attach_postprocessing() de ambos os solvers
    concretos para alimentar *DynamicEquivalentReferenceLoad."""

    def test_radial_family_gives_Cr_only(self):
        bearing = SimpleNamespace(label="b1", family=DeepGrooveBallFamily(), C=1234.0)
        Cr, Ca = SolverBase._Cr_Ca(bearing)
        assert Cr == 1234.0
        assert Ca is None

    def test_thrust_family_gives_Ca_only(self):
        bearing = SimpleNamespace(label="b1", family=ThrustBallSingleRowFamily(), C=999.0)
        Cr, Ca = SolverBase._Cr_Ca(bearing)
        assert Cr is None
        assert Ca == 999.0


# =====================================================================
# LineContactSolverBase -- check extra de laminae (Sec 5.2.2)
# =====================================================================

class TestLineContactSolverExtraReadyChecks:
    def test_rejects_fewer_than_min_laminae(self, roller_bearing):
        solver = ISO16281RollerSolver()
        roller_bearing.n_s = 20
        roller_bearing.x_k = np.linspace(-2.0, 2.0, 20)
        with pytest.raises(ValueError, match="Sec 5.2.2"):
            solver._extra_ready_checks(roller_bearing, "b2")

    def test_rejects_x_k_length_mismatch(self, roller_bearing):
        solver = ISO16281RollerSolver()
        roller_bearing.x_k = np.linspace(-2.0, 2.0, roller_bearing.n_s - 1)
        with pytest.raises(ValueError, match="x_k must have exactly one lamina"):
            solver._extra_ready_checks(roller_bearing, "b2")

    def test_passes_with_valid_laminae(self, roller_bearing):
        solver = ISO16281RollerSolver()
        solver._extra_ready_checks(roller_bearing, "b2")  # não deve levantar


# =====================================================================
# ISO16281BallSolver -- point contact
# =====================================================================

class TestISO16281BallSolverRegistration:
    def test_capability(self):
        assert ISO16281BallSolver.CAPABILITY == "point_contact"

    def test_linked_to_its_multirow_sibling(self):
        assert ISO16281BallSolver.MULTIROW_SOLVER is ISO16281MultiRowBallSolverSharedDisplacement


class TestISO16281BallSolverElements:
    def test_delta_j_never_negative(self, ball_bearing):
        delta_j, alpha_j, ca, sa, cp_j, d32 = ISO16281BallSolver.elements(
            ball_bearing, delta_r=-10.0, delta_a=-10.0, Vpsi=np.zeros(ball_bearing.Z),
        )
        assert (delta_j >= 0.0).all()

    def test_shapes_match_Z(self, ball_bearing):
        out = ISO16281BallSolver.elements(ball_bearing, delta_r=0.1, delta_a=0.0,
                                          Vpsi=np.zeros(ball_bearing.Z))
        for arr in out:
            assert np.shape(arr) == (ball_bearing.Z,)


class TestISO16281BallSolverSolveContact:
    def test_pure_radial_load_converges_and_balances(self, ball_bearing):
        solver = ISO16281BallSolver()
        Fr_xz, Fr_xy, Fa = 500.0, 0.0, 0.0
        result = solver.solve_contact(ball_bearing, Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3
        assert result.delta_r > 0.0
        assert (result.row.delta_j >= 0.0).all()

    def test_pure_axial_load_gives_positive_delta_a(self, ball_bearing):
        solver = ISO16281BallSolver()
        result = solver.solve_contact(ball_bearing, Fr_xz=0.0, Fr_xy=0.0, Fa=800.0,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3
        assert result.delta_a > 0.0

    def test_negative_axial_load_gives_negative_delta_a(self, ball_bearing):
        solver = ISO16281BallSolver()
        result = solver.solve_contact(ball_bearing, Fr_xz=0.0, Fr_xy=0.0, Fa=-800.0,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.delta_a < 0.0

    def test_zero_load_gives_near_zero_residual(self, ball_bearing):
        """A carga nula não tem raiz única em (delta_r, delta_a) --
        qualquer folga sem contacto (delta_j=0 em todo o lado) dá
        resíduo zero, por isso não se fixa um valor exato, só que o
        solver converge com resíduo desprezável."""
        solver = ISO16281BallSolver()
        result = solver.solve_contact(ball_bearing, Fr_xz=0.0, Fr_xy=0.0, Fa=0.0,
                                      delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3

    def test_larger_radial_load_gives_larger_delta_r(self, ball_bearing):
        solver = ISO16281BallSolver()
        small = solver.solve_contact(ball_bearing, Fr_xz=200.0, Fr_xy=0.0, Fa=0.0,
                                     delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        large = solver.solve_contact(ball_bearing, Fr_xz=800.0, Fr_xy=0.0, Fa=0.0,
                                     delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
        assert large.delta_r > small.delta_r


class TestISO16281BallSolverMinimumAxialLoad:
    def test_trivial_branch_when_already_engaged(self, ball_bearing):
        """Com alpha_0=0 e psi=0, delta_a=0 já satisfaz Fa=0 (V_j=0 em
        todo o lado, logo sin(alpha_j)=0) -- já está 'engaged' sem
        pré-carga nenhuma, então minimum_axial_load devolve 0.0
        diretamente, sem chamar brentq."""
        solver = ISO16281BallSolver()
        ball_bearing.alpha_0 = 0.0
        Fa_min, result = solver.minimum_axial_load(ball_bearing, Fr_xz=100.0, Fr_xy=0.0, psi=0.0)
        assert Fa_min == 0.0
        assert result.delta_a == pytest.approx(0.0, abs=1e-9)

    def test_finds_minimum_axial_load_that_closes_the_clearance(self, ball_bearing):
        """Com alpha_0=15deg a folga axial não está fechada a Fa=0 --
        minimum_axial_load tem de encontrar, via brentq, o Fa que traz
        delta_a para a fronteira de fecho (delta_a ~ 0)."""
        solver = ISO16281BallSolver()
        Fa_min, result = solver.minimum_axial_load(ball_bearing, Fr_xz=100.0, Fr_xy=0.0, psi=0.0)
        assert Fa_min > 0.0
        assert result.delta_a == pytest.approx(0.0, abs=1e-4)

    def test_raises_when_bracket_too_narrow(self, ball_bearing):
        """Forçar um alpha_0 muito raso para exigir mais pré-carga do
        que o bracket por omissão cobre."""
        solver = ISO16281BallSolver()
        ball_bearing.A = 5.0  # gap gigante -- precisa de muito mais Fa do que o bracket default
        with pytest.raises(ValueError, match="widen Fa_bracket"):
            solver.minimum_axial_load(ball_bearing, Fr_xz=100.0, Fr_xy=0.0, psi=0.0,
                                      Fa_bracket=(0.0, 1.0))


class _FakeBallFamily:
    """Stub de bearing.family só para exercitar a ligação em
    _attach_postprocessing() -- devolve (Q_ci, Q_ce) fixos, não uma
    capacidade real; a fórmula de per_element_dynamic_capacity() já tem
    os seus próprios testes em bearings/families/tests. Não é uma
    family registada (is_point_contact_family/is_thrust_family
    devolvem False para ela) -- inofensivo aqui porque estes testes
    chamam _solve_one_bearing()/_attach_postprocessing() diretamente,
    nunca solve() (que é o único caminho que passa por
    check_bearing_ready/is_*_family)."""
    @staticmethod
    def per_element_dynamic_capacity(bearing):
        return 500.0, 500.0


class TestISO16281BallSolverPostprocess:
    """_attach_postprocessing() é chamado por solve() quando
    postprocess=True -- nunca por _solve_one_bearing(), que devolve
    sempre o *BearingResult "cru" (Q_j já preenchido, mas sem
    stiffness). Aqui só se verifica a ligação (campos presentes e
    plausíveis com postprocess=True), os valores em si já estão
    cobertos em test_contact_postprocessing.py."""

    def test_raw_result_has_no_stiffness_before_postprocessing(self, ball_bearing):
        solver = ISO16281BallSolver(postprocess=False)
        node = make_node(Fr_xz=500.0, Fa=100.0, label="B1")
        result = solver._solve_one_bearing(ball_bearing, node, phi_Fr=0.0, psi=0.0)

        assert result.row.Q_j is not None
        assert result.row.Q_j.shape == (ball_bearing.Z,)
        assert (result.row.Q_j >= 0.0).all()
        assert result.stiffness is None

    def test_attach_postprocessing_fills_stiffness_and_life_for_radial_family(self, ball_bearing):
        ball_bearing.family = _FakeBallFamily()
        ball_bearing.C = 5000.0
        solver = ISO16281BallSolver(postprocess=True)
        node = make_node(Fr_xz=500.0, Fa=100.0, label="B1")
        result = solver._solve_one_bearing(ball_bearing, node, phi_Fr=0.0, psi=0.0)
        analysis = solver._attach_postprocessing(ball_bearing, result)

        assert analysis.label == "B1"
        assert analysis.load_distribution.stiffness is not None
        assert analysis.load_distribution.stiffness.label == "B1"
        assert analysis.basic_life is not None
        assert analysis.basic_life.L10r > 0.0
        assert analysis.basic_life.Pref_r is not None and analysis.basic_life.Pref_r > 0.0
        assert analysis.basic_life.Pref_a is None  # family não-thrust -- só Cr foi passado

    def test_attach_postprocessing_thrust_family_gives_axial_pref_only(self, ball_bearing, monkeypatch):
        ball_bearing.family = _FakeBallFamily()
        ball_bearing.C = 5000.0
        monkeypatch.setattr(contact_solver_module, "is_thrust_family", lambda family: True)
        solver = ISO16281BallSolver(postprocess=True)
        node = make_node(Fr_xz=500.0, Fa=100.0, label="B1")
        result = solver._solve_one_bearing(ball_bearing, node, phi_Fr=0.0, psi=0.0)
        analysis = solver._attach_postprocessing(ball_bearing, result)

        assert analysis.basic_life.Pref_r is None
        assert analysis.basic_life.Pref_a is not None and analysis.basic_life.Pref_a > 0.0


# =====================================================================
# ISO16281RollerSolver -- line contact
# =====================================================================

class TestISO16281RollerSolverRegistration:
    def test_capability(self):
        assert ISO16281RollerSolver.CAPABILITY == "line_contact"

    def test_linked_to_its_multirow_sibling(self):
        assert ISO16281RollerSolver.MULTIROW_SOLVER is ISO16281MultiRowRollerSolverSharedDisplacement


class TestISO16281RollerSolverElements:
    def test_delta_jk_never_negative(self, roller_bearing):
        _, _, delta_jk, q_jk, _ = ISO16281RollerSolver.elements(roller_bearing, delta_r=0.0, psi=0.0)
        assert (delta_jk >= 0.0).all()
        assert (q_jk >= 0.0).all()

    def test_shapes(self, roller_bearing):
        delta_j, psi_j, delta_jk, q_jk, cp_j = ISO16281RollerSolver.elements(
            roller_bearing, delta_r=0.05, psi=0.0)
        Z, n_s = roller_bearing.Z, roller_bearing.n_s
        assert delta_j.shape == (Z,)
        assert psi_j.shape == (Z,)
        assert delta_jk.shape == (Z, n_s)
        assert q_jk.shape == (Z, n_s)
        assert cp_j.shape == (Z,)


class TestISO16281RollerSolverSolveContact:
    def test_pure_radial_load_converges_and_balances(self, roller_bearing):
        solver = ISO16281RollerSolver()
        result = solver.solve_contact(roller_bearing, Fr_xz=1000.0, Fr_xy=0.0,
                                      delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-2
        assert result.delta_r > 0.0
        assert result.delta_a == 0.0  # roller radial não tem capacidade axial

    def test_zero_load_gives_near_zero_residual(self, roller_bearing):
        """Tal como no ball solver, carga nula não tem raiz única em
        delta_r -- qualquer delta_r que deixe todos os rolos fora de
        contacto (delta_jk=0 em todo o lado) dá resíduo zero. Só se
        verifica que o solver converge com resíduo desprezável."""
        solver = ISO16281RollerSolver()
        result = solver.solve_contact(roller_bearing, Fr_xz=0.0, Fr_xy=0.0,
                                      delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert result.ok
        assert result.residual < 1e-3

    def test_larger_radial_load_gives_larger_delta_r(self, roller_bearing):
        solver = ISO16281RollerSolver()
        small = solver.solve_contact(roller_bearing, Fr_xz=500.0, Fr_xy=0.0,
                                     delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        large = solver.solve_contact(roller_bearing, Fr_xz=2000.0, Fr_xy=0.0,
                                     delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert large.delta_r > small.delta_r

    def test_alpha_j_is_constant_equal_to_bearing_alpha_0(self, roller_bearing):
        solver = ISO16281RollerSolver()
        result = solver.solve_contact(roller_bearing, Fr_xz=500.0, Fr_xy=0.0,
                                      delta_r_init=0.0, psi=0.0, phi_Fr=0.0)
        assert np.all(result.row.alpha_j == roller_bearing.alpha_0)


class _FakeRollerFamily:
    """Stub de bearing.family, espelha _FakeBallFamily mas para
    per_lamina_dynamic_capacity() -- devolve (q_ci, q_ce) escalares
    fixos, como a família real (ver capacity.py: per_lamina é um par
    escalar, não arrays por lamina). Mesma ressalva de não-registo que
    _FakeBallFamily."""
    @staticmethod
    def per_lamina_dynamic_capacity(bearing):
        return 200.0, 200.0


class TestISO16281RollerSolverPostprocess:
    """Espelha TestISO16281BallSolverPostprocess para o lado roller --
    mesma ligação (_attach_postprocessing()), valores só verificados
    contract-level, física certificada em test_contact_postprocessing.py."""

    def test_raw_result_has_no_stiffness_before_postprocessing(self, roller_bearing):
        solver = ISO16281RollerSolver(postprocess=False)
        node = make_node(Fr_xz=1000.0, label="B2")
        result = solver._solve_one_bearing(roller_bearing, node, phi_Fr=0.0, psi=0.0)

        assert result.row.Q_j is not None
        assert result.row.Q_j.shape == (roller_bearing.Z,)
        assert (result.row.Q_j >= 0.0).all()
        assert result.stiffness is None

    def test_attach_postprocessing_fills_stiffness_and_life_for_radial_family(self, roller_bearing):
        roller_bearing.family = _FakeRollerFamily()
        roller_bearing.C = 8000.0
        solver = ISO16281RollerSolver(postprocess=True)
        node = make_node(Fr_xz=1000.0, label="B2")
        result = solver._solve_one_bearing(roller_bearing, node, phi_Fr=0.0, psi=0.0)
        analysis = solver._attach_postprocessing(roller_bearing, result)

        assert analysis.load_distribution.stiffness is not None
        assert analysis.basic_life is not None
        assert analysis.basic_life.L10r > 0.0
        assert analysis.basic_life.Pref_r is not None and analysis.basic_life.Pref_r > 0.0
        assert analysis.basic_life.Pref_a is None  # duty radial -- roller radial não leva axial de qualquer forma

    def test_attach_postprocessing_thrust_family_gives_axial_pref_only(self, roller_bearing, monkeypatch):
        roller_bearing.family = _FakeRollerFamily()
        roller_bearing.C = 8000.0
        monkeypatch.setattr(contact_solver_module, "is_thrust_family", lambda family: True)
        solver = ISO16281RollerSolver(postprocess=True)
        node = make_node(Fr_xz=1000.0, label="B2")
        result = solver._solve_one_bearing(roller_bearing, node, phi_Fr=0.0, psi=0.0)
        analysis = solver._attach_postprocessing(roller_bearing, result)

        assert analysis.basic_life.Pref_r is None
        assert analysis.basic_life.Pref_a is not None and analysis.basic_life.Pref_a > 0.0


# =====================================================================
# MultiRowSolverBase -- shared-displacement multi-row (não revisto para
# acoplamento fila-a-fila -- ver docstring do próprio módulo)
# =====================================================================

class TestMultiRowSolverBaseRowViews:
    def test_rejects_fewer_than_two_rows(self):
        bearing = SimpleNamespace(label="b1", rows=[{"cp": 1.0}])
        with pytest.raises(ValueError, match=">= 2 rows"):
            MultiRowSolverBase._row_views(bearing)

    def test_rejects_missing_rows_attribute(self):
        bearing = SimpleNamespace(label="b1")
        with pytest.raises(ValueError, match=">= 2 rows"):
            MultiRowSolverBase._row_views(bearing)

    def test_wraps_dict_rows_as_simplenamespace(self):
        bearing = SimpleNamespace(label="b1", rows=[{"cp": 1.0}, {"cp": 2.0}])
        views = MultiRowSolverBase._row_views(bearing)
        assert all(isinstance(v, SimpleNamespace) for v in views)
        assert [v.cp for v in views] == [1.0, 2.0]

    def test_leaves_object_rows_untouched(self):
        row_obj = SimpleNamespace(cp=3.0)
        bearing = SimpleNamespace(label="b1", rows=[row_obj, SimpleNamespace(cp=4.0)])
        views = MultiRowSolverBase._row_views(bearing)
        assert views[0] is row_obj


class TestISO16281MultiRowBallSolver:
    def test_check_row_ready_rejects_non_positive_cp(self):
        solver = ISO16281MultiRowBallSolverSharedDisplacement()
        row = SimpleNamespace(cp=0.0)
        with pytest.raises(ValueError, match="cp must be positive"):
            solver._check_row_ready(row, "b1[row0]")

    def test_check_row_ready_accepts_positive_cp(self):
        solver = ISO16281MultiRowBallSolverSharedDisplacement()
        solver._check_row_ready(SimpleNamespace(cp=1.0), "b1[row0]")  # não deve levantar

    def test_two_identical_rows_each_take_half_the_load(self):
        solver = ISO16281MultiRowBallSolverSharedDisplacement()
        row_views = [SimpleNamespace(**BALL_BEARING_KWARGS), SimpleNamespace(**BALL_BEARING_KWARGS)]
        row_results, delta_r, delta_a, Fa, nfev, res, ok = solver._solve_rows(
            row_views, Fr_xz=1000.0, Fr_xy=0.0, Fa=0.0, psi=0.0, phi_Fr=0.0,
        )
        assert ok
        assert len(row_results) == 2
        assert row_results[0].Fr_row == pytest.approx(row_results[1].Fr_row, rel=1e-6)


class TestMultiRowSolverBasePostprocessGuard:
    """postprocess=True nunca foi desenhado/revisto para o caso
    shared-displacement multi-row (Cr/Ca por fila, combinação de L10r
    entre filas) -- _extra_ready_checks() recusa de imediato, antes de
    sequer validar bearing.rows."""

    def test_ball_multirow_rejects_postprocess_true(self):
        solver = ISO16281MultiRowBallSolverSharedDisplacement(postprocess=True)
        bearing = SimpleNamespace(label="b1")  # nem sequer .rows -- não chega a ser lido
        with pytest.raises(NotImplementedError, match="postprocess=True is not supported"):
            solver._extra_ready_checks(bearing, "b1")

    def test_roller_multirow_rejects_postprocess_true(self):
        solver = ISO16281MultiRowRollerSolverSharedDisplacement(postprocess=True)
        bearing = SimpleNamespace(label="b2")
        with pytest.raises(NotImplementedError, match="postprocess=True is not supported"):
            solver._extra_ready_checks(bearing, "b2")

    def test_ball_multirow_postprocess_false_unaffected(self):
        """postprocess=False continua a validar as filas normalmente --
        o guard não interfere com o caminho já testado acima."""
        solver = ISO16281MultiRowBallSolverSharedDisplacement(postprocess=False)
        bearing = SimpleNamespace(label="b1", rows=[dict(BALL_BEARING_KWARGS), dict(BALL_BEARING_KWARGS)])
        solver._extra_ready_checks(bearing, "b1")  # não deve levantar


class TestISO16281MultiRowRollerSolver:
    def test_check_row_ready_rejects_too_few_laminae(self):
        solver = ISO16281MultiRowRollerSolverSharedDisplacement()
        row = SimpleNamespace(n_s=10, x_k=np.zeros(10))
        with pytest.raises(ValueError, match="Sec 5.2.2"):
            solver._check_row_ready(row, "b1[row0]")

    def test_check_row_ready_rejects_x_k_length_mismatch(self):
        solver = ISO16281MultiRowRollerSolverSharedDisplacement()
        row = SimpleNamespace(n_s=30, x_k=np.zeros(29))
        with pytest.raises(ValueError, match=r"len\(x_k\)"):
            solver._check_row_ready(row, "b1[row0]")

    def test_two_identical_rows_each_take_half_the_load(self):
        solver = ISO16281MultiRowRollerSolverSharedDisplacement()
        row_views = [SimpleNamespace(**ROLLER_BEARING_KWARGS), SimpleNamespace(**ROLLER_BEARING_KWARGS)]
        row_results, delta_r, delta_a, Fa, nfev, res, ok = solver._solve_rows(
            row_views, Fr_xz=1000.0, Fr_xy=0.0, Fa=0.0, psi=0.0, phi_Fr=0.0,
        )
        assert ok
        assert len(row_results) == 2
        assert row_results[0].Fr_row == pytest.approx(row_results[1].Fr_row, rel=1e-6)
        assert row_results[0].Fa_row == 0.0 and row_results[1].Fa_row == 0.0
        assert delta_a == 0.0
        assert Fa == 0.0