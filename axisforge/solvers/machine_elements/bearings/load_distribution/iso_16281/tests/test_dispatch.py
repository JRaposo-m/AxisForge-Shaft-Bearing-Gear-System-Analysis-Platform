# test_dispatch.py
"""Testes de dispatch.py -- register_contact_solver (decorator factory),
family_matches_capability e resolve_solver_cls.

Usa um _REGISTRY isolado (monkeypatch) em vez do global: dispatch.py é
populado a import-time por contact_solver.py (ISO16281BallSolver,
ISO16281RollerSolver, ...); registar classes de teste diretamente no
_REGISTRY partilhado poluiria esse registo para o resto da sessão de
testes.

REQUIRED_ATTRS / _most_specific / resolve_solver_cls_for_attrs foram
removidos do dispatch.py atual -- o dispatch já não é por conjunto de
atributos exigidos, é por capability (point_contact/line_contact via
is_point_contact_family/is_line_contact_family) + iso16281_analysis +
presença de bearing.rows. Usam-se aqui famílias REAIS e já registadas
(DeepGrooveBallFamily / CylindricalRollerFamily), porque
family_matches_capability faz lookup nos registos reais de family.py
(_POINT_CONTACT/_LINE_CONTACT), não em atributos fabricados numa
SimpleNamespace fake."""
from types import SimpleNamespace

import pytest

from axisforge.core.machine_elements.bearings.families.family import (
    CylindricalRollerFamily, DeepGrooveBallFamily,
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281 import dispatch as dp


@pytest.fixture
def isolated_registry(monkeypatch):
    fresh: list = []
    monkeypatch.setattr(dp, "_REGISTRY", fresh)
    return fresh


class TestFamilyMatchesCapability:
    def test_point_contact_family_matches_point_contact(self):
        assert dp.family_matches_capability(DeepGrooveBallFamily(), "point_contact") is True

    def test_point_contact_family_does_not_match_line_contact(self):
        assert dp.family_matches_capability(DeepGrooveBallFamily(), "line_contact") is False

    def test_line_contact_family_matches_line_contact(self):
        assert dp.family_matches_capability(CylindricalRollerFamily(), "line_contact") is True

    def test_line_contact_family_does_not_match_point_contact(self):
        assert dp.family_matches_capability(CylindricalRollerFamily(), "point_contact") is False

    def test_unknown_capability_returns_false_instead_of_raising(self):
        """Uma capability desconhecida (typo, ou nova capability registada
        sem entrada em _CAPABILITY_CHECK) falha como 'nenhum solver
        corresponde', não como KeyError -- ver docstring da própria
        função em dispatch.py."""
        assert dp.family_matches_capability(DeepGrooveBallFamily(), "torque_contact") is False


class TestRegisterContactSolver:
    def test_rejects_empty_capability(self, isolated_registry):
        with pytest.raises(TypeError, match="capability must be non-empty"):
            dp.register_contact_solver(capability="")

    def test_rejects_unknown_capability(self, isolated_registry):
        with pytest.raises(TypeError, match="unknown capability"):
            dp.register_contact_solver(capability="torque_contact")

    def test_decorator_sets_class_attr_and_registers(self, isolated_registry):
        @dp.register_contact_solver(capability="point_contact")
        class _Solver:
            MULTIROW_SOLVER = None

        assert _Solver.CAPABILITY == "point_contact"
        assert _Solver in isolated_registry

    def test_decorator_returns_the_same_class(self, isolated_registry):
        @dp.register_contact_solver(capability="point_contact")
        class _Solver:
            """docstring preservada"""
            MULTIROW_SOLVER = None

        assert _Solver.__doc__ == "docstring preservada"


class TestResolveSolverCls:
    """resolve_solver_cls() faz 3 verificações em sequência: capability
    (family da bearing registada no capability do candidato) ->
    iso16281_analysis (truthy na bearing) -> exactamente 1 candidato com
    MULTIROW_SOLVER wired (não None) para essa capability. O candidato
    registado em _REGISTRY é sempre o solver SINGLE-ROW; MULTIROW_SOLVER
    é o atributo nele que aponta para o sibling multi-row -- resolve
    devolve esse sibling só quando a bearing tem `.rows`, senão devolve
    o próprio single-row."""

    def test_resolves_single_row_by_capability(self, isolated_registry):
        class _MultiRow:
            pass

        @dp.register_contact_solver(capability="point_contact")
        class _Solver:
            MULTIROW_SOLVER = _MultiRow

        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=True)
        assert dp.resolve_solver_cls(bearing) is _Solver

    def test_resolves_multirow_when_bearing_has_rows(self, isolated_registry):
        class _MultiRow:
            pass

        @dp.register_contact_solver(capability="point_contact")
        class _Solver:
            MULTIROW_SOLVER = _MultiRow

        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=True, rows=())
        assert dp.resolve_solver_cls(bearing) is _MultiRow

    def test_raises_when_no_capability_matches(self, isolated_registry):
        class _MultiRow:
            pass

        @dp.register_contact_solver(capability="line_contact")
        class _Solver:
            MULTIROW_SOLVER = _MultiRow

        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=True)
        with pytest.raises(dp.SolverDispatchError, match="nenhuma capability registada"):
            dp.resolve_solver_cls(bearing, label="b1")

    def test_raises_when_iso16281_analysis_not_true(self, isolated_registry):
        class _MultiRow:
            pass

        @dp.register_contact_solver(capability="point_contact")
        class _Solver:
            MULTIROW_SOLVER = _MultiRow

        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=False)
        with pytest.raises(dp.SolverDispatchError, match="iso16281_analysis"):
            dp.resolve_solver_cls(bearing, label="b1")

    def test_raises_when_iso16281_analysis_attribute_absent(self, isolated_registry):
        class _MultiRow:
            pass

        @dp.register_contact_solver(capability="point_contact")
        class _Solver:
            MULTIROW_SOLVER = _MultiRow

        bearing = SimpleNamespace(family=DeepGrooveBallFamily())
        with pytest.raises(dp.SolverDispatchError, match="iso16281_analysis"):
            dp.resolve_solver_cls(bearing)

    def test_raises_when_two_single_row_solvers_registered_for_capability(self, isolated_registry):
        @dp.register_contact_solver(capability="point_contact")
        class _A:
            MULTIROW_SOLVER = object()

        @dp.register_contact_solver(capability="point_contact")
        class _B:
            MULTIROW_SOLVER = object()

        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=True)
        with pytest.raises(dp.SolverDispatchError, match="exactamente 1"):
            dp.resolve_solver_cls(bearing, label="b1")

    def test_raises_when_matching_candidate_has_no_multirow_solver_wired(self, isolated_registry):
        """Único candidato registado, mas com MULTIROW_SOLVER=None (nunca
        ligado ao sibling multi-row) -- conta como 0 single_row
        (o filtro exige MULTIROW_SOLVER is not None), não como 1."""
        @dp.register_contact_solver(capability="point_contact")
        class _Solver:
            MULTIROW_SOLVER = None

        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=True)
        with pytest.raises(dp.SolverDispatchError, match="exactamente 1"):
            dp.resolve_solver_cls(bearing, label="b1")