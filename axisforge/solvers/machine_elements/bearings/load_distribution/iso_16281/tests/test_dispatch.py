# test_dispatch.py
"""Testes de dispatch.py -- register_contact_solver (decorator factory),
_most_specific, resolve_solver_cls_for_attrs e resolve_solver_cls.

Usa um _REGISTRY isolado (monkeypatch) em vez do global: dispatch.py é
populado a import-time por contact_solver.py (ISO16281BallSolver,
ISO16281RollerSolver, ...); registar classes de teste diretamente no
_REGISTRY partilhado poluiria esse registo para o resto da sessão de
testes."""
from types import SimpleNamespace

import pytest

from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281 import dispatch as dp


@pytest.fixture
def isolated_registry(monkeypatch):
    fresh: list = []
    monkeypatch.setattr(dp, "_REGISTRY", fresh)
    return fresh


class TestRegisterContactSolver:
    def test_rejects_empty_capability(self, isolated_registry):
        with pytest.raises(TypeError, match="capability must be non-empty"):
            dp.register_contact_solver(capability="", required_attrs=("a",))

    def test_rejects_empty_required_attrs(self, isolated_registry):
        with pytest.raises(TypeError, match="required_attrs must be non-empty"):
            dp.register_contact_solver(capability="point_contact", required_attrs=())

    def test_decorator_sets_class_attrs_and_registers(self, isolated_registry):
        @dp.register_contact_solver(capability="point_contact", required_attrs=("a", "b"))
        class _Solver:
            pass

        assert _Solver.CAPABILITY == "point_contact"
        assert _Solver.REQUIRED_ATTRS == ("a", "b")
        assert _Solver in isolated_registry

    def test_decorator_returns_the_same_class(self, isolated_registry):
        @dp.register_contact_solver(capability="c", required_attrs=("a",))
        class _Solver:
            """docstring preservada"""

        assert _Solver.__doc__ == "docstring preservada"


class TestMostSpecific:
    def test_single_match_returned_as_is(self, isolated_registry):
        @dp.register_contact_solver(capability="c", required_attrs=("a",))
        class _A:
            pass

        assert dp._most_specific([_A]) == [_A]

    def test_subset_requirement_is_filtered_out(self, isolated_registry):
        @dp.register_contact_solver(capability="c", required_attrs=("a",))
        class _Narrow:
            pass

        @dp.register_contact_solver(capability="c", required_attrs=("a", "b"))
        class _Wide:
            pass

        assert dp._most_specific([_Narrow, _Wide]) == [_Wide]

    def test_disjoint_requirements_both_kept(self, isolated_registry):
        @dp.register_contact_solver(capability="c", required_attrs=("a",))
        class _A:
            pass

        @dp.register_contact_solver(capability="c", required_attrs=("b",))
        class _B:
            pass

        assert set(dp._most_specific([_A, _B])) == {_A, _B}

    def test_empty_list_returns_empty(self, isolated_registry):
        assert dp._most_specific([]) == []


class TestResolveSolverClsForAttrs:
    def test_resolves_unique_match(self, isolated_registry):
        @dp.register_contact_solver(capability="c", required_attrs=("a", "b"))
        class _Solver:
            pass

        assert dp.resolve_solver_cls_for_attrs({"a", "b", "c"}) is _Solver

    def test_raises_when_no_match(self, isolated_registry):
        @dp.register_contact_solver(capability="c", required_attrs=("a",))
        class _Solver:
            pass

        with pytest.raises(dp.SolverDispatchError, match="nenhum solver cobre"):
            dp.resolve_solver_cls_for_attrs({"z"}, label="bearing-1")

    def test_raises_when_ambiguous(self, isolated_registry):
        @dp.register_contact_solver(capability="c1", required_attrs=("a",))
        class _A:
            pass

        @dp.register_contact_solver(capability="c2", required_attrs=("b",))
        class _B:
            pass

        with pytest.raises(dp.SolverDispatchError, match="ambiguo"):
            dp.resolve_solver_cls_for_attrs({"a", "b"})

    def test_prefers_most_specific_match(self, isolated_registry):
        @dp.register_contact_solver(capability="c", required_attrs=("a",))
        class _Narrow:
            pass

        @dp.register_contact_solver(capability="c", required_attrs=("a", "b"))
        class _Wide:
            pass

        assert dp.resolve_solver_cls_for_attrs({"a", "b"}) is _Wide


class TestResolveSolverCls:
    def test_resolves_by_capability_and_family_required_for(self, isolated_registry):
        @dp.register_contact_solver(capability="point_contact", required_attrs=("a",))
        class _Solver:
            pass

        family = SimpleNamespace(name="fake",
                                  REQUIRED_FOR={"point_contact": frozenset({"a", "b"})},
                                  CAPABILITIES=frozenset({"point_contact"}))
        bearing = SimpleNamespace(family=family, is_enabled=lambda cap: cap == "point_contact")

        assert dp.resolve_solver_cls(bearing) is _Solver

    def test_ignores_disabled_capability(self, isolated_registry):
        @dp.register_contact_solver(capability="point_contact", required_attrs=("a",))
        class _Solver:
            pass

        family = SimpleNamespace(name="fake",
                                  REQUIRED_FOR={"point_contact": frozenset({"a"})},
                                  CAPABILITIES=frozenset())
        bearing = SimpleNamespace(family=family, is_enabled=lambda cap: False)

        with pytest.raises(dp.SolverDispatchError, match="nenhum solver registado cobre"):
            dp.resolve_solver_cls(bearing, label="b1")

    def test_ignores_solver_whose_required_attrs_exceed_family_required_for(self, isolated_registry):
        """O solver está habilitado (is_enabled=True) mas a family não
        promete, para essa capability, os atributos que ele exige --
        não deve ser escolhido."""
        @dp.register_contact_solver(capability="point_contact", required_attrs=("a", "b"))
        class _Solver:
            pass

        family = SimpleNamespace(name="fake",
                                  REQUIRED_FOR={"point_contact": frozenset({"a"})},
                                  CAPABILITIES=frozenset({"point_contact"}))
        bearing = SimpleNamespace(family=family, is_enabled=lambda cap: True)

        with pytest.raises(dp.SolverDispatchError, match="nenhum solver registado cobre"):
            dp.resolve_solver_cls(bearing)

    def test_raises_when_ambiguous(self, isolated_registry):
        @dp.register_contact_solver(capability="point_contact", required_attrs=("a",))
        class _A:
            pass

        @dp.register_contact_solver(capability="point_contact", required_attrs=("b",))
        class _B:
            pass

        family = SimpleNamespace(name="fake",
                                  REQUIRED_FOR={"point_contact": frozenset({"a", "b"})},
                                  CAPABILITIES=frozenset({"point_contact"}))
        bearing = SimpleNamespace(family=family, is_enabled=lambda cap: True)

        with pytest.raises(dp.SolverDispatchError, match="ambiguo"):
            dp.resolve_solver_cls(bearing, label="b1")
