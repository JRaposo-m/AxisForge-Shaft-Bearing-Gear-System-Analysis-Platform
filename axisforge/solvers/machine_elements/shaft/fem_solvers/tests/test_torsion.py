# test_torsion.py
"""Testes de global_solver/torsion.py::TorsionSolver.

_torque_and_stress()/validate_equilibrium() não precisam de material
nenhum -- testados de ponta a ponta com um FakeShaftSystem local.
solve() (que chama _twist_angle() internamente) e _twist_angle()
diretamente NÃO conseguem completar: _twist_angle() chama
shaft_system.shaft.J_at(x), que ainda não existe em Shaft (ver o
TODO/MISSING PIECE no docstring do próprio módulo) -- isto está
documentado abaixo como test_KNOWN_GAP, não escondido/contornado.
get_material é monkeypatched (não a versão real -- nunca vi
core/materials.py nesta conversa) só para isolar esse gap
especificamente ao J_at e não a um material_id inválido."""
import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver import torsion as torsion_module
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.torsion import TorsionSolver

from axisforge.solvers.machine_elements.shaft.fem_solvers.tests.conftest import (
    FakeShaft, FakeShaftSystem,
)


@dataclass
class FakeTorqueLoad:
    position: float
    magnitude: float
    label: str | None = None


@dataclass
class FakeSection:
    material_id: str = "fake-steel"


class _TorsionCapableShaft(FakeShaft):
    """Estende FakeShaft com section_at() -- só usado aqui, porque
    _twist_angle() é o único consumidor deste método em fem_solvers/."""
    def section_at(self, x: float):
        return FakeSection(), None


@dataclass
class FakeMaterial:
    E: float = 210_000.0
    poisson_ratio: float = 0.3


class TestTorqueSourcesAndStress:
    def test_torque_sources_sorted_by_position(self):
        sys_ = FakeShaftSystem(torque_loads=[
            FakeTorqueLoad(position=50.0, magnitude=-100.0, label="driven"),
            FakeTorqueLoad(position=0.0, magnitude=100.0, label="driver"),
        ])
        sources = TorsionSolver()._torque_sources(sys_)
        assert [s[0] for s in sources] == [0.0, 50.0]

    def test_T_accumulates_only_up_to_x(self):
        sys_ = FakeShaftSystem(
            shaft=_TorsionCapableShaft(Wt=12_000.0),
            torque_loads=[
                FakeTorqueLoad(position=0.0, magnitude=100.0, label="driver"),
                FakeTorqueLoad(position=50.0, magnitude=-100.0, label="driven"),
            ],
        )
        x_nodes = [0.0, 25.0, 50.0, 75.0]
        T_total, tau_total, contributions = TorsionSolver()._torque_and_stress(sys_, x_nodes)

        assert math.isclose(T_total[0], 100.0)   # só a fonte em x=0
        assert math.isclose(T_total[1], 100.0)   # ainda só a de x=0
        assert math.isclose(T_total[2], 0.0)     # +100 -100 já incluído
        assert math.isclose(T_total[3], 0.0)     # equilíbrio mantém-se

    def test_tau_uses_Wt_at_in_MPa(self):
        sys_ = FakeShaftSystem(
            shaft=_TorsionCapableShaft(Wt=1000.0),
            torque_loads=[FakeTorqueLoad(position=0.0, magnitude=1.0, label="t")],
        )
        x_nodes = [0.0]
        T_total, tau_total, _ = TorsionSolver()._torque_and_stress(sys_, x_nodes)
        # tau = T[N*m]*1000/Wt[mm^3] = 1*1000/1000 = 1 MPa
        assert math.isclose(tau_total[0], 1.0)

    def test_contributions_breakdown_per_node(self):
        sys_ = FakeShaftSystem(
            shaft=_TorsionCapableShaft(),
            torque_loads=[FakeTorqueLoad(position=0.0, magnitude=50.0, label="only")],
        )
        x_nodes = [0.0, 10.0]
        _, _, contributions = TorsionSolver()._torque_and_stress(sys_, x_nodes)
        assert len(contributions) == 2
        assert contributions[0][0]["label"] == "only"
        assert contributions[1][0]["label"] == "only"


class TestValidateEquilibrium:
    def test_balanced_system_no_errors(self):
        sys_ = FakeShaftSystem(torque_loads=[
            FakeTorqueLoad(position=0.0, magnitude=100.0),
            FakeTorqueLoad(position=50.0, magnitude=-100.0),
        ])
        assert TorsionSolver().validate_equilibrium(sys_) == []

    def test_unbalanced_system_reports_error(self):
        sys_ = FakeShaftSystem(torque_loads=[
            FakeTorqueLoad(position=0.0, magnitude=100.0),
            FakeTorqueLoad(position=50.0, magnitude=-40.0),
        ])
        errors = TorsionSolver().validate_equilibrium(sys_)
        assert len(errors) == 1
        assert "equilibrium violated" in errors[0]


class TestTwistAngleKnownGap:
    def test_KNOWN_GAP_twist_angle_requires_missing_shaft_J_at(self, monkeypatch):
        """MISSING PIECE documentado no docstring do módulo:
        _twist_angle() chama shaft_system.shaft.J_at(x), que não existe
        em Shaft (nem em FakeShaft -- de propósito, para espelhar o
        estado real). get_material é monkeypatched só para garantir
        que o AttributeError vem mesmo do J_at em falta, e não de
        qualquer outra dependência de materiais que eu não vi.
        Substitui este teste por um teste real quando Shaft.J_at()
        existir."""
        monkeypatch.setattr(torsion_module, "get_material", lambda material_id: FakeMaterial())

        sys_ = FakeShaftSystem(shaft=_TorsionCapableShaft())
        x_nodes = [0.0, 10.0]
        T_total = np.array([5.0, 5.0])

        with pytest.raises(AttributeError, match="J_at"):
            TorsionSolver()._twist_angle(sys_, x_nodes, T_total)

    def test_KNOWN_GAP_solve_blocked_by_same_gap(self, monkeypatch):
        """solve() chama _torque_and_stress() (que funciona) e depois
        _twist_angle() (que não) -- confirma que o TODO se propaga até
        ao entry point público."""
        monkeypatch.setattr(torsion_module, "get_material", lambda material_id: FakeMaterial())

        sys_ = FakeShaftSystem(
            shaft=_TorsionCapableShaft(),
            torque_loads=[FakeTorqueLoad(position=0.0, magnitude=10.0)],
        )
        with pytest.raises(AttributeError, match="J_at"):
            TorsionSolver().solve(sys_, [0.0, 10.0])
