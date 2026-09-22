# test_boundary_conditions.py
"""Testes de constraints/boundary_conditions.py. boundary_dofs() só
precisa de shaft_system.bearings (position/arrangement) -- usa
FakeBearing/FakeShaftSystem do conftest, não o ShaftSystem real.
Depende de Elem.find_node_index (import direto de
axisforge.mesh.shaft.element_type.elem.Elem dentro do módulo sob
teste) -- isso corre contra o Elem REAL do repositório, não um fake,
porque find_node_index é um staticmethod puro sobre uma lista de
floats."""
import pytest

from axisforge.solvers.machine_elements.shaft.fem_solvers.constraints.boundary_conditions import (
    boundary_dofs, boundary_dofs_explicit,
)

from axisforge.solvers.machine_elements.shaft.fem_solvers.tests.conftest import (
    FakeBearing, FakeShaftSystem,
)


class TestBoundaryDofs:
    def test_two_bearings_one_locating(self, fake_shaft_system_2brg, x_nodes_2elem):
        free, constrained = boundary_dofs(x_nodes_2elem, fake_shaft_system_2brg)
        n_dofs = 3 * len(x_nodes_2elem)
        assert sorted(free + constrained) == list(range(n_dofs))
        # locating (nó 0): u=0 e v=0 -> dofs 0, 1
        assert 0 in constrained
        assert 1 in constrained
        # floating (nó 2): só v=0 -> dof 3*2+1 = 7 ; u (dof 6) livre
        assert 7 in constrained
        assert 6 in free

    def test_no_locating_bearing_raises(self, x_nodes_2elem):
        sys_ = FakeShaftSystem(bearings=[
            FakeBearing(position=x_nodes_2elem[0], arrangement="floating"),
            FakeBearing(position=x_nodes_2elem[-1], arrangement="floating"),
        ])
        with pytest.raises(ValueError, match="no bearing with arrangement == 'locating'"):
            boundary_dofs(x_nodes_2elem, sys_)

    def test_single_bearing_position_raises(self, x_nodes_2elem):
        sys_ = FakeShaftSystem(bearings=[
            FakeBearing(position=x_nodes_2elem[0], arrangement="locating"),
        ])
        with pytest.raises(ValueError, match="at least two are required"):
            boundary_dofs(x_nodes_2elem, sys_)

    def test_two_bearings_same_node_counts_as_one_position(self, x_nodes_2elem):
        """Duas chumaceiras no MESMO nó (mesmo find_node_index) só
        contam como uma posição distinta -- deve continuar a levantar
        o erro de 'menos de duas posições'."""
        sys_ = FakeShaftSystem(bearings=[
            FakeBearing(position=x_nodes_2elem[0], arrangement="locating"),
            FakeBearing(position=x_nodes_2elem[0], arrangement="floating"),
        ])
        with pytest.raises(ValueError, match="at least two are required"):
            boundary_dofs(x_nodes_2elem, sys_)


class TestBoundaryDofsExplicit:
    def test_partitions_all_dofs(self, x_nodes_2elem):
        n_dofs = 3 * len(x_nodes_2elem)
        free, constrained = boundary_dofs_explicit(x_nodes_2elem, [0, 1, 2])
        assert constrained == [0, 1, 2]
        assert sorted(free + constrained) == list(range(n_dofs))

    def test_out_of_range_dof_raises(self, x_nodes_2elem):
        n_dofs = 3 * len(x_nodes_2elem)
        with pytest.raises(ValueError, match="out of range"):
            boundary_dofs_explicit(x_nodes_2elem, [n_dofs + 5])

    def test_negative_dof_raises(self, x_nodes_2elem):
        with pytest.raises(ValueError, match="out of range"):
            boundary_dofs_explicit(x_nodes_2elem, [-1])

    def test_duplicate_dofs_deduplicated(self, x_nodes_2elem):
        free, constrained = boundary_dofs_explicit(x_nodes_2elem, [0, 0, 1])
        assert constrained == [0, 1]
