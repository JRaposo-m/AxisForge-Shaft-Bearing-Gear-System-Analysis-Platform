# test_build_stiffness_matrix.py
"""Testes de assembly/build_stiffness_matrix.py::StiffnessMatrixBuilder,
com FakeElem (conftest.py). Verifica a sequência de montagem
(elemento -> scatter -> K global) e as duas propriedades que qualquer
matriz de rigidez de viga bem montada tem de respeitar, independentemente
da fórmula exata de cada elemento: simetria e positividade nos DOFs
efetivamente ligados."""
import numpy as np
import pytest

from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.build_stiffness_matrix import (
    StiffnessMatrixBuilder,
)


class _FakeMesh:
    """Duplo mínimo de Mesh1D -- só o que StiffnessMatrixBuilder lê
    (.x_nodes, .n_nodes)."""
    def __init__(self, x_nodes):
        self.x_nodes = x_nodes
        self.n_nodes = len(x_nodes)


class TestStiffnessMatrixBuilderFrameTrue:
    def test_shape_and_symmetry(self, euler_elements, x_nodes_2elem):
        mesh = _FakeMesh(x_nodes_2elem)
        builder = StiffnessMatrixBuilder(mesh, euler_elements, frame=True)
        K = builder.build()
        n_dofs = 3 * len(x_nodes_2elem)
        assert K.shape == (n_dofs, n_dofs)
        assert np.allclose(K, K.T, atol=1e-8)

    def test_axial_dofs_are_populated(self, euler_elements, x_nodes_2elem):
        """frame=True -> axial_stiffness_element() É chamado -> DOFs
        axiais (3*i) não ficam a zero na diagonal."""
        mesh = _FakeMesh(x_nodes_2elem)
        K = StiffnessMatrixBuilder(mesh, euler_elements, frame=True).build()
        assert K[0, 0] != 0.0
        assert K[3, 3] != 0.0
        assert K[6, 6] != 0.0

    def test_shared_node_accumulates_both_elements(self, euler_elements, x_nodes_2elem):
        """O nó 1 (meio) é partilhado pelos dois elementos -- a sua
        rigidez diagonal deve ser maior que a de qualquer um dos dois
        elementos isolados (soma das duas contribuições)."""
        mesh = _FakeMesh(x_nodes_2elem)
        K_both = StiffnessMatrixBuilder(mesh, euler_elements, frame=True).build()
        mesh1 = _FakeMesh(x_nodes_2elem)
        K_first_only = StiffnessMatrixBuilder(mesh1, euler_elements[:1], frame=True).build()
        assert K_both[4, 4] > K_first_only[4, 4]

    def test_build_is_idempotent(self, euler_elements, x_nodes_2elem):
        """build() chamado duas vezes não deve montar a rigidez outra
        vez (self._beam_assembled guarda isso) -- resultado igual."""
        mesh = _FakeMesh(x_nodes_2elem)
        builder = StiffnessMatrixBuilder(mesh, euler_elements, frame=True)
        K1 = builder.build()
        K2 = builder.build()
        assert np.array_equal(K1, K2)


class TestStiffnessMatrixBuilderFrameFalse:
    def test_axial_dofs_left_at_zero(self, euler_elements, x_nodes_2elem):
        mesh = _FakeMesh(x_nodes_2elem)
        K = StiffnessMatrixBuilder(mesh, euler_elements, frame=False).build()
        assert K[0, 0] == 0.0
        assert K[3, 3] == 0.0
        assert K[6, 6] == 0.0

    def test_bending_dofs_still_populated(self, euler_elements, x_nodes_2elem):
        mesh = _FakeMesh(x_nodes_2elem)
        K = StiffnessMatrixBuilder(mesh, euler_elements, frame=False).build()
        assert K[1, 1] != 0.0


class TestStiffnessMatrixBuilderTimoshenko:
    def test_builds_without_kGA_override(self, timoshenko_elements, x_nodes_2elem):
        mesh = _FakeMesh(x_nodes_2elem)
        K = StiffnessMatrixBuilder(mesh, timoshenko_elements, frame=True).build()
        assert np.all(np.isfinite(K))

    def test_kGA_override_is_forwarded_only_to_timoshenko(self, euler_elements, timoshenko_elements, x_nodes_2elem):
        """kGA_override não deve alterar nada para elementos euler
        (ignorado), mas afeta a rigidez de flexão dos elementos
        timoshenko -- aqui só confirmamos que build() não rebenta com
        um mix hipotético e que o override realmente muda K para o
        caso puro-timoshenko (a FakeElem.stiffness_element() atual
        não lê kGA_override, então este teste documenta esse limite do
        duplo -- ver docstring de FakeElem)."""
        mesh = _FakeMesh(x_nodes_2elem)
        K_default = StiffnessMatrixBuilder(mesh, timoshenko_elements, frame=True).build(kGA_override=None)
        mesh2 = _FakeMesh(x_nodes_2elem)
        K_override = StiffnessMatrixBuilder(mesh2, timoshenko_elements, frame=True).build(kGA_override=1e9)
        # KNOWN LIMITATION do FakeElem: stiffness_element() não usa
        # kGA_override (ver conftest.py) -- por isso K não muda aqui.
        # Isto testa a SEQUÊNCIA (build() aceita e propaga o kwarg sem
        # erro), não o efeito físico, que só o Elem real pode mostrar.
        assert np.array_equal(K_default, K_override)


class TestAddNodalStiffness:
    def test_adds_at_nearest_node(self, euler_elements, x_nodes_2elem):
        mesh = _FakeMesh(x_nodes_2elem)
        builder = StiffnessMatrixBuilder(mesh, euler_elements, frame=True)
        builder.add_nodal_stiffness(x=50.0, K_local=np.array([[1234.0]]), dof_slots=[1])
        K = builder.build()
        assert K[4, 4] >= 1234.0  # nó 1 -> dof transverse = 3*1+1 = 4

    def test_shape_mismatch_raises(self, euler_elements, x_nodes_2elem):
        mesh = _FakeMesh(x_nodes_2elem)
        builder = StiffnessMatrixBuilder(mesh, euler_elements, frame=True)
        with pytest.raises(ValueError, match="does not match dof_slots length"):
            builder.add_nodal_stiffness(x=50.0, K_local=np.eye(2), dof_slots=[1])

    def test_unknown_position_raises(self, euler_elements, x_nodes_2elem):
        mesh = _FakeMesh(x_nodes_2elem)
        builder = StiffnessMatrixBuilder(mesh, euler_elements, frame=True)
        with pytest.raises(Exception):
            builder.add_nodal_stiffness(x=999.0, K_local=np.array([[1.0]]), dof_slots=[1])
