# test_point_loads.py
"""Testes de assembly/load_assembly/point_loads.py. Puro sobre
x_nodes/listas (radial, axial, moments) -- usa Elem.find_node_index
REAL (import direto dentro do módulo sob teste), não um fake, porque é
um staticmethod puro sobre uma lista de floats."""
import numpy as np
import pytest

from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.load_assembly.point_loads import (
    assemble_point_load_vector,
)


class TestAssemblePointLoadVector:
    def test_empty_lists_return_zero_vector(self):
        x_nodes = [0.0, 10.0, 20.0]
        f = assemble_point_load_vector(x_nodes, [], [], [])
        assert f.shape == (9,)
        assert np.allclose(f, 0.0)

    def test_radial_load_goes_to_transverse_dof(self):
        x_nodes = [0.0, 10.0, 20.0]
        f = assemble_point_load_vector(x_nodes, [(10.0, -500.0)], [], [])
        # nó 1 -> DOFs [3, 4, 5] = [axial, transverse, moment]
        assert f[4] == -500.0
        assert f[3] == 0.0 and f[5] == 0.0

    def test_axial_load_goes_to_axial_dof(self):
        x_nodes = [0.0, 10.0, 20.0]
        f = assemble_point_load_vector(x_nodes, [], [(0.0, 1200.0)], [])
        assert f[0] == 1200.0

    def test_moment_goes_to_moment_dof(self):
        x_nodes = [0.0, 10.0, 20.0]
        f = assemble_point_load_vector(x_nodes, [], [], [(20.0, 3000.0)])
        assert f[8] == 3000.0

    def test_multiple_loads_at_same_node_accumulate(self):
        x_nodes = [0.0, 10.0]
        f = assemble_point_load_vector(x_nodes, [(0.0, 100.0), (0.0, 50.0)], [], [])
        assert f[1] == 150.0

    def test_all_three_kinds_combined(self):
        x_nodes = [0.0, 10.0]
        f = assemble_point_load_vector(
            x_nodes, radial=[(10.0, -200.0)], axial=[(0.0, 400.0)], moments=[(10.0, 1000.0)],
        )
        assert f[0] == 400.0   # axial no nó 0
        assert f[4] == -200.0  # radial no nó 1
        assert f[5] == 1000.0  # moment no nó 1
