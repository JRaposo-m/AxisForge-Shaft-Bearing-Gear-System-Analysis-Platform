# test_submodel_extraction.py
"""Testes de constraints/submodel_extraction.py -- puro sobre arrays
sintéticos, sem depender de Elem/ShaftSystem/solve()."""
import numpy as np
import pytest

from axisforge.solvers.machine_elements.shaft.fem_solvers.constraints.submodel_extraction import (
    extract_submodel_values,
)


def _make_arrays(n_nodes: int):
    n_dofs = 3 * n_nodes
    d_xz = np.arange(n_dofs, dtype=float)
    d_xy = np.arange(n_dofs, dtype=float) + 100.0
    f_xz = np.arange(n_dofs, dtype=float) + 1000.0
    f_xy = np.arange(n_dofs, dtype=float) + 2000.0
    return d_xz, d_xy, f_xz, f_xy


class TestExtractSubmodelValues:
    def test_raises_if_solve_not_run(self):
        with pytest.raises(RuntimeError, match="before solve"):
            extract_submodel_values([0.0, 10.0], [0.0, 10.0], None, None, None, None)

    def test_raises_if_x_not_length_2(self):
        x_nodes = [0.0, 10.0, 20.0]
        d_xz, d_xy, f_xz, f_xy = _make_arrays(3)
        with pytest.raises(ValueError, match="exactly \\[x_lo, x_hi\\]"):
            extract_submodel_values(x_nodes, [0.0], d_xz, d_xy, f_xz, f_xy)

    def test_returns_only_nodes_within_interval(self):
        x_nodes = [0.0, 10.0, 20.0, 30.0]
        d_xz, d_xy, f_xz, f_xy = _make_arrays(4)
        result = extract_submodel_values(x_nodes, [10.0, 20.0], d_xz, d_xy, f_xz, f_xy)
        assert set(result.keys()) == {10.0, 20.0}

    def test_full_interval_returns_every_node(self):
        x_nodes = [0.0, 10.0, 20.0]
        d_xz, d_xy, f_xz, f_xy = _make_arrays(3)
        result = extract_submodel_values(x_nodes, [0.0, 20.0], d_xz, d_xy, f_xz, f_xy)
        assert set(result.keys()) == {0.0, 10.0, 20.0}

    def test_values_read_from_correct_dof_stride(self):
        """u/v_xz/theta_xz devem vir de d_xz no stride-3 [u, v, theta];
        v_xy/theta_xy de d_xy no mesmo stride (sem ler u de d_xy --
        essa é a armadilha do DOF axial documentada em todo o código)."""
        x_nodes = [0.0, 10.0]
        d_xz, d_xy, f_xz, f_xy = _make_arrays(2)
        result = extract_submodel_values(x_nodes, [0.0, 10.0], d_xz, d_xy, f_xz, f_xy)

        # nó 1 (x=10.0) -> DOFs globais 3, 4, 5
        node1 = result[10.0]
        assert node1["u"] == d_xz[3]
        assert node1["v_xz"] == d_xz[4]
        assert node1["theta_xz"] == d_xz[5]
        assert node1["v_xy"] == d_xy[4]
        assert node1["theta_xy"] == d_xy[5]
        # confirma que "u" NUNCA é lido de d_xy
        assert "u" not in {k for k in node1 if node1[k] == d_xy[3]}

    def test_keys_present_per_node(self):
        x_nodes = [0.0]
        d_xz, d_xy, f_xz, f_xy = _make_arrays(1)
        result = extract_submodel_values(x_nodes, [0.0, 0.0], d_xz, d_xy, f_xz, f_xy)
        expected_keys = {
            "u", "v_xz", "theta_xz", "f_u", "f_v_xz", "f_theta_xz",
            "v_xy", "theta_xy", "f1_xy", "f2_xy",
        }
        assert set(result[0.0].keys()) == expected_keys
