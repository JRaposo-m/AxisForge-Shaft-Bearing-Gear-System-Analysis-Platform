# test_mesh_grade.py
"""Testes de mesh_generation/mesh_grade.py -- Grader gera grades por
bisecção sucessiva de um subdomínio [x_lo, x_hi] dentro do mesh base."""
import pytest

from axisforge.mesh.shaft.mesh_generation.mesh_grade import Grader
from axisforge.config import SOLVER_TOLERANCE

BASE_NODES = [0.0, 10.0, 20.0, 30.0, 40.0]


class TestParseGrade:
    def test_grade_0_returns_base_nodes(self):
        grader = Grader(x_lo=10.0, x_hi=30.0, x_nodes=BASE_NODES)
        assert grader.get_grade("grade_0") == [10.0, 20.0, 30.0]

    @pytest.mark.parametrize("bad_grade", ["gradeX", "grade", "1_grade", "GRADE_1"])
    def test_rejects_grade_without_valid_prefix(self, bad_grade):
        """Falha logo no prefixo -- não começa por 'grade_'."""
        grader = Grader(x_lo=10.0, x_hi=30.0, x_nodes=BASE_NODES)
        with pytest.raises(ValueError, match="Invalid grade string"):
            grader.get_grade(bad_grade)

    def test_rejects_empty_suffix_after_prefix(self):
        """'grade_' passa o teste de prefixo (startswith 'grade_'), mas
        o sufixo vazio não é dígito -- cai no ramo 'Invalid grade
        level', não em 'Invalid grade string'."""
        grader = Grader(x_lo=10.0, x_hi=30.0, x_nodes=BASE_NODES)
        with pytest.raises(ValueError, match="Invalid grade level"):
            grader.get_grade("grade_")

    def test_rejects_non_digit_suffix(self):
        grader = Grader(x_lo=10.0, x_hi=30.0, x_nodes=BASE_NODES)
        with pytest.raises(ValueError, match="Invalid grade level"):
            grader.get_grade("grade_N")

    def test_rejects_negative_looking_level(self):
        """'grade_-1' falha em isdigit() (o '-' não é dígito), por isso
        cai no ramo 'Invalid grade level', não produz N negativo."""
        grader = Grader(x_lo=10.0, x_hi=30.0, x_nodes=BASE_NODES)
        with pytest.raises(ValueError, match="Invalid grade level"):
            grader.get_grade("grade_-1")


class TestBaseNodesWindowing:
    def test_excludes_nodes_outside_bounds(self):
        grader = Grader(x_lo=10.0, x_hi=30.0, x_nodes=BASE_NODES)
        nodes = grader.get_grade("grade_0")
        assert 0.0 not in nodes and 40.0 not in nodes

    def test_includes_boundary_within_solver_tolerance(self):
        x_lo = 10.0 + SOLVER_TOLERANCE / 2.0
        grader = Grader(x_lo=x_lo, x_hi=30.0, x_nodes=BASE_NODES)
        assert 10.0 in grader.get_grade("grade_0")

    def test_base_nodes_are_sorted(self):
        grader = Grader(x_lo=0.0, x_hi=40.0, x_nodes=list(reversed(BASE_NODES)))
        assert grader.get_grade("grade_0") == BASE_NODES


class TestBisection:
    def test_grade_1_adds_midpoints(self):
        grader = Grader(x_lo=0.0, x_hi=20.0, x_nodes=BASE_NODES)
        assert grader.get_grade("grade_1") == [0.0, 5.0, 10.0, 15.0, 20.0]

    def test_grade_2_bisects_again(self):
        grader = Grader(x_lo=0.0, x_hi=20.0, x_nodes=BASE_NODES)
        grade_2 = grader.get_grade("grade_2")
        # grade_0: 2 elementos -> grade_1: 4 elementos (5 nos) ->
        # grade_2: 8 elementos (9 nos)
        assert len(grade_2) == 9
        assert grade_2 == sorted(grade_2)

    def test_each_grade_level_doubles_element_count(self):
        grader = Grader(x_lo=0.0, x_hi=40.0, x_nodes=BASE_NODES)
        n0 = len(grader.get_grade("grade_0"))
        n1 = len(grader.get_grade("grade_1"))
        n2 = len(grader.get_grade("grade_2"))
        assert (n1 - 1) == 2 * (n0 - 1)
        assert (n2 - 1) == 2 * (n1 - 1)

    def test_grade_result_always_sorted(self):
        grader = Grader(x_lo=0.0, x_hi=40.0, x_nodes=BASE_NODES)
        assert grader.get_grade("grade_3") == sorted(grader.get_grade("grade_3"))

    def test_repeated_calls_are_independent_and_deterministic(self):
        """get_grade recomeça de _base_nodes() de cada vez -- chamar
        grade_2 duas vezes não deve acumular bisecções."""
        grader = Grader(x_lo=0.0, x_hi=20.0, x_nodes=BASE_NODES)
        assert grader.get_grade("grade_2") == grader.get_grade("grade_2")

class TestRefinementFloor:
    def test_raises_before_bisecting_below_the_floor(self):
        from axisforge.config import MESH_MIN_NODE_DIST_MM
        from axisforge.mesh.shaft.mesh_generation.mesh_grade import RefinementFloorReached
        h0 = 1.0
        grader = Grader(x_lo=0.0, x_hi=h0, x_nodes=[0.0, h0])
        # h after N bisections = h0 / 2**N; last allowed N keeps h > floor
        n_ok = 0
        while h0 / 2 ** (n_ok + 1) > MESH_MIN_NODE_DIST_MM:
            n_ok += 1
        assert len(grader.get_grade(f"grade_{n_ok}")) == 2 ** n_ok + 1
        with pytest.raises(RefinementFloorReached) as exc:
            grader.get_grade(f"grade_{n_ok + 1}")
        assert exc.value.grade == f"grade_{n_ok + 1}"
        assert exc.value.floor == MESH_MIN_NODE_DIST_MM
        assert exc.value.h_min == pytest.approx(h0 / 2 ** n_ok)

    def test_is_a_value_error(self):
        from axisforge.mesh.shaft.mesh_generation.mesh_grade import RefinementFloorReached
        grader = Grader(x_lo=0.0, x_hi=0.15, x_nodes=[0.0, 0.15])
        with pytest.raises(ValueError):
            grader.get_grade("grade_1")

    def test_grade_0_never_raises(self):
        grader = Grader(x_lo=0.0, x_hi=0.05, x_nodes=[0.0, 0.05])
        assert grader.get_grade("grade_0") == [0.0, 0.05]