# test_mesh_convergence_study.py
"""Testes de MeshConvergenceStudy -- construção, add_level de ponta a
ponta (com SubmodelResult sintéticos do conftest.py, desenhados para
bater com o caso p=2/r=2 exato de test_richardson_gci.py), e
finalize_unconverged/build_mesh_refinement_result."""
import math
import numpy as np
import pytest

from axisforge.solvers.mesh.convergence_solver import (
    MeshConvergenceStudy, build_mesh_refinement_result,
)
from axisforge.results.convergence_results.convergence_results import ConvergenceRecord

from axisforge.solvers.mesh.tests.conftest import make_result, array_with_exact_max_minus_mean


class TestMeshConvergenceStudyConstruction:
    def test_unknown_form_raises_at_construction(self):
        with pytest.raises(ValueError, match="unknown eval form"):
            MeshConvergenceStudy(requests=[("M_xz", "does_not_exist")])

    def test_point_form_without_point_name_raises(self):
        with pytest.raises(ValueError, match="needs a point"):
            MeshConvergenceStudy(requests=[("v_xz", "at_point")])

    def test_nonpoint_form_with_point_name_raises(self):
        with pytest.raises(ValueError, match="takes no point"):
            MeshConvergenceStudy(requests=[("M_xz", "max_minus_mean", "centroid")])

    def test_valid_requests_construct_cleanly(self):
        MeshConvergenceStudy(requests=[
            ("M_xz", "max_minus_mean"),
            ("v_xz", "at_point", "centroid"),
        ])


def _build_three_grades(target_values):
    """target_values: (f_coarse, f_medium, f_fine) para
    max_minus_mean(M_xz) -- devolve os 3 SubmodelResult sintéticos,
    com n_nodes a dobrar por nível (r=2 uniforme, mesmo caso de
    test_richardson_gci.py::test_second_order_convergence_exact_p)."""
    f_c, f_m, f_f = target_values
    x_c = list(range(3))  # n=2
    x_m = list(range(5))  # n=4
    x_f = list(range(9))  # n=8
    return [
        ("grade_0", make_result(x_c, M_xz=array_with_exact_max_minus_mean(3, f_c))),
        ("grade_1", make_result(x_m, M_xz=array_with_exact_max_minus_mean(5, f_m))),
        ("grade_2", make_result(x_f, M_xz=array_with_exact_max_minus_mean(9, f_f))),
    ]


class TestAddLevelSingleMetric:
    def test_first_two_levels_do_not_compute_gci_yet(self):
        study = MeshConvergenceStudy(requests=[("M_xz", "max_minus_mean")])
        rec = study.new_record("test-load", 0.0, 8.0)
        grades = _build_three_grades((110.0, 102.5, 100.625))

        converged = study.add_level(rec, grades[0][0], grades[0][1])
        assert converged is False
        assert len(rec.levels) == 1
        assert rec.gci_history == []

        converged = study.add_level(rec, grades[1][0], grades[1][1])
        assert converged is False
        assert len(rec.levels) == 2
        assert rec.gci_history == []

    def test_third_level_computes_gci(self):
        study = MeshConvergenceStudy(requests=[("M_xz", "max_minus_mean")], gci_threshold=1e9)
        rec = study.new_record("test-load", 0.0, 8.0)
        grades = _build_three_grades((110.0, 102.5, 100.625))

        for grade, result in grades[:2]:
            study.add_level(rec, grade, result)
        converged = study.add_level(rec, grades[2][0], grades[2][1])

        assert len(rec.gci_history) == 1
        assert "M_xz:max_minus_mean" in rec.gci_history[0]
        assert converged is True
        assert rec.converged is True
        assert rec.x_final == list(range(9))

    def test_point_metrics_history_keyed_by_request_label(self):
        study = MeshConvergenceStudy(requests=[("M_xz", "max_minus_mean")])
        rec = study.new_record("test-load", 0.0, 8.0)
        grades = _build_three_grades((110.0, 102.5, 100.625))
        study.add_level(rec, grades[0][0], grades[0][1])
        assert math.isclose(rec.point_metrics_history[0]["M_xz:max_minus_mean"], 110.0)

    def test_high_threshold_stays_unconverged(self):
        study = MeshConvergenceStudy(requests=[("M_xz", "max_minus_mean")], gci_threshold=1e-30)
        rec = study.new_record("test-load", 0.0, 8.0)
        grades = _build_three_grades((110.0, 102.5, 100.625))
        for grade, result in grades[:2]:
            study.add_level(rec, grade, result)
        converged = study.add_level(rec, grades[2][0], grades[2][1])
        assert converged is False
        assert rec.converged is False


class TestAddLevelMultipleMetrics:
    def test_all_requests_must_converge(self):
        """M_xz converge (sequência bem comportada); M_xy é
        não-monotónica (RichardsonGCI cai para _DummyGCI,
        converged=False) -- a transição toda tem de reportar
        não-convergido."""
        study = MeshConvergenceStudy(
            requests=[("M_xz", "max_minus_mean"), ("M_xy", "max_minus_mean")],
            gci_threshold=1e9,
        )
        rec = study.new_record("test-load", 0.0, 8.0)

        x_c, x_m, x_f = list(range(3)), list(range(5)), list(range(9))
        grades = [
            ("grade_0", make_result(
                x_c,
                M_xz=array_with_exact_max_minus_mean(3, 110.0),
                M_xy=array_with_exact_max_minus_mean(3, 100.0),
            )),
            ("grade_1", make_result(
                x_m,
                M_xz=array_with_exact_max_minus_mean(5, 102.5),
                M_xy=array_with_exact_max_minus_mean(5, 110.0),  # sobe -- não monotónico
            )),
            ("grade_2", make_result(
                x_f,
                M_xz=array_with_exact_max_minus_mean(9, 100.625),
                M_xy=array_with_exact_max_minus_mean(9, 105.0),
            )),
        ]

        for grade, result in grades[:2]:
            study.add_level(rec, grade, result)
        converged = study.add_level(rec, grades[2][0], grades[2][1])

        assert converged is False
        assert rec.converged is False
        gci_this = rec.gci_history[0]
        assert gci_this["M_xz:max_minus_mean"].converged is True
        assert gci_this["M_xy:max_minus_mean"].converged is False


class TestAddLevelWithPoint:
    def test_missing_points_raises(self):
        study = MeshConvergenceStudy(requests=[("v_xz", "at_point", "centroid")])
        rec = study.new_record("test-load", 0.0, 10.0)
        result = make_result([0.0, 10.0], v_xz=np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="needs point"):
            study.add_level(rec, "grade_0", result)

    def test_point_resolved_and_read_correctly(self):
        study = MeshConvergenceStudy(requests=[("v_xz", "at_point", "centroid")])
        rec = study.new_record("test-load", 0.0, 10.0)
        result = make_result([0.0, 5.0, 10.0], v_xz=np.array([1.0, 2.5, 4.0]))
        study.add_level(rec, "grade_0", result, points={"centroid": 5.0})
        assert math.isclose(rec.point_metrics_history[0]["v_xz:at_point:centroid"], 2.5)


class TestFinalizeUnconverged:
    def test_sets_x_final_without_marking_converged(self):
        study = MeshConvergenceStudy(requests=[("M_xz", "max_minus_mean")])
        rec = study.new_record("test-load", 0.0, 8.0)
        result = make_result([0.0, 4.0, 8.0], M_xz=array_with_exact_max_minus_mean(3, 50.0))
        study.finalize_unconverged(rec, result)
        assert rec.x_final == [0.0, 4.0, 8.0]
        assert rec.converged is False


class TestBuildMeshRefinementResult:
    def test_wraps_records_under_shaft_name(self):
        rec = ConvergenceRecord(label="load-1", x_lo=0.0, x_hi=10.0)
        result = build_mesh_refinement_result("shaft-A", {"load-1": rec})
        assert result.shaft_name == "shaft-A"
        assert result.per_load["load-1"] is rec
