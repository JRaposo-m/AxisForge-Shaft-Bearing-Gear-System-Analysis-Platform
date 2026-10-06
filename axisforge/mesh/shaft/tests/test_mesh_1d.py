# test_mesh_1d.py
"""Testes de mesh_generation/mesh_1D.py -- Mesh1D constrói o grid de
nós FEM a partir de posições MANDATÓRIAS (fronteiras de secção,
bearings, gears, loads). Usa FakeShaftSystem/FakeShaft/FakeBearing/
FakeGear/FakeDistributedLoad (ver conftest.py) para exercitar
_mandatory_positions()/_create_mesh() sem montar a hierarquia real de
ShaftSystem/GearElement/materiais -- Mesh1D só lê estes objetos por
duck typing, nunca por isinstance."""
import pytest

from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D, MergeEvent
from axisforge.config import MESH_MIN_NODE_DIST_MM, NODE_LOOKUP_TOL_MM
from axisforge.mesh.shaft.tests.conftest import (
    FakeShaft, FakeShaftSystem, FakeBearing, FakeGear, FakeDistributedLoad, _Positioned,
)


class TestMandatoryPositionsBasic:
    def test_section_boundaries_become_nodes(self):
        shaft_system = FakeShaftSystem(shaft=FakeShaft([0.0, 40.0, 100.0]))
        mesh = Mesh1D(shaft_system)
        assert mesh.x_nodes == [0.0, 40.0, 100.0]

    def test_single_section_gives_two_nodes(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system)
        assert mesh.x_nodes == [0.0, 100.0]
        assert mesh.n_nodes == 2


class TestMandatoryPositionsBearingsAndGears:
    def test_bearing_position_and_extent_become_nodes(self):
        shaft = FakeShaft([0.0, 100.0])
        bearing = FakeBearing(position=20.0, extent=(15.0, 25.0))
        shaft_system = FakeShaftSystem(shaft=shaft, bearings=[bearing])
        mesh = Mesh1D(shaft_system)
        assert {0.0, 15.0, 20.0, 25.0, 100.0} <= set(mesh.x_nodes)

    def test_gear_position_and_extent_become_nodes(self):
        shaft = FakeShaft([0.0, 100.0])
        gear = FakeGear(position=60.0, extent=(55.0, 65.0))
        shaft_system = FakeShaftSystem(shaft=shaft, gears=[gear])
        mesh = Mesh1D(shaft_system)
        assert {0.0, 55.0, 60.0, 65.0, 100.0} <= set(mesh.x_nodes)

    def test_multiple_bearings_all_contribute_nodes(self):
        shaft = FakeShaft([0.0, 100.0])
        bearings = [FakeBearing(10.0, extent=(8.0, 12.0)), FakeBearing(90.0, extent=(88.0, 92.0))]
        shaft_system = FakeShaftSystem(shaft=shaft, bearings=bearings)
        mesh = Mesh1D(shaft_system)
        assert {8.0, 10.0, 12.0, 88.0, 90.0, 92.0} <= set(mesh.x_nodes)


class TestMandatoryPositionsLoads:
    @pytest.mark.parametrize("load_attr", ["radial_loads", "axial_loads", "torque_loads", "external_moments"])
    def test_load_position_becomes_node(self, load_attr):
        shaft = FakeShaft([0.0, 100.0])
        load = _Positioned(position=33.0)
        shaft_system = FakeShaftSystem(shaft=shaft, **{load_attr: [load]})
        mesh = Mesh1D(shaft_system)
        assert 33.0 in mesh.x_nodes

    def test_distributed_load_bounds_and_centroid_become_nodes(self):
        shaft = FakeShaft([0.0, 100.0])
        dload = FakeDistributedLoad(x_lo=10.0, x_hi=30.0, centroid=20.0)
        shaft_system = FakeShaftSystem(shaft=shaft, distributed_radial_loads=[dload])
        mesh = Mesh1D(shaft_system)
        assert {10.0, 20.0, 30.0} <= set(mesh.x_nodes)

    def test_distributed_load_centroid_appended_twice_is_harmless(self):
        """Mesh1D._mandatory_positions() chama ld.centroid(LoadPlane.XY)
        duas vezes seguidas para o mesmo load (parece copy-paste de um
        centroid XZ que nunca foi adicionado -- ver código-fonte). Como
        o resultado final passa por set(), duplicar o mesmo valor não
        muda o mesh; só regista aqui o comportamento atual, não que
        esteja correto."""
        shaft = FakeShaft([0.0, 100.0])
        dload = FakeDistributedLoad(x_lo=10.0, x_hi=30.0, centroid=20.0)
        shaft_system = FakeShaftSystem(shaft=shaft, distributed_radial_loads=[dload])
        mesh = Mesh1D(shaft_system)
        assert mesh.x_nodes.count(20.0) == 1


class TestExtraMandatory:
    def test_extra_mandatory_positions_become_nodes(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system, extra_mandatory=[42.0])
        assert 42.0 in mesh.x_nodes

    def test_no_extra_mandatory_defaults_to_empty(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system)
        assert mesh.x_nodes == [0.0, 100.0]


class TestDedupAndMerge:
    def test_duplicate_positions_collapse_to_one_node(self):
        shaft = FakeShaft([0.0, 100.0])
        bearing = FakeBearing(position=0.0, extent=(0.0, 0.0))
        shaft_system = FakeShaftSystem(shaft=shaft, bearings=[bearing])
        mesh = Mesh1D(shaft_system)
        assert mesh.x_nodes == [0.0, 100.0]

    def test_mandatory_points_closer_than_min_dist_raise(self):
        """Two physical features closer than the minimum element length
        cannot be merged without moving one of them -> error, not a
        silent merge."""
        shaft = FakeShaft([0.0, 100.0])
        too_close = MESH_MIN_NODE_DIST_MM / 2.0
        bearing = FakeBearing(position=too_close)
        shaft_system = FakeShaftSystem(shaft=shaft, bearings=[bearing])
        mesh = Mesh1D(shaft_system)
        with pytest.raises(ValueError, match="mandatory-node conflicts"):
            mesh.build()

    def test_validate_lists_mandatory_conflict_without_raising(self):
        shaft = FakeShaft([0.0, 100.0])
        bearing = FakeBearing(position=MESH_MIN_NODE_DIST_MM / 2.0)
        mesh = Mesh1D(FakeShaftSystem(shaft=shaft, bearings=[bearing]))
        errors = mesh.validate()
        assert len(errors) == 1
        assert "bearing" in errors[0] and "section 0 start" in errors[0]

    def test_coincident_mandatory_points_within_lookup_tol_are_not_conflicts(self):
        shaft = FakeShaft([0.0, 100.0])
        bearing = FakeBearing(position=NODE_LOOKUP_TOL_MM / 2.0, extent=(0.0, 0.0))
        mesh = Mesh1D(FakeShaftSystem(shaft=shaft, bearings=[bearing]))
        assert mesh.validate() == []
        assert mesh.x_nodes == [0.0, 100.0]
        assert mesh.merge_report() == []

    def test_points_farther_than_min_dist_stay_separate(self):
        shaft = FakeShaft([0.0, 100.0])
        far_enough = MESH_MIN_NODE_DIST_MM * 2.0
        bearing = FakeBearing(position=far_enough)
        shaft_system = FakeShaftSystem(shaft=shaft, bearings=[bearing])
        mesh = Mesh1D(shaft_system)
        assert far_enough in mesh.x_nodes


class TestCachingAndPublicApi:
    def test_build_is_cached(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system)
        first = mesh.build()
        assert mesh.build() is first  # mesma lista, não recalculada

    def test_x_nodes_property_matches_build(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system)
        assert mesh.x_nodes == mesh.build()

    def test_n_nodes_matches_x_nodes_length(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system)
        assert mesh.n_nodes == len(mesh.x_nodes)

    def test_show_nodes_returns_indexed_positions(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system)
        result = mesh.show_nodes(print_output=False)
        assert result == [(0, 0.0), (1, 100.0)]

    def test_show_nodes_does_not_print_when_disabled(self, fake_shaft_system, capsys):
        mesh = Mesh1D(fake_shaft_system)
        mesh.show_nodes(print_output=False)
        assert capsys.readouterr().out == ""

    def test_show_nodes_prints_when_enabled(self, fake_shaft_system, capsys):
        mesh = Mesh1D(fake_shaft_system)
        mesh.show_nodes(print_output=True)
        assert capsys.readouterr().out != ""


class TestGraders:
    def test_add_grader_invalidates_cache_and_injects_nodes(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system)
        mesh.build()  # popula a cache com [0.0, 100.0]

        class _StubGrader:
            def get_grade(self, grade):
                return [50.0]

        mesh.add_grader(_StubGrader(), "grade_1")
        assert 50.0 in mesh.x_nodes

    def test_clear_graders_removes_injected_nodes(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system)

        class _StubGrader:
            def get_grade(self, grade):
                return [50.0]

        mesh.add_grader(_StubGrader(), "grade_1")
        assert 50.0 in mesh.x_nodes

        mesh.clear_graders()
        assert mesh.x_nodes == [0.0, 100.0]

    def test_grader_receives_the_grade_string_it_was_added_with(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system)
        seen = []

        class _StubGrader:
            def get_grade(self, grade):
                seen.append(grade)
                return []

        mesh.add_grader(_StubGrader(), "grade_3")
        mesh.x_nodes
        assert seen == ["grade_3"]


class TestMergePriorityAndReport:
    def test_extra_close_to_mandatory_is_absorbed_and_reported(self, fake_shaft_system):
        x_extra = 100.0 - MESH_MIN_NODE_DIST_MM / 2.0
        mesh = Mesh1D(fake_shaft_system, extra_mandatory=[x_extra])
        assert mesh.x_nodes == [0.0, 100.0]          # mandatory survives, not the extra
        report = mesh.merge_report()
        assert report == [MergeEvent(100.0, "shaft end", x_extra, "extra_mandatory",
                                     pytest.approx(MESH_MIN_NODE_DIST_MM / 2.0))]

    def test_extra_close_to_extra_keeps_the_first(self, fake_shaft_system):
        a, b = 50.0, 50.0 + MESH_MIN_NODE_DIST_MM / 2.0
        mesh = Mesh1D(fake_shaft_system, extra_mandatory=[b, a])   # order irrelevant
        assert mesh.x_nodes == [0.0, a, 100.0]
        (event,) = mesh.merge_report()
        assert (event.kept_x, event.dropped_x) == (a, b)
        assert event.kept_origin == event.dropped_origin == "extra_mandatory"

    def test_extra_coincident_with_mandatory_is_not_reported(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system, extra_mandatory=[100.0])
        assert mesh.x_nodes == [0.0, 100.0]
        assert mesh.merge_report() == []

    def test_extra_never_triggers_a_conflict(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system, extra_mandatory=[MESH_MIN_NODE_DIST_MM / 2.0])
        assert mesh.validate() == []
        assert mesh.x_nodes == [0.0, 100.0]

    def test_grader_nodes_are_low_priority_and_labelled(self, fake_shaft_system):
        x_g = 100.0 - MESH_MIN_NODE_DIST_MM / 2.0

        class _StubGrader:
            def get_grade(self, grade):
                return [x_g]

        mesh = Mesh1D(fake_shaft_system)
        mesh.add_grader(_StubGrader(), "grade_4")
        assert mesh.x_nodes == [0.0, 100.0]
        (event,) = mesh.merge_report()
        assert event.dropped_origin == "grader grade_4"


class TestProtectedNodes:
    def test_protected_position_becomes_node(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system, protected=[37.5])
        assert 37.5 in mesh.x_nodes

    def test_protected_survives_nearby_extra(self, fake_shaft_system):
        """Regression: a submodel cut node at 107.5 mm was deleted by a
        grader midpoint within MESH_MIN_NODE_DIST_MM, so the prescribed BC
        had no node to act on. A protected node now wins over any extra."""
        x_cut = 37.5
        x_mid = x_cut - MESH_MIN_NODE_DIST_MM / 2.0          # sorts BEFORE the cut node
        mesh = Mesh1D(fake_shaft_system, extra_mandatory=[x_mid], protected=[x_cut])
        assert x_cut in mesh.x_nodes and x_mid not in mesh.x_nodes
        (event,) = mesh.merge_report()
        assert (event.kept_x, event.kept_origin) == (x_cut, "protected")

    def test_protected_close_to_mandatory_is_a_conflict(self, fake_shaft_system):
        mesh = Mesh1D(fake_shaft_system, protected=[MESH_MIN_NODE_DIST_MM / 2.0])
        with pytest.raises(ValueError, match="protected"):
            mesh.build()