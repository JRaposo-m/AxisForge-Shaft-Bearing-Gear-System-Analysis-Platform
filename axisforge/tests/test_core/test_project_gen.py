"""
tests/test_core/test_project_gen.py
Unit and integration tests for core.project_gen.ProjectManager.

Coverage:
  - create_project: structure (axf + pipeline only, no shaft)
  - generate_shaft_file: creation, FileExistsError, name validation
  - generate_bearing_file: creation, FileExistsError
  - generate_gear_file: creation, FileExistsError
  - generate_gear_pair_file: creation, sequential, directly in stage dir
  - generate_stage_file: creation, FileExistsError
  - scan_project: shaft, stage, bearing, gear, gear_pair discovery
  - make_create_shaft: closure behaviour, error logging (no raise)
  - make_create_gear_pair: closure, auto-numbering, multi-pair same stage
  - project.axf: written on create, read by scan_project
  - Template correctness: no unreplaced $vars, required symbols present

No PySide6. No solver calls. All tests use tmp_path.
"""

import pytest
from pathlib import Path

from core.project_gen import ProjectManager, ProjectState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(tmp_path: Path, name: str = "test_proj") -> Path:
    return ProjectManager.create_project(name, tmp_path)


# ---------------------------------------------------------------------------
# create_project
# ---------------------------------------------------------------------------

class TestCreateProject:

    def test_returns_project_root(self, tmp_path):
        root = _make_project(tmp_path)
        assert root == tmp_path / "test_proj"
        assert root.is_dir()

    def test_creates_project_axf(self, tmp_path):
        root = _make_project(tmp_path)
        assert (root / "project.axf").exists()

    def test_axf_contains_project_name(self, tmp_path):
        root = ProjectManager.create_project("my_reducer", tmp_path)
        import json
        axf = json.loads((root / "project.axf").read_text(encoding="utf-8"))
        assert axf["name"] == "my_reducer"

    def test_axf_contains_version_and_created(self, tmp_path):
        root = _make_project(tmp_path)
        import json
        axf = json.loads((root / "project.axf").read_text(encoding="utf-8"))
        assert "version" in axf
        assert "created" in axf

    def test_creates_full_pipeline(self, tmp_path):
        root = _make_project(tmp_path)
        assert (root / "scripts" / "full_pipeline.py").exists()

    def test_pipeline_contains_project_name(self, tmp_path):
        root = ProjectManager.create_project("my_reducer", tmp_path)
        text = (root / "scripts" / "full_pipeline.py").read_text(encoding="utf-8")
        assert "my_reducer" in text

    def test_pipeline_contains_create_shaft(self, tmp_path):
        root = _make_project(tmp_path)
        text = (root / "scripts" / "full_pipeline.py").read_text(encoding="utf-8")
        assert "create_shaft" in text

    def test_pipeline_contains_run_function(self, tmp_path):
        root = _make_project(tmp_path)
        text = (root / "scripts" / "full_pipeline.py").read_text(encoding="utf-8")
        assert "_run(" in text

    def test_no_shaft_created_at_init(self, tmp_path):
        """New project must NOT create shaft_1 — that is done by create_shaft()."""
        root = _make_project(tmp_path)
        assert not (root / "shafts" / "shaft_1").exists()

    def test_creates_required_directories(self, tmp_path):
        root = _make_project(tmp_path)
        assert (root / "shafts").is_dir()
        assert (root / "stages").is_dir()
        assert (root / "scripts").is_dir()
        assert (root / "results").is_dir()

    def test_raises_if_directory_exists(self, tmp_path):
        _make_project(tmp_path)
        with pytest.raises(FileExistsError, match="already exists"):
            _make_project(tmp_path)

    def test_no_unreplaced_template_vars_in_pipeline(self, tmp_path):
        root = ProjectManager.create_project("demo", tmp_path)
        text = (root / "scripts" / "full_pipeline.py").read_text(encoding="utf-8")
        assert "$project_name" not in text


# ---------------------------------------------------------------------------
# generate_shaft_file
# ---------------------------------------------------------------------------

class TestGenerateShaftFile:

    def test_creates_shaft_directory(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        assert (root / "shafts" / "shaft_1").is_dir()

    def test_creates_shaft_py(self, tmp_path):
        root = _make_project(tmp_path)
        result = ProjectManager.generate_shaft_file("shaft_1", root, "proj")
        assert result.exists()
        assert result.name == "shaft_1.py"

    def test_creates_elements_dir(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        assert (root / "shafts" / "shaft_1" / "elements").is_dir()

    def test_creates_results_dir(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        assert (root / "shafts" / "shaft_1" / "results").is_dir()

    def test_raises_if_shaft_exists(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        with pytest.raises(FileExistsError, match="shaft_1"):
            ProjectManager.generate_shaft_file("shaft_1", root)

    def test_raises_on_invalid_name(self, tmp_path):
        root = _make_project(tmp_path)
        with pytest.raises(ValueError, match="shaft_<id>"):
            ProjectManager.generate_shaft_file("veio_1", root)

    def test_raises_on_name_without_prefix(self, tmp_path):
        root = _make_project(tmp_path)
        with pytest.raises(ValueError):
            ProjectManager.generate_shaft_file("1", root)

    def test_shaft_py_contains_shaft_name(self, tmp_path):
        root = _make_project(tmp_path)
        result = ProjectManager.generate_shaft_file("shaft_1", root, "proj")
        assert "shaft_1" in result.read_text(encoding="utf-8")

    def test_shaft_py_contains_mechanical_system(self, tmp_path):
        root = _make_project(tmp_path)
        result = ProjectManager.generate_shaft_file("shaft_1", root)
        assert "MechanicalSystem" in result.read_text(encoding="utf-8")

    def test_shaft_py_contains_statics_solver(self, tmp_path):
        root = _make_project(tmp_path)
        result = ProjectManager.generate_shaft_file("shaft_1", root)
        assert "StaticsSolver" in result.read_text(encoding="utf-8")

    def test_shaft_py_contains_shaft_position(self, tmp_path):
        root = _make_project(tmp_path)
        result = ProjectManager.generate_shaft_file("shaft_1", root)
        assert "shaft_position" in result.read_text(encoding="utf-8")

    def test_no_unreplaced_template_vars(self, tmp_path):
        root = _make_project(tmp_path)
        result = ProjectManager.generate_shaft_file("shaft_1", root, "proj")
        text = result.read_text(encoding="utf-8")
        assert "$shaft_name" not in text
        assert "$project_name" not in text

    def test_different_shaft_names_coexist(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        ProjectManager.generate_shaft_file("shaft_2", root)
        assert (root / "shafts" / "shaft_1").is_dir()
        assert (root / "shafts" / "shaft_2").is_dir()


# ---------------------------------------------------------------------------
# generate_bearing_file
# ---------------------------------------------------------------------------

class TestGenerateBearingFile:

    def test_creates_file(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        result = ProjectManager.generate_bearing_file("shaft_1", "A", root, "proj")
        assert result.exists()
        assert result.name == "bearing_A.py"

    def test_creates_results_dir(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        ProjectManager.generate_bearing_file("shaft_1", "A", root)
        assert (root / "shafts" / "shaft_1" / "elements" / "bearing_A" / "results").is_dir()

    def test_raises_on_collision(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        ProjectManager.generate_bearing_file("shaft_1", "A", root)
        with pytest.raises(FileExistsError, match="bearing_A"):
            ProjectManager.generate_bearing_file("shaft_1", "A", root)

    def test_same_label_different_shafts(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        ProjectManager.generate_shaft_file("shaft_2", root)
        r1 = ProjectManager.generate_bearing_file("shaft_1", "A", root)
        r2 = ProjectManager.generate_bearing_file("shaft_2", "A", root)
        assert r1.exists() and r2.exists()

    def test_file_contains_label(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        result = ProjectManager.generate_bearing_file("shaft_1", "A", root, "proj")
        assert "bearing_A" in result.read_text(encoding="utf-8")

    def test_no_unreplaced_vars(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        result = ProjectManager.generate_bearing_file("shaft_1", "A", root, "proj")
        text = result.read_text(encoding="utf-8")
        assert "$label" not in text
        assert "$shaft_name" not in text


# ---------------------------------------------------------------------------
# generate_gear_file
# ---------------------------------------------------------------------------

class TestGenerateGearFile:

    def test_creates_file(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        result = ProjectManager.generate_gear_file("shaft_1", "C14", root)
        assert result.exists()
        assert result.name == "gear_C14.py"

    def test_raises_on_collision(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        ProjectManager.generate_gear_file("shaft_1", "C14", root)
        with pytest.raises(FileExistsError, match="gear_C14"):
            ProjectManager.generate_gear_file("shaft_1", "C14", root)

    def test_no_unreplaced_vars(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        result = ProjectManager.generate_gear_file("shaft_1", "C14", root, "proj")
        text = result.read_text(encoding="utf-8")
        assert "$label" not in text
        assert "$shaft_name" not in text


# ---------------------------------------------------------------------------
# generate_gear_pair_file
# ---------------------------------------------------------------------------

class TestGenerateGearPairFile:

    def test_creates_file_in_stage_dir(self, tmp_path):
        root = _make_project(tmp_path)
        result = ProjectManager.generate_gear_pair_file("stage_1", 1, root)
        assert result.exists()
        assert result.parent == root / "stages" / "stage_1"
        assert result.name == "gear_pair_1.py"

    def test_not_in_elements_subdir(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_gear_pair_file("stage_1", 1, root)
        assert not (root / "stages" / "stage_1" / "elements").exists()

    def test_sequential_indexing(self, tmp_path):
        root = _make_project(tmp_path)
        r1 = ProjectManager.generate_gear_pair_file("stage_1", 1, root)
        r2 = ProjectManager.generate_gear_pair_file("stage_1", 2, root)
        assert r1.name == "gear_pair_1.py"
        assert r2.name == "gear_pair_2.py"

    def test_raises_on_collision(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_gear_pair_file("stage_1", 1, root)
        with pytest.raises(FileExistsError):
            ProjectManager.generate_gear_pair_file("stage_1", 1, root)

    def test_file_contains_pair_index(self, tmp_path):
        root = _make_project(tmp_path)
        result = ProjectManager.generate_gear_pair_file("stage_1", 1, root, "proj")
        assert "gear_pair_1" in result.read_text(encoding="utf-8")

    def test_no_unreplaced_vars(self, tmp_path):
        root = _make_project(tmp_path)
        result = ProjectManager.generate_gear_pair_file("stage_1", 1, root, "proj")
        text = result.read_text(encoding="utf-8")
        assert "$stage_name" not in text
        assert "$pair_index" not in text

    def test_same_gear_label_multiple_pairs_allowed(self, tmp_path):
        root = _make_project(tmp_path)
        r1 = ProjectManager.generate_gear_pair_file("stage_1", 1, root)
        r2 = ProjectManager.generate_gear_pair_file("stage_1", 2, root)
        assert r1.exists() and r2.exists()

    def test_creates_results_dir(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_gear_pair_file("stage_1", 1, root)
        assert (root / "stages" / "stage_1" / "results").is_dir()


# ---------------------------------------------------------------------------
# generate_stage_file
# ---------------------------------------------------------------------------

class TestGenerateStageFile:

    def test_creates_file(self, tmp_path):
        root = _make_project(tmp_path)
        result = ProjectManager.generate_stage_file("stage_1", 1, root, "proj")
        assert result.exists()
        assert result.name == "stage_1.py"

    def test_raises_on_collision(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_stage_file("stage_1", 1, root)
        with pytest.raises(FileExistsError):
            ProjectManager.generate_stage_file("stage_1", 1, root)

    def test_no_unreplaced_vars(self, tmp_path):
        root = _make_project(tmp_path)
        result = ProjectManager.generate_stage_file("stage_1", 1, root, "proj")
        text = result.read_text(encoding="utf-8")
        assert "$stage_name" not in text
        assert "$project_name" not in text
        assert "$stage_index" not in text


# ---------------------------------------------------------------------------
# scan_project
# ---------------------------------------------------------------------------

class TestScanProject:

    def test_reads_project_name_from_axf(self, tmp_path):
        root = ProjectManager.create_project("my_proj", tmp_path)
        state = ProjectManager.scan_project(root)
        assert state.project_name == "my_proj"

    def test_fallback_project_name(self, tmp_path):
        state = ProjectManager.scan_project(tmp_path)
        assert state.project_name == tmp_path.name

    def test_detects_shaft_dirs(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        ProjectManager.generate_shaft_file("shaft_2", root)
        state = ProjectManager.scan_project(root)
        assert "shaft_1" in state.shaft_dirs
        assert "shaft_2" in state.shaft_dirs

    def test_detects_bearing_labels(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        ProjectManager.generate_bearing_file("shaft_1", "A", root)
        ProjectManager.generate_bearing_file("shaft_1", "B", root)
        state = ProjectManager.scan_project(root)
        assert state.bearing_labels["shaft_1"] == {"A", "B"}

    def test_detects_gear_labels(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        ProjectManager.generate_gear_file("shaft_1", "C14", root)
        state = ProjectManager.scan_project(root)
        assert "C14" in state.gear_labels["shaft_1"]

    def test_counts_gear_pairs_directly_in_stage_dir(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_gear_pair_file("stage_1", 1, root)
        ProjectManager.generate_gear_pair_file("stage_1", 2, root)
        state = ProjectManager.scan_project(root)
        assert state.gear_pair_counts["stage_1"] == 2

    def test_empty_shaft_has_empty_sets(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        state = ProjectManager.scan_project(root)
        assert state.bearing_labels["shaft_1"] == set()
        assert state.gear_labels["shaft_1"] == set()

    def test_ignores_files_in_shafts_dir(self, tmp_path):
        root = _make_project(tmp_path)
        (root / "shafts" / "README.txt").write_text("ignored", encoding="utf-8")
        state = ProjectManager.scan_project(root)
        assert "README.txt" not in state.shaft_dirs

    def test_no_shafts_dir(self, tmp_path):
        state = ProjectManager.scan_project(tmp_path)
        assert state.shaft_dirs == {}


# ---------------------------------------------------------------------------
# make_create_shaft closure
# ---------------------------------------------------------------------------

class TestMakeCreateShaft:

    def test_creates_shaft(self, tmp_path):
        root = _make_project(tmp_path)
        cs = ProjectManager.make_create_shaft(root, "proj")
        cs("shaft_1")
        assert (root / "shafts" / "shaft_1" / "shaft_1.py").exists()

    def test_duplicate_logs_error_not_raises(self, tmp_path):
        root = _make_project(tmp_path)
        logs = []
        cs = ProjectManager.make_create_shaft(root, "proj", logs.append)
        cs("shaft_1")
        cs("shaft_1")
        assert any("ERRO" in l for l in logs)

    def test_invalid_name_logs_error(self, tmp_path):
        root = _make_project(tmp_path)
        logs = []
        cs = ProjectManager.make_create_shaft(root, "proj", logs.append)
        cs("veio_1")
        assert any("ERRO" in l for l in logs)

    def test_success_logs_created(self, tmp_path):
        root = _make_project(tmp_path)
        logs = []
        cs = ProjectManager.make_create_shaft(root, "proj", logs.append)
        cs("shaft_1")
        assert any("Created" in l for l in logs)

    def test_multiple_shafts(self, tmp_path):
        root = _make_project(tmp_path)
        cs = ProjectManager.make_create_shaft(root, "proj")
        cs("shaft_1")
        cs("shaft_2")
        assert (root / "shafts" / "shaft_2" / "shaft_2.py").exists()


# ---------------------------------------------------------------------------
# make_create_gear_pair closure
# ---------------------------------------------------------------------------

class TestMakeCreateGearPair:

    def test_creates_gear_pair_file(self, tmp_path):
        root = _make_project(tmp_path)
        cgp = ProjectManager.make_create_gear_pair(root, "proj")
        cgp(None, "G1", None, "G2", stage_name="stage_1")
        assert (root / "stages" / "stage_1" / "gear_pair_1.py").exists()

    def test_creates_stage_file(self, tmp_path):
        root = _make_project(tmp_path)
        cgp = ProjectManager.make_create_gear_pair(root, "proj")
        cgp(None, "G1", None, "G2", stage_name="stage_1")
        assert (root / "stages" / "stage_1" / "stage_1.py").exists()

    def test_sequential_pair_index(self, tmp_path):
        root = _make_project(tmp_path)
        cgp = ProjectManager.make_create_gear_pair(root, "proj")
        cgp(None, "G1", None, "G2", stage_name="stage_1")
        cgp(None, "G1", None, "G3", stage_name="stage_1")
        assert (root / "stages" / "stage_1" / "gear_pair_1.py").exists()
        assert (root / "stages" / "stage_1" / "gear_pair_2.py").exists()

    def test_same_gear_label_multiple_pairs_allowed(self, tmp_path):
        root = _make_project(tmp_path)
        logs = []
        cgp = ProjectManager.make_create_gear_pair(root, "proj", logs.append)
        cgp(None, "G1", None, "G1", stage_name="stage_1")
        cgp(None, "G1", None, "G1", stage_name="stage_1")
        assert (root / "stages" / "stage_1" / "gear_pair_2.py").exists()
        assert not any("ERRO" in l for l in logs)

    def test_auto_stage_numbering(self, tmp_path):
        root = _make_project(tmp_path)
        cgp = ProjectManager.make_create_gear_pair(root, "proj")
        cgp(None, "G1", None, "G2")
        state = ProjectManager.scan_project(root)
        assert "stage_1" in state.stage_dirs

    def test_gear_pair_not_in_elements_subdir(self, tmp_path):
        root = _make_project(tmp_path)
        cgp = ProjectManager.make_create_gear_pair(root, "proj")
        cgp(None, "G1", None, "G2", stage_name="stage_1")
        assert not (root / "stages" / "stage_1" / "elements").exists()
        assert (root / "stages" / "stage_1" / "gear_pair_1.py").exists()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_project_cycle(self, tmp_path):
        root = ProjectManager.create_project("full_test", tmp_path)
        logs = []
        cs  = ProjectManager.make_create_shaft(root, "full_test", logs.append)
        cgp = ProjectManager.make_create_gear_pair(root, "full_test", logs.append)

        cs("shaft_1")
        cs("shaft_2")
        cgp(None, "G1", None, "G1", stage_name="stage_1")

        ProjectManager.generate_bearing_file("shaft_1", "A", root, "full_test")
        ProjectManager.generate_bearing_file("shaft_1", "B", root, "full_test")
        ProjectManager.generate_gear_file("shaft_1", "Z1", root, "full_test")

        state = ProjectManager.scan_project(root)
        assert state.project_name == "full_test"
        assert "shaft_1" in state.shaft_dirs
        assert "shaft_2" in state.shaft_dirs
        assert state.bearing_labels["shaft_1"] == {"A", "B"}
        assert "Z1" in state.gear_labels["shaft_1"]
        assert state.gear_pair_counts["stage_1"] == 1

    def test_collision_does_not_overwrite(self, tmp_path):
        root = _make_project(tmp_path)
        ProjectManager.generate_shaft_file("shaft_1", root)
        target = root / "shafts" / "shaft_1" / "shaft_1.py"
        target.write_text(target.read_text(encoding="utf-8") + "\n# USER EDIT", encoding="utf-8")
        with pytest.raises(FileExistsError):
            ProjectManager.generate_shaft_file("shaft_1", root)
        assert "# USER EDIT" in target.read_text(encoding="utf-8")

    def test_two_stage_project(self, tmp_path):
        root = ProjectManager.create_project("two_stage", tmp_path)
        cgp = ProjectManager.make_create_gear_pair(root, "two_stage")
        cgp(None, "G1", None, "G2", stage_name="stage_1")
        cgp(None, "G2", None, "G3", stage_name="stage_2")
        state = ProjectManager.scan_project(root)
        assert "stage_1" in state.stage_dirs
        assert "stage_2" in state.stage_dirs