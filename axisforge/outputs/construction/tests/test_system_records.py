"""``system_records`` of a resolved gear system (baseline case 7a): the mesh loads are described
only when the loaded copies returned by ``resolve`` are passed as ``shafts=``."""
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc
from axisforge.outputs.construction.system import system_records


def _describes(records, load_label):
    return any(load_label in r["parameter"] for r in records)


def test_mesh_loads_are_described_with_the_loaded_copies_only():
    case = bc.resolved_case7("pinion", 1)
    assert _describes(system_records(case.system, shafts=(case.sP, case.sW)), "g_wheel:Ft")
    assert not _describes(system_records(case.system), "g_wheel:Ft")
