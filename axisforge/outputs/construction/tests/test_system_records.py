"""``system_records`` and the texts of a ``Transmission`` (baseline case 7a): the mesh loads and
the resolved torques are described only for the ``ResolvedTransmission`` returned by ``resolve``."""
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc
from axisforge.outputs.construction.system import system_full_text, system_records, system_text


def _describes(records, load_label):
    return any(load_label in r["parameter"] for r in records)


def test_mesh_loads_are_described_with_the_loaded_copies_only():
    case = bc.resolved_case7("pinion", 1)
    assert _describes(system_records(case.resolved), "g_wheel:Ft")
    assert not _describes(system_records(case.system), "g_wheel:Ft")


def test_records_read_the_resolved_transmission():
    """Torque entering each shaft [N m] and its sense of rotation come from the ResolvedTransmission:
    60.75 N m on the pinion shaft, 91.125 N m on the wheel shaft (case 7a); the stage is described
    from the written path."""
    case = bc.resolved_case7("pinion", 1)
    rows = {(r["section"], r["parameter"]): r["value"] for r in system_records(case.resolved, case.P)}
    assert rows[("gear system", "source shaft")] == "shaft_pinion"
    assert (rows[("gear system", "number of stages")], rows[("gear system", "resolved")]) == (1, True)
    assert rows[("resolved shaft shaft_pinion", "torque entering the shaft")] == case.resolved.torque_Nm[0]
    assert rows[("resolved shaft shaft_wheel", "torque entering the shaft")] == case.resolved.torque_Nm[1]
    assert rows[("resolved shaft shaft_wheel", "rotation direction")] == -1
    mesh = "mesh stage (shaft_pinion -> shaft_wheel)"
    assert (rows[(mesh, "share of the torque")], rows[(mesh, "line of centres angle phi")]) == (1.0, 30.0)
    assert (rows[(mesh, "driver gear")], rows[(mesh, "driven gear")]) == ("g_pinion", "g_wheel")
    assert rows[(mesh, "working centre distance")] == case.link.pair.al
    unresolved = [r["section"] for r in system_records(case.system)]
    assert not any(sec.startswith("resolved shaft") for sec in unresolved)


def test_texts_of_a_resolved_transmission():
    case = bc.resolved_case7("pinion", 1)
    full = system_full_text(case.resolved, case.P)
    assert case.resolved.summary() in full and repr(case.link) in full
    assert "g_wheel:Ft" in system_text(case.resolved, case.P)
    assert "g_wheel:Ft" not in system_text(case.system, case.P)
