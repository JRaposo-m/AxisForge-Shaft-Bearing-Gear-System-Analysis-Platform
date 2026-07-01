"""
AxisForge — Mesh Generator Tests

Tests six shaft configurations that cover the full range of mesh-quality
challenges:

    1. Simple solid shaft        — uniform diameter, 3 equal sections.
    2. Stepped solid shaft       — multiple diameter steps (transmission shaft).
    3. Mixed solid/hollow        — hollow central section.
    4. Extreme aspect-ratio      — very long thin journal pins beside a wide
                                   gear seat.  Stresses d/L-aware lc formula.
    5. Multi-step hollow shaft   — alternating hollow/solid sections with
                                   varying diameters.
    6. Second-order elements     — stepped shaft re-meshed as mesh_order=2
                                   (27-node hexahedra H27) to validate
                                   quadratic element promotion.

Each test prints a compact summary table and flags any element below a
quality threshold.  A non-zero exit code is returned if any test fails the
quality gate.

Element type
------------
All meshes use pure hexahedral elements via gmsh SubdivisionAlgorithm=2:
  mesh_order=1  →  Hexahedron 8  (H8,  linear,    8 nodes)
  mesh_order=2  →  Hexahedron 27 (H27, quadratic, 27 nodes)

Quality metric
--------------
minSICN — minimum signed scaled Jacobian per element (range −1..1).
This is the only SICN-family metric accepted by gmsh ≥ 4.11's Python API.
Gate values are unchanged from the tet-mesh baseline:
  min  ≥ 0.1  (flag badly distorted elements)
  mean ≥ 0.5  (overall mesh health)

Location
--------
axisforge/tests/test_mesh/test_mesh_shaft/global_mesh/test_mesh_generator.py

Run from the project root (axisforge/):
    python -m tests.test_mesh.test_mesh_shaft.global_mesh.test_mesh_generator

Output .msh files are written to the output/ subdirectory next to this file.
Inspect them with:
    gmsh <path>.msh
"""

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — add axisforge/ to sys.path so imports resolve correctly.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[4]   # axisforge/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from core.shaft import Shaft, ShaftSection  # noqa: E402
from mesh.mesh_shaft.global_mesh.mesh_generator import (  # noqa: E402
    MeshResult,
    QUALITY_NAME,
    generate_mesh,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Quality gate — minSICN in [−1, 1]; values below 0.1 indicate badly skewed
# hex elements (large Jacobian variation within the element).
QUALITY_GATE_MIN  : float = 0.1
QUALITY_GATE_MEAN : float = 0.5

SEPARATOR = "─" * 70


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_result(result: MeshResult, elapsed: float) -> None:
    """Print a structured summary of a mesh result."""
    min_ok  = result.quality_min  >= QUALITY_GATE_MIN
    mean_ok = result.quality_mean >= QUALITY_GATE_MEAN

    print(f"\n{SEPARATOR}")
    print(f"  File         : {result.output_path.name}")
    print(f"  Sections     : {result.n_sections}")
    print(f"  Total length : {result.total_length_mm:.1f} mm")
    print(f"  Wall time    : {elapsed:.1f} s")
    print()
    print(f"  Quality ({QUALITY_NAME})")
    _flag = lambda ok: "✓" if ok else "✗ BELOW GATE"
    print(f"    min  = {result.quality_min :.4f}   {_flag(min_ok )}")
    print(f"    mean = {result.quality_mean:.4f}   {_flag(mean_ok)}")
    print()
    print(f"  {'Idx':<4} {'Label':<22} {'d [mm]':>8} {'L [mm]':>9} {'d/L':>6} {'lc [mm]':>8}")
    for geo in result.section_geometries:
        hollow_mark = " (hollow)" if geo.is_hollow else ""
        print(
            f"  [{geo.index}]  {geo.label + hollow_mark:<22}  "
            f"{geo.diameter:7.1f}  {geo.length:8.1f}  "
            f"{geo.d_over_L:5.3f}  {geo.lc:7.2f}"
        )
    print()
    print(f"  Physical groups ({len(result.physical_group_tags)} total):")
    for name, tag in result.physical_group_tags.items():
        print(f"    tag {tag:5d}  →  {name}")
    print(SEPARATOR)


def check_quality(result: MeshResult) -> bool:
    """Return True if the mesh passes both quality gates."""
    return (
        result.quality_min  >= QUALITY_GATE_MIN and
        result.quality_mean >= QUALITY_GATE_MEAN
    )


def run_test(name: str, shaft: Shaft, filename: str, mesh_order: int = 1) -> MeshResult:
    """Run a single mesh test with timing, printing, and quality check."""
    print(f"\n{'═' * 70}")
    print(f"  TEST: {name}")
    print(f"{'═' * 70}")
    t0 = time.perf_counter()
    result = generate_mesh(
        shaft,
        OUTPUT_DIR / filename,
        mesh_order = mesh_order,
        verbose    = False,
    )
    elapsed = time.perf_counter() - t0
    print_result(result, elapsed)
    if not check_quality(result):
        print(f"  ⚠  Quality gate FAILED for {name}")
    else:
        print(f"  ✓  Quality gate PASSED for {name}")
    return result


# ---------------------------------------------------------------------------
# Test 1 — Simple uniform solid shaft
# ---------------------------------------------------------------------------

def test_simple_solid() -> MeshResult:
    """
    Three equal sections at uniform diameter.
    Baseline test — should always produce near-cubic hex elements.
    """
    shaft = Shaft(label="shaft_simple")
    shaft.add_section(ShaftSection(length=50.0,  diameter=40.0, label="left_journal"))
    shaft.add_section(ShaftSection(length=120.0, diameter=40.0, label="body"))
    shaft.add_section(ShaftSection(length=50.0,  diameter=40.0, label="right_journal"))
    return run_test("Simple uniform solid shaft", shaft, "shaft_simple.msh")


# ---------------------------------------------------------------------------
# Test 2 — Stepped solid shaft (typical transmission shaft)
# ---------------------------------------------------------------------------

def test_stepped_solid() -> MeshResult:
    """
    Five sections with three distinct diameters.
    Tests transition refinement and conforming fragment across diameter steps.
    """
    shaft = Shaft(label="shaft_stepped")
    shaft.add_section(ShaftSection(length=40.0,  diameter=30.0, label="journal_left"))
    shaft.add_section(ShaftSection(length=20.0,  diameter=35.0, label="step_left"))
    shaft.add_section(ShaftSection(length=100.0, diameter=50.0, label="gear_seat"))
    shaft.add_section(ShaftSection(length=20.0,  diameter=35.0, label="step_right"))
    shaft.add_section(ShaftSection(length=40.0,  diameter=30.0, label="journal_right"))
    return run_test("Stepped solid shaft", shaft, "shaft_stepped.msh")


# ---------------------------------------------------------------------------
# Test 3 — Mixed solid/hollow shaft
# ---------------------------------------------------------------------------

def test_mixed_hollow() -> MeshResult:
    """
    Solid end caps with a hollow central section.
    Tests boolean cut, inner surface Physical Groups, and element quality
    in thin-walled annular hex regions.
    """
    shaft = Shaft(label="shaft_mixed")
    shaft.add_section(ShaftSection(
        length=50.0, diameter=60.0, inner_diameter=0.0,  label="solid_left"))
    shaft.add_section(ShaftSection(
        length=150.0, diameter=60.0, inner_diameter=30.0, label="hollow_centre"))
    shaft.add_section(ShaftSection(
        length=50.0, diameter=60.0, inner_diameter=0.0,  label="solid_right"))
    return run_test("Mixed solid/hollow shaft", shaft, "shaft_mixed.msh")


# ---------------------------------------------------------------------------
# Test 4 — Extreme aspect-ratio shaft
# ---------------------------------------------------------------------------

def test_extreme_aspect_ratio() -> MeshResult:
    """
    Very long thin journal pins beside a wide, short gear seat.

    d/L ratios:
      journal pins:  30/200 = 0.15  (very elongated section)
      gear seat:    100/30  = 3.33  (very flat/wide section)

    Tests that the lc formula min(d, L) * k prevents both
    under-refinement of thin long sections and over-refinement of
    flat short sections.
    """
    shaft = Shaft(label="shaft_extreme")
    shaft.add_section(ShaftSection(length=200.0, diameter=30.0,  label="pin_left"))
    shaft.add_section(ShaftSection(length=30.0,  diameter=100.0, label="gear_seat"))
    shaft.add_section(ShaftSection(length=200.0, diameter=30.0,  label="pin_right"))
    return run_test("Extreme aspect-ratio shaft", shaft, "shaft_extreme.msh")


# ---------------------------------------------------------------------------
# Test 5 — Multi-step hollow shaft
# ---------------------------------------------------------------------------

def test_multi_step_hollow() -> MeshResult:
    """
    Seven sections alternating solid/hollow with varying outer diameters.
    Combines all mesh-quality challenges simultaneously:
      - multiple diameter transitions
      - hollow sections with thin walls
      - conforming fragment across all interfaces
    """
    shaft = Shaft(label="shaft_complex")
    shaft.add_section(ShaftSection(
        length=30.0,  diameter=40.0, inner_diameter=0.0,  label="end_left"))
    shaft.add_section(ShaftSection(
        length=10.0,  diameter=50.0, inner_diameter=0.0,  label="fillet_left"))
    shaft.add_section(ShaftSection(
        length=80.0,  diameter=50.0, inner_diameter=25.0, label="hollow_left"))
    shaft.add_section(ShaftSection(
        length=20.0,  diameter=60.0, inner_diameter=0.0,  label="hub"))
    shaft.add_section(ShaftSection(
        length=80.0,  diameter=50.0, inner_diameter=25.0, label="hollow_right"))
    shaft.add_section(ShaftSection(
        length=10.0,  diameter=50.0, inner_diameter=0.0,  label="fillet_right"))
    shaft.add_section(ShaftSection(
        length=30.0,  diameter=40.0, inner_diameter=0.0,  label="end_right"))
    return run_test("Multi-step hollow shaft", shaft, "shaft_complex.msh")


# ---------------------------------------------------------------------------
# Test 6 — Quadratic (second-order) hex elements
# ---------------------------------------------------------------------------

def test_quadratic_elements() -> MeshResult:
    """
    Re-meshes the stepped solid shaft with mesh_order=2.
    Produces 27-node hexahedra (Hexahedron 27 / H27).
    Validates quadratic element promotion via gmsh.model.mesh.setOrder(2).

    Note: the HighOrder optimiser is intentionally skipped for H27 elements
    (see mesh_generator.py for rationale).
    """
    shaft = Shaft(label="shaft_stepped_p2")
    shaft.add_section(ShaftSection(length=40.0,  diameter=30.0, label="journal_left"))
    shaft.add_section(ShaftSection(length=20.0,  diameter=35.0, label="step_left"))
    shaft.add_section(ShaftSection(length=100.0, diameter=50.0, label="gear_seat"))
    shaft.add_section(ShaftSection(length=20.0,  diameter=35.0, label="step_right"))
    shaft.add_section(ShaftSection(length=40.0,  diameter=30.0, label="journal_right"))
    return run_test(
        "Quadratic elements (H27) — stepped shaft",
        shaft,
        "shaft_stepped_p2.msh",
        mesh_order = 2,
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[tuple[str, MeshResult]]) -> None:
    """Print a compact pass/fail table for all tests."""
    print(f"\n{'═' * 70}")
    print("  SUMMARY")
    print(f"{'═' * 70}")
    print(f"  {'Test':<42} {'Q min':>7} {'Q mean':>7} {'Gate':>6}")
    print(f"  {'─'*42} {'─'*7} {'─'*7} {'─'*6}")
    all_passed = True
    for name, r in results:
        ok = check_quality(r)
        all_passed = all_passed and ok
        mark = "PASS ✓" if ok else "FAIL ✗"
        print(
            f"  {name:<42} {r.quality_min:7.4f} {r.quality_mean:7.4f} {mark:>6}"
        )
    print(f"{'═' * 70}")
    if all_passed:
        print("  ✓ All tests passed quality gate.")
    else:
        print("  ✗ One or more tests FAILED the quality gate.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("AxisForge — Mesh Generator Tests  [hex elements]")
    print(f"Project root : {ROOT}")
    print(f"Output dir   : {OUTPUT_DIR.resolve()}")
    print(f"Element type : H8 (mesh_order=1) / H27 (mesh_order=2)")
    print(f"Quality gate : {QUALITY_NAME} min ≥ {QUALITY_GATE_MIN},"
          f" mean ≥ {QUALITY_GATE_MEAN}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    named_results: list[tuple[str, MeshResult]] = []

    tests = [
        ("Simple uniform solid shaft",          test_simple_solid),
        ("Stepped solid shaft",                  test_stepped_solid),
        ("Mixed solid/hollow shaft",             test_mixed_hollow),
        ("Extreme aspect-ratio shaft",           test_extreme_aspect_ratio),
        ("Multi-step hollow shaft",              test_multi_step_hollow),
        ("Quadratic elements (H27)",             test_quadratic_elements),
    ]

    for test_name, test_fn in tests:
        try:
            r = test_fn()
            named_results.append((test_name, r))
        except Exception as exc:
            print(f"\n  ERROR in '{test_name}': {exc}")
            sys.exit(2)

    print_summary(named_results)

    print("  Open meshes in gmsh:")
    for _, r in named_results:
        print(f"    gmsh {r.output_path}")
    print()

    all_ok = all(check_quality(r) for _, r in named_results)
    sys.exit(0 if all_ok else 1)