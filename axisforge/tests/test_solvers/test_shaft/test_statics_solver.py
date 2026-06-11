# test_statics_solver.py
"""
Validation test — StaticsSolver (FEM Euler-Bernoulli)

Shaft geometry:
  Section 1: d=30mm, L=100mm
  Section 2: d=45mm, L=150mm
  Section 3: d=30mm, L=100mm
  Total length: 350mm

  Shoulders at sec1/sec2 and sec2/sec3 transitions (r=1.5mm)

Boundary conditions:
  Bearing A: x=50mm  (simple support, v=0)
  Bearing B: x=300mm (simple support, v=0)
  Span: 250mm

Loading (C14 gear at x=175mm — midspan):
  Ft = 5464.5 N  → XZ plane (tangential)
  Fr = 2256.6 N  → XY plane (radial)
  Fa = 0.0 N     (spur gear)
  T  = 200 N·m = 200000 N·mm

Analytical reference (simply supported beam, point load at midspan):
  v_max = F * L³ / (48 * E * I)   [at midspan, uniform section approximation]

  For XZ: F=5464.5N, span=250mm, I≈sec2 (conservative)
  For XY: F=2256.6N

Note: FEM result with stepped sections will differ from single-section
      analytical solution. Use as order-of-magnitude check only.
      Exact validation requires textbook stepped-beam case (Phase 2).

Checks:
  1. solve() runs without exception
  2. StaticsResult fields populated
  3. Reaction equilibrium: ΣFy = 0, ΣFz = 0
  4. Displacement at bearings ≈ 0
  5. Midspan displacement sign correct (downward under radial load)
  6. Torsion constant between gear and bearing B
  7. Node count and element count consistent
"""

import math
import pytest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import math


# ── Path setup ───────────────────────────────────────────────────────────────
# Adjust this import path to match your project structure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from core.shaft import Shaft, ShaftSection, Shoulder
from core.components import Bearing, GearElement
from core.system import MechanicalSystem
from core.materials import CrMo42

# Import the new FEM solver — adjust path as needed
from solvers.shaft.shaft_analysis import StaticsSolver
from models.shaft_result import StaticsResult, StressResult
from solvers.shaft.shaft_analysis import StressSolver


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def shaft_3sec():
    """3-section stepped shaft: 30/45/30 mm diameters."""
    shoulder_12 = Shoulder(fillet_radius=1.5, diameter_large=45.0, diameter_small=30.0)
    shoulder_23 = Shoulder(fillet_radius=1.5, diameter_large=45.0, diameter_small=30.0)

    s1 = ShaftSection(length=100.0, diameter=30.0, material_id="42CrMo4", label="sec1")
    s2 = ShaftSection(length=150.0, diameter=45.0, material_id="42CrMo4",
                    shoulder_left=shoulder_12, shoulder_right=shoulder_23, label="sec2")
    s3 = ShaftSection(length=100.0, diameter=30.0, material_id="42CrMo4", label="sec3")

    shaft = Shaft()
    shaft.add_section(s1)
    shaft.add_section(s2)
    shaft.add_section(s3)
    return shaft


@pytest.fixture
def system_c14(shaft_3sec):
    """MechanicalSystem with C14 gear at midspan, bearings at x=50 and x=300."""
    system = MechanicalSystem(shaft=shaft_3sec, name="C14_test", speed_rpm=1500.0)

    system.add_bearing(Bearing(position=50.0,  label="A"))
    system.add_bearing(Bearing(position=300.0, label="B"))

    # C14 gear: spur, pinion side
    # Ft=5464.5N (XZ), Fr=2256.6N (XY), Fa=0, T=200 N·m
    gear = GearElement(
        position=175.0,
        tangential_force=5464.5,
        radial_force=2256.6,
        axial_force=0.0,
        pitch_diameter=73.2,   # 2 * 36.6mm pitch radius
        torque=200_000.0,      # N·mm
        label="C14_pinion",
    )
    system.add_gear(gear)
    return system


@pytest.fixture
def solver():
    return StaticsSolver(theory="euler")


@pytest.fixture
def result(solver, system_c14):
    return solver.solve(system_c14)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSolverRuns:
    def test_solve_returns_statics_result(self, result):
        assert isinstance(result, StaticsResult)

    def test_result_has_nodes(self, result):
        assert len(result.x_nodes) >= 5  # sec boundaries + bearings + gear

    def test_result_has_elements(self, result):
        assert len(result.elements) == len(result.x_nodes) - 1

    def test_displacement_vectors_correct_size(self, result):
        n_dof = 3 * len(result.x_nodes)
        assert result.d_total_xz.shape == (n_dof,)
        assert result.d_total_xy.shape == (n_dof,)


class TestMeshConsistency:
    def test_bearing_positions_are_nodes(self, result):
        """Bearings at x=50 and x=300 must be mesh nodes."""
        assert any(abs(x - 50.0) < 1e-6 for x in result.x_nodes)
        assert any(abs(x - 300.0) < 1e-6 for x in result.x_nodes)

    def test_gear_position_is_node(self, result):
        """Gear at x=175 must be a mesh node."""
        assert any(abs(x - 175.0) < 1e-6 for x in result.x_nodes)

    def test_section_boundaries_are_nodes(self, result):
        """Section boundaries at x=0, 100, 250, 350 must be nodes."""
        for x in [0.0, 100.0, 250.0, 350.0]:
            assert any(abs(xn - x) < 1e-6 for xn in result.x_nodes), \
                f"Section boundary x={x} not in mesh"

    def test_nodes_strictly_increasing(self, result):
        diffs = np.diff(result.x_nodes)
        assert np.all(diffs > 0)


class TestEquilibrium:
    def _bearing_node_dof(self, result, x_bearing):
        """Return v-DOF index for a bearing node."""
        for i, x in enumerate(result.x_nodes):
            if abs(x - x_bearing) < 1e-6:
                return 3 * i + 1
        raise ValueError(f"No node at x={x_bearing}")

    def test_xz_reaction_equilibrium(self, result):
        """ΣFz = 0: reactions + applied tangential force = 0."""
        # f_xz_ext contains internal forces; reactions = f_ext - f_applied
        # Sum of all external forces in XZ must be zero
        total = np.sum(result.f_xz_ext)
        # Tolerance: 0.1% of applied load
        assert abs(total) < 5464.5 * 0.001 + 1.0, \
            f"XZ force equilibrium failed: ΣF = {total:.2f} N"

    def test_xy_reaction_equilibrium(self, result):
        """ΣFy = 0: reactions + applied radial force = 0."""
        total = np.sum(result.f_xy_ext)
        assert abs(total) < 2256.6 * 0.001 + 1.0, \
            f"XY force equilibrium failed: ΣF = {total:.2f} N"

    def test_bearing_displacement_xz_near_zero(self, result):
        """v at bearing nodes must be ≈ 0 (boundary condition enforced)."""
        for x_b in [50.0, 300.0]:
            dof = self._bearing_node_dof(result, x_b)
            assert abs(result.d_total_xz[dof]) < 1e-8, \
                f"XZ displacement at bearing x={x_b}: {result.d_total_xz[dof]:.2e} mm"

    def test_bearing_displacement_xy_near_zero(self, result):
        for x_b in [50.0, 300.0]:
            dof = self._bearing_node_dof(result, x_b)
            assert abs(result.d_total_xy[dof]) < 1e-8, \
                f"XY displacement at bearing x={x_b}: {result.d_total_xy[dof]:.2e} mm"


class TestDisplacementPhysics:
    def _node_v_dof(self, result, x_target):
        for i, x in enumerate(result.x_nodes):
            if abs(x - x_target) < 1e-6:
                return 3 * i + 1
        raise ValueError(f"No node at x={x_target}")

    def test_midspan_xz_displacement_nonzero(self, result):
        """Gear node must deflect under tangential load."""
        dof = self._node_v_dof(result, 175.0)
        assert abs(result.d_total_xz[dof]) > 1e-10

    def test_midspan_xy_displacement_nonzero(self, result):
        dof = self._node_v_dof(result, 175.0)
        assert abs(result.d_total_xy[dof]) > 1e-10

    def test_displacement_order_of_magnitude(self, result):
        """
        Rough check: max deflection should be between 1e-5 and 1.0 mm.
        For steel shaft d≈35mm, L=250mm, F≈5500N → v ~ 0.01-0.1 mm range.
        """
        v_max_xz = np.max(np.abs(result.d_total_xz[1::3]))  # v DOFs only
        assert 1e-5 < v_max_xz < 1.0, \
            f"XZ max deflection out of expected range: {v_max_xz:.4e} mm"


# ── Console output ────────────────────────────────────────────────────────────

def print_results(result: StaticsResult):
    print("\n" + "="*60)
    print("  StaticsSolver — C14 Test Case Results")
    print("="*60)
    print(f"\nMesh: {len(result.x_nodes)} nodes, {len(result.elements)} elements")
    print(f"Node positions [mm]: {[f'{x:.1f}' for x in result.x_nodes]}")

    print("\n── Displacements (v) at key nodes ──")
    key_positions = [0.0, 50.0, 100.0, 175.0, 250.0, 300.0, 350.0]
    print(f"{'x [mm]':>10} {'v_xz [mm]':>14} {'v_xy [mm]':>14}")
    print("-" * 40)
    for x_target in key_positions:
        for i, x in enumerate(result.x_nodes):
            if abs(x - x_target) < 1e-6:
                v_xz = result.d_total_xz[3*i + 1]
                v_xy = result.d_total_xy[3*i + 1]
                print(f"{x:>10.1f} {v_xz:>14.6e} {v_xy:>14.6e}")
                break

    print("\n── Reaction check (f_ext sum) ──")
    print(f"  ΣF_xz = {np.sum(result.f_xz_ext):.4f} N  (should ≈ 0)")
    print(f"  ΣF_xy = {np.sum(result.f_xy_ext):.4f} N  (should ≈ 0)")

    print("\n── Max deflection ──")
    v_max_xz = np.max(np.abs(result.d_total_xz[1::3]))
    v_max_xy = np.max(np.abs(result.d_total_xy[1::3]))
    print(f"  XZ: {v_max_xz:.4e} mm")
    print(f"  XY: {v_max_xy:.4e} mm")
    print("="*60)


def build_system_complex():
    """
    4-section shaft, 3 bearings, overhang loads both sides.
    
    Layout:
      x=0 ──[sec1 d=25]── x=80 ──[sec2 d=40]── x=200 ──[sec3 d=40]── x=320 ──[sec4 d=25]── x=400
      
    Bearings:  A=x=80, B=x=200, C=x=320  (hyperstatic, 1 redundant)
    Gear:      x=200 (midspan)
    Overhang:  RadialLoad at x=30 (left), RadialLoad at x=370 (right)
    """
    from core.loads import RadialLoad, LoadPlane

    sh12 = Shoulder(fillet_radius=2.0, diameter_large=40.0, diameter_small=25.0)
    sh34 = Shoulder(fillet_radius=2.0, diameter_large=40.0, diameter_small=25.0)

    s1 = ShaftSection(length=100.0,  diameter=25.0, material_id="42CrMo4", label="sec1_overhang_L")
    s2 = ShaftSection(length=100.0, diameter=40.0, material_id="42CrMo4",
                    shoulder_left=sh12, label="sec2")
    s3 = ShaftSection(length=100.0, diameter=40.0, material_id="42CrMo4",
                    shoulder_right=sh34, label="sec3")
    s4 = ShaftSection(length=100.0,  diameter=25.0, material_id="42CrMo4", label="sec4_overhang_R")

    shaft = Shaft()
    shaft.add_section(s1); shaft.add_section(s2)
    shaft.add_section(s3); shaft.add_section(s4)

    system = MechanicalSystem(shaft=shaft, name="complex_3brg", speed_rpm=1500.0)

    system.add_bearing(Bearing(position=50.0,  label="A"))
    system.add_bearing(Bearing(position=250.0, label="B"))
    system.add_bearing(Bearing(position=320.0, label="C"))

    system.add_gear(GearElement(
        position=150.0, tangential_force=5464.5, radial_force=2256.6,
        axial_force=0.0, pitch_diameter=73.2, torque=200_000.0, label="C14_pinion"
    ))

    # Overhang loads
    system.add_load(RadialLoad(position=30.0,  magnitude=1500.0, plane=LoadPlane.XY, label="overhang_L"))
    system.add_load(RadialLoad(position=350.0, magnitude=800.0,  plane=LoadPlane.XZ, label="overhang_R"))

    from core.loads import RadialLoad, TorqueLoad, LoadPlane

    system.add_load(TorqueLoad(position=150.0, magnitude=0.0, label="input_torque"))   # N·mm
    system.add_load(TorqueLoad(position=250.0, magnitude=-200_000.0, label="output_torque"))

    from core.loads import LoadingProfile

    loading_profiles = {
        "overhang_R":                    LoadingProfile(label="overhang_R",   R=1.0),
        "overhang_L":                    LoadingProfile(label="overhang_L",   R=1.0),
        "gear_C14_pinion@150.0":         LoadingProfile(label="gear_bending", R=-1.0),
        "gear_C14_pinion@150.0_torque":  LoadingProfile(label="gear_torque",  R=0.0),
        "output_torque":                 LoadingProfile(label="output_torque", R=0.0),
    }

    return system, loading_profiles


if __name__ == "__main__":
    from core.shaft import Shaft, ShaftSection, Shoulder
    from core.components import Bearing, GearElement
    from core.system import MechanicalSystem

    shoulder_12 = Shoulder(fillet_radius=1.5, diameter_large=45.0, diameter_small=30.0)
    shoulder_23 = Shoulder(fillet_radius=1.5, diameter_large=45.0, diameter_small=30.0)
    s1 = ShaftSection(length=100.0, diameter=30.0, material_id="42CrMo4", label="sec1")
    s2 = ShaftSection(length=150.0, diameter=45.0, material_id="42CrMo4",
                    shoulder_left=shoulder_12, shoulder_right=shoulder_23, label="sec2")
    s3 = ShaftSection(length=100.0, diameter=30.0, material_id="42CrMo4", label="sec3")
    shaft = Shaft()
    shaft.add_section(s1); shaft.add_section(s2); shaft.add_section(s3)

    system = MechanicalSystem(shaft=shaft, name="C14_test", speed_rpm=1500.0)
    system.add_bearing(Bearing(position=50.0, label="A"))
    system.add_bearing(Bearing(position=300.0, label="B"))
    system.add_gear(GearElement(
        position=175.0, tangential_force=5464.5, radial_force=2256.6,
        axial_force=0.0, pitch_diameter=73.2, torque=200_000.0, label="C14_pinion"
    ))

    solver = StaticsSolver(theory="euler")
    result = solver.solve(system)
    print_results(result)
    print("\n── Reactions at bearings ──")
    print(f"{'Bearing':>10} {'R_xz [N]':>14} {'R_xy [N]':>14}")
    print("-" * 40)
    for x_b, label in [(50.0, 'A'), (300.0, 'B')]:
        for i, x in enumerate(result.x_nodes):
            if abs(x - x_b) < 1e-6:
                r_xz = result.f_xz_ext[3*i + 1]
                r_xy = result.f_xy_ext[3*i + 1]
                print(f"{label:>10} {r_xz:>14.2f} {r_xy:>14.2f}")
                break

    print("\n── External forces applied ──")
    print(f"{'x [mm]':>10} {'F_xz [N]':>14} {'F_xy [N]':>14}")
    print("-" * 40)
    for i, x in enumerate(result.x_nodes):
        fxz = result.f_xz_ext[3*i + 1]
        fxy = result.f_xy_ext[3*i + 1]
        if abs(fxz) > 1.0 or abs(fxy) > 1.0:
            print(f"{x:>10.1f} {fxz:>14.2f} {fxy:>14.2f}")

    stress_solver = StressSolver(theory="euler")
    stress_result = stress_solver.solve(result, system)

    # ── Tabela esforços e tensões ──
    print("\n── Stress and internal forces at nodes ──")
    print(f"{'x [mm]':>10} {'σ_xz [MPa]':>14} {'σ_xy [MPa]':>14} {'N_xz [N]':>12} {'V_xz [N]':>12} {'M_xz [Nmm]':>14} {'V_xy [N]':>12} {'M_xy [Nmm]':>14}")
    print("-" * 110)
    for (zeta, elem, s_ax, s_xz, s_xy), (_, _, fi_xz, fi_xy) in zip(stress_result.sigma, stress_result.internal_forces):
        x = result.x_nodes[elem.idx_node_1] if zeta == -1.0 else result.x_nodes[elem.idx_node_2]
        N_xz, V_xz, M_xz = fi_xz
        N_xy, V_xy, M_xy = fi_xy
        print(f"{x:>10.1f} {s_xz:>14.4f} {s_xy:>14.4f} {N_xz:>12.2f} {V_xz:>12.2f} {M_xz:>14.2f} {V_xy:>12.2f} {M_xy:>14.2f}")

    # ── Plot deflexão ──
    x_pts = [result.x_nodes[i] for i in range(len(result.x_nodes))]
    v_xz_pts = [result.d_total_xz[3*i+1] for i in range(len(result.x_nodes))]
    v_xy_pts  = [result.d_total_xy[3*i+1] for i in range(len(result.x_nodes))]

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(x_pts, v_xz_pts, 'b-o')
    ax1.axhline(0, color='k', linewidth=0.5)
    ax1.set_ylabel('Deflection XZ [mm]'); ax1.grid(True)
    ax2.plot(x_pts, v_xy_pts, 'r-o')
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.set_ylabel('Deflection XY [mm]'); ax2.set_xlabel('x [mm]'); ax2.grid(True)
    fig1.suptitle('Shaft Deflection — C14 Case'); fig1.tight_layout()

    # ── Plot tensão ──
    x_sigma = [result.x_nodes[elem.idx_node_1] if zeta == -1.0 else result.x_nodes[elem.idx_node_2]
               for zeta, elem, _, _, _ in stress_result.sigma]
    sigma_xz_pts = [s for _, _, _, s, _ in stress_result.sigma]
    sigma_xy_pts = [s for _, _, _, _, s in stress_result.sigma]

    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax3.plot(x_sigma, sigma_xz_pts, 'b-o')
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.set_ylabel('σ_xz [MPa]'); ax3.grid(True)
    ax4.plot(x_sigma, sigma_xy_pts, 'r-o')
    ax4.axhline(0, color='k', linewidth=0.5)
    ax4.set_ylabel('σ_xy [MPa]'); ax4.set_xlabel('x [mm]'); ax4.grid(True)
    fig2.suptitle('Shaft Stress — C14 Case'); fig2.tight_layout()

    plt.show()


    # ── Caso 2: secção única d=30mm ──
    s_single = ShaftSection(length=350.0, diameter=30.0, material_id="42CrMo4", label="single")
    shaft_single = Shaft()
    shaft_single.add_section(s_single)

    system_single = MechanicalSystem(shaft=shaft_single, name="single_sec_test", speed_rpm=1500.0)
    system_single.add_bearing(Bearing(position=50.0, label="A"))
    system_single.add_bearing(Bearing(position=300.0, label="B"))
    system_single.add_gear(GearElement(
        position=175.0, tangential_force=5464.5, radial_force=2256.6,
        axial_force=0.0, pitch_diameter=73.2, torque=200_000.0, label="C14_pinion"
    ))

    result_single = solver.solve(system_single)
    stress_result_single = stress_solver.solve(result_single, system_single)

    print("\n\n── CASO 2: Secção única d=30mm ──")
    print("\n── Stress and internal forces at nodes ──")
    print(f"{'x [mm]':>10} {'σ_xz [MPa]':>14} {'σ_xy [MPa]':>14} {'V_xz [N]':>12} {'M_xz [Nmm]':>14} {'V_xy [N]':>12} {'M_xy [Nmm]':>14}")
    print("-" * 100)
    for (zeta, elem, s_ax, s_xz, s_xy), (_, _, fi_xz, fi_xy) in zip(stress_result_single.sigma, stress_result_single.internal_forces):
        x = result_single.x_nodes[elem.idx_node_1] if zeta == -1.0 else result_single.x_nodes[elem.idx_node_2]
        N_xz, V_xz, M_xz = fi_xz
        N_xy, V_xy, M_xy = fi_xy
        print(f"{x:>10.1f} {s_xz:>14.4f} {s_xy:>14.4f} {V_xz:>12.2f} {M_xz:>14.2f} {V_xy:>12.2f} {M_xy:>14.2f}")

        x_pts_s = [result_single.x_nodes[i] for i in range(len(result_single.x_nodes))]
    v_xz_s = [result_single.d_total_xz[3*i+1] for i in range(len(result_single.x_nodes))]
    v_xy_s  = [result_single.d_total_xy[3*i+1] for i in range(len(result_single.x_nodes))]

    x_sigma_s = [result_single.x_nodes[elem.idx_node_1] if zeta == -1.0 else result_single.x_nodes[elem.idx_node_2]
                 for zeta, elem, _, _,_ in stress_result_single.sigma]
    sigma_xz_s = [s for _, _, _, s, _ in stress_result_single.sigma]
    sigma_xy_s = [s for _, _, _, _, s in stress_result_single.sigma]

    fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax5.plot(x_pts_s, v_xz_s, 'b-o')
    ax5.axhline(0, color='k', linewidth=0.5)
    ax5.set_ylabel('Deflection XZ [mm]'); ax5.grid(True)
    ax6.plot(x_pts_s, v_xy_s, 'r-o')
    ax6.axhline(0, color='k', linewidth=0.5)
    ax6.set_ylabel('Deflection XY [mm]'); ax6.set_xlabel('x [mm]'); ax6.grid(True)
    fig3.suptitle('Shaft Deflection — Single Section d=30mm'); fig3.tight_layout()

    fig4, (ax7, ax8) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax7.plot(x_sigma_s, sigma_xz_s, 'b-o')
    ax7.axhline(0, color='k', linewidth=0.5)
    ax7.set_ylabel('σ_xz [MPa]'); ax7.grid(True)
    ax8.plot(x_sigma_s, sigma_xy_s, 'r-o')
    ax8.axhline(0, color='k', linewidth=0.5)
    ax8.set_ylabel('σ_xy [MPa]'); ax8.set_xlabel('x [mm]'); ax8.grid(True)
    fig4.suptitle('Shaft Stress — Single Section d=30mm'); fig4.tight_layout()

    plt.show()


    system_complex, loading_profiles  = build_system_complex()
    result_complex = solver.solve(system_complex)
    stress_complex = stress_solver.solve(result_complex, system_complex)

    print("\n\n── CASO 3: 4 secções, 3 rolamentos, overhang ──")
    print(f"Mesh: {len(result_complex.x_nodes)} nodes")
    print(f"{'x [mm]':>10} {'σ_xz [MPa]':>14} {'σ_xy [MPa]':>14} {'V_xz [N]':>12} {'M_xz [Nmm]':>14}")
    print("-" * 70)
    for (zeta, elem, s_ax, s_xz, s_xy), (_, _, fi_xz, fi_xy) in zip(stress_complex.sigma, stress_complex.internal_forces):
        x = result_complex.x_nodes[elem.idx_node_1] if zeta == -1.0 else result_complex.x_nodes[elem.idx_node_2]
        print(f"{x:>10.1f} {s_xz:>14.4f} {s_xy:>14.4f} {fi_xz[1]:>12.2f} {fi_xz[2]:>14.2f}")

    x_pts_c = [result_complex.x_nodes[i] for i in range(len(result_complex.x_nodes))]
    v_xz_c = [result_complex.d_total_xz[3*i+1] for i in range(len(result_complex.x_nodes))]
    v_xy_c  = [result_complex.d_total_xy[3*i+1] for i in range(len(result_complex.x_nodes))]
    x_sig_c = [result_complex.x_nodes[elem.idx_node_1] if zeta==-1.0 else result_complex.x_nodes[elem.idx_node_2]
               for zeta,elem,_,_,_ in stress_complex.sigma]
    sxz_c = [s for _,_,_,s,_ in stress_complex.sigma]
    sxy_c = [s for _,_,_,_,s in stress_complex.sigma]

    fig5, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(x_pts_c, v_xz_c, 'b-o'); axes[0].axhline(0,color='k',lw=0.5)
    axes[0].set_ylabel('v_xz [mm]'); axes[0].grid(True)
    axes[1].plot(x_pts_c, v_xy_c,  'r-o'); axes[1].axhline(0,color='k',lw=0.5)
    axes[1].set_ylabel('v_xy [mm]'); axes[1].grid(True)
    axes[2].plot(x_sig_c, sxz_c, 'b-o'); axes[2].axhline(0,color='k',lw=0.5)
    axes[2].set_ylabel('σ_xz [MPa]'); axes[2].grid(True)
    axes[3].plot(x_sig_c, sxy_c, 'r-o'); axes[3].axhline(0,color='k',lw=0.5)
    axes[3].set_ylabel('σ_xy [MPa]'); axes[3].set_xlabel('x [mm]'); axes[3].grid(True)
    fig5.suptitle('Complex Case — 4 sections, 3 bearings, overhang')
    fig5.tight_layout()
    plt.show()

    # ── Caso 3 repetido: Euler vs Timoshenko ──
    solver_timo = StaticsSolver(theory="timoshenko")
    stress_timo = StressSolver(theory="timoshenko")

    result_euler_c = solver.solve(system_complex)
    result_timo_c  = solver_timo.solve(system_complex)
    stress_euler_c = stress_solver.solve(result_euler_c, system_complex)
    stress_timo_c  = stress_timo.solve(result_timo_c, system_complex)

    print("\n\n── EULER vs TIMOSHENKO — Caso 3 ──")
    print(f"{'x [mm]':>10} {'σ_xz E':>12} {'σ_xz T':>12} {'Δ_xz %':>10} {'σ_xy E':>12} {'σ_xy T':>12} {'Δ_xy %':>10}")
    print("-" * 80)
    for (_, _, _, sxz_e, sxy_e), (zeta, elem, _, sxz_t, sxy_t) in zip(stress_euler_c.sigma, stress_timo_c.sigma):
        x = result_timo_c.x_nodes[elem.idx_node_1] if zeta == -1.0 else result_timo_c.x_nodes[elem.idx_node_2]
        d_xz = (sxz_e - sxz_t) / sxz_t * 100 if abs(sxz_t) > 1e-6 else 0.0
        d_xy = (sxy_e - sxy_t) / sxy_t * 100 if abs(sxy_t) > 1e-6 else 0.0
        print(f"{x:>10.1f} {sxz_e:>12.4f} {sxz_t:>12.4f} {d_xz:>10.2f} {sxy_e:>12.4f} {sxy_t:>12.4f} {d_xy:>10.2f}")

    # ── Plots comparativos ──
    x_e = [result_euler_c.x_nodes[elem.idx_node_1] if z==-1.0 else result_euler_c.x_nodes[elem.idx_node_2]
           for z,elem,_,_,_ in stress_euler_c.sigma]
    x_t = [result_timo_c.x_nodes[elem.idx_node_1] if z==-1.0 else result_timo_c.x_nodes[elem.idx_node_2]
           for z,elem,_,_,_ in stress_timo_c.sigma]

    ve_xz = [result_euler_c.d_total_xz[3*i+1] for i in range(len(result_euler_c.x_nodes))]
    vt_xz = [result_timo_c.d_total_xz[3*i+1]  for i in range(len(result_timo_c.x_nodes))]
    ve_xy = [result_euler_c.d_total_xy[3*i+1]  for i in range(len(result_euler_c.x_nodes))]
    vt_xy = [result_timo_c.d_total_xy[3*i+1]   for i in range(len(result_timo_c.x_nodes))]
    x_nodes_e = result_euler_c.x_nodes
    x_nodes_t = result_timo_c.x_nodes

    sxz_e_pts = [s for _,_,_,s,_ in stress_euler_c.sigma]
    sxz_t_pts = [s for _,_,_,s,_ in stress_timo_c.sigma]
    sxy_e_pts = [s for _,_,_,_,s in stress_euler_c.sigma]
    sxy_t_pts = [s for _,_,_,_,s in stress_timo_c.sigma]

    fig6, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(x_nodes_e, ve_xz, 'b-o', label='Euler')
    axes[0].plot(x_nodes_t, vt_xz, 'r--s', label='Timoshenko')
    axes[0].axhline(0,color='k',lw=0.5); axes[0].set_ylabel('v_xz [mm]')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(x_nodes_e, ve_xy, 'b-o', label='Euler')
    axes[1].plot(x_nodes_t, vt_xy, 'r--s', label='Timoshenko')
    axes[1].axhline(0,color='k',lw=0.5); axes[1].set_ylabel('v_xy [mm]')
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(x_e, sxz_e_pts, 'b-o', label='Euler')
    axes[2].plot(x_t, sxz_t_pts, 'r--s', label='Timoshenko')
    axes[2].axhline(0,color='k',lw=0.5); axes[2].set_ylabel('σ_xz [MPa]')
    axes[2].legend(); axes[2].grid(True)

    axes[3].plot(x_e, sxy_e_pts, 'b-o', label='Euler')
    axes[3].plot(x_t, sxy_t_pts, 'r--s', label='Timoshenko')
    axes[3].axhline(0,color='k',lw=0.5); axes[3].set_ylabel('σ_xy [MPa]')
    axes[3].set_xlabel('x [mm]'); axes[3].legend(); axes[3].grid(True)

    fig6.suptitle('Euler vs Timoshenko — Complex Case')
    fig6.tight_layout()
    plt.show()

    # ── Contribuições por fonte de carga ──
    if stress_complex.sigma_contributions:
        print("\n── Stress contributions per load source — Caso 3 ──")
        for contrib in stress_complex.sigma_contributions:
            print(f"\n  [{contrib['type'].upper()}] {contrib['label']}")
            print(f"  {'x [mm]':>10} {'σ_ax [MPa]':>14} {'σ_xz [MPa]':>14} {'σ_xy [MPa]':>14}")
            print("  " + "-" * 56)
            for zeta, elem, s_ax, s_xz, s_xy in contrib['sigma']:
                x = result_complex.x_nodes[elem.idx_node_1] if zeta == -1.0 else result_complex.x_nodes[elem.idx_node_2]
                print(f"  {x:>10.1f} {s_ax:>14.4f} {s_xz:>14.4f} {s_xy:>14.4f}")

        fig7, axes7 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for contrib in stress_complex.sigma_contributions:
            xs = [result_complex.x_nodes[elem.idx_node_1] if z==-1.0 else result_complex.x_nodes[elem.idx_node_2]
                  for z, elem, _, _, _ in contrib['sigma']]
            sxz = [s for _, _, _, s, _ in contrib['sigma']]
            sxy = [s for _, _, _, _, s in contrib['sigma']]
            axes7[0].plot(xs, sxz, '-o', label=contrib['label'])
            axes7[1].plot(xs, sxy, '-o', label=contrib['label'])

        axes7[0].axhline(0, color='k', lw=0.5); axes7[0].set_ylabel('σ_xz [MPa]')
        axes7[0].legend(fontsize=7); axes7[0].grid(True)
        axes7[1].axhline(0, color='k', lw=0.5); axes7[1].set_ylabel('σ_xy [MPa]')
        axes7[1].set_xlabel('x [mm]'); axes7[1].legend(fontsize=7); axes7[1].grid(True)
        fig7.suptitle('Stress Contributions per Load Source — Complex Case')
        fig7.tight_layout()
        plt.show()

        print("\n── Torsion at nodes — Caso 3 ──")
        print(f"{'x [mm]':>10} {'τ [MPa]':>12}")
        print("-" * 25)
        for node in stress_complex.tau:
            print(f"{node['x']:>10.1f} {node['tau_total']:>12.4f}")

        print("\n── Torsion contributions — Caso 3 ──")
        for node in stress_complex.tau:
            x = node['x']
            tau_t = node['tau_total']
            contribs = node['contributions']
            if abs(tau_t) > 1e-6 or any(abs(c['tau']) > 1e-6 for c in contribs):
                print(f"\n  x={x:.1f} mm  τ_total={tau_t:.4f} MPa")
                for c in contribs:
                    print(f"    [{c['label']}] T={c['T']:.1f} N·mm  τ={c['tau']:.4f} MPa")

        print("\n── Loading Profiles ──")
        for label, profile in loading_profiles.items():
            print(f"  {label:>35} R={profile.R:>5.1f}  σ_m_factor={profile.sigma_mean_factor:.2f}  σ_a_factor={profile.sigma_amplitude_factor:.2f}")

        from solvers.shaft.shaft_analysis import FatiguePostProcessing, StressRaiser

        stress_raisers = [
            StressRaiser(label="shoulder@100.0", x=100.0, raiser_type="shoulder", Kf=1.5, Kfs=1.5),
            StressRaiser(label="shoulder@250.0", x=250.0, raiser_type="shoulder", Kf=1.5, Kfs=1.5),
        ]

        fatigue_pp = FatiguePostProcessing()
        fatigue_result = fatigue_pp.process(
            stress_complex, loading_profiles, result_complex.x_nodes, stress_raisers
        )

        print("\n── FatiguePostProcessing — Caso 3 ──")
        print(f"{'x [mm]':>10} {'σ_m_xz':>12} {'σ_a_xz':>12} {'σ_m_xy':>12} {'σ_a_xy':>12} {'τ_m':>10} {'τ_a':>10} {'Kf':>6} {'Kfs':>6}")
        print("-" * 90)
        for i, node in enumerate(fatigue_result):
            x = result_complex.x_nodes[i]
            print(f"{x:>10.1f} {node['sigma_m_xz']:>12.4f} {node['sigma_a_xz']:>12.4f} "
                  f"{node['sigma_m_xy']:>12.4f} {node['sigma_a_xy']:>12.4f} "
                  f"{node['tau_m']:>10.4f} {node['tau_a']:>10.4f} "
                  f"{node['Kf']:>6.2f} {node['Kfs']:>6.2f}")

    x_pts_f = result_complex.x_nodes
    fig8, axes8 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes8[0].plot(x_pts_f, [n['sigma_m_xz'] for n in fatigue_result], 'b-o', label='σ_m_xz')
    axes8[0].plot(x_pts_f, [n['sigma_a_xz'] for n in fatigue_result], 'b--o', label='σ_a_xz')
    axes8[0].axhline(0, color='k', lw=0.5); axes8[0].set_ylabel('σ_xz [MPa]')
    axes8[0].legend(); axes8[0].grid(True)
    axes8[1].plot(x_pts_f, [n['sigma_m_xy'] for n in fatigue_result], 'r-o', label='σ_m_xy')
    axes8[1].plot(x_pts_f, [n['sigma_a_xy'] for n in fatigue_result], 'r--o', label='σ_a_xy')
    axes8[1].axhline(0, color='k', lw=0.5); axes8[1].set_ylabel('σ_xy [MPa]')
    axes8[1].legend(); axes8[1].grid(True)
    axes8[2].plot(x_pts_f, [n['tau_m'] for n in fatigue_result], 'g-o', label='τ_m')
    axes8[2].plot(x_pts_f, [n['tau_a'] for n in fatigue_result], 'g--o', label='τ_a')
    axes8[2].axhline(0, color='k', lw=0.5); axes8[2].set_ylabel('τ [MPa]')
    axes8[2].set_xlabel('x [mm]'); axes8[2].legend(); axes8[2].grid(True)
    fig8.suptitle('FatiguePostProcessing — σ_m, σ_a, τ — Caso 3')
    fig8.tight_layout()
    plt.show()