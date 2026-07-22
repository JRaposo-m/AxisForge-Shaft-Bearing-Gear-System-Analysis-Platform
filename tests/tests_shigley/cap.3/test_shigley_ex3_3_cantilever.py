"""
tests/validation/test_shigley_ex3_3_cantilever.py

Validation of SimpleFEMSolver against Shigley Ex. 3-3 (10th ed., §3-3).
[... cabeçalho do problema inalterado ...]
"""

from __future__ import annotations

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LBF_TO_N       = 4.44822
IN_TO_MM       = 25.4
LBFIN_TO_Nmm   = LBF_TO_N * IN_TO_MM


def mac(x: np.ndarray, a: float, n: int) -> np.ndarray:
    diff = x - a
    return np.where(diff > 0, diff**n, 0.0)


def shigley_ex33_analytical(x_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x_mm / IN_TO_MM
    R1_lbf = 80.0
    M1_lbfin = 160.0
    q_lbf_in = 20.0

    V_lbf = (
        + R1_lbf   * mac(x, 0.0, 0)
        - q_lbf_in * mac(x, 3.0, 1)
        + q_lbf_in * mac(x, 7.0, 1)
    )
    M_lbfin = (
        - M1_lbfin     * mac(x, 0.0, 0)
        + R1_lbf       * mac(x, 0.0, 1)
        - (q_lbf_in/2) * mac(x, 3.0, 2)
        + (q_lbf_in/2) * mac(x, 7.0, 2)
        - 240.0        * mac(x, 10.0, 0)
    )
    return V_lbf * LBF_TO_N, M_lbfin * LBFIN_TO_Nmm


def analytical_reactions() -> dict:
    return {
        "R1_lbf": 80.0, "M1_lbfin": 160.0,
        "R1_N": 80.0 * LBF_TO_N, "M1_Nmm": 160.0 * LBFIN_TO_Nmm,
    }


def test_analytical_reactions():
    rxn = analytical_reactions()
    x_check = np.array([10.001])
    V, M = shigley_ex33_analytical(x_check * IN_TO_MM)
    assert abs(float(V[0])) < 1.0
    assert abs(float(M[0])) < 50.0
    print(f"  ✓ Analytical reactions: R1={rxn['R1_N']:.2f} N, M1={rxn['M1_Nmm']:.1f} N·mm")


def test_analytical_key_points():
    tol_V, tol_M = 1.0, 100.0
    check_points = [
        (0.001, 80.0,  -160.0, "just right of A"),
        (1.0,   80.0,   -80.0, "x=1"),
        (3.001, 80.0,    80.0, "just right of B"),
        (5.0,   40.0,   200.0, "midspan dist load"),
        (7.0,    0.0,   240.0, "C: dist ends"),
        (9.0,    0.0,   240.0, "between C and D"),
        (9.999,  0.0,   240.0, "just before D"),
    ]
    all_pass = True
    for x_in, V_ref_lbf, M_ref_lbfin, desc in check_points:
        x_mm = np.array([x_in * IN_TO_MM])
        V_N, M_Nmm = shigley_ex33_analytical(x_mm)
        err_V = abs(float(V_N[0]) - V_ref_lbf * LBF_TO_N)
        err_M = abs(float(M_Nmm[0]) - M_ref_lbfin * LBFIN_TO_Nmm)
        print(f"  x={x_in:.3f} in  err_V={err_V:.2f} N  err_M={err_M:.1f} N·mm  ({desc})")
        if err_V >= tol_V or err_M >= tol_M:
            all_pass = False
    assert all_pass


# ---------------------------------------------------------------------------
# FEM — cantilever BC + mesh refinement via Mesh1D.add_mandatory_positions
# ---------------------------------------------------------------------------

def _build_shaft_system(shaft_length_mm: float = 254.0):
    """Constrói o ShaftSystem do Shigley Ex.3-3, sem qualquer node hack."""
    from axisforge.core.machine_elements.Shaft.shaft import Shaft, ShaftSection
    from axisforge.core.machine_elements.Bearings.bearing import Bearing, BearingType
    from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem
    from axisforge.core.loads import DistributedRadialLoad, ExternalMoment

    L_mm = shaft_length_mm
    shaft = Shaft(label="shigley_ex33_shaft")
    shaft.add_section(ShaftSection(length=L_mm, diameter=30.0, label="sec1"))

    sys = ShaftSystem(shaft=shaft, name="Shigley_Ex3_3", speed_rpm=0.0)
    sys.add_bearing(Bearing(
        label="A", position=0.0, bearing_type=BearingType.CYLINDRICAL_ROLLER,
        C=50_000.0, C0=30_000.0, arrangement="locating",
    ))
    sys.add_bearing(Bearing(
        label="D", position=L_mm, bearing_type=BearingType.CYLINDRICAL_ROLLER,
        C=50_000.0, C0=30_000.0, arrangement="floating",
    ))

    F_total = (20.0 * LBF_TO_N / IN_TO_MM) * (4.0 * IN_TO_MM)
    sys.add_load(DistributedRadialLoad(
        x_lo=3.0 * IN_TO_MM, x_hi=7.0 * IN_TO_MM,
        magnitude=F_total, theta_deg=180.0,
        label="uniform_q", source="user",
    ))
    sys.add_load(ExternalMoment(
        position=L_mm, magnitude=240.0 * LBFIN_TO_Nmm,
        theta_deg=0.0, label="CCW_moment_D", source="user",
    ))
    return sys


def _cantilever_solver_class():
    """
    CantileverFEMSolver: mesmo BC monkey-patch de sempre (encastrement em A),
    mas agora com _build_mesh sobreposto para injetar nós extra dentro do
    intervalo da carga distribuída via Mesh1D.add_mandatory_positions —
    substitui por completo o truque antigo de RadialLoad(magnitude=0.0).
    """
    from axisforge.mesh.oneD.shaft.Elements.elem import Elem
    from axisforge.mesh.oneD.shaft.mesh_generation.mesh_1D import Mesh1D
    from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver

    class CantileverFEMSolver(SimpleFEMSolver):
        def __init__(self, *args, extra_nodes: list[float] | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self._extra_nodes = extra_nodes or []

        def _build_mesh(self, shaft_system):
            mesh = Mesh1D(shaft_system)
            if self._extra_nodes:
                mesh.add_mandatory_positions(self._extra_nodes)
            return mesh

        def _boundary_dofs(self, x_nodes, shaft_system):
            constrained = []
            for b in shaft_system.bearings:
                if b.arrangement != "locating":
                    continue  # floating = free end, sem constraint
                i = Elem.find_node_index(x_nodes, b.position)
                constrained += [3 * i, 3 * i + 1, 3 * i + 2]  # u, v, theta = 0
            n_dofs = 3 * len(x_nodes)
            return [d for d in range(n_dofs) if d not in constrained], constrained

    return CantileverFEMSolver


def _intermediate_nodes(x_lo: float, x_hi: float, n_intermediate: int) -> list[float]:
    """n_intermediate pontos igualmente espaçados dentro de (x_lo, x_hi), excluindo os extremos.

    NOTA: mantida apenas como fallback manual. A malha "oficial" de produção
    passa agora a vir de `_extra_nodes_from_convergence`, que usa o
    MeshConvergenceStudy em vez de um número de nós arbitrário.
    """
    return [x_lo + k * (x_hi - x_lo) / n_intermediate for k in range(1, n_intermediate)]


def _extra_nodes_from_convergence(
    shaft_system,
    theory: str = "timoshenko",
    tol: float = 1e-3,
    max_levels: int = 4,
    metric: str = "sigma_b",
    verbose: bool = True,
):
    """
    Corre o MeshConvergenceStudy sobre o ShaftSystem e devolve
    (extra_nodes_mm, MeshRefinementResult).

    Esta é agora a fonte oficial dos extra_nodes usados no cantilever:
    em vez de escolher `n_intermediate` à mão, deixamos o estudo de
    convergência decidir onde a malha precisa de refinamento dentro de
    cada distributed_radial_load, com base no critério `metric`/`tol`.
    """
    from axisforge.mesh.oneD.shaft.mesh_generation.mesh_convergence_study import MeshConvergenceStudy
    from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver

    base_solver = SimpleFEMSolver(theory=theory)
    study = MeshConvergenceStudy(base_solver, tol=tol, max_levels=max_levels, metric=metric)
    result = study.run(shaft_system)

    if verbose:
        result.print_report(unit_label="mm")

    return result.all_extra_nodes, result


def _try_fem_solve(
    shaft_length_mm: float = 254.0,
    n_intermediate: int = 8,
    use_convergence: bool = False,
    convergence_kwargs: dict | None = None,
):
    """
    Resolve o cantilever.

    - use_convergence=False (default, compatibilidade): nós extra igualmente
      espaçados dentro de [3,7] in, via `_intermediate_nodes`.
    - use_convergence=True: nós extra vêm de `_extra_nodes_from_convergence`
      (MeshConvergenceStudy), isto é, do próprio `mesh_convergence_study.py`,
      em vez de um número de subdivisões escolhido à mão.
    """
    from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import ShaftResultsReader

    sys = _build_shaft_system(shaft_length_mm)

    conv_result = None
    if use_convergence:
        extra_nodes, conv_result = _extra_nodes_from_convergence(sys, **(convergence_kwargs or {}))
    else:
        extra_nodes = _intermediate_nodes(3.0 * IN_TO_MM, 7.0 * IN_TO_MM, n_intermediate)

    CantileverFEMSolver = _cantilever_solver_class()
    solver = CantileverFEMSolver(theory="timoshenko", extra_nodes=extra_nodes)
    solver.solve(sys)

    reader  = ShaftResultsReader(solver, sys)
    results = reader.read()

    x_nodes = results.x
    V_fem   = results.V_xy
    M_fem   = results.M_xy

    f_xy_rxn = solver.f_xy_reaction
    R1_fem = abs(float(f_xy_rxn[1]))
    M1_fem = abs(float(f_xy_rxn[2]))

    print(f"\n  [diag] n_nodes={len(x_nodes)}  R1={R1_fem:.3f} N  M1={M1_fem:.3f} N·mm"
          f"  (use_convergence={use_convergence})")

    if conv_result is not None:
        return x_nodes, V_fem, M_fem, R1_fem, M1_fem, conv_result
    return x_nodes, V_fem, M_fem, R1_fem, M1_fem


def test_fem_reactions():
    try:
        _, _, _, R1_fem, M1_fem = _try_fem_solve()
    except ImportError as e:
        print(f"\n  [SKIP] axisforge not importable: {e}")
        return

    rxn = analytical_reactions()
    tol_R, tol_M = 2.0, 200.0

    err_R = abs(R1_fem - rxn["R1_N"])
    print(f"  R1: FEM={R1_fem:.2f} N, ref={rxn['R1_N']:.2f} N, err={err_R:.3f} N")
    assert err_R < tol_R

    err_M = abs(M1_fem - rxn["M1_Nmm"])
    print(f"  M1: FEM={M1_fem:.1f} N·mm, ref={rxn['M1_Nmm']:.1f} N·mm, err={err_M:.1f} N·mm")
    assert err_M < tol_M
    print("  ✓ Reações dentro da tolerância")


def test_mesh_convergence_report():
    """
    Corre o MeshConvergenceStudy (mesh_convergence_study.py) sobre o
    ShaftSystem do Ex.3-3, imprime o relatório completo (por load: níveis,
    métrica, erro relativo, nodes finais) e valida as reações FEM resolvidas
    com esses extra_nodes, exatamente como nos outros testes deste ficheiro.
    """
    try:
        sys_ = _build_shaft_system()
        extra_nodes, conv_result = _extra_nodes_from_convergence(sys_, verbose=True)
    except ImportError as e:
        print(f"\n  [SKIP] axisforge not importable: {e}")
        return

    assert len(extra_nodes) > 0, "MeshConvergenceStudy não devolveu nenhum extra_node"

    x_nodes, V_fem, M_fem, R1_fem, M1_fem, conv_result = _try_fem_solve(
        use_convergence=True,
    )

    rxn = analytical_reactions()
    tol_R, tol_M = 2.0, 200.0

    err_R = abs(R1_fem - rxn["R1_N"])
    err_M = abs(M1_fem - rxn["M1_Nmm"])
    print(f"  R1 (malha da convergência): FEM={R1_fem:.2f} N, ref={rxn['R1_N']:.2f} N, err={err_R:.3f} N")
    print(f"  M1 (malha da convergência): FEM={M1_fem:.1f} N·mm, ref={rxn['M1_Nmm']:.1f} N·mm, err={err_M:.1f} N·mm")

    assert err_R < tol_R
    assert err_M < tol_M
    print("  ✓ Reações dentro da tolerância, usando extra_nodes do MeshConvergenceStudy")


# ---------------------------------------------------------------------------
# PLOTTING — malha grosseira vs. malha refinada (add_mandatory_positions)
# ---------------------------------------------------------------------------

def plot_shigley_ex33(show: bool = True):
    x_mm = np.linspace(0.0, 254.0, 2000)
    V_ref, M_ref = shigley_ex33_analytical(x_mm)

    fem_available = False
    x_coarse = V_coarse = M_coarse = None
    x_fine   = V_fine   = M_fine   = None
    try:
        x_coarse, V_coarse, M_coarse, _, _ = _try_fem_solve(n_intermediate=0)   # sem refinamento
        # malha "refinada": agora vinda do MeshConvergenceStudy, não de n_intermediate=24
        x_fine, V_fine, M_fine, _, _, _ = _try_fem_solve(use_convergence=True)
        fem_available = True
    except ImportError as e:
        print(f"  [INFO] FEM overlay not available: {e}")

    x_in = x_mm / IN_TO_MM

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(
        "Shigley Example 3-3 — Cantilever: V(x) e M(x)\n"
        "(malha grosseira vs. malha do MeshConvergenceStudy)",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 1, hspace=0.45, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x_in, V_ref / LBF_TO_N, "b-", lw=2.0, label="Macaulay (ref)")
    if fem_available:
        ax1.plot(x_coarse / IN_TO_MM, V_coarse / LBF_TO_N, "g--", lw=1.2, label="FEM grosseiro")
        ax1.plot(x_fine / IN_TO_MM, V_fine / LBF_TO_N, "r-", lw=1.2, label="FEM (convergência)")
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    for xv in (3, 7, 10):
        ax1.axvline(xv, color="gray", lw=0.7, ls=":")
    ax1.set_xlabel("x [in]"); ax1.set_ylabel("V [lbf]")
    ax1.set_title("Shear Force V(x)", fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_xticks([0, 3, 7, 10]); ax1.set_xticklabels(["A\n0", "B\n3", "C\n7", "D\n10"])

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x_in, M_ref / LBFIN_TO_Nmm, "b-", lw=2.0, label="Macaulay (ref)")
    if fem_available:
        ax2.plot(x_coarse / IN_TO_MM, M_coarse / LBFIN_TO_Nmm, "g--", lw=1.2, label="FEM grosseiro")
        ax2.plot(x_fine / IN_TO_MM, M_fine / LBFIN_TO_Nmm, "r-", lw=1.2, label="FEM (convergência)")
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    for xv in (3, 7, 10):
        ax2.axvline(xv, color="gray", lw=0.7, ls=":")
    ax2.set_xlabel("x [in]"); ax2.set_ylabel("M [lbf·in]")
    ax2.set_title("Bending Moment M(x)", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_xticks([0, 3, 7, 10]); ax2.set_xticklabels(["A\n0", "B\n3", "C\n7", "D\n10"])

    fig.subplots_adjust(bottom=0.1)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    print("=" * 65)
    print("  Shigley Ex. 3-3 — Cantilever Validation")
    print("=" * 65)

    print("\n[1] Analytical: free-end boundary condition check")
    test_analytical_reactions()

    print("\n[2] Analytical: key section point checks")
    test_analytical_key_points()

    print("\n[3] FEM: reaction comparison (malha manual, requires axisforge)")
    test_fem_reactions()

    print("\n[4] Mesh convergence study: relatório + reações com extra_nodes da convergência")
    test_mesh_convergence_report()

    print("\n[5] Plotting: malha grosseira vs. malha do MeshConvergenceStudy...")
    plot_shigley_ex33(show=True)