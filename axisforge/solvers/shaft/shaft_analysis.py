# shaft_analysis.py

from __future__ import annotations
import math
import warnings
import numpy as np
from dataclasses import dataclass

from core.system import MechanicalSystem
from core.components import Bearing
from core.loads import AxialLoad, RadialLoad, TorqueLoad, ExternalMoment, LoadPlane
from core.materials import get_material
from models.shaft_result import (
    StaticsResult,
    #StressRaiserType,
    #CriticalSection,
    StressResult,
    #StaticFailureSection,
    #StaticFailureResult,
)
from config import SOLVER_RESOLUTION, SOLVER_TOLERANCE, BOUNDARY_MOMENT_TOLERANCE, MESH_MIN_NODE_DIST_MM

from models.elem import Elem

class StaticsSolver:
    def __init__(self, theory: str = "euler"):
        self.theory = theory

    def solve(self, system:MechanicalSystem) -> StaticsResult: 

        system.validate_or_raise()
        
        x_bearing = [b.position for b in system.bearings]

    # ── Collect loads with metadata ──────────────────────────────────────
        load_cases = []

        # Cargas externas radiais XZ — uma por uma
        for l in system.radial_loads_xz:
            load_cases.append({
                'label': l.label or f"radial_xz@{l.position}",
                'type': 'external',
                'radial_xz': [(l.position, -l.magnitude)],
                'radial_xy': [], 'axial': [], 'moments_xz': [], 'moments_xy': [],
            })

        # Cargas externas radiais XY
        for l in system.radial_loads_xy:
            load_cases.append({
                'label': l.label or f"radial_xy@{l.position}",
                'type': 'external',
                'radial_xz': [], 'radial_xy': [(l.position, -l.magnitude)],
                'axial': [], 'moments_xz': [], 'moments_xy': [],
            })

        # Cargas axiais
        for l in system.axial_loads:
            load_cases.append({
                'label': l.label or f"axial@{l.position}",
                'type': 'external',
                'radial_xz': [], 'radial_xy': [],
                'axial': [(l.position, l.magnitude)],
                'moments_xz': [], 'moments_xy': [],
            })

        # Momentos externos
        for m in system.external_moments:
            plane_key = 'moments_xz' if m.plane == LoadPlane.XZ else 'moments_xy'
            load_cases.append({
                'label': m.label or f"moment@{m.position}",
                'type': 'external',
                'radial_xz': [], 'radial_xy': [], 'axial': [],
                'moments_xz': [(m.position, m.magnitude)] if m.plane == LoadPlane.XZ else [],
                'moments_xy': [(m.position, m.magnitude)] if m.plane == LoadPlane.XY else [],
            })

        # Engrenagens — todas as forças de um gear numa única contribuição
        for gear in system.gears:
            lc = {
                'label': f"gear_{gear.label}@{gear.position}",
                'type': 'gear',
                'radial_xz': [(gear.position, -gear.tangential_force)],
                'radial_xy': [(gear.position, gear.radial_force)],
                'axial': [],
                'moments_xz': [],
                'moments_xy': [],
            }
            if gear.axial_force != 0.0:
                lc['axial'].append((gear.position, gear.axial_force))
            load_cases.append(lc)
        # ── Get sections positions and informations ──────────────
        shaft_sec = [
            (system.shaft.axial_start(i), s.second_moment_of_area, s.length, get_material(s.material_id).E)
            for i, s in enumerate(system.shaft.sections)
        ]

        # ── Create mesh and assemble system ────────────────────────────────────────────────────────
        x_nodes = self._create_mesh(system)
        elements = self._build_elements(x_nodes, system)

        # ── Assemble global stiffness matrix ────────────────────────────────────────────────────────
        k = self._assemble_stiffness(elements, len(x_nodes))

        # ── Solve per load contribution ──────────────────────────────────────
        K_red, _, free_dofs, constrained_dofs = self._apply_boundary_conditions(
            k, np.zeros(k.shape[0]), x_nodes, system.bearings
        )

        d_contributions = []
        for lc in load_cases:
            f_xz_i = self._assemble_load_vector(x_nodes, lc['radial_xz'], lc['axial'], lc['moments_xz'])
            f_xy_i = self._assemble_load_vector(x_nodes, lc['radial_xy'], [], lc['moments_xy'])
            d_xz_i = np.zeros(k.shape[0])
            d_xy_i  = np.zeros(k.shape[0])
            d_xz_i[free_dofs] = self._solve_system(K_red, f_xz_i[free_dofs])
            d_xy_i[free_dofs]  = self._solve_system(K_red, f_xy_i[free_dofs])
            d_contributions.append({
                'label': lc['label'], 'type': lc['type'],
                'd_xz': d_xz_i, 'd_xy': d_xy_i,
            })

        d_total_xz = sum(c['d_xz'] for c in d_contributions)
        d_total_xy  = sum(c['d_xy'] for c in d_contributions)
        f_xz_ext = k @ d_total_xz
        f_xy_ext  = k @ d_total_xy

        return StaticsResult(
            x_nodes=x_nodes,
            elements=elements,
            d_total_xz=d_total_xz,
            d_total_xy=d_total_xy,
            f_xz_ext=f_xz_ext,
            f_xy_ext=f_xy_ext,
            d_contributions=d_contributions,
        )


    # ── Private methods ───────────────────────────────────────────────────────
    # FEM StaticsSolver — apontamentos de implementação
    #
    # Elemento: frame element 3 DOF/nó (u, v, θ)
    # Matriz elemento: 6×6 (Euler-Bernoulli + axial)
    # Matriz global: 3(n+1) × 3(n+1), n = número de secções
    #
    # Pipeline:
    #   1. _euler_bernoulli_beam_equations(section) → K_elem 6×6
    #   2. _assemble_stiffness(shaft) → K_global
    #   3. _apply_boundary_conditions(K, bearings) → condensação nos nós de apoio
    #   4. _solve(K, f) → deslocamentos nodais
    #   5. _compute_internal_forces(displacements, shaft) → V, M, T, Fa
    #
    # Malha:
    #   Nós obrigatórios em:
    #     - fronteiras entre ShaftSections
    #     - posições de rolamentos (boundary conditions)
    #     - posições de cargas externas (RadialLoad, TorqueLoad, AxialLoad, GearElement)
    #   Secções subdivididas automaticamente quando necessário
    #   Malha não uniforme — sem problema, cada elemento usa o seu comprimento real

    # ── Future improvements ───────────────────────────────────────────────────────
    # TODO (futuro): suporte a cargas distribuídas em loads.py exige reavaliação
    #   do critério de malha (refinamento local por span) e funções auxiliares
    #   de integração para o vector de forças nodais equivalentes.


    def _create_mesh(self, system: MechanicalSystem) -> list[float]:
    # Recolher todas as posições obrigatórias
        x_mandatory = []

        # Fronteiras de secção
        x_mandatory += [system.shaft.axial_start(i) for i in range(system.shaft.n_sections)]
        x_mandatory.append(system.shaft.axial_end(system.shaft.n_sections - 1))

        # Rolamentos
        x_mandatory += [b.position for b in system.bearings]

        # Engrenagens
        x_mandatory += [g.position for g in system.gears]

        # Cargas
        x_mandatory += [l.position for l in system.radial_loads_xz]
        x_mandatory += [l.position for l in system.radial_loads_xy]
        x_mandatory += [l.position for l in system.axial_loads]
        x_mandatory += [l.position for l in system.torque_loads]
        x_mandatory += [m.position for m in system.external_moments]

        # Ordenar e remover duplicados/pontos próximos
        x_nodes = sorted(set(x_mandatory))
        x_nodes = [x for i, x in enumerate(x_nodes) if i == 0 or x - x_nodes[i-1] > MESH_MIN_NODE_DIST_MM]
        return x_nodes

    def _find_node_index(self, x_nodes: list[float], x: float, tol: float = MESH_MIN_NODE_DIST_MM) -> int:
        for i, xn in enumerate(x_nodes):
            if abs(xn - x) <= tol:
                return i
        raise ValueError(f"No node found at x={x:.4f} mm within tolerance {tol} mm")
    
    def _build_elements(self, x_nodes: list[float], system: MechanicalSystem) -> list[Elem]:
        elements = []
        for i, s in enumerate(system.shaft.sections):
            idx_start = self._find_node_index(x_nodes, system.shaft.axial_start(i))
            idx_end   = self._find_node_index(x_nodes, system.shaft.axial_end(i))
            for j in range(idx_start, idx_end):
                mat = get_material(s.material_id)
                elements.append(Elem(
                    length=x_nodes[j+1] - x_nodes[j],
                    E=mat.E * 1000, # N/mm^2
                    I=s.second_moment_of_area,
                    A=s.area,
                    v=mat.poisson_ratio,
                    idx_node_1=j,
                    idx_node_2=j + 1,
                ))
        return elements

    def _assemble_load_vector(self, x_nodes, radial_loads, axial_loads, moments) -> np.ndarray:
        f = np.zeros(3 * len(x_nodes))  # 3 DOF/nó: u, v, θ
        # Mapear cargas para o vetor de forças nodal
        for x, mag in axial_loads:
            i = self._find_node_index(x_nodes, x)
            f[3*i] += mag  # u DOF        
        for x, mag in radial_loads:
            i = self._find_node_index(x_nodes, x)
            f[3*i + 1] += mag  # v DOF
        for x, mag in moments:
            i = self._find_node_index(x_nodes, x)
            f[3*i + 2] += mag  # θ DOF
        return f
    
    def _assemble_stiffness(self, elements: list[Elem], n_nodes: int) -> np.ndarray:
        n_dofs = 3 * n_nodes
        K = np.zeros((n_dofs, n_dofs))
        _beam_theories = {
            "euler": EulerBernoulliBeam,
            "timoshenko": TimoshenkoBeam,
        }
        beam = _beam_theories[self.theory]()
        for elem in elements:
            K_elem = beam.stiffness_element(elem)
            dofs = [
                3*elem.idx_node_1,     # u1
                3*elem.idx_node_1 + 1, # v1
                3*elem.idx_node_1 + 2, # θ1
                3*elem.idx_node_2,     # u2
                3*elem.idx_node_2 + 1, # v2
                3*elem.idx_node_2 + 2, # θ2
            ]
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    K[gi, gj] += K_elem[i, j]
        return K

    def _apply_boundary_conditions(self, K: np.ndarray, f: np.ndarray, x_nodes: list[float], bearings: list[Bearing]) -> tuple[np.ndarray, np.ndarray]:
        constrained_dofs = []
        for b in bearings:
            i = self._find_node_index(x_nodes, b.position)
            constrained_dofs.append(3*i + 1)  # v = 0 sempre
            if b.arrangement == "fixed":
                constrained_dofs.append(3*i)  # u = 0
        
        free_dofs = [d for d in range(K.shape[0]) if d not in constrained_dofs]
        K_red = K[np.ix_(free_dofs, free_dofs)]
        f_red = f[free_dofs]
        return K_red, f_red, free_dofs, constrained_dofs

    def _solve_system(self, K_red: np.ndarray, f_red: np.ndarray) -> np.ndarray:
        return np.linalg.solve(K_red, f_red)


class EulerBernoulliBeam:
    def stiffness_element(self, elem: Elem) -> np.ndarray:
        le = elem.length
        E  = elem.E
        I  = elem.I
        A  = elem.A
        k  = np.zeros((6, 6))
        Rod_const   = E * A / le
        Beam_const  = E * I / (le**3)
        k[0, 0] = k[3, 3] = Rod_const
        k[3, 0] = k[0, 3] = - Rod_const
        k[1, 1] = k[4, 4] = Beam_const * 12
        k[1, 2] = k[1, 5] = k[2, 1] = k[5, 1] = Beam_const * 6 * le
        k[2, 2] = k[5, 5] = Beam_const * 4 * le**2
        k[4, 1] = k[1, 4] = - Beam_const * 12
        k[4, 2] = k[2, 4] = k[5, 4] = k[4, 5] = - Beam_const * 6 * le
        k[5, 2] = k[2, 5] = Beam_const * 2 * le**2
        return k
        
    def shape_functions(self, zeta: float, elem: Elem) -> np.ndarray:
        le = elem.length
        N = np.zeros(6)
        N[0] = 1/2 * (1 - zeta)  # axial
        N[1] = 1/4 * (2 - 3*zeta + zeta**3)  # transverse v
        N[2] = le/8 * (1 - zeta - zeta**2 + zeta**3)  # rotation θ
        N[3] = 1/2 * (1 + zeta)  # axial
        N[4] = 1/4 * (2 + 3*zeta - zeta**3)  # transverse v
        N[5] = le/8 * (-1 - zeta + zeta**2 + zeta**3)  # rotation θ
        return N
    
    def deformation_matrix(self, zeta: float, elem: Elem) -> np.ndarray:
        le = elem.length
        B = np.zeros((2, 6))
        B[0, 0] = -1/2  # axial strain ε_x = du/dx
        B[1, 1] = -3/4 * (1 - zeta**2) # curvature κ = d²v/dx²]
        B[1, 2] = -le/8 * (-1 - 2*zeta + 3*zeta**2) # curvature κ = d²v/dx²
        B[0, 3] = 1/2
        B[1, 4] = 3/4 * (1 - zeta**2) # curvature κ = d²v/dx²
        B[1, 5] = le/8 * (-1 + 2*zeta + 3*zeta**2) # curvature κ = d²v/dx²
        return B

    def elasticity_matrix(self, elem: Elem) -> np.ndarray:
        E = elem.E
        A = elem.A
        I = elem.I
        D = np.zeros((2, 2))
        D[0, 0] = E
        D[1, 1] = E
        return D
    # def consistent_load(...) -> np.ndarray: ...
    # def recover_internal_forces(...) -> tuple: ...

class TimoshenkoBeam:
    def stiffness_element(self, elem: Elem) -> np.ndarray:
        le = elem.length
        E  = elem.E
        I  = elem.I
        A  = elem.A
        shear_factor = 5/6
        v = elem.v
        G = E / (2 * (1 + v))
        k  = np.zeros((6, 6))
        Rod_const   = E * A / le
        Bending_const  = E * I / (le)
        Shear_const = shear_factor * E * G * A / le
        k[0, 0] = k[3, 3] = Rod_const
        k[3, 0] = k[0, 3] = - Rod_const
        k[1, 1] = k[4, 4] = Shear_const
        k[1, 2] = k[1, 5] = k[2, 1] = k[5, 1] = 1/2 * Shear_const * le
        k[2, 2] = k[5, 5] = Bending_const + 1/4 * Shear_const * le**2
        k[4, 1] = k[1, 4] = - Shear_const
        k[4, 2] = k[2, 4] = k[5, 4] = k[4, 5] = - 1/2 * Shear_const * le
        k[5, 2] = k[2, 5] = - Bending_const + 1/4 * Shear_const * le**2
        return k
    
    def shape_functions(self, zeta: float, elem: Elem) -> np.ndarray:
        # Implementation for Timoshenko beam shape functions
        le = elem.length
        N = np.zeros(6)
        N[0] = 1/2 * (1 - zeta)  # axial  
        N[1] = 1/2 * (1 - zeta)  # transverse v
        N[2] = 1/2 * (1 - zeta)  # rotation θ
        N[3] = 1/2 * (1 + zeta)  # axial     
        N[4] = 1/2 * (1 + zeta)  # transverse v
        N[5] = 1/2 * (1 + zeta)  # rotation θ
        return N 
        
    def deformation_matrix(self, zeta: float, elem: Elem) -> np.ndarray:
        le = elem.length
        B = np.zeros((3, 6))
        B[0, 0] = -1/2  # axial strain ε_x = du/dx
        B[1, 1] = -1/2 # curvature κ = d²v/dx²]
        B[2, 2] = -1/2 # curvature κ = d²v/dx²
        B[0, 3] = 1/2
        B[1, 4] = 1/2 # curvature κ = d²v/dx²
        B[2, 5] = 1/2 # curvature κ = d²v/dx²
        return B
    
    def elasticity_matrix(self, elem: Elem) -> np.ndarray:
        E = elem.E
        A = elem.A
        I = elem.I
        shear_factor = 5/6
        D = np.zeros((3, 3))
        D[0, 0] = E
        D[1, 1] = E
        D[2, 2] = shear_factor * E * A / (2 * (1 + elem.v))
        return D
    # def consistent_load(...) -> np.ndarray: ...
    # def recover_internal_forces(...) -> tuple: ...
    

# TODO (futuro): class BeamTheorySelector: — compara resultados Euler-Bernoulli vs Timoshenko
#   e selecciona automaticamente a teoria adequada com base no critério L/D do eixo.
    

class StressSolver:
    def __init__(self, theory: str = "euler"):
        self.theory = theory

    def solve(self, statics_result: StaticsResult, system: MechanicalSystem) -> StressResult:
        sigma = self._compute_stress_from_displacements(
            statics_result.d_total_xz, statics_result.d_total_xy, statics_result.elements
        )

        # internal forces — mantém lógica actual
        internal_forces = []
        _beam_theories = {"euler": EulerBernoulliBeam, "timoshenko": TimoshenkoBeam}
        beam = _beam_theories[self.theory]()
        calculated_nodes = set()
        for elem in statics_result.elements:
            dofs = [
                3*elem.idx_node_1, 3*elem.idx_node_1+1, 3*elem.idx_node_1+2,
                3*elem.idx_node_2, 3*elem.idx_node_2+1, 3*elem.idx_node_2+2,
            ]
            K_e = beam.stiffness_element(elem)
            f_xz = K_e @ statics_result.d_total_xz[dofs]
            f_xy  = K_e @ statics_result.d_total_xy[dofs]
            for idx, fi_xz, fi_xy, node_id, zeta in [
                (0, f_xz[:3], f_xy[:3], elem.idx_node_1, -1.0),
                (1, f_xz[3:], f_xy[3:], elem.idx_node_2,  1.0),
            ]:
                if node_id in calculated_nodes:
                    continue
                calculated_nodes.add(node_id)
                internal_forces.append((zeta, elem, fi_xz.copy(), fi_xy.copy()))

        # contribuições por fonte de carga
        sigma_contributions = None
        if statics_result.d_contributions:
            sigma_contributions = []
            for c in statics_result.d_contributions:
                s = self._compute_stress_from_displacements(
                    c['d_xz'], c['d_xy'], statics_result.elements
                )
                sigma_contributions.append({
                    'label': c['label'], 'type': c['type'], 'sigma': s
                })

        tau = self._compute_torsion(statics_result, system)

        return StressResult(sigma=sigma, internal_forces=internal_forces,
                            sigma_contributions=sigma_contributions, tau=tau)
        
    def _compute_stress_from_displacements(self, d_xz: np.ndarray, d_xy: np.ndarray, elements: list[Elem]) -> list[tuple]:
        _beam_theories = {
            "euler": EulerBernoulliBeam,
            "timoshenko": TimoshenkoBeam,
        }
        beam = _beam_theories[self.theory]()
        sigma = []
        calculated_nodes = set()

        for elem in elements:
            d = 2 * math.sqrt(elem.A / math.pi)
            y = d / 2
            dofs = [
                3*elem.idx_node_1, 3*elem.idx_node_1+1, 3*elem.idx_node_1+2,
                3*elem.idx_node_2, 3*elem.idx_node_2+1, 3*elem.idx_node_2+2,
            ]
            K_e = beam.stiffness_element(elem)
            f_xz = K_e @ d_xz[dofs]
            f_xy  = K_e @ d_xy[dofs]

            pairs = [(0, f_xz[:3], f_xy[:3], elem.idx_node_1, -1.0),
                    (1, f_xz[3:], f_xy[3:], elem.idx_node_2,  1.0)]

            for _, fi_xz, fi_xy, node_id, zeta in pairs:
                if node_id in calculated_nodes:
                    continue
                calculated_nodes.add(node_id)
                N_xz, V_xz, M_xz = fi_xz
                N_xy, V_xy, M_xy = fi_xy
                sigma_ax = N_xz / elem.A
                sigma_xz = M_xz * y / elem.I
                sigma_xy  = M_xy * y / elem.I
                sigma.append((zeta, elem, sigma_ax, sigma_xz, sigma_xy))

        return sigma
    
    def _compute_torsion(self, statics_result: StaticsResult, system: MechanicalSystem) -> list[tuple[float, float]]:
        torque_loads = [(l.position, l.magnitude) for l in system.torque_loads]
        for gear in system.gears:
            if gear.torque != 0.0:
                torque_loads.append((gear.position, gear.torque))

        tau_nodes = []
        for i, x in enumerate(statics_result.x_nodes):
            # torque acumulado à esquerda de x
            T = sum(mag for pos, mag in torque_loads if pos <= x)
            # secção no nó
            elem = next((e for e in statics_result.elements if e.idx_node_1 == i or e.idx_node_2 == i), None)
            if elem is None:
                tau_nodes.append((x, 0.0))
                continue
            d = 2 * math.sqrt(elem.A / math.pi)
            J = math.pi * d**4 / 32
            Wt = J / (d / 2)
            tau = T / Wt if Wt > 0 else 0.0
            tau_nodes.append((x, tau))


        return tau_nodes
    

    # com isto tenho as tensoes nos nós importantes onde existirá concentração de tensões ou pontos importantes
        # depois tenho de usar estes valores de sigma guardados e "brincar" com eles na parte de fadiga e fazer combinações com eles enquanto aplico as concentrações de tensão
            # Isto no postProcessing
        # depois coisas de material e analise será na FatigueSolver

        # Falta ainda fazer uma analise rápida à falha estática em que ja tenho as tensoes entao é so aplicar formulas rápidas e siga siga

class StaticFailureSolver: ...
    # analise de falha estática (yielding) usando os resultados da analise estática e propriedades do material 

class FatiguePostProcessing: ...
    # post-processing dos resultados de tensoes e aplicação dos coeficientes de concentração de tensao e das deformações reais causadas por eles
        # depois teremos ainda a aplicação dos fatores como R (sigma_m/sigma_a) de modo a ter as tensoes sentidas na fadiga e depois poder usar no seu calculo de vida util

class FatigueSolver: ...
    # analise de fadiga usando os resultados da analise de tensoes e deformações e propriedades do material, incluindo a vida util estimada e fatores de segurança contra fadiga

