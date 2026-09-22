"""
axisforge/solvers/machine_elements/shaft/fem_solvers/global_solver/torsion.py

No DOF, no stiffness matrix, no dependency on Timoshenko vs.
Euler-Bernoulli -- it is pure statics over the same x_nodes: T(x) =
cumulative sum of TorqueLoad up to x, tau(x) = T(x)/Wt(x) via
Shaft.Wt_at(), and phi(x) = twist angle, integrated node-to-node from
d(phi)/dx = T(x)/(G*J(x)).

MOVED (this pass): out of static_solvers/ into global_solver/, sibling
to rigid_support.py and postprocessing.py -- still genuinely
independent of everything else here (doesn't even need the FEM
elements), the move is only about physical location matching the rest
of this pass's folder cleanup.

Called from global_solver/postprocessing.py's build_shaft_result() --
NOT from rigid_support.py's solve(), which stays bending+axial only
(see that module's docstring). TorsionSolver.solve() is called
independently there, the same way ShaftResultsReader.read() used to
call it directly.

MISSING PIECE -- flagging rather than guessing: _twist_angle() below
calls shaft_system.shaft.J_at(x), a polar-second-moment-of-area accessor
that does not exist yet alongside diameter_at()/W_at()/Wt_at() (as far
as I've seen of the Shaft class). This file will raise AttributeError
until that's added. If Wt_at() is already J/r_outer for your section
definitions, J_at() could just be Wt_at(x) * (diameter_at(x) / 2) --
but I don't have Shaft's source to confirm that relationship holds for
both solid and hollow sections, so I'm not assuming it here. Still open
from the previous pass -- unchanged.
"""

from __future__ import annotations

import numpy as np

from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem
from axisforge.core.materials import get_material
from axisforge.config import SOLVER_TOLERANCE


class TorsionSolver:
    """
    Pure statics over x_nodes -- see module docstring. solve() returns
    everything build_shaft_result() needs for the torsional part of
    ShaftResults: T(x), tau(x), phi(x), and the per-node load
    contributions breakdown.
    """

    def solve(
        self, shaft_system: ShaftSystem, x_nodes: list[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[dict]]]:
        """
        Returns (T_total, tau_total, phi_total, contributions), each of
        T_total/tau_total/phi_total a length-n np.ndarray aligned with
        x_nodes.
        """
        T_total, tau_total, contributions = self._torque_and_stress(shaft_system, x_nodes)
        phi_total = self._twist_angle(shaft_system, x_nodes, T_total)
        return T_total, tau_total, phi_total, contributions

    def _torque_sources(self, shaft_system: ShaftSystem) -> list[tuple[float, float, str]]:
        return sorted(
            (
                (ld.position, ld.magnitude, ld.label or f"torque@{ld.position:.1f}")
                for ld in shaft_system.torque_loads
            ),
            key=lambda t: t[0],
        )

    def _torque_and_stress(self, shaft_system: ShaftSystem, x_nodes: list[float]):
        """
        T(x) = sum of TorqueLoad.magnitude for every source at position <= x.

        Sign convention fixed upstream in GearSystem._forces_to_loads:
        driver mesh point -> +T_in ; driven mesh point -> -T_out. A shaft
        in equilibrium (net torque ~0) should return T(x)->0 at the free
        end -- see validate_equilibrium() for the explicit check.

        Units: TorqueLoad.magnitude is N*m (per loads.py); Wt_at returns
        mm^3 -> tau = T[N*m]*1000 / Wt[mm^3] = N/mm^2 = MPa.
        """
        torque_sources = self._torque_sources(shaft_system)

        n = len(x_nodes)
        T_total = np.zeros(n)
        tau_total = np.zeros(n)
        contributions: list[list[dict]] = []

        for i, x in enumerate(x_nodes):
            Wt_x = shaft_system.shaft.Wt_at(x)
            node_contribs = []
            for pos, mag, label in torque_sources:
                if pos <= x + SOLVER_TOLERANCE:
                    node_contribs.append({
                        "label": label,
                        "T": mag,
                        "tau": mag * 1000.0 / Wt_x,
                    })
            contributions.append(node_contribs)
            T_total[i] = sum(c["T"] for c in node_contribs)
            tau_total[i] = T_total[i] * 1000.0 / Wt_x

        return T_total, tau_total, contributions

    def _twist_angle(self, shaft_system: ShaftSystem, x_nodes: list[float], T_total: np.ndarray) -> np.ndarray:
        """
        phi(x_nodes[0]) = 0 -- reference at the first node. A
        free-floating shaft in torsion has no absolute reference point
        the way a cantilever's fixed end does, so this is arbitrary;
        say if you want it referenced elsewhere (e.g. zeroed at a
        specific bearing) instead.

        Node-to-node cumulative integration, mirroring how T(x) is
        already a cumulative sum: phi(x_i+1) = phi(x_i) + T_i * dx_i /
        (G_i * J_i), with T_i = T_total[i] and G_i, J_i evaluated at the
        segment's start node x_i (piecewise-constant across each
        segment -- same assumption _torque_and_stress() already makes
        for T(x) itself between torque application points).

        G_i = E/(2*(1+v)) at x_i -- same formula already used in
        TimoshenkoBeam.stiffness_element() -- via the section's material
        at x_i (shaft.section_at(x) already used elsewhere, e.g.
        Elem.from_x_nodes()).

        J_i = shaft.J_at(x_i) -- see the MISSING PIECE note in the
        module docstring; this call does not exist yet.

        Units: T_total is N*m -> *1000 = N*mm. x_nodes assumed mm
        (consistent with E in MPa = N/mm^2, I/J in mm^4 elsewhere in
        this codebase). phi_segment [rad] = T[N*mm] * dx[mm] /
        (G[N/mm^2] * J[mm^4]).
        """
        n = len(x_nodes)
        phi = np.zeros(n)

        for i in range(n - 1):
            x_i, x_ip1 = x_nodes[i], x_nodes[i + 1]
            dx = x_ip1 - x_i

            section, _ = shaft_system.shaft.section_at(x_i)
            mat = get_material(section.material_id)
            G = mat.E / (2.0 * (1.0 + mat.poisson_ratio))
            J = shaft_system.shaft.J_at(x_i)

            T_i_N_mm = T_total[i] * 1000.0
            phi[i + 1] = phi[i] + T_i_N_mm * dx / (G * J)

        return phi

    def validate_equilibrium(self, shaft_system: ShaftSystem, tol: float = 1e-6) -> list[str]:
        """
        Checks that the net applied torque (sum of every
        TorqueLoad.magnitude) is ~0 -- a non-zero net usually means a
        missing or mis-signed load (e.g. the driven-side reaction torque
        was never added, or GearSystem._forces_to_loads produced the
        wrong sign for one of the sources).
        """
        errors: list[str] = []
        net = sum(ld.magnitude for ld in shaft_system.torque_loads)

        if abs(net) > tol:
            errors.append(
                f"Torsional equilibrium violated: net torque = {net:.6f} N*m "
                f"(tol = {tol:.1e}). Check that every TorqueLoad source "
                f"(driver and driven) was added with the correct sign."
            )

        return errors