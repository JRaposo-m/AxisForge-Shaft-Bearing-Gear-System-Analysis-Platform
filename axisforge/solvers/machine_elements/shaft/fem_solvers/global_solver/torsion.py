"""Torsion of the shaft by statics: torque, nominal torsional shear stress and twist angle at the
mesh nodes. No DOF and no stiffness matrix are involved, so the result does not depend on the beam
theory. Called by ``build_shaft_result`` (not by ``RigidSupportFEMSolver.solve`` directly).
"""

from __future__ import annotations

import numpy as np

from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem
from axisforge.core.materials import get_material
from axisforge.config import SOLVER_TOLERANCE


class TorsionSolver:
    """Torque T(x), torsional shear stress tau(x) and twist angle phi(x) along the shaft.

    Warnings
    --------
    Torque units: ``TorqueLoad.magnitude`` and the returned T are [N m]; the module docstring of
    ``core/loads.py`` says N mm (open review item).

    Notes
    -----
        T(x) = sum of T_k over the sources with pos <= x;  tau(x) = T(x) / W_t(x);
        d(phi)/dx = T(x) / (G J(x))

    with ``T_k`` = ``TorqueLoad.magnitude`` [N m], ``W_t`` = ``Shaft.Wt_at`` [mm^3], ``G`` [MPa] and
    ``J`` = ``Shaft.J_at`` [mm^4]. The torque-twist relation ``torque = G J dtheta_x/dx`` is
    Onate2013_vol2, 4.3, Eq. (4.27b), (4.34), (4.37a). ``tau = T / W_t`` with ``W_t = J / (d/2)`` is
    the maximum shear stress of a circular section in free torsion: Onate2013_vol2, Fig. 4.11 (thick
    cylinder, ``tau = 2 r_o M_x / (pi (r_o^4 - r_i^4))``; the full-circle entry of that figure is
    misprinted). Nominal value: no stress concentration.

    At the node of a torque source, T and tau are the values just to the right of the source (the
    source is already included).

    Sign: fixed upstream in ``GearSystem._forces_to_loads`` (driver mesh point +T_in, driven mesh
    point -T_out); a shaft in equilibrium returns T -> 0 at the free end.
    """

    def solve(
        self, shaft_system: ShaftSystem, x_nodes: list[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[dict]]]:
        """Torque, shear stress and twist angle at every node.

        Parameters
        ----------
        shaft_system : ShaftSystem
            Provides ``torque_loads`` ([N m] at positions [mm]) and the shaft sections.
        x_nodes : list of float
            Node positions [mm].

        Returns
        -------
        T_total : ndarray, shape (n,)
            Torque transmitted at each node [N m].
        tau_total : ndarray, shape (n,)
            Nominal torsional shear stress ``T / W_t`` at each node [MPa].
        phi_total : ndarray, shape (n,)
            Twist angle relative to the first node [rad].
        contributions : list of list of dict
            Per node, one dict per torque source already applied (``"label"``, ``"T"`` [N m],
            ``"tau"`` [MPa]).
        """
        T_total, tau_total, contributions = self._torque_and_stress(shaft_system, x_nodes)
        phi_total = self._twist_angle(shaft_system, x_nodes, T_total)
        return T_total, tau_total, phi_total, contributions

    def _torque_sources(self, shaft_system: ShaftSystem) -> list[tuple[float, float, str]]:
        """Torque loads as ``(position [mm], magnitude [N m], label)`` tuples sorted by position.
        """
        return sorted(
            (
                (ld.position, ld.magnitude, ld.label or f"torque@{ld.position:.1f}")
                for ld in shaft_system.torque_loads
            ),
            key=lambda t: t[0],
        )

    def _torque_and_stress(self, shaft_system: ShaftSystem, x_nodes: list[float]):
        """Torque, shear stress and per-source contributions at every node.

        ``T(x)`` is the sum of ``TorqueLoad.magnitude`` over the sources at ``pos <= x + SOLVER_TOLERANCE``.
        Units: T [N m], ``W_t`` [mm^3], ``tau = T * 1000 / W_t`` [MPa] (the factor 1000 converts N m to
        N mm).

        Parameters
        ----------
        shaft_system : ShaftSystem
            Torque loads and shaft sections.
        x_nodes : list of float
            Node positions [mm].

        Returns
        -------
        T_total, tau_total : ndarray, shape (n,)
            Torque [N m] and shear stress [MPa] at each node.
        contributions : list of list of dict
            Per node, the sources included in T.
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
        """Twist angle at every node by node-to-node integration of ``d phi/dx = T / (G J)``.

        ``phi(x_0) = 0`` (arbitrary reference: a free shaft has no fixed end). For each segment
        ``phi_{i+1} = phi_i + T_i * dx_i / (G_i * J_i)`` with ``T_i = T_total[i]`` (the torque at the
        segment start node) and ``G_i``, ``J_i`` evaluated at that node, so T, G and J are piecewise
        constant over each segment (Onate2013_vol2, 4.3, Eq. (4.27b), (4.34), (4.37a)).

        Parameters
        ----------
        shaft_system : ShaftSystem
            Provides the sections (material) and ``shaft.J_at``.
        x_nodes : list of float
            Node positions [mm].
        T_total : ndarray, shape (n,)
            Torque at each node [N m]; converted internally to N mm (times 1000).

        Returns
        -------
        ndarray, shape (n,)
            Twist angle [rad]; ``G = E / (2 (1 + nu))`` [MPa] of the section material at the start node,
            ``J`` [mm^4].
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
        """Check that the net applied torque is zero.

        A non-zero net torque usually means a missing or mis-signed load (for example the driven-side
        reaction torque was not added).

        Parameters
        ----------
        shaft_system : ShaftSystem
            Provides ``torque_loads`` ([N m]).
        tol : float, optional
            Accepted absolute net torque [N m]. Default 1e-6.

        Returns
        -------
        list of str
            Empty if ``abs(sum of magnitudes) <= tol``, otherwise one error message.
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