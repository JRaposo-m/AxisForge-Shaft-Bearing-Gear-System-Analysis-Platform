"""
axisforge/solvers/machine_elements/shaft/static_solvers/torsion.py

No DOF, no stiffness matrix, no dependency on Timoshenko vs.
Euler-Bernoulli -- it is pure statics over the same x_nodes: T(x) =
cumulative sum of TorqueLoad up to x, tau(x) = T(x)/Wt(x) via
Shaft.Wt_at(). Kept in its own module (a sibling of fem_solvers/, not
inside it) because it is genuinely independent of everything else --
it does not even need the FEM elements.

No longer called from rigid_support.py::RigidSupportFEMSolver.solve()
-- that solver only produces bending + axial results now (see its own
docstring). solve_torsion() is called independently by whoever
orchestrates a full result set for a shaft (e.g.
ShaftResultsReader.read(), which takes its (T_total, tau_total,
torsion_contributions) as explicit arguments -- see
results_reader.py). validate_torsion_equilibrium() is meant to be
called separately too, same as before.
"""

from __future__ import annotations

import numpy as np

from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem
from axisforge.config import SOLVER_TOLERANCE


def solve_torsion(
    shaft_system: ShaftSystem, x_nodes: list[float]
) -> tuple[np.ndarray, np.ndarray, list[list[dict]]]:
    """
    T(x) = sum of TorqueLoad.magnitude for every source at position <= x.

    Sign convention fixed upstream in GearSystem._forces_to_loads:
    driver mesh point -> +T_in ; driven mesh point -> -T_out. A shaft
    in equilibrium (net torque ~0) should return T(x)->0 at the free
    end -- see validate_torsion_equilibrium() for the explicit check.

    Units: TorqueLoad.magnitude is N*m (per loads.py); Wt_at returns
    mm^3 -> tau = T[N*m]*1000 / Wt[mm^3] = N/mm^2 = MPa.
    """
    torque_sources = sorted(
        (
            (ld.position, ld.magnitude, ld.label or f"torque@{ld.position:.1f}")
            for ld in shaft_system.torque_loads
        ),
        key=lambda t: t[0],
    )

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


def validate_torsion_equilibrium(
    shaft_system: ShaftSystem, tol: float = 1e-6
) -> list[str]:
    """
    Checks that the net applied torque (sum of every
    TorqueLoad.magnitude) is ~0 -- a non-zero net usually means a
    missing or mis-signed load (e.g. the driven-side reaction torque
    was never added, or GearSystem._forces_to_loads produced the wrong
    sign for one of the sources).
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