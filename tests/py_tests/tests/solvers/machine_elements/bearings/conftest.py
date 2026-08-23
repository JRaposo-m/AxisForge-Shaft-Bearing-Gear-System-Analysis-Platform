"""
tests/solvers/machine_elements/bearings/conftest.py

Fixtures for the ISO/TS 16281 solver package.

The FEM boundary
----------------
The bearing solvers read exactly two things out of a
SimpleFEMResultsLibrary: `library.get(shaft_system.name)` and, off that,
`.bearing_nodes` -- a list of BearingNodeData matched to bearings by label.
Nothing else. So these fixtures build a REAL SimpleFEMResultsLibrary holding
a REAL ShaftResults whose bearing_nodes are prescribed by hand, instead of
running SimpleFEMSolver to produce them.

That is deliberate, and it is the right seam: a bearing-solver test that
needs the FEM solver to be correct first cannot tell you which of the two
broke. Prescribing the nodal state makes the input exactly known, so any
discrepancy in the output belongs to the bearing solver. Coupled FEM ->
bearing behaviour is a separate, integration-level question.

ASCII only.
"""

from __future__ import annotations

import numpy as np
import pytest

from axisforge.core.machine_elements.Shaft.shaft import Shaft, ShaftSection
from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.families.ball_bearing.radial.subtypes.deep_groove_ball import (
    DeepGrooveBallFamily,
)
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import (
    ShaftSystem,
)
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import (
    BearingNodeData,
    ShaftResults,
    SimpleFEMResultsLibrary,
)


SHAFT_NAME = "brg_test_shaft"
DGBB_GEOMETRY = dict(Dw=7.94, Dpw=33.5, Z=8, E=206_000.0, nu=0.3, s=0.010)


def _make_dgbb(label: str, position: float, arrangement: str = "locating") -> Bearing:
    """A solver-ready 6204 at `position`, assembled with point_contact on."""
    return Bearing.assemble(
        family=DeepGrooveBallFamily(),
        catalog=BearingCatalog(
            d=20.0, D=47.0, b=14.0, C=12_700.0, C0=6_550.0,
            designation="6204", position=position,
            arrangement=arrangement, label=label,
        ),
        geometry=DGBB_GEOMETRY,
        analyses={"point_contact": True},
    )


@pytest.fixture
def dgbb_factory():
    """Factory fixture -- call it to build extra bearings inside a test."""
    return _make_dgbb


@pytest.fixture
def brg_A() -> Bearing:
    """Locating 6204 at x = 50 mm."""
    return _make_dgbb("A", 50.0, "locating")


@pytest.fixture
def brg_B() -> Bearing:
    """Floating 6204 at x = 260 mm."""
    return _make_dgbb("B", 260.0, "floating")


@pytest.fixture
def brg_shaft_system(brg_A, brg_B) -> ShaftSystem:
    """
    Uniform 300 mm shaft carrying both bearings. Extents [43, 57] and
    [253, 267] -- no overlap, so ShaftSystem.validate() is clean on the
    axial side.
    """
    shaft = Shaft(label="brg_shaft")
    shaft.add_section(ShaftSection(length=300.0, diameter=50.0, label="body"))

    sys = ShaftSystem(shaft, name=SHAFT_NAME, speed_rpm=1500.0)
    sys.add_bearing(brg_A)
    sys.add_bearing(brg_B)
    return sys


def _node(label: str, position: float, Fr_xz: float, Fr_xy: float,
          Fa: float = 0.0, psi_xz: float = 0.0, psi_xy: float = 0.0,
          v_xz: float = 0.0, v_xy: float = 0.0, u: float = 0.0) -> BearingNodeData:
    return BearingNodeData(
        label=label, position=position,
        u=u, v_xz=v_xz, v_xy=v_xy,
        Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fr=float(np.hypot(Fr_xz, Fr_xy)), Fa=Fa,
        psi_xz=psi_xz, psi_xy=psi_xy,
    )


@pytest.fixture
def bearing_node_factory():
    """Factory fixture for building BearingNodeData inside a test."""
    return _node


@pytest.fixture
def fem_library() -> SimpleFEMResultsLibrary:
    """
    A real SimpleFEMResultsLibrary holding a prescribed nodal state:

        A : Fr_xz = 2000 N, Fr_xy =    0 N, Fa = 300 N   (locating)
        B : Fr_xz =  800 N, Fr_xy = 600 N, Fa =   0 N    (floating)

    A is loaded purely in XZ so phi_Fr = 0 there, which makes the
    local/global frame distinction easy to reason about in assertions.
    B is loaded off-axis so the phi_Fr rotation is actually exercised.
    """
    library = SimpleFEMResultsLibrary()
    library.store(ShaftResults(
        name=SHAFT_NAME,
        bearing_nodes=[
            _node("A", 50.0, Fr_xz=2000.0, Fr_xy=0.0, Fa=300.0,
                  psi_xz=1.0e-4, v_xz=0.004),
            _node("B", 260.0, Fr_xz=800.0, Fr_xy=600.0, Fa=0.0,
                  psi_xz=0.5e-4, psi_xy=0.3e-4, v_xz=0.002, v_xy=0.0015),
        ],
    ))
    return library


@pytest.fixture
def bearings_by_label(brg_A, brg_B) -> dict:
    """The {label: Bearing} mapping every solver entry point takes."""
    return {"A": brg_A, "B": brg_B}


# ===========================================================================
# The independent equilibrium check
# ===========================================================================

def recover_loads(bearing, row) -> tuple[float, float]:
    """
    Recompute (Fr, Fa) from a converged row's own delta_j / alpha_j, using
    ISO/TS 16281 Sec 4.2.2.1 equilibrium:

        Fr = cp * sum(delta_j^1.5 * cos(alpha_j) * cos(phi_j))
        Fa = cp * sum(delta_j^1.5 * sin(alpha_j))

    This is the INDEPENDENT reference for the whole solver suite. It uses
    only the solver's published output (delta_j, alpha_j) and the bearing's
    own geometry (cp, phi_j) -- never an intermediate the solver kept to
    itself. Comparing the result against the loads that went in is the one
    assertion here that cannot pass by being self-consistently wrong.
    """
    d32 = np.maximum(row.delta_j, 0.0) ** 1.5
    Fr = float(bearing.cp * np.sum(d32 * np.cos(row.alpha_j) * np.cos(bearing.phi_j)))
    Fa = float(bearing.cp * np.sum(d32 * np.sin(row.alpha_j)))
    return Fr, Fa


@pytest.fixture
def recovered_loads():
    """Expose recover_loads() as a fixture for readability inside tests."""
    return recover_loads
