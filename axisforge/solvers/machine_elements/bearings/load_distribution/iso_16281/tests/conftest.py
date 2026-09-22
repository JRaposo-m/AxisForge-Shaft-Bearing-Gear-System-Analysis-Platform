# conftest.py
"""Fixtures partilhadas pelos testes de
solvers/machine_elements/bearings/load_distribution/iso_16281/.

Usa SimpleNamespace como "Bearing"/"node"/"ShaftResults" fake em vez de
montar a hierarquia real (Bearing, ShaftSystem, ShaftResults) -- os
solvers e o postprocessing aqui só fazem duck typing sobre estes
objetos (getattr, nunca isinstance), tal como o resto do projeto."""
from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------
# geometrias de referência -- plausíveis, não de catálogo real, só para
# exercitar o código e verificar contrato/equilíbrio, não valores
# certificados contra a norma.
# ---------------------------------------------------------------------

BALL_BEARING_KWARGS = dict(
    Z=8,
    Dpw=40.0,
    cp=5.0e4,
    A=0.05,
    alpha_0=np.radians(15.0),
    Ri=20.0,
    phi_j=np.arange(8) * (2.0 * np.pi / 8),
    label="B1",
    arrangement=None,
)

ROLLER_BEARING_KWARGS = dict(
    Z=12,
    Dwe=8.0,
    Lwe=8.0,
    Dpw=60.0,
    phi_j=np.arange(12) * (2.0 * np.pi / 12),
    s=0.01,
    n_s=30,
    x_k=np.linspace(-3.5, 3.5, 30),
    cL=2.0e4,
    cs=2.0e4 / 30,
    alpha_0=0.0,
    P_xk=np.zeros(30),
    label="B2",
    arrangement=None,
)


@pytest.fixture
def ball_bearing():
    return SimpleNamespace(**BALL_BEARING_KWARGS)


@pytest.fixture
def roller_bearing():
    return SimpleNamespace(**ROLLER_BEARING_KWARGS)


def make_node(Fr_xz=0.0, Fr_xy=0.0, Fa=0.0, v_xz=0.0, v_xy=0.0, u=0.0,
              psi_xz=0.0, psi_xy=0.0, label="B1"):
    return SimpleNamespace(Fr_xz=Fr_xz, Fr_xy=Fr_xy, Fa=Fa, v_xz=v_xz, v_xy=v_xy,
                            u=u, psi_xz=psi_xz, psi_xy=psi_xy, label=label)


def make_shaft_results(*nodes):
    return SimpleNamespace(bearing_nodes=list(nodes))
