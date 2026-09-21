"""Fixtures partilhadas pelos testes de families/."""
import pytest
import numpy as np

# geometria de referência para uma thrust ball bearing "normal" --
# valores plausíveis, não de catálogo real; só para exercitar o código.
THRUST_ROW_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, E=210_000.0, alpha_0_deg=90.0, nu=0.3)
THRUST_ROW_KWARGS_NON_90 = dict(Dw=8.0, Dpw=40.0, Z=12, E=210_000.0, alpha_0_deg=60.0, nu=0.3)


@pytest.fixture
def catalog():
    """Placeholder -- assemble_geometry recebe `catalog` mas (por agora)
    não o usa no ramo thrust. Ajusta quando isso deixar de ser verdade."""
    return object()


# conftest.py
@pytest.fixture
def thrust_single_row_family():
    from axisforge.core.machine_elements.bearings.families.family import ThrustBallSingleRowFamily
    return ThrustBallSingleRowFamily()

@pytest.fixture
def thrust_multi_row_family():
    from axisforge.core.machine_elements.bearings.families.family import ThrustBallMultiRowFamily
    return ThrustBallMultiRowFamily()


@pytest.fixture
def assembled_row_90deg(thrust_single_row_family, catalog):
    return thrust_single_row_family.assemble_geometry(catalog, **THRUST_ROW_KWARGS)


@pytest.fixture
def assembled_row_non_90deg(thrust_single_row_family, catalog):
    return thrust_single_row_family.assemble_geometry(catalog, **THRUST_ROW_KWARGS_NON_90)