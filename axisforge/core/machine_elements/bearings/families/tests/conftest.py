# conftest.py
"""Fixtures partilhadas pelos testes de families/."""
import pytest
import numpy as np

# ---------------------------------------------------------------------
# geometrias de referência -- plausíveis, não de catálogo real, só para
# exercitar o código.
# ---------------------------------------------------------------------

DEEP_GROOVE_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, E=210_000.0, s=0.02, nu=0.3, i=1)
ANGULAR_CONTACT_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, E=210_000.0, alpha_0_deg=25.0, nu=0.3, i=1)

THRUST_ROW_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, E=210_000.0, alpha_0_deg=90.0, nu=0.3)
THRUST_ROW_KWARGS_NON_90 = dict(Dw=8.0, Dpw=40.0, Z=12, E=210_000.0, alpha_0_deg=60.0, nu=0.3)

CYL_ROLLER_KWARGS = dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40, alpha_0_deg=0.0, i=1)

THRUST_CYL_ROLLER_KWARGS = dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40, alpha_0_deg=90.0, i=1)
THRUST_CYL_ROLLER_KWARGS_NON_90 = dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40, alpha_0_deg=70.0, i=1)


# ---------------------------------------------------------------------
# catalog fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def catalog():
    """Placeholder genérico -- assemble_geometry recebe `catalog` mas a
    maioria dos ramos ainda não o lê. Ajusta quando isso deixar de ser
    verdade."""
    return object()


@pytest.fixture
def catalog_floating():
    """CylindricalRollerFamily valida catalog.arrangement -- só esta
    família precisa de mais do que um object() vazio."""
    class _Catalog:
        arrangement = "floating"
        label = None
        designation = "TEST-NU"
    return _Catalog()


# ---------------------------------------------------------------------
# families -- uma instância por família concreta
# ---------------------------------------------------------------------

@pytest.fixture
def deep_groove_family():
    from axisforge.core.machine_elements.bearings.families.family import DeepGrooveBallFamily
    return DeepGrooveBallFamily()

@pytest.fixture
def angular_contact_family():
    from axisforge.core.machine_elements.bearings.families.family import AngularContactFamily
    return AngularContactFamily()

@pytest.fixture
def self_aligning_family():
    from axisforge.core.machine_elements.bearings.families.family import SelfAligningBallFamily
    return SelfAligningBallFamily()

@pytest.fixture
def thrust_single_row_family():
    from axisforge.core.machine_elements.bearings.families.family import ThrustBallSingleRowFamily
    return ThrustBallSingleRowFamily()

@pytest.fixture
def thrust_multi_row_family():
    from axisforge.core.machine_elements.bearings.families.family import ThrustBallMultiRowFamily
    return ThrustBallMultiRowFamily()

@pytest.fixture
def cylindrical_roller_family():
    from axisforge.core.machine_elements.bearings.families.family import CylindricalRollerFamily
    return CylindricalRollerFamily()

@pytest.fixture
def thrust_cyl_roller_family():
    from axisforge.core.machine_elements.bearings.families.family import ThrustCylindricalRollerFamily
    return ThrustCylindricalRollerFamily()

@pytest.fixture
def thrust_cyl_roller_multi_row_family():
    from axisforge.core.machine_elements.bearings.families.family import RollerThrustMultiRowFamily
    return RollerThrustMultiRowFamily()


# ---------------------------------------------------------------------
# geometrias já montadas -- atalhos usados em vários testes
# ---------------------------------------------------------------------

@pytest.fixture
def assembled_deep_groove(deep_groove_family, catalog):
    return deep_groove_family.assemble_geometry(catalog, **DEEP_GROOVE_KWARGS)

@pytest.fixture
def assembled_angular_contact(angular_contact_family, catalog):
    return angular_contact_family.assemble_geometry(catalog, **ANGULAR_CONTACT_KWARGS)

@pytest.fixture
def assembled_row_90deg(thrust_single_row_family, catalog):
    return thrust_single_row_family.assemble_geometry(catalog, **THRUST_ROW_KWARGS)

@pytest.fixture
def assembled_row_non_90deg(thrust_single_row_family, catalog):
    return thrust_single_row_family.assemble_geometry(catalog, **THRUST_ROW_KWARGS_NON_90)

@pytest.fixture
def assembled_cyl_roller(cylindrical_roller_family, catalog_floating):
    return cylindrical_roller_family.assemble_geometry(catalog_floating, **CYL_ROLLER_KWARGS)

@pytest.fixture
def assembled_thrust_cyl_roller_90(thrust_cyl_roller_family, catalog):
    return thrust_cyl_roller_family.assemble_geometry(catalog, **THRUST_CYL_ROLLER_KWARGS)

@pytest.fixture
def assembled_thrust_cyl_roller_non90(thrust_cyl_roller_family, catalog):
    return thrust_cyl_roller_family.assemble_geometry(catalog, **THRUST_CYL_ROLLER_KWARGS_NON_90)


class RowAsBearing:
    """Adaptador: um dict de assemble_geometry() servido como se fosse
    um Bearing (getattr em vez de __getitem__), para chamar
    dynamic_capacity(bearing)/per_element_dynamic_capacity(bearing)."""
    def __init__(self, row: dict):
        self.__dict__.update(row)