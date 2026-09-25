# conftest.py
"""Shared fixtures for the tests in families/.

Bearing-side geometry parameters (Dw, Dpw, Z, ...) are kept in the
KWARGS dictionaries below, plausible but not real catalogue values --
they only need to exercise the code. Material/contact-surface data is
no longer part of these dictionaries: since base.py/family.py were
updated to receive materials as a ``surfaces`` argument (a
``RadialSurfaces`` built from ``core.materials.Material``) instead of
raw ``E``/``nu`` floats, a single reusable steel surface set is
provided below via the ``steel_material``/``steel_surfaces`` fixtures
and merged into each assembled-geometry fixture that needs it.
"""
import pytest
import numpy as np

from axisforge.core.materials import Material, IsotropicElastic
from axisforge.core.machine_elements.bearings.families.bearing_properties import RadialSurfaces

# ---------------------------------------------------------------------
# geometrias de referência -- plausíveis, não de catálogo real, só para
# exercitar o código. E/nu deixaram de viver aqui: entram via
# `surfaces` (ver steel_surfaces, abaixo) para as famílias radiais que
# já foram migradas (DeepGroove, AngularContact, SelfAligning,
# CylindricalRoller). As famílias de thrust ainda recebem E/nu
# diretamente -- não foram tocadas nesta mudança.
# ---------------------------------------------------------------------

DEEP_GROOVE_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, s=0.02, i=1)
ANGULAR_CONTACT_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=25.0, i=1)
SELF_ALIGNING_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=25.0, i=1)

THRUST_ROW_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, E=210_000.0, alpha_0_deg=90.0, nu=0.3)
THRUST_ROW_KWARGS_NON_90 = dict(Dw=8.0, Dpw=40.0, Z=12, E=210_000.0, alpha_0_deg=60.0, nu=0.3)

CYL_ROLLER_KWARGS = dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40, alpha_0_deg=0.0, i=1)

THRUST_CYL_ROLLER_KWARGS = dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40, alpha_0_deg=90.0, i=1)
THRUST_CYL_ROLLER_KWARGS_NON_90 = dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40, alpha_0_deg=70.0, i=1)


# ---------------------------------------------------------------------
# materials / surfaces
# ---------------------------------------------------------------------

@pytest.fixture
def steel_material():
    """Plain isotropic steel, only E/nu are exercised by contact code."""
    return Material(
        material_id="test_steel",
        density=7850.0,
        elastic=IsotropicElastic(E=210_000.0, poisson_ratio=0.3),
    )


@pytest.fixture
def steel_surfaces(steel_material):
    """Uniform RadialSurfaces (rolling element == inner == outer == steel),
    the same material on every surface, matching what the old flat
    E=210_000.0/nu=0.3 kwargs implied before the surfaces-based rewrite."""
    return RadialSurfaces.uniform(steel_material)


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
def assembled_deep_groove(deep_groove_family, catalog, steel_surfaces):
    return deep_groove_family.assemble_geometry(catalog, surfaces=steel_surfaces, **DEEP_GROOVE_KWARGS)

@pytest.fixture
def assembled_angular_contact(angular_contact_family, catalog, steel_surfaces):
    return angular_contact_family.assemble_geometry(catalog, surfaces=steel_surfaces, **ANGULAR_CONTACT_KWARGS)

@pytest.fixture
def assembled_row_90deg(thrust_single_row_family, catalog):
    return thrust_single_row_family.assemble_geometry(catalog, **THRUST_ROW_KWARGS)

@pytest.fixture
def assembled_row_non_90deg(thrust_single_row_family, catalog):
    return thrust_single_row_family.assemble_geometry(catalog, **THRUST_ROW_KWARGS_NON_90)

@pytest.fixture
def assembled_cyl_roller(cylindrical_roller_family, catalog_floating, steel_surfaces):
    return cylindrical_roller_family.assemble_geometry(catalog_floating, surfaces=steel_surfaces, **CYL_ROLLER_KWARGS)

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
