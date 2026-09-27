# conftest.py
"""Shared fixtures for the tests in families/.

Bearing-side geometry parameters (Dw, Dpw, Z, ...) are kept in the
KWARGS dictionaries below, plausible but not real catalogue values --
they only need to exercise the code. Material inputs are the raw
e1/e2/nu1/nu2 floats that assemble_geometry() actually accepts today --
an earlier draft of this conftest assumed a `surfaces`/`RadialSurfaces`
object (built from `core.materials.Material`) that was never merged
into the source; that version is gone, this one matches what's here.
"""
import pytest
import numpy as np

from axisforge.core.machine_elements.bearings.contact_models.analysis import ContactAnalysis

# ---------------------------------------------------------------------
# geometrias de referência -- plausíveis, não de catálogo real, só para
# exercitar o código.
# ---------------------------------------------------------------------

DEEP_GROOVE_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, s=0.02, i=1)
ANGULAR_CONTACT_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=25.0, i=1)
SELF_ALIGNING_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=25.0, i=1)

THRUST_ROW_KWARGS = dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=90.0)
THRUST_ROW_KWARGS_NON_90 = dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=60.0)

CYL_ROLLER_KWARGS = dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40, i=1)

# ThrustCylindricalRollerFamily is known-stale (see review) -- these two
# dicts are kept only for the xfail tests in test_family.py that document
# the current, broken call shape. Do not build new fixtures on them until
# the family is migrated to the contact=/e1..nu2 pattern.
THRUST_CYL_ROLLER_KWARGS = dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40, alpha_0_deg=90.0, i=1)
THRUST_CYL_ROLLER_KWARGS_NON_90 = dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40, alpha_0_deg=70.0, i=1)


# ---------------------------------------------------------------------
# materiais -- e1/e2/nu1/nu2 crus (aço em ambos os lados do contacto),
# só para exercitar o ramo contact=ContactAnalysis.ISO16281.
# ---------------------------------------------------------------------

STEEL_CONTACT_KWARGS = dict(contact=ContactAnalysis.ISO16281,
                             e1=210_000.0, e2=210_000.0, nu1=0.3, nu2=0.3)


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
# geometrias já montadas -- atalhos usados em vários testes.
#
# Cada família com contact=ISO16281 suportado ganha duas variantes: a
# "base" (contact=NONE, o default) para testes de geometria/capacidade
# que não precisam de materiais, e a "_contact" (contact=ISO16281 +
# STEEL_CONTACT_KWARGS) para testes que verificam cp/Ri/cl/cs.
# ---------------------------------------------------------------------

@pytest.fixture
def assembled_deep_groove(deep_groove_family, catalog):
    return deep_groove_family.assemble_geometry(catalog, **DEEP_GROOVE_KWARGS)

@pytest.fixture
def assembled_deep_groove_contact(deep_groove_family, catalog):
    return deep_groove_family.assemble_geometry(catalog, **DEEP_GROOVE_KWARGS, **STEEL_CONTACT_KWARGS)

@pytest.fixture
def assembled_angular_contact(angular_contact_family, catalog):
    return angular_contact_family.assemble_geometry(catalog, **ANGULAR_CONTACT_KWARGS)

@pytest.fixture
def assembled_angular_contact_contact(angular_contact_family, catalog):
    return angular_contact_family.assemble_geometry(catalog, **ANGULAR_CONTACT_KWARGS, **STEEL_CONTACT_KWARGS)

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
def assembled_cyl_roller_contact(cylindrical_roller_family, catalog_floating):
    return cylindrical_roller_family.assemble_geometry(catalog_floating, **CYL_ROLLER_KWARGS, **STEEL_CONTACT_KWARGS)

# Sem fixtures "assembled_thrust_cyl_roller_*" -- ThrustCylindricalRollerFamily
# está sabidamente desatualizada (chama LineContactStiffness com a
# assinatura antiga) e rebenta sempre; os testes que documentam isso em
# test_family.py chamam assemble_geometry() diretamente para poder
# marcar xfail no corpo do teste, não numa fixture.


class RowAsBearing:
    """Adaptador: um dict de assemble_geometry() servido como se fosse
    um Bearing (getattr em vez de __getitem__), para chamar
    dynamic_capacity(bearing)/per_element_dynamic_capacity(bearing)."""
    def __init__(self, row: dict):
        self.__dict__.update(row)