# test_bearing.py
"""Testes de bearing.py -- Bearing.assemble()/imutabilidade/C/validate()/
is_locating(), e _validate_thrust_arrangement() (a validação de
arrangement='thrust' vs família thrust/não-thrust que assemble() corre
antes de construir o Bearing -- ver o comentário no próprio bearing.py
sobre mover isto para family.py um dia).

Usa DeepGrooveBallFamily (não-thrust) e ThrustBallSingleRowFamily
(thrust) -- famílias reais já registadas via @register_family/@thrust --
em vez de inventar uma família fake, para não duplicar o registo de
_THRUST_BEARING/_FAMILY_REGISTRY num mock."""
import pytest

from axisforge.core.machine_elements.bearings.base import BearingCatalog
from axisforge.core.machine_elements.bearings.bearing import Bearing, _validate_thrust_arrangement
from axisforge.core.machine_elements.bearings.families.family import (
    DeepGrooveBallFamily, ThrustBallSingleRowFamily,
)


DEEP_GROOVE_GEOMETRY = dict(Dw=8.0, Dpw=40.0, Z=12, s=0.02, i=1)
THRUST_ROW_GEOMETRY = dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=90.0)


@pytest.fixture
def locating_catalog():
    return BearingCatalog(d=30.0, D=62.0, b=16.0, designation="6206",
                           label="B1", position=0.0, arrangement="locating")


@pytest.fixture
def thrust_catalog():
    return BearingCatalog(d=30.0, D=62.0, b=16.0, designation="TEST-THRUST",
                           label="B2", position=0.0, arrangement="thrust")


@pytest.fixture
def assembled_bearing(locating_catalog):
    return Bearing.assemble(DeepGrooveBallFamily(), locating_catalog, DEEP_GROOVE_GEOMETRY)


# =====================================================================
# _validate_thrust_arrangement
# =====================================================================

class TestValidateThrustArrangement:
    def test_thrust_family_with_thrust_arrangement_ok(self, thrust_catalog):
        _validate_thrust_arrangement(ThrustBallSingleRowFamily(), thrust_catalog)  # não deve levantar

    def test_radial_family_with_locating_arrangement_ok(self, locating_catalog):
        _validate_thrust_arrangement(DeepGrooveBallFamily(), locating_catalog)  # não deve levantar

    def test_thrust_family_with_non_thrust_arrangement_raises(self, locating_catalog):
        with pytest.raises(ValueError, match="is a thrust family"):
            _validate_thrust_arrangement(ThrustBallSingleRowFamily(), locating_catalog)

    def test_non_thrust_family_with_thrust_arrangement_raises(self, thrust_catalog):
        with pytest.raises(ValueError, match="is not registered as a thrust family"):
            _validate_thrust_arrangement(DeepGrooveBallFamily(), thrust_catalog)


# =====================================================================
# Bearing.assemble()
# =====================================================================

class TestBearingAssemble:
    def test_assemble_returns_assembled_bearing(self, assembled_bearing):
        assert assembled_bearing.has_internal_geometry()

    def test_assemble_mirrors_geometry_onto_instance(self, assembled_bearing):
        assert assembled_bearing.Dw == DEEP_GROOVE_GEOMETRY["Dw"]
        assert assembled_bearing.Z == DEEP_GROOVE_GEOMETRY["Z"]

    def test_assemble_copies_catalog_fields(self, assembled_bearing, locating_catalog):
        assert assembled_bearing.d == locating_catalog.d
        assert assembled_bearing.D == locating_catalog.D
        assert assembled_bearing.label == locating_catalog.label

    def test_assemble_computes_dm_as_mean_diameter(self, assembled_bearing, locating_catalog):
        assert assembled_bearing.dm == pytest.approx(0.5 * (locating_catalog.d + locating_catalog.D))

    def test_assemble_rejects_thrust_family_mismatch(self, locating_catalog):
        with pytest.raises(ValueError, match="is a thrust family"):
            Bearing.assemble(ThrustBallSingleRowFamily(), locating_catalog, THRUST_ROW_GEOMETRY)

    def test_family_property_returns_the_assembled_family(self, locating_catalog):
        family = DeepGrooveBallFamily()
        bearing = Bearing.assemble(family, locating_catalog, DEEP_GROOVE_GEOMETRY)
        assert bearing.family is family


# =====================================================================
# imutabilidade pós-assemble()
# =====================================================================

class TestBearingImmutability:
    def test_cannot_set_existing_attribute_after_assemble(self, assembled_bearing):
        with pytest.raises(AttributeError, match="immutable"):
            assembled_bearing.Dw = 999.0

    def test_cannot_set_new_attribute_after_assemble(self, assembled_bearing):
        with pytest.raises(AttributeError, match="immutable"):
            assembled_bearing.some_new_field = 1.0


# =====================================================================
# Bearing.C
# =====================================================================

class TestBearingC:
    def test_C_positive_once_assembled(self, assembled_bearing):
        assert assembled_bearing.C > 0.0

    def test_C_matches_family_dynamic_capacity(self, assembled_bearing):
        expected = DeepGrooveBallFamily.dynamic_capacity(assembled_bearing)
        assert assembled_bearing.C == expected

    def test_C_raises_before_assemble(self, locating_catalog):
        bearing = Bearing(locating_catalog, DeepGrooveBallFamily())
        with pytest.raises(RuntimeError, match="requires Bearing.assemble"):
            _ = bearing.C


# =====================================================================
# Bearing.validate() / is_locating()
# =====================================================================

class TestBearingValidateAndHelpers:
    def test_validate_returns_no_errors_for_healthy_bearing(self, assembled_bearing):
        assert assembled_bearing.validate() == []

    def test_validate_flags_not_assembled(self, locating_catalog):
        bearing = Bearing(locating_catalog, DeepGrooveBallFamily())
        errors = bearing.validate()
        assert any("not assembled" in e for e in errors)

    def test_is_locating_true_for_locating_arrangement(self, assembled_bearing):
        assert assembled_bearing.is_locating()

    def test_is_locating_false_for_thrust_arrangement(self, thrust_catalog):
        bearing = Bearing.assemble(ThrustBallSingleRowFamily(), thrust_catalog, THRUST_ROW_GEOMETRY)
        assert not bearing.is_locating()


# =====================================================================
# repr / summary -- só fumo (não rebentam, contêm o essencial)
# =====================================================================

class TestBearingReprAndSummary:
    def test_repr_contains_family_name_and_arrangement(self, assembled_bearing):
        text = repr(assembled_bearing)
        assert DeepGrooveBallFamily().name in text
        assert assembled_bearing.arrangement in text

    def test_summary_contains_computed_C(self, assembled_bearing):
        text = assembled_bearing.summary()
        assert "C (computed)" in text
        assert f"{assembled_bearing.C:.0f} N" in text