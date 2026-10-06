# test_base.py
"""Testes de base.py -- BearingCatalog (validação own-construction via
__post_init__) e o contrato abstrato BearingFamily.

O mecanismo de BEARING_TYPE/DUTY/CAPABILITIES/REQUIRED_FOR que a
docstring de __init_subclass__ ainda descreve foi removido do código
-- __init_subclass__ hoje só chama super(), não valida nada. Não há
testes aqui para essa parte porque deixou de existir; ver
families/tests/test_family.py para o contrato mínimo que sobrou."""
import dataclasses
import pytest

from axisforge.core.machine_elements.bearings.base import BearingCatalog, BearingFamily


# =====================================================================
# BearingCatalog
# =====================================================================

VALID_KWARGS = dict(d=30.0, D=62.0, b=16.0, designation="6206", label="B1",
                     position=0.0, arrangement="locating")


class TestBearingCatalogValidation:
    def test_valid_catalog_constructs_without_error(self):
        BearingCatalog(**VALID_KWARGS)

    def test_defaults_when_only_d_D_given(self):
        cat = BearingCatalog(d=30.0, D=62.0)
        assert cat.b == 0.0
        assert cat.position == 0.0
        assert cat.arrangement == "locating"

    @pytest.mark.parametrize("bad_kwargs, match", [
        (dict(d=0.0), "d must be > 0"),
        (dict(d=-5.0), "d must be > 0"),
        (dict(D=0.0), "D must be > 0"),
        (dict(D=20.0), "D must be > d"),   # D < d, com d=30 do VALID_KWARGS
        (dict(b=-1.0), "b must be >= 0"),
        (dict(position=-1.0), "position must be >= 0"),
        (dict(arrangement="bogus"), "arrangement must be"),
    ])
    def test_rejects_invalid_field_at_construction(self, bad_kwargs, match):
        """__post_init__ chama validate_or_raise() -- não é preciso
        chamar .validate() manualmente para apanhar um catálogo
        fisicamente impossível; o construtor já recusa."""
        kwargs = {**VALID_KWARGS, **bad_kwargs}
        with pytest.raises(ValueError, match=match):
            BearingCatalog(**kwargs)

    def test_D_equal_to_d_is_rejected(self):
        with pytest.raises(ValueError, match="D must be > d"):
            BearingCatalog(d=30.0, D=30.0)

    @pytest.mark.parametrize("arrangement", ["locating", "floating", "non-locating", "thrust"])
    def test_accepts_every_documented_arrangement(self, arrangement):
        BearingCatalog(d=30.0, D=62.0, arrangement=arrangement)

    def test_validate_without_raise_returns_all_errors_at_once(self):
        """validate() (sem o _or_raise) deve devolver a lista completa de
        problemas, não parar no primeiro -- útil para reportar tudo de
        uma vez. Construído via object.__new__ para contornar
        __post_init__/validate_or_raise (o dataclass é frozen, por isso
        os campos são postos com object.__setattr__)."""
        cat = object.__new__(BearingCatalog)
        object.__setattr__(cat, "d", -1.0)
        object.__setattr__(cat, "D", -1.0)
        object.__setattr__(cat, "b", -1.0)
        object.__setattr__(cat, "designation", "")
        object.__setattr__(cat, "label", "")
        object.__setattr__(cat, "position", -1.0)
        object.__setattr__(cat, "arrangement", "bogus")
        errors = cat.validate()
        # d, D, D>d, b, position, arrangement -- os 6 problemas de uma vez
        assert len(errors) == 6

    def test_frozen_dataclass_rejects_mutation(self):
        cat = BearingCatalog(**VALID_KWARGS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cat.d = 999.0


# =====================================================================
# BearingFamily -- contrato abstrato
# =====================================================================

class TestBearingFamilyContract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BearingFamily()

    def test_concrete_subclass_missing_members_cannot_be_instantiated(self):
        class _Incomplete(BearingFamily):
            @property
            def name(self):
                return "incomplete"
            # assemble_geometry / dynamic_capacity / per_element_dynamic_capacity em falta

        with pytest.raises(TypeError):
            _Incomplete()

    def test_fully_implemented_subclass_can_be_instantiated(self):
        class _Complete(BearingFamily):
            @property
            def name(self):
                return "complete"

            def assemble_geometry(self, catalog, **kwargs):
                return {}

            @staticmethod
            def dynamic_capacity(bearing):
                return 1.0

            @staticmethod
            def per_element_dynamic_capacity(bearing, capacity=None):
                return (1.0, 1.0)

        _Complete()  # não deve levantar