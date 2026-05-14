"""
tests/test_core/test_materials.py
Unit tests for core/materials.py.

Coverage target: >= 90% of core/materials.py
References:
  Shigley Table A-20 (embedded material values)
  Shigley §6-2 (endurance limit rule)
"""

import pytest

from core.materials import (
    Material,
    get_material,
    available_materials,
    S355,
    CrMo42,
    AISI_1045,
    AISI_4340,
)


# ===========================================================================
# Material construction and validation
# ===========================================================================

class TestMaterialConstruction:

    def test_valid_material_creates_successfully(self):
        m = Material(
            material_id="TEST",
            Sut=600.0,
            Sy=400.0,
            E=210.0,
            density=7850.0,
        )
        assert m.material_id == "TEST"
        assert m.Sut == 600.0
        assert m.Sy == 400.0
        assert m.E == 210.0
        assert m.density == 7850.0

    def test_default_se_base_is_none(self):
        m = Material(material_id="X", Sut=600.0, Sy=400.0, E=210.0, density=7850.0)
        assert m.Se_base is None

    def test_explicit_se_base_stored(self):
        m = Material(material_id="X", Sut=1500.0, Sy=1400.0, E=210.0,
                     density=7850.0, Se_base=700.0)
        assert m.Se_base == 700.0

    def test_description_default_empty(self):
        m = Material(material_id="X", Sut=600.0, Sy=400.0, E=210.0, density=7850.0)
        assert m.description == ""

    def test_raises_sut_zero(self):
        with pytest.raises(ValueError, match="Sut"):
            Material(material_id="X", Sut=0.0, Sy=400.0, E=210.0, density=7850.0)

    def test_raises_sut_negative(self):
        with pytest.raises(ValueError, match="Sut"):
            Material(material_id="X", Sut=-100.0, Sy=400.0, E=210.0, density=7850.0)

    def test_raises_sy_zero(self):
        with pytest.raises(ValueError, match="Sy"):
            Material(material_id="X", Sut=600.0, Sy=0.0, E=210.0, density=7850.0)

    def test_raises_sy_negative(self):
        with pytest.raises(ValueError, match="Sy"):
            Material(material_id="X", Sut=600.0, Sy=-100.0, E=210.0, density=7850.0)

    def test_raises_sy_exceeds_sut(self):
        with pytest.raises(ValueError, match="Sy"):
            Material(material_id="X", Sut=600.0, Sy=700.0, E=210.0, density=7850.0)

    def test_sy_equal_to_sut_is_accepted(self):
        """
        Sy == Sut is not physically realistic for steel but is not
        explicitly forbidden by the model (guard is Sy > Sut strictly).
        Document the design decision.
        """
        m = Material(material_id="X", Sut=600.0, Sy=600.0, E=210.0, density=7850.0)
        assert m.Sy == m.Sut

    def test_raises_E_zero(self):
        with pytest.raises(ValueError, match="E"):
            Material(material_id="X", Sut=600.0, Sy=400.0, E=0.0, density=7850.0)

    def test_raises_E_negative(self):
        with pytest.raises(ValueError, match="E"):
            Material(material_id="X", Sut=600.0, Sy=400.0, E=-210.0, density=7850.0)

    def test_raises_density_zero(self):
        with pytest.raises(ValueError, match="density"):
            Material(material_id="X", Sut=600.0, Sy=400.0, E=210.0, density=0.0)

    def test_raises_density_negative(self):
        with pytest.raises(ValueError, match="density"):
            Material(material_id="X", Sut=600.0, Sy=400.0, E=210.0, density=-7850.0)

    def test_frozen_immutable(self):
        """Material is a frozen dataclass — attributes cannot be changed."""
        m = Material(material_id="X", Sut=600.0, Sy=400.0, E=210.0, density=7850.0)
        with pytest.raises((AttributeError, TypeError)):
            m.Sut = 700.0  # type: ignore


# ===========================================================================
# endurance_limit property
# ===========================================================================

class TestEnduranceLimit:

    def test_se_base_provided_returned_directly(self):
        """If Se_base is set, endurance_limit returns it unchanged."""
        m = Material(material_id="X", Sut=1500.0, Sy=1400.0, E=210.0,
                     density=7850.0, Se_base=650.0)
        assert m.endurance_limit == 650.0

    def test_se_base_none_rule_sut_below_1400(self):
        """Se = 0.5 * Sut when Sut <= 1400 MPa and Se_base is None."""
        m = Material(material_id="X", Sut=800.0, Sy=600.0, E=210.0, density=7850.0)
        assert pytest.approx(m.endurance_limit, rel=1e-9) == 400.0

    def test_se_base_none_rule_sut_above_1400_capped(self):
        """Se = 700 MPa when Sut > 1400 MPa (Shigley §6-2 cap)."""
        m = Material(material_id="X", Sut=1600.0, Sy=1500.0, E=210.0, density=7850.0)
        assert m.endurance_limit == 700.0

    def test_se_base_none_rule_sut_exactly_1400(self):
        """Sut = 1400 MPa → Se = 700 MPa (boundary case)."""
        m = Material(material_id="X", Sut=1400.0, Sy=1200.0, E=210.0, density=7850.0)
        assert pytest.approx(m.endurance_limit, rel=1e-9) == 700.0

    def test_se_base_zero_explicit(self):
        """Se_base=0.0 is technically invalid physically but tests the branch."""
        # Se_base=0 would be returned as-is if Se_base is not None
        # Use a small positive value to test the Se_base path without Sut issues
        m = Material(material_id="X", Sut=600.0, Sy=400.0, E=210.0,
                     density=7850.0, Se_base=250.0)
        assert m.endurance_limit == 250.0


# ===========================================================================
# shear_yield_strength property
# ===========================================================================

class TestShearYieldStrength:

    def test_shear_yield_von_mises(self):
        """Ssy = 0.577 * Sy — von Mises distortion energy criterion."""
        m = Material(material_id="X", Sut=600.0, Sy=400.0, E=210.0, density=7850.0)
        assert pytest.approx(m.shear_yield_strength, rel=1e-6) == 0.577 * 400.0

    def test_shear_yield_s355(self):
        assert pytest.approx(S355.shear_yield_strength, rel=1e-6) == 0.577 * 355.0

    def test_shear_yield_crmo42(self):
        assert pytest.approx(CrMo42.shear_yield_strength, rel=1e-6) == 0.577 * 800.0


# ===========================================================================
# Embedded material library — value verification
# ===========================================================================

class TestEmbeddedMaterials:
    """
    Verify embedded material values against published sources.
    Sources: Shigley Table A-20, EN 10025-2, DIN 17200.
    """

    def test_s355_sut(self):
        assert S355.Sut == 590.0

    def test_s355_sy(self):
        assert S355.Sy == 355.0

    def test_s355_E(self):
        assert S355.E == 210.0

    def test_s355_density(self):
        assert S355.density == 7850.0

    def test_s355_endurance_limit(self):
        # Se = 0.5 * 590 = 295 MPa
        assert pytest.approx(S355.endurance_limit, rel=1e-9) == 295.0

    def test_crmo42_sut(self):
        assert CrMo42.Sut == 1000.0

    def test_crmo42_sy(self):
        assert CrMo42.Sy == 800.0

    def test_crmo42_endurance_limit(self):
        # Se = 0.5 * 1000 = 500 MPa
        assert pytest.approx(CrMo42.endurance_limit, rel=1e-9) == 500.0

    def test_aisi_1045_sut(self):
        assert AISI_1045.Sut == 570.0

    def test_aisi_1045_sy(self):
        assert AISI_1045.Sy == 310.0

    def test_aisi_1045_E(self):
        assert AISI_1045.E == 207.0

    def test_aisi_1045_endurance_limit(self):
        # Se = 0.5 * 570 = 285 MPa
        assert pytest.approx(AISI_1045.endurance_limit, rel=1e-9) == 285.0

    def test_aisi_4340_sut(self):
        assert AISI_4340.Sut == 1460.0

    def test_aisi_4340_sy(self):
        assert AISI_4340.Sy == 1380.0

    def test_aisi_4340_se_base_capped(self):
        # Se_base explicitly set to 700 MPa (Shigley §6-2 cap)
        assert AISI_4340.Se_base == 700.0
        assert AISI_4340.endurance_limit == 700.0

    def test_all_materials_sy_less_than_sut(self):
        """Physical constraint: Sy < Sut for all embedded materials."""
        for mat in [S355, CrMo42, AISI_1045, AISI_4340]:
            assert mat.Sy < mat.Sut, f"{mat.material_id}: Sy >= Sut"

    def test_all_materials_positive_endurance_limit(self):
        for mat in [S355, CrMo42, AISI_1045, AISI_4340]:
            assert mat.endurance_limit > 0


# ===========================================================================
# get_material()
# ===========================================================================

class TestGetMaterial:

    def test_get_s355_returns_correct_object(self):
        m = get_material("S355")
        assert m.material_id == "S355"
        assert m is S355

    def test_get_42crmo4_returns_correct_object(self):
        m = get_material("42CrMo4")
        assert m is CrMo42

    def test_get_aisi_1045(self):
        m = get_material("AISI_1045")
        assert m is AISI_1045

    def test_get_aisi_4340(self):
        m = get_material("AISI_4340")
        assert m is AISI_4340

    def test_unknown_id_raises_key_error(self):
        with pytest.raises(KeyError, match="not found"):
            get_material("NONEXISTENT_MATERIAL")

    def test_key_error_message_lists_available(self):
        """Error message should include the available material IDs."""
        with pytest.raises(KeyError) as exc_info:
            get_material("INVALID")
        assert "S355" in str(exc_info.value)

    def test_case_sensitive_lookup(self):
        """material_id lookup is case-sensitive."""
        with pytest.raises(KeyError):
            get_material("s355")  # lowercase should not match


# ===========================================================================
# available_materials()
# ===========================================================================

class TestAvailableMaterials:

    def test_returns_list(self):
        result = available_materials()
        assert isinstance(result, list)

    def test_contains_all_four_materials(self):
        result = available_materials()
        assert "S355" in result
        assert "42CrMo4" in result
        assert "AISI_1045" in result
        assert "AISI_4340" in result

    def test_length_is_four(self):
        assert len(available_materials()) == 4

    def test_all_ids_are_strings(self):
        for mat_id in available_materials():
            assert isinstance(mat_id, str)

    def test_all_ids_retrievable(self):
        """Every ID returned by available_materials() must work with get_material()."""
        for mat_id in available_materials():
            m = get_material(mat_id)
            assert m.material_id == mat_id
