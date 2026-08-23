"""
tests/core/test_materials.py

Unit tests for axisforge.core.materials

Covers Material, GearMaterial, the embedded libraries and the lookup helpers.

Reference: Shigley Table A-20 (steel properties), Shigley Sec. 6-2
(endurance limit), ISO 6336-5:2016 (gear material grades).

ASCII only.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from axisforge.core.materials import (
    Material,
    GearMaterial,
    S355,
    CrMo42,
    AISI_1045,
    AISI_4340,
    GEAR_STEEL,
    GEAR_ADI,
    GEAR_POM,
    GEAR_PA66,
    get_material,
    get_gear_material,
    available_materials,
    available_gear_materials,
)


# ===========================================================================
# Material -- construction guards
# ===========================================================================

class TestMaterialConstruction:

    @pytest.mark.parametrize("field,value,message", [
        ("Sut", 0.0, "Sut"),
        ("Sut", -1.0, "Sut"),
        ("Sy", 0.0, "Sy"),
        ("Sy", -1.0, "Sy"),
        ("E", 0.0, "E must be"),
        ("density", 0.0, "density"),
    ])
    def test_non_positive_values_rejected(self, field, value, message):
        kwargs = dict(material_id="X", Sut=600.0, Sy=400.0,
                      E=210_000.0, density=7850.0)
        kwargs[field] = value
        with pytest.raises(ValueError, match=message):
            Material(**kwargs)

    def test_yield_above_ultimate_rejected(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            Material(material_id="X", Sut=500.0, Sy=600.0,
                     E=210_000.0, density=7850.0)

    def test_yield_equal_to_ultimate_accepted(self):
        """Sy == Sut is the limiting case, not an error."""
        m = Material(material_id="X", Sut=500.0, Sy=500.0,
                     E=210_000.0, density=7850.0)
        assert m.Sy == pytest.approx(m.Sut)

    def test_is_frozen(self, material_s355):
        with pytest.raises(FrozenInstanceError):
            material_s355.Sut = 1.0

    def test_default_poisson_ratio(self, material_custom):
        assert material_custom.poisson_ratio == pytest.approx(0.3)


# ===========================================================================
# Material -- derived properties
# ===========================================================================

class TestMaterialDerivedProperties:

    def test_endurance_limit_is_half_sut_below_the_cap(self, material_custom):
        """Shigley Sec. 6-2: Se = 0.5*Sut for Sut <= 1400 MPa."""
        assert material_custom.endurance_limit == pytest.approx(400.0)

    def test_endurance_limit_capped_at_700(self, material_high_sut):
        """Sut = 1600 MPa, no Se_base -- Se saturates at 700 MPa."""
        assert material_high_sut.endurance_limit == pytest.approx(700.0)

    def test_explicit_se_base_wins_over_the_formula(self):
        """Se_base is authoritative: it is not re-derived, nor re-capped."""
        m = Material(material_id="X", Sut=600.0, Sy=400.0, E=210_000.0,
                     density=7850.0, Se_base=123.0)
        assert m.endurance_limit == pytest.approx(123.0)

    def test_shear_yield_strength_is_von_mises(self, material_s355):
        assert material_s355.shear_yield_strength == pytest.approx(0.577 * 355.0)


# ===========================================================================
# Embedded shaft library -- reference values
# ===========================================================================

class TestEmbeddedShaftLibrary:

    @pytest.mark.parametrize("material,material_id,Sut,Sy,E", [
        (S355,      "S355",      590.0,  355.0,  210_000.0),
        (CrMo42,    "42CrMo4",  1000.0,  800.0,  210_000.0),
        (AISI_1045, "AISI_1045", 570.0,  310.0,  207_000.0),
        (AISI_4340, "AISI_4340",1460.0, 1380.0,  207_000.0),
    ])
    def test_reference_values(self, material, material_id, Sut, Sy, E):
        assert material.material_id == material_id
        assert material.Sut == pytest.approx(Sut)
        assert material.Sy == pytest.approx(Sy)
        assert material.E == pytest.approx(E)

    def test_s355_endurance_limit(self):
        assert S355.endurance_limit == pytest.approx(295.0)

    def test_aisi_4340_uses_its_declared_se_base(self):
        """Sut = 1460 > 1400, and Se_base is set explicitly to the cap."""
        assert AISI_4340.Se_base == pytest.approx(700.0)
        assert AISI_4340.endurance_limit == pytest.approx(700.0)

    def test_every_embedded_material_is_self_consistent(self):
        for mid in available_materials():
            m = get_material(mid)
            assert m.Sy <= m.Sut
            assert m.E > 0.0
            assert m.density > 0.0
            assert m.endurance_limit > 0.0


# ===========================================================================
# GearMaterial
# ===========================================================================

class TestGearMaterial:

    def test_reference_gear_steel(self):
        assert GEAR_STEEL.sigma_Hlim == pytest.approx(1500.0)
        assert GEAR_STEEL.sigma_Flim == pytest.approx(430.0)
        assert GEAR_STEEL.material_class == "ME"

    @pytest.mark.parametrize("polymer", [GEAR_POM, GEAR_PA66])
    def test_polymers_carry_zero_fatigue_limits(self, polymer):
        """
        Documented convention: polymer sigma_Hlim/sigma_Flim are temperature-
        and cycle-dependent, so they are stored as 0.0 rather than guessed.
        """
        assert polymer.sigma_Hlim == 0.0
        assert polymer.sigma_Flim == 0.0
        assert polymer.material_class == "ML"

    @pytest.mark.parametrize("field,value,message", [
        ("E", 0.0, "E must be"),
        ("poisson_ratio", 0.0, "poisson_ratio"),
        ("density", 0.0, "density"),
        ("sigma_Hlim", -1.0, "sigma_Hlim"),
        ("sigma_Flim", -1.0, "sigma_Flim"),
    ])
    def test_invalid_values_rejected(self, field, value, message):
        kwargs = dict(material_id="X", E=200_000.0, poisson_ratio=0.3,
                      density=7800.0, cp=460.0, k_thermal=45.0,
                      sigma_Hlim=1000.0, sigma_Flim=300.0)
        kwargs[field] = value
        with pytest.raises(ValueError, match=message):
            GearMaterial(**kwargs)

    def test_unknown_material_class_rejected(self):
        with pytest.raises(ValueError, match="material_class"):
            GearMaterial(material_id="X", E=200_000.0, poisson_ratio=0.3,
                         density=7800.0, cp=460.0, k_thermal=45.0,
                         sigma_Hlim=1000.0, sigma_Flim=300.0,
                         material_class="XX")

    def test_equivalent_modulus_against_closed_form(self, gear_material_steel):
        """E* = 1 / ((1-v1^2)/E1 + (1-v2^2)/E2)."""
        other = GEAR_ADI
        a, b = gear_material_steel, other
        expected = 1.0 / (
            (1 - a.poisson_ratio ** 2) / a.E + (1 - b.poisson_ratio ** 2) / b.E
        )
        assert a.equivalent_modulus(b) == pytest.approx(expected)

    def test_equivalent_modulus_is_symmetric(self, gear_material_steel):
        a, b = gear_material_steel, GEAR_ADI
        assert a.equivalent_modulus(b) == pytest.approx(b.equivalent_modulus(a))

    def test_equivalent_modulus_of_identical_steels(self, gear_material_steel):
        """Same material both sides: E* = E / (2*(1 - v^2))."""
        m = gear_material_steel
        expected = m.E / (2.0 * (1 - m.poisson_ratio ** 2))
        assert m.equivalent_modulus(m) == pytest.approx(expected)

    def test_soft_polymer_dominates_the_pair_stiffness(self):
        """
        E* of steel-on-POM must sit below E* of POM-on-POM's steel partner
        but above POM-on-POM -- the compliant body governs.
        """
        steel_pom = GEAR_STEEL.equivalent_modulus(GEAR_POM)
        pom_pom = GEAR_POM.equivalent_modulus(GEAR_POM)
        steel_steel = GEAR_STEEL.equivalent_modulus(GEAR_STEEL)
        assert pom_pom < steel_pom < steel_steel


# ===========================================================================
# Lookups
# ===========================================================================

class TestLookups:

    def test_get_material_returns_the_library_instance(self):
        assert get_material("S355") is S355

    def test_get_material_uses_material_id_not_variable_name(self):
        """CrMo42 is registered under '42CrMo4'."""
        assert get_material("42CrMo4") is CrMo42
        with pytest.raises(KeyError):
            get_material("CrMo42")

    def test_get_material_unknown_id_raises_with_the_available_list(self):
        with pytest.raises(KeyError, match="not found"):
            get_material("UNOBTAINIUM")

    def test_available_materials(self):
        assert set(available_materials()) == {
            "S355", "42CrMo4", "AISI_1045", "AISI_4340"
        }

    def test_get_gear_material(self):
        assert get_gear_material("GEAR_STEEL") is GEAR_STEEL

    def test_get_gear_material_unknown_id_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_gear_material("GEAR_UNOBTAINIUM")

    def test_available_gear_materials(self):
        assert set(available_gear_materials()) == {
            "GEAR_STEEL", "GEAR_ADI", "GEAR_POM", "GEAR_PA66"
        }

    def test_the_two_libraries_are_disjoint(self):
        """A shaft material id must never resolve as a gear material."""
        assert not set(available_materials()) & set(available_gear_materials())

    def test_available_list_is_a_copy_not_the_registry(self):
        """Mutating the returned list must not corrupt the library."""
        listed = available_materials()
        listed.append("BOGUS")
        assert "BOGUS" not in available_materials()
