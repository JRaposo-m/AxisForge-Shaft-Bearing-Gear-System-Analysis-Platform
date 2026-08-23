"""
tests/core/machine_elements/test_bearings.py

Unit tests for the bearing core:
  catalog.py                       BearingCatalog
  bearing.py                       Bearing (the assemble-once orchestrator)
  families/.../deep_groove_ball.py DeepGrooveBallFamily

These tests cover the ASSEMBLY CONTRACT and the geometry bookkeeping, not
the ISO/TS 16281 contact math itself -- Hertz constants and per-element
capacities belong to a validation suite with literature reference values,
which is a separate file and a separate job.

Reference:
  ISO 281:2007 Table 1  -- raceway groove radii (0.52*Dw for DGBB),
                           reduction factor lambda {1 row: 0.95, 2 rows: 0.90}

ASCII only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from axisforge.core.machine_elements.Bearings.catalog import BearingCatalog
from axisforge.core.machine_elements.Bearings.bearing import Bearing
from axisforge.core.machine_elements.Bearings.bearing_types import BearingType
from axisforge.core.machine_elements.Bearings.families.ball_bearing.radial.subtypes.deep_groove_ball import (
    DeepGrooveBallFamily,
)


# ===========================================================================
# BearingCatalog
# ===========================================================================

class TestBearingCatalog:

    def test_clean_catalog_validates(self, catalog_6204):
        assert catalog_6204.validate() == []
        catalog_6204.validate_or_raise()

    def test_minimal_catalog_only_needs_bore_and_outer_diameter(self):
        cat = BearingCatalog(d=20.0, D=47.0)
        assert cat.b == 0.0
        assert cat.C == 0.0
        assert cat.position == 0.0
        assert cat.arrangement == "locating"
        assert cat.validate() == []

    def test_negative_position_reported(self):
        errors = BearingCatalog(d=20.0, D=47.0, position=-1.0).validate()
        assert any("position must be >= 0" in e for e in errors)

    @pytest.mark.parametrize("field", ["C", "C0"])
    def test_negative_ratings_reported(self, field):
        cat = BearingCatalog(d=20.0, D=47.0, **{field: -1.0})
        assert any(f"{field} must be >= 0" in e for e in cat.validate())

    @pytest.mark.parametrize("arrangement", ["locating", "floating", "non-locating"])
    def test_allowed_arrangements(self, arrangement):
        cat = BearingCatalog(d=20.0, D=47.0, arrangement=arrangement)
        assert cat.validate() == []

    @pytest.mark.parametrize("arrangement", ["fixed", "free", "LOCATING", ""])
    def test_rejected_arrangements(self, arrangement):
        cat = BearingCatalog(d=20.0, D=47.0, arrangement=arrangement)
        assert any("arrangement must be" in e for e in cat.validate())

    def test_error_tag_prefers_label_then_designation(self):
        by_label = BearingCatalog(d=20.0, D=47.0, position=-1.0,
                                  label="brg_A", designation="6204")
        by_designation = BearingCatalog(d=20.0, D=47.0, position=-1.0,
                                        designation="6204")
        anonymous = BearingCatalog(d=20.0, D=47.0, position=-1.0)

        assert by_label.validate()[0].startswith("brg_A:")
        assert by_designation.validate()[0].startswith("6204:")
        assert anonymous.validate()[0].startswith("BearingCatalog:")

    def test_validate_or_raise_collects_every_error(self):
        cat = BearingCatalog(d=20.0, D=47.0, position=-1.0, C=-1.0,
                             arrangement="bogus")
        with pytest.raises(ValueError) as excinfo:
            cat.validate_or_raise()
        text = str(excinfo.value)
        assert "position" in text and "C must be" in text and "arrangement" in text

    def test_is_frozen(self, catalog_6204):
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            catalog_6204.position = 10.0


# ===========================================================================
# DeepGrooveBallFamily -- declared contract
# ===========================================================================

class TestDeepGrooveBallFamilyContract:

    def test_identity(self, dgbb_family):
        assert dgbb_family.name == "deep_groove_ball"
        assert dgbb_family.BEARING_TYPE is BearingType.DEEP_GROOVE_BALL
        assert dgbb_family.DUTY == "radial"

    def test_capabilities(self, dgbb_family):
        assert dgbb_family.CAPABILITIES == frozenset({"point_contact"})

    def test_required_for_lists_every_field_the_solver_reads(self, dgbb_family):
        required = dgbb_family.REQUIRED_FOR["point_contact"]
        assert {"ri", "re", "Dw", "Dpw", "Z", "E", "nu",
                "A", "alpha_0", "Ri", "phi_j", "gamma", "cp"} <= required

    def test_reference_raceway_radii_iso281_table1(self, dgbb_family):
        ri, re = dgbb_family.reference_raceway_radii(10.0)
        assert ri == pytest.approx(5.2)
        assert re == pytest.approx(5.2)

    def test_reduction_factor_table(self, dgbb_family):
        assert dgbb_family.REDUCTION_FACTOR_BY_ROWS == {1: 0.95, 2: 0.90}


class TestDeepGrooveBallGeometryAssembly:

    def test_returned_fields_cover_required_for(self, dgbb_family, catalog_6204,
                                                dgbb_6204_geometry):
        geom = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry)
        missing = dgbb_family.REQUIRED_FOR["point_contact"] - set(geom)
        assert missing == set()

    def test_raceway_radii_are_derived_not_taken_from_the_caller(
        self, dgbb_family, catalog_6204, dgbb_6204_geometry
    ):
        geom = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry)
        assert geom["ri"] == pytest.approx(0.52 * 7.94)
        assert geom["re"] == pytest.approx(0.52 * 7.94)
        assert geom["raceway_radii_from_reference"] is True

    def test_total_curvature_A(self, dgbb_family, catalog_6204, dgbb_6204_geometry):
        """A = ri + re - Dw."""
        geom = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry)
        assert geom["A"] == pytest.approx(2 * 0.52 * 7.94 - 7.94)

    def test_element_angles_are_evenly_spaced_over_the_full_circle(
        self, dgbb_family, catalog_6204, dgbb_6204_geometry
    ):
        geom = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry)
        phi_j = geom["phi_j"]

        assert len(phi_j) == 8
        assert phi_j[0] == pytest.approx(0.0)
        assert np.allclose(np.diff(phi_j), 2 * math.pi / 8)
        assert phi_j[-1] < 2 * math.pi          # endpoint excluded, no duplicate

    def test_gamma_is_a_dimensionless_ratio_below_one(
        self, dgbb_family, catalog_6204, dgbb_6204_geometry
    ):
        geom = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry)
        assert 0.0 < geom["gamma"] < 1.0

    def test_hertz_spring_constant_is_positive(
        self, dgbb_family, catalog_6204, dgbb_6204_geometry
    ):
        geom = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry)
        assert geom["cp"] > 0.0

    def test_zero_clearance_gives_zero_contact_angle(
        self, dgbb_family, catalog_6204
    ):
        """s = 0 is a legal input and must place the contact on the radial plane."""
        geom = dgbb_family.assemble_geometry(
            catalog_6204, Dw=7.94, Dpw=33.5, Z=8, E=206_000.0, s=0.0
        )
        assert geom["alpha_0"] == pytest.approx(0.0, abs=1e-9)

    def test_clearance_opens_the_contact_angle(self, dgbb_family, catalog_6204):
        tight = dgbb_family.assemble_geometry(
            catalog_6204, Dw=7.94, Dpw=33.5, Z=8, E=206_000.0, s=0.005)
        loose = dgbb_family.assemble_geometry(
            catalog_6204, Dw=7.94, Dpw=33.5, Z=8, E=206_000.0, s=0.050)
        assert loose["alpha_0"] > tight["alpha_0"]

    def test_negative_clearance_rejected(self, dgbb_family, catalog_6204):
        with pytest.raises(ValueError, match="s must be >= 0"):
            dgbb_family.assemble_geometry(
                catalog_6204, Dw=7.94, Dpw=33.5, Z=8, E=206_000.0, s=-0.001)

    def test_row_count_selects_the_reduction_factor(self, dgbb_family, catalog_6204,
                                                    dgbb_6204_geometry):
        single = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry, i=1)
        double = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry, i=2)
        assert single["reduction_factor"] == pytest.approx(0.95)
        assert double["reduction_factor"] == pytest.approx(0.90)

    def test_row_count_does_not_change_the_raceway_radii(self, dgbb_family,
                                                         catalog_6204,
                                                         dgbb_6204_geometry):
        single = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry, i=1)
        double = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry, i=2)
        assert single["ri"] == pytest.approx(double["ri"])
        assert single["re"] == pytest.approx(double["re"])

    @pytest.mark.parametrize("i", [0, 3, -1])
    def test_unsupported_row_count_rejected(self, dgbb_family, catalog_6204,
                                            dgbb_6204_geometry, i):
        with pytest.raises(ValueError, match="number of rows"):
            dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry, i=i)

    def test_assemble_geometry_is_pure(self, dgbb_family, catalog_6204,
                                       dgbb_6204_geometry):
        """Same inputs, same outputs -- no accumulated state on the family."""
        first = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry)
        second = dgbb_family.assemble_geometry(catalog_6204, **dgbb_6204_geometry)
        assert first["alpha_0"] == pytest.approx(second["alpha_0"])
        assert first["cp"] == pytest.approx(second["cp"])


# ===========================================================================
# Bearing.assemble -- the orchestrator contract
# ===========================================================================

class TestBearingAssembly:

    def test_catalogue_fields_are_mirrored(self, bearing_6204):
        b = bearing_6204
        assert b.d == pytest.approx(20.0)
        assert b.D == pytest.approx(47.0)
        assert b.b == pytest.approx(14.0)
        assert b.C == pytest.approx(12_700.0)
        assert b.C0 == pytest.approx(6_550.0)
        assert b.designation == "6204"
        assert b.label == "brg_A"
        assert b.position == pytest.approx(50.0)
        assert b.arrangement == "locating"

    def test_mean_diameter(self, bearing_6204):
        assert bearing_6204.dm == pytest.approx(0.5 * (20.0 + 47.0))

    def test_geometry_fields_are_mirrored_onto_the_instance(self, bearing_6204):
        for field in ("ri", "re", "Dw", "Dpw", "Z", "E", "nu",
                      "A", "alpha_0", "Ri", "phi_j", "gamma", "cp"):
            assert getattr(bearing_6204, field) is not None, field

    def test_bearing_type_label_is_mirrored(self, bearing_6204):
        assert bearing_6204.bearing_type is BearingType.DEEP_GROOVE_BALL

    def test_family_is_exposed_for_dispatch(self, bearing_6204):
        assert isinstance(bearing_6204.family, DeepGrooveBallFamily)
        assert bearing_6204.family.name == "deep_groove_ball"

    def test_state_flags(self, bearing_6204):
        assert bearing_6204.has_internal_geometry() is True
        assert bearing_6204.is_enabled("point_contact") is True
        assert bearing_6204.is_enabled("line_contact") is False
        assert bearing_6204.is_locating() is True

    def test_assembled_bearing_validates_clean(self, bearing_6204):
        assert bearing_6204.validate() == []

    def test_summary_is_ascii(self, bearing_6204):
        bearing_6204.summary().encode("ascii")

    def test_repr_names_the_family_and_designation(self, bearing_6204):
        text = repr(bearing_6204)
        assert "deep_groove_ball" in text
        assert "6204" in text


class TestBearingAssemblyGuards:

    def test_invalid_catalogue_stops_before_any_geometry_work(
        self, dgbb_family, dgbb_6204_geometry
    ):
        bad = BearingCatalog(d=20.0, D=47.0, position=-5.0, designation="6204")
        with pytest.raises(ValueError, match="position must be >= 0"):
            Bearing.assemble(family=dgbb_family, catalog=bad,
                             geometry=dgbb_6204_geometry,
                             analyses={"point_contact": True})

    def test_unsupported_analysis_rejected(self, dgbb_family, catalog_6204,
                                           dgbb_6204_geometry):
        with pytest.raises(NotImplementedError, match="line_contact"):
            Bearing.assemble(family=dgbb_family, catalog=catalog_6204,
                             geometry=dgbb_6204_geometry,
                             analyses={"line_contact": True})

    def test_analysis_switched_off_is_not_requested(self, dgbb_family, catalog_6204,
                                                    dgbb_6204_geometry):
        """analyses={"x": False} must not count as a request for x."""
        bearing = Bearing.assemble(
            family=dgbb_family, catalog=catalog_6204,
            geometry=dgbb_6204_geometry,
            analyses={"point_contact": True, "line_contact": False},
        )
        assert bearing.is_enabled("point_contact") is True
        assert bearing.is_enabled("line_contact") is False

    def test_no_analyses_still_assembles_the_geometry(self, dgbb_family, catalog_6204,
                                                      dgbb_6204_geometry):
        bearing = Bearing.assemble(family=dgbb_family, catalog=catalog_6204,
                                   geometry=dgbb_6204_geometry)
        assert bearing.has_internal_geometry() is True
        assert bearing.is_enabled("point_contact") is False

    def test_missing_geometry_kwarg_is_a_type_error(self, dgbb_family, catalog_6204):
        """geometry is splatted straight into assemble_geometry()."""
        with pytest.raises(TypeError):
            Bearing.assemble(family=dgbb_family, catalog=catalog_6204,
                             geometry=dict(Dw=7.94, Dpw=33.5),
                             analyses={"point_contact": True})


class TestBearingImmutability:
    """Assembled once, sealed thereafter -- the single-source-of-truth rule."""

    @pytest.mark.parametrize("field,value", [
        ("position", 999.0),
        ("arrangement", "floating"),
        ("C", 1.0),
        ("Dw", 1.0),
    ])
    def test_writes_are_refused(self, bearing_6204, field, value):
        with pytest.raises(AttributeError):
            setattr(bearing_6204, field, value)

    def test_new_attributes_are_refused(self, bearing_6204):
        with pytest.raises(AttributeError):
            bearing_6204.some_solver_result = 42.0

    def test_the_value_is_unchanged_after_a_refused_write(self, bearing_6204):
        with pytest.raises(AttributeError):
            bearing_6204.position = 999.0
        assert bearing_6204.position == pytest.approx(50.0)

    def test_two_bearings_from_one_family_are_independent(
        self, dgbb_family, dgbb_6204_geometry
    ):
        a = Bearing.assemble(
            family=dgbb_family,
            catalog=BearingCatalog(d=20.0, D=47.0, b=14.0, designation="6204",
                                   position=40.0, label="A"),
            geometry=dgbb_6204_geometry, analyses={"point_contact": True})
        b = Bearing.assemble(
            family=dgbb_family,
            catalog=BearingCatalog(d=20.0, D=47.0, b=14.0, designation="6204",
                                   position=260.0, label="B"),
            geometry=dgbb_6204_geometry, analyses={"point_contact": True})

        assert a is not b
        assert a.position == pytest.approx(40.0)
        assert b.position == pytest.approx(260.0)
        assert a.family is b.family      # the family instance is shared, by design
