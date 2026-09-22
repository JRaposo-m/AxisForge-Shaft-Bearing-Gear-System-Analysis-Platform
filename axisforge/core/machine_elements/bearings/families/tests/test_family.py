# test_family.py
"""Testes de families/family.py -- contrato genérico (via
_FAMILY_REGISTRY, para cobrir qualquer família nova automaticamente sem
tocar neste ficheiro) mais testes específicos por família onde o
comportamento diverge (multi-row, branch 90/não-90, validações)."""
import math
import pytest
import numpy as np

from axisforge.core.machine_elements.bearings.families.family import _FAMILY_REGISTRY
from axisforge.core.machine_elements.bearings.families import capacity as bcap
from axisforge.core.machine_elements.bearings.families.tests.conftest import (
    DEEP_GROOVE_KWARGS, ANGULAR_CONTACT_KWARGS,
    THRUST_ROW_KWARGS, THRUST_ROW_KWARGS_NON_90,
    THRUST_CYL_ROLLER_KWARGS, THRUST_CYL_ROLLER_KWARGS_NON_90,
    RowAsBearing,
)


# =====================================================================
# contrato genérico -- corre para TODA família registada via
# @register_family, sem precisar de saber os nomes de antemão.
# =====================================================================

@pytest.mark.parametrize("family_cls", list(_FAMILY_REGISTRY.values()), ids=list(_FAMILY_REGISTRY.keys()))
class TestBearingFamilyContract:
    def test_name_is_nonempty_string(self, family_cls):
        assert isinstance(family_cls().name, str) and family_cls().name

    def test_bearing_type_is_set(self, family_cls):
        assert family_cls.BEARING_TYPE is not None

    def test_duty_is_radial_or_thrust(self, family_cls):
        assert family_cls.DUTY in ("radial", "thrust")

    def test_capabilities_is_nonempty_frozenset(self, family_cls):
        assert isinstance(family_cls.CAPABILITIES, frozenset) and family_cls.CAPABILITIES

    def test_required_for_covers_every_capability(self, family_cls):
        assert set(family_cls.REQUIRED_FOR) == set(family_cls.CAPABILITIES)


# =====================================================================
# DeepGrooveBallFamily / AngularContactFamily -- radial, point contact
# =====================================================================

class TestDeepGrooveBallFamily:
    def test_returns_all_required_fields(self, assembled_deep_groove):
        for key in ("ri", "re", "Dw", "Dpw", "Z", "E", "nu", "A", "alpha_0", "Ri", "phi_j", "gamma", "cp"):
            assert key in assembled_deep_groove

    def test_dynamic_capacity_positive(self, assembled_deep_groove):
        from axisforge.core.machine_elements.bearings.families.family import DeepGrooveBallFamily
        Ca = DeepGrooveBallFamily.dynamic_capacity(RowAsBearing(assembled_deep_groove))
        assert Ca > 0.0

    @pytest.mark.parametrize("i", [0, 3])
    def test_rejects_unsupported_row_count(self, deep_groove_family, catalog, i):
        with pytest.raises(ValueError, match="i must be one of"):
            deep_groove_family.assemble_geometry(catalog, **{**DEEP_GROOVE_KWARGS, "i": i})


class TestAngularContactFamily:
    def test_returns_all_required_fields(self, assembled_angular_contact):
        for key in ("ri", "re", "Dw", "Dpw", "Z", "E", "nu", "A", "alpha_0", "Ri", "phi_j", "gamma", "cp"):
            assert key in assembled_angular_contact

    @pytest.mark.parametrize("alpha_0_deg", [0.0, 45.1, -5.0])
    def test_rejects_alpha_0_deg_out_of_range(self, angular_contact_family, catalog, alpha_0_deg):
        with pytest.raises(ValueError, match="alpha_0_deg must be in"):
            angular_contact_family.assemble_geometry(catalog, **{**ANGULAR_CONTACT_KWARGS, "alpha_0_deg": alpha_0_deg})


class TestSelfAligningBallFamily:
    def test_KNOWN_BUG_assemble_geometry_missing_gamma_arg(self, self_aligning_family, catalog):
        """reference_raceway_radii(cls, Dw, gamma) exige gamma, mas
        assemble_geometry chama self.reference_raceway_radii(Dw) sem ele
        -- e é circular mesmo que o passasses (gamma só existe depois de
        re/PointContactStiffness). Ver review -- não corrigido aqui.
        Substitui este teste por asserts reais quando resolveres."""
        with pytest.raises(TypeError):
            self_aligning_family.assemble_geometry(
                catalog, Dw=8.0, Dpw=40.0, Z=12, E=210_000.0, alpha_0_deg=25.0)


# =====================================================================
# ThrustBallSingleRowFamily / ThrustBallMultiRowFamily
# =====================================================================

class TestThrustBallAssembleGeometry:
    def test_returns_all_required_fields(self, assembled_row_90deg):
        from axisforge.core.machine_elements.bearings.families.family import ThrustBallSingleRowFamily
        for key in ThrustBallSingleRowFamily.REQUIRED_FOR["point_contact"]:
            assert key in assembled_row_90deg

    def test_also_returns_eta_and_lam(self, assembled_row_90deg):
        from axisforge.core.machine_elements.bearings.families.family import ThrustBallSingleRowFamily
        assert assembled_row_90deg["lam"] == ThrustBallSingleRowFamily.LAM
        assert "eta" in assembled_row_90deg

    def test_alpha_0_deg_none_defaults_to_90deg(self, thrust_single_row_family, catalog):
        row = thrust_single_row_family.assemble_geometry(catalog, **{**THRUST_ROW_KWARGS, "alpha_0_deg": None})
        assert math.isclose(row["alpha_0"], np.pi / 2)
        assert row["s"] == 0.0

    def test_rejects_non_scalar_Z(self, thrust_single_row_family, catalog):
        with pytest.raises(TypeError, match="Z must be a scalar int"):
            thrust_single_row_family.assemble_geometry(catalog, **{**THRUST_ROW_KWARGS, "Z": [12, 12]})

    @pytest.mark.parametrize("alpha_0_deg", [0.0, 45.0, 90.1, -10.0])
    def test_rejects_alpha_0_deg_out_of_range(self, thrust_single_row_family, catalog, alpha_0_deg):
        with pytest.raises(ValueError, match="alpha_0_deg must be in"):
            thrust_single_row_family.assemble_geometry(catalog, **{**THRUST_ROW_KWARGS, "alpha_0_deg": alpha_0_deg})


class TestThrustBallCapacityDispatch:
    def test_90deg_row_uses_90deg_capacity_class(self, assembled_row_90deg):
        from axisforge.core.machine_elements.bearings.families.family import ThrustBallSingleRowFamily
        assert ThrustBallSingleRowFamily._capacity_class(assembled_row_90deg["alpha_0"]) is bcap.PointContactCapacityThrust_90deg

    def test_non_90deg_row_uses_non_90deg_capacity_class(self, assembled_row_non_90deg):
        from axisforge.core.machine_elements.bearings.families.family import ThrustBallSingleRowFamily
        assert ThrustBallSingleRowFamily._capacity_class(assembled_row_non_90deg["alpha_0"]) is bcap.PointContactCapacityThrust_Non_90deg

    def test_dynamic_capacity_positive(self, assembled_row_90deg):
        from axisforge.core.machine_elements.bearings.families.family import ThrustBallSingleRowFamily
        assert ThrustBallSingleRowFamily.dynamic_capacity(RowAsBearing(assembled_row_90deg)) > 0.0

    def test_per_element_dynamic_capacity_returns_positive_pair(self, assembled_row_90deg):
        from axisforge.core.machine_elements.bearings.families.family import ThrustBallSingleRowFamily
        Q_ci, Q_ce = ThrustBallSingleRowFamily.per_element_dynamic_capacity(RowAsBearing(assembled_row_90deg))
        assert Q_ci > 0.0 and Q_ce > 0.0


class TestThrustBallMultiRow:
    def test_two_identical_rows_combine_without_error(self, thrust_multi_row_family, catalog):
        result = thrust_multi_row_family.assemble_geometry(catalog, rows=[THRUST_ROW_KWARGS, THRUST_ROW_KWARGS])
        assert result["Ca"] > 0.0
        assert len(result["rows"]) == len(result["Q_elements"]) == 2

    def test_mixed_90_and_non_90_rows_raise(self, thrust_multi_row_family, catalog):
        with pytest.raises(ValueError, match="mixed"):
            thrust_multi_row_family.assemble_geometry(catalog, rows=[THRUST_ROW_KWARGS, THRUST_ROW_KWARGS_NON_90])


# =====================================================================
# CylindricalRollerFamily -- radial, line contact
# =====================================================================

class TestCylindricalRollerFamily:
    def test_returns_all_required_fields(self, assembled_cyl_roller):
        from axisforge.core.machine_elements.bearings.families.family import CylindricalRollerFamily
        for key in CylindricalRollerFamily.REQUIRED_FOR["line_contact"]:
            assert key in assembled_cyl_roller

    def test_rejects_locating_arrangement(self, cylindrical_roller_family, catalog):
        class _Locating:
            arrangement, label, designation = "locating", None, "TEST"
        with pytest.raises(ValueError, match="cannot be 'locating'"):
            cylindrical_roller_family.assemble_geometry(_Locating(), **{
                "Dwe": 8.0, "Lwe": 8.0, "Dpw": 72.5, "Z": 18, "s": 0.01, "n_s": 40, "alpha_0_deg": 0.0})

    def test_rejects_n_s_below_30(self, cylindrical_roller_family, catalog_floating):
        with pytest.raises(ValueError, match="n_s must be"):
            cylindrical_roller_family.assemble_geometry(catalog_floating, **{
                "Dwe": 8.0, "Lwe": 8.0, "Dpw": 72.5, "Z": 18, "s": 0.01, "n_s": 20, "alpha_0_deg": 0.0})


# =====================================================================
# ThrustCylindricalRollerFamily / RollerThrustMultiRowFamily
# =====================================================================

class TestThrustCylindricalRollerFamily:
    def test_returns_all_required_fields(self, assembled_thrust_cyl_roller_90):
        from axisforge.core.machine_elements.bearings.families.family import ThrustCylindricalRollerFamily
        for key in ThrustCylindricalRollerFamily.REQUIRED_FOR["line_contact"]:
            assert key in assembled_thrust_cyl_roller_90

    def test_90deg_row_uses_90deg_capacity_class(self, assembled_thrust_cyl_roller_90):
        from axisforge.core.machine_elements.bearings.families.family import ThrustCylindricalRollerFamily
        assert ThrustCylindricalRollerFamily._capacity_class(assembled_thrust_cyl_roller_90["alpha_0"]) is bcap.LineContactCapacityThrust_90deg

    def test_non_90deg_row_uses_non_90deg_capacity_class(self, assembled_thrust_cyl_roller_non90):
        from axisforge.core.machine_elements.bearings.families.family import ThrustCylindricalRollerFamily
        assert ThrustCylindricalRollerFamily._capacity_class(assembled_thrust_cyl_roller_non90["alpha_0"]) is bcap.LineContactCapacityThrust_Non_90deg

    def test_dynamic_capacity_positive(self, assembled_thrust_cyl_roller_90):
        from axisforge.core.machine_elements.bearings.families.family import ThrustCylindricalRollerFamily
        assert ThrustCylindricalRollerFamily.dynamic_capacity(RowAsBearing(assembled_thrust_cyl_roller_90)) > 0.0


class TestRollerThrustMultiRow:
    def test_two_identical_rows_combine_without_error(self, thrust_cyl_roller_multi_row_family, catalog):
        result = thrust_cyl_roller_multi_row_family.assemble_geometry(
            catalog, rows=[THRUST_CYL_ROLLER_KWARGS, THRUST_CYL_ROLLER_KWARGS])
        assert result["Ca"] > 0.0
        assert len(result["rows"]) == len(result["Q_elements"]) == 2

    def test_mixed_90_and_non_90_rows_raise(self, thrust_cyl_roller_multi_row_family, catalog):
        with pytest.raises(ValueError, match="mixed"):
            thrust_cyl_roller_multi_row_family.assemble_geometry(
                catalog, rows=[THRUST_CYL_ROLLER_KWARGS, THRUST_CYL_ROLLER_KWARGS_NON_90])