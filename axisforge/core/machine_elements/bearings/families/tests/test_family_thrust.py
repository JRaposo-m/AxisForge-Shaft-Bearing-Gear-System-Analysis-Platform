"""Testes de ThrustBallSingleRowFamily / ThrustBallMultiRowFamily --
families/family.py. Foco: contrato (campos presentes, tipos, exceções
esperadas), não os valores numéricos finais (esses ainda por validar
contra a norma -- ver notas nos testes marcados com TODO)."""
import math
import pytest
import numpy as np

from axisforge.core.machine_elements.bearings.families.family import (
    ThrustBallSingleRowFamily, ThrustBallMultiRowFamily,
)
from axisforge.core.machine_elements.bearings.families.tests.conftest import THRUST_ROW_KWARGS, THRUST_ROW_KWARGS_NON_90


class TestAssembleGeometry:
    def test_returns_all_required_fields(self, assembled_row_90deg):
        for key in ThrustBallSingleRowFamily.REQUIRED_FOR["point_contact"]:
            assert key in assembled_row_90deg, f"missing required field: {key}"

    def test_also_returns_eta_and_lam(self, assembled_row_90deg):
        # eta/lam não estão em REQUIRED_FOR mas são necessários para a
        # capacity -- ver "precisa de confirmação" sobre Bearing.assemble()
        assert "eta" in assembled_row_90deg
        assert "lam" in assembled_row_90deg
        assert assembled_row_90deg["lam"] == ThrustBallSingleRowFamily.LAM

    def test_alpha_0_deg_none_defaults_to_90deg(self, thrust_single_row_family, catalog):
        row = thrust_single_row_family.assemble_geometry(
            catalog, **{**THRUST_ROW_KWARGS, "alpha_0_deg": None})
        assert math.isclose(row["alpha_0"], np.pi / 2)
        assert row["s"] == 0.0

    def test_rejects_non_scalar_Z(self, thrust_single_row_family, catalog):
        with pytest.raises(TypeError, match="Z must be a scalar int"):
            thrust_single_row_family.assemble_geometry(
                catalog, **{**THRUST_ROW_KWARGS, "Z": [12, 12]})

    @pytest.mark.parametrize("alpha_0_deg", [0.0, 45.0, 90.1, -10.0])
    def test_rejects_alpha_0_deg_out_of_range(self, thrust_single_row_family, catalog, alpha_0_deg):
        with pytest.raises(ValueError, match="alpha_0_deg must be in"):
            thrust_single_row_family.assemble_geometry(
                catalog, **{**THRUST_ROW_KWARGS, "alpha_0_deg": alpha_0_deg})

    def test_gamma_in_open_unit_interval(self, assembled_row_90deg):
        # PointContactCapacityRadial já valida isto no __init__; aqui só
        # confirmamos que a geometria produz um gamma fisicamente plausível
        assert 0.0 < assembled_row_90deg["gamma"] < 1.0


class TestCapacityDispatch:
    def test_90deg_row_uses_90deg_capacity_class(self, assembled_row_90deg):
        import axisforge.core.machine_elements.bearings.families.capacity as bcap
        cls_ = ThrustBallSingleRowFamily._capacity_class(assembled_row_90deg["alpha_0"])
        assert cls_ is bcap.PointContactCapacityThrust_90deg

    def test_non_90deg_row_uses_non_90deg_capacity_class(self, assembled_row_non_90deg):
        import axisforge.core.machine_elements.bearings.families.capacity as bcap
        cls_ = ThrustBallSingleRowFamily._capacity_class(assembled_row_non_90deg["alpha_0"])
        assert cls_ is bcap.PointContactCapacityThrust_Non_90deg

    def test_dynamic_capacity_returns_positive_float(self, thrust_single_row_family, assembled_row_90deg):
        Ca = ThrustBallSingleRowFamily.dynamic_capacity(_RowAsBearing(assembled_row_90deg))
        assert isinstance(Ca, float)
        assert Ca > 0.0

    def test_per_element_dynamic_capacity_returns_pair(self, assembled_row_90deg):
        Q_ci, Q_ce = ThrustBallSingleRowFamily.per_element_dynamic_capacity(
            _RowAsBearing(assembled_row_90deg))
        assert Q_ci > 0.0 and Q_ce > 0.0


class TestMultiRow:
    def test_two_identical_rows_combine_without_error(self, thrust_multi_row_family, catalog):
        result = thrust_multi_row_family.assemble_geometry(
            catalog, rows=[THRUST_ROW_KWARGS, THRUST_ROW_KWARGS])
        assert "Ca" in result
        assert result["Ca"] > 0.0
        assert len(result["rows"]) == 2
        assert len(result["Q_elements"]) == 2

    def test_mixed_90_and_non_90_rows_raise(self, thrust_multi_row_family, catalog):
        with pytest.raises(ValueError, match="mixed"):
            thrust_multi_row_family.assemble_geometry(
                catalog, rows=[THRUST_ROW_KWARGS, THRUST_ROW_KWARGS_NON_90])


class _RowAsBearing:
    """Pequeno adaptador para os testes chamarem dynamic_capacity()/
    per_element_dynamic_capacity() com um dict de assemble_geometry()
    como se fosse um Bearing (getattr em vez de __getitem__)."""
    def __init__(self, row: dict):
        self.__dict__.update(row)