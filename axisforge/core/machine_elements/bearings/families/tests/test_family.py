# test_family.py
"""Testes de families/family.py -- por família concreta (assemble_geometry,
validações, dispatch de capacidade, combinação multi-fila).

O antigo `TestBearingFamilyContract` genérico (parametrizado sobre
_FAMILY_REGISTRY, verificando BEARING_TYPE/DUTY/CAPABILITIES/REQUIRED_FOR
em toda família) testava um mecanismo que já não existe no código --
`BearingFamily.__init_subclass__` hoje só chama super(), não valida nada,
e nenhuma família concreta define esses atributos. Substituído abaixo por
`TestBearingFamilyMinimalContract`, que só verifica o que ainda é
verdade (nome não-vazio, registo correto).

Nota sobre materiais: as famílias com contact=ContactAnalysis.ISO16281
recebem e1/e2/nu1/nu2 como floats crus -- não há Surfaces/Material aqui.

SelfAligningBallFamily e ThrustCylindricalRollerFamily/RollerThrustMultiRowFamily
estão sabidamente desatualizadas (ver review). Os testes abaixo para
essas documentam o bug atual via xfail(strict=True) -- se começarem a
passar sem que o teste tenha sido atualizado, o xfail falha alto e avisa."""
import math
import pytest
import numpy as np

from axisforge.core.machine_elements.bearings.families.family import _FAMILY_REGISTRY
from axisforge.core.machine_elements.bearings.families import capacity as bcap
from axisforge.core.machine_elements.bearings.families.tests.conftest import (
    DEEP_GROOVE_KWARGS, ANGULAR_CONTACT_KWARGS, SELF_ALIGNING_KWARGS,
    THRUST_ROW_KWARGS, THRUST_ROW_KWARGS_NON_90,
    THRUST_CYL_ROLLER_KWARGS,
    RowAsBearing,
)


# =====================================================================
# contrato mínimo -- corre para toda família registada via
# @register_family, sem precisar de saber os nomes de antemão.
# =====================================================================

@pytest.mark.parametrize("family_cls", list(_FAMILY_REGISTRY.values()), ids=list(_FAMILY_REGISTRY.keys()))
class TestBearingFamilyMinimalContract:
    def test_name_is_nonempty_string(self, family_cls):
        assert isinstance(family_cls().name, str) and family_cls().name

    def test_registered_under_its_own_class_name(self, family_cls):
        assert _FAMILY_REGISTRY[family_cls.__name__] is family_cls


# =====================================================================
# DeepGrooveBallFamily / AngularContactFamily -- radial, point contact
# =====================================================================

class TestDeepGrooveBallFamily:
    def test_returns_all_required_fields(self, assembled_deep_groove_contact):
        for key in ("ri", "re", "Dw", "Dpw", "Z", "e1", "e2", "nu1", "nu2",
                    "A", "alpha_0", "Ri", "phi_j", "gamma", "cp"):
            assert key in assembled_deep_groove_contact

    def test_contact_none_omits_material_and_hertz_fields(self, assembled_deep_groove):
        """Sem contact=ISO16281, nada relacionado com materiais/Hertz
        deve aparecer -- nem sequer como None (ver ContactAnalysis.NONE
        em contact_models/analysis.py)."""
        for key in ("e1", "e2", "nu1", "nu2", "cp", "Ri", "iso16281_analysis"):
            assert key not in assembled_deep_groove

    def test_dynamic_capacity_positive(self, assembled_deep_groove):
        from axisforge.core.machine_elements.bearings.families.family import DeepGrooveBallFamily
        Ca = DeepGrooveBallFamily.dynamic_capacity(RowAsBearing(assembled_deep_groove))
        assert Ca > 0.0

    @pytest.mark.parametrize("i", [0, 3])
    def test_rejects_unsupported_row_count(self, deep_groove_family, catalog, i):
        with pytest.raises(ValueError, match="i must be one of"):
            deep_groove_family.assemble_geometry(catalog, **{**DEEP_GROOVE_KWARGS, "i": i})

    def test_rejects_negative_s(self, deep_groove_family, catalog):
        with pytest.raises(ValueError, match="s must be >= 0"):
            deep_groove_family.assemble_geometry(catalog, **{**DEEP_GROOVE_KWARGS, "s": -0.01})


class TestAngularContactFamily:
    def test_returns_all_required_fields(self, assembled_angular_contact_contact):
        for key in ("ri", "re", "Dw", "Dpw", "Z", "e1", "e2", "nu1", "nu2",
                    "A", "alpha_0", "Ri", "phi_j", "gamma", "cp"):
            assert key in assembled_angular_contact_contact

    @pytest.mark.parametrize("alpha_0_deg", [0.0, 45.1, -5.0])
    def test_rejects_alpha_0_deg_out_of_range(self, angular_contact_family, catalog, alpha_0_deg):
        with pytest.raises(ValueError, match="alpha_0_deg must be in"):
            angular_contact_family.assemble_geometry(
                catalog, **{**ANGULAR_CONTACT_KWARGS, "alpha_0_deg": alpha_0_deg})

    def test_dynamic_capacity_positive(self, assembled_angular_contact):
        from axisforge.core.machine_elements.bearings.families.family import AngularContactFamily
        Ca = AngularContactFamily.dynamic_capacity(RowAsBearing(assembled_angular_contact))
        assert Ca > 0.0


class TestSelfAligningBallFamily:
    """SelfAligningBallFamily.assemble_geometry tem um bug de ordem
    conhecido e não corrigido: `A = ri + re - Dw` lê `ri`/`re` antes de
    `self.reference_raceway_radii(Dw, gamma_ref)` os definir, algumas
    linhas abaixo -- toda chamada rebenta com UnboundLocalError. Sabido,
    tratado à parte (ver review). Substitui este teste por asserts reais
    quando a ordem for corrigida -- o strict=True garante que o
    esqueces-te-de-atualizar não passa despercebido."""

    @pytest.mark.xfail(raises=UnboundLocalError, strict=True,
                        reason="ri/re usados antes de atribuídos em SelfAligningBallFamily.assemble_geometry")
    def test_known_bug_ri_re_used_before_assignment(self, self_aligning_family, catalog):
        self_aligning_family.assemble_geometry(catalog, **SELF_ALIGNING_KWARGS)


# =====================================================================
# ThrustBallSingleRowFamily / ThrustBallMultiRowFamily
# =====================================================================

class TestThrustBallAssembleGeometry:
    def test_returns_all_required_fields(self, assembled_row_90deg):
        for key in ("ri", "re", "Dw", "Dpw", "gamma", "Z", "s", "A", "alpha_0",
                    "eta", "phi_j", "P_e", "f_i", "f_o", "r_rolling_el",
                    "r_inner", "r_outer", "i", "lam"):
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
        for key in ("Dwe", "Lwe", "Dpw", "Z", "s", "n_s", "x_k", "phi_j",
                    "r_rolling_el", "alpha_0", "r_inner", "r_outer",
                    "gamma", "lambda_v", "i"):
            assert key in assembled_cyl_roller

    def test_returns_hertz_fields_when_contact_enabled(self, assembled_cyl_roller_contact):
        for key in ("e1", "e2", "nu1", "nu2", "cl", "cs", "P_xk", "iso16281_analysis"):
            assert key in assembled_cyl_roller_contact

    def test_rejects_locating_arrangement(self, cylindrical_roller_family, catalog):
        class _Locating:
            arrangement, label, designation = "locating", None, "TEST"
        with pytest.raises(ValueError, match="has no flange"):
            cylindrical_roller_family.assemble_geometry(_Locating(), **{
                "Dwe": 8.0, "Lwe": 8.0, "Dpw": 72.5, "Z": 18, "s": 0.01, "n_s": 40})

    def test_rejects_n_s_below_30(self, cylindrical_roller_family, catalog_floating):
        with pytest.raises(ValueError, match="n_s must be"):
            cylindrical_roller_family.assemble_geometry(catalog_floating, **{
                "Dwe": 8.0, "Lwe": 8.0, "Dpw": 72.5, "Z": 18, "s": 0.01, "n_s": 20})

    def test_rejects_negative_s(self, cylindrical_roller_family, catalog_floating):
        with pytest.raises(ValueError, match="s must be >= 0"):
            cylindrical_roller_family.assemble_geometry(catalog_floating, **{
                "Dwe": 8.0, "Lwe": 8.0, "Dpw": 72.5, "Z": 18, "s": -0.01, "n_s": 40})

    def test_dynamic_capacity_positive(self, assembled_cyl_roller):
        from axisforge.core.machine_elements.bearings.families.family import CylindricalRollerFamily
        assert CylindricalRollerFamily.dynamic_capacity(RowAsBearing(assembled_cyl_roller)) > 0.0


# =====================================================================
# ThrustCylindricalRollerFamily / RollerThrustMultiRowFamily -- sabidamente
# desatualizadas (nunca migradas para o padrão contact=/e1..nu2; a
# chamada a LineContactStiffness usa a assinatura antiga, (Dwe, Dpw,
# alpha_0, Lwe, n_s), que já não existe -- a classe atual tem campos
# (Lwe, n_s, Dwe, x_k, _LOG_ARG_EPS) e não tem .gamma/.lamina_positions).
# Documentado como xfail, não corrigido aqui -- ver review.
# =====================================================================

class TestThrustCylindricalRollerFamily:
    @pytest.mark.xfail(raises=AttributeError, strict=True,
                        reason="ThrustCylindricalRollerFamily ainda chama LineContactStiffness "
                               "com a assinatura antiga -- a classe atual não tem .gamma")
    def test_known_bug_stale_line_contact_stiffness_call(self, thrust_cyl_roller_family, catalog):
        thrust_cyl_roller_family.assemble_geometry(catalog, **THRUST_CYL_ROLLER_KWARGS)


class TestRollerThrustMultiRow:
    @pytest.mark.xfail(raises=AttributeError, strict=True,
                        reason="delega em ThrustCylindricalRollerFamily -- mesmo bug conhecido")
    def test_known_bug_propagates_from_single_row(self, thrust_cyl_roller_multi_row_family, catalog):
        thrust_cyl_roller_multi_row_family.assemble_geometry(
            catalog, rows=[THRUST_CYL_ROLLER_KWARGS, THRUST_CYL_ROLLER_KWARGS])