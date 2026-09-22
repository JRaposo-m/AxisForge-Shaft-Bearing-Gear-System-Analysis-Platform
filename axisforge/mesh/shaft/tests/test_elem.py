# test_elem.py
"""Testes de element_type/elem.py -- BeamFormulation/
ShearDeformableBeamFormulation (contrato), EulerBernoulliBeam,
TimoshenkoBeam, FrameElement, o registry (_FORMULATION_REGISTRY /
available_formulations), ElemBase (dispatch, concreto e partilhado por
qualquer topologia), Elem (2 nós, o único usado hoje), e
QuadraticTimoshenkoElem (placeholder de design, ainda não funcional --
só se testa que continua a levantar NotImplementedError onde deve)."""
import math
import pytest

from axisforge.mesh.shaft.element_type.elem import (
    _FORMULATION_REGISTRY, EulerBernoulliBeam, TimoshenkoBeam, Elem, QuadraticTimoshenkoElem,
)
from axisforge.mesh.shaft.element_type import available_formulations
from axisforge.mesh.shaft.beam_model_settings import VALID_BEAM_THEORIES
import axisforge.mesh.shaft.element_type.shear_factor as sf_module
from axisforge.mesh.shaft.tests.conftest import ELEM_KWARGS


# =====================================================================
# Registry -- (beam_theory, element_order) -> instância singleton
# =====================================================================

class TestFormulationRegistry:
    def test_euler_bernoulli_linear_registered(self):
        assert isinstance(_FORMULATION_REGISTRY[("euler_bernoulli", "linear")], EulerBernoulliBeam)

    def test_timoshenko_linear_registered(self):
        assert isinstance(_FORMULATION_REGISTRY[("timoshenko", "linear")], TimoshenkoBeam)

    def test_every_valid_beam_theory_has_a_linear_formulation(self):
        registered = available_formulations()
        for theory in VALID_BEAM_THEORIES:
            assert (theory, "linear") in registered

    def test_available_formulations_returns_classes_not_instances(self):
        for key, cls in available_formulations().items():
            assert isinstance(cls, type)

    def test_registry_entries_are_shared_singletons(self):
        """register_formulation instancia uma vez só -- duas chamadas a
        _formulation() para o mesmo (beam_theory, element_order) têm de
        devolver o mesmo objeto, não instâncias novas."""
        a = _FORMULATION_REGISTRY[("timoshenko", "linear")]
        b = _FORMULATION_REGISTRY[("timoshenko", "linear")]
        assert a is b


# =====================================================================
# EulerBernoulliBeam
# =====================================================================

class TestEulerBernoulliBeam:
    def test_shape_functions_at_node_1_isolate_transverse_dof(self, euler_elem):
        formulation = euler_elem._formulation()
        N = formulation.shape_functions(-1.0, euler_elem)
        assert N[0] == pytest.approx(1.0)
        assert N[1] == pytest.approx(0.0, abs=1e-12)
        assert N[2] == pytest.approx(0.0, abs=1e-12)
        assert N[3] == pytest.approx(0.0, abs=1e-12)

    def test_shape_functions_at_node_2_isolate_transverse_dof(self, euler_elem):
        formulation = euler_elem._formulation()
        N = formulation.shape_functions(1.0, euler_elem)
        assert N[0] == pytest.approx(0.0, abs=1e-12)
        assert N[2] == pytest.approx(1.0)

    def test_stiffness_element_is_symmetric(self, euler_elem):
        k = euler_elem.stiffness_element()
        assert k == pytest.approx(k.T)

    def test_stiffness_element_diagonal_terms_positive(self, euler_elem):
        k = euler_elem.stiffness_element()
        assert (k.diagonal() > 0).all()

    def test_stiffness_element_translational_terms_match_closed_form(self, euler_elem):
        le = euler_elem.length
        c = euler_elem.E * euler_elem.I / le ** 3
        k = euler_elem.stiffness_element()
        assert k[0, 0] == pytest.approx(12 * c)
        assert k[0, 2] == pytest.approx(-12 * c)
        assert k[1, 1] == pytest.approx(4 * c * le ** 2)


# =====================================================================
# TimoshenkoBeam
# =====================================================================

class TestTimoshenkoBeam:
    def test_shape_functions_are_linear_2node(self, timoshenko_elem):
        formulation = timoshenko_elem._formulation()
        assert formulation.shape_functions(-1.0, timoshenko_elem) == pytest.approx([1.0, 0.0])
        assert formulation.shape_functions(1.0, timoshenko_elem) == pytest.approx([0.0, 1.0])

    def test_shear_strain_matrix_rejects_unknown_integration(self, timoshenko_elem):
        formulation = timoshenko_elem._formulation()
        with pytest.raises(ValueError, match="unknown integration method"):
            formulation.shear_strain_matrix(0.0, timoshenko_elem, "bogus")

    def test_exact_and_single_point_shear_strain_agree_at_zeta_zero(self, timoshenko_elem):
        formulation = timoshenko_elem._formulation()
        single = formulation.shear_strain_matrix(0.0, timoshenko_elem, "single_point")
        exact = formulation.shear_strain_matrix(0.0, timoshenko_elem, "exact")
        assert exact == pytest.approx(single)

    def test_stiffness_element_rejects_unknown_integration(self, timoshenko_elem):
        formulation = timoshenko_elem._formulation()
        with pytest.raises(ValueError, match="unknown integration method"):
            formulation.stiffness_element(timoshenko_elem, integration="bogus")

    def test_shear_rigidity_matches_shear_factor_module(self, timoshenko_elem):
        k_expected = sf_module.ShearFactor().cowper_factor(timoshenko_elem.v, timoshenko_elem.radius_ratio)
        G = timoshenko_elem.E / (2 * (1 + timoshenko_elem.v))
        assert timoshenko_elem.shear_rigidity() == pytest.approx(k_expected * G * timoshenko_elem.A)

    def test_shear_rigidity_honours_kGA_override(self, timoshenko_elem):
        assert timoshenko_elem.shear_rigidity(kGA_override=12345.0) == pytest.approx(12345.0)

    def test_stiffness_element_is_symmetric(self, timoshenko_elem):
        k = timoshenko_elem.stiffness_element()
        assert k == pytest.approx(k.T)

    def test_single_point_and_exact_integration_give_different_rotational_stiffness(
        self, timoshenko_elem, timoshenko_elem_exact,
    ):
        k_sp = timoshenko_elem.stiffness_element()
        k_ex = timoshenko_elem_exact.stiffness_element()
        assert not math.isclose(k_sp[1, 1], k_ex[1, 1])


# =====================================================================
# FrameElement -- axial, unconditional, fora da hierarquia BeamFormulation
# =====================================================================

class TestFrameElement:
    def test_axial_strain_matrix(self, euler_elem):
        B = euler_elem.axial_strain_matrix()
        le = euler_elem.length
        assert B == pytest.approx([-1.0 / le, 1.0 / le])

    def test_axial_stiffness_matches_closed_form(self, euler_elem):
        k = euler_elem.axial_stiffness_element()
        expected = euler_elem.E * euler_elem.A / euler_elem.length
        assert k[0, 0] == pytest.approx(expected)
        assert k[0, 1] == pytest.approx(-expected)
        assert k == pytest.approx(k.T)

    def test_axial_behaviour_is_unconditional_across_beam_theories(self, euler_elem, timoshenko_elem):
        """FrameElement não depende de beam_theory -- euler_bernoulli e
        timoshenko com a mesma geometria têm de dar a mesma rigidez
        axial."""
        assert euler_elem.axial_stiffness_element() == pytest.approx(timoshenko_elem.axial_stiffness_element())
        assert euler_elem.axial_shape_functions(0.3) == pytest.approx(timoshenko_elem.axial_shape_functions(0.3))


# =====================================================================
# ElemBase -- dispatch concreto e partilhado, independente da topologia
# =====================================================================

class TestElemBaseDispatch:
    def test_natural_coordenates_endpoints_and_midpoint(self, euler_elem):
        assert euler_elem.natural_coordenates(euler_elem.x_a) == pytest.approx(-1.0)
        assert euler_elem.natural_coordenates(euler_elem.x_b) == pytest.approx(1.0)
        midpoint = (euler_elem.x_a + euler_elem.x_b) / 2.0
        assert euler_elem.natural_coordenates(midpoint) == pytest.approx(0.0)

    def test_shear_rigidity_rejects_non_shear_deformable_theory(self, euler_elem):
        with pytest.raises(ValueError, match="only applies to shear-deformable"):
            euler_elem.shear_rigidity()

    def test_shear_strain_matrix_rejects_non_shear_deformable_theory(self, euler_elem):
        with pytest.raises(ValueError, match="only applies"):
            euler_elem.shear_strain_matrix(0.0)

    def test_stiffness_element_shape_for_euler_bernoulli(self, euler_elem):
        assert euler_elem.stiffness_element().shape == (4, 4)

    def test_stiffness_element_shape_for_timoshenko(self, timoshenko_elem):
        assert timoshenko_elem.stiffness_element().shape == (4, 4)

    def test_stiffness_element_forwards_kGA_override_only_for_shear_deformable(self, timoshenko_elem):
        k_default = timoshenko_elem.stiffness_element()
        k_override = timoshenko_elem.stiffness_element(kGA_override=1.0)
        assert not (k_default == pytest.approx(k_override))


# =====================================================================
# Elem -- validação
# =====================================================================

class TestElemValidation:
    def test_valid_euler_bernoulli_has_no_errors(self, euler_elem):
        assert euler_elem.validate() == []

    def test_valid_timoshenko_has_no_errors(self, timoshenko_elem):
        assert timoshenko_elem.validate() == []

    @pytest.mark.parametrize("overrides,expected_substr", [
        (dict(length=0.0), "length must be > 0"),
        (dict(length=-5.0), "length must be > 0"),
        (dict(E=0.0), "E must be > 0"),
        (dict(E=500.0), "may be in GPa"),
        (dict(I=-1.0), "I must be >= 0"),
        (dict(A=0.0), "A must be > 0"),
        (dict(v=0.5), "v out of physical range"),
        (dict(v=-1.0), "v out of physical range"),
        (dict(idx_node_1=1, idx_node_2=1), "must differ"),
        (dict(idx_node_1=-1), "must be >= 0"),
        (dict(x_a=50.0, x_b=50.0), "x_b must be > x_a"),
        (dict(x_a=60.0, x_b=50.0), "x_b must be > x_a"),
    ])
    def test_rejects_invalid_field(self, settings_euler_bernoulli, overrides, expected_substr):
        kwargs = {**ELEM_KWARGS, "settings": settings_euler_bernoulli, **overrides}
        elem = Elem(**kwargs)
        errors = elem.validate()
        assert any(expected_substr in e for e in errors)

    def test_validate_or_raise_raises_on_bad_field(self, settings_euler_bernoulli):
        kwargs = {**ELEM_KWARGS, "settings": settings_euler_bernoulli, "length": -1.0}
        elem = Elem(**kwargs)
        with pytest.raises(ValueError):
            elem.validate_or_raise()

    def test_validate_or_raise_does_not_raise_when_valid(self, euler_elem):
        euler_elem.validate_or_raise()  # não deve levantar

    def test_hutchinson_with_hollow_section_rejected(self, settings_timoshenko_hutchinson):
        kwargs = {**ELEM_KWARGS, "settings": settings_timoshenko_hutchinson, "radius_ratio": 0.5}
        elem = Elem(**kwargs)
        errors = elem.validate()
        assert any("hutchinson" in e and "not yet implemented" in e for e in errors)

    def test_hutchinson_with_solid_section_is_fine(self, settings_timoshenko_hutchinson):
        kwargs = {**ELEM_KWARGS, "settings": settings_timoshenko_hutchinson, "radius_ratio": 0.0}
        elem = Elem(**kwargs)
        assert elem.validate() == []


# =====================================================================
# Elem -- construção rejeita combinações não registadas no
# _FORMULATION_REGISTRY (settings genuínas do BeamModelSettings nunca
# produzem isto -- só chega aqui com um objeto settings "à mão").
# =====================================================================

class _FakeSettings:
    def __init__(self, beam_theory, shear_theory=None, integration_method=None, element_order=None):
        self.beam_theory = beam_theory
        self.shear_theory = shear_theory
        self.integration_method = integration_method
        if element_order is not None:
            self.element_order = element_order


class TestElemFormulationRegistryLookup:
    def test_rejects_unregistered_beam_theory(self):
        settings = _FakeSettings(beam_theory="bogus")
        with pytest.raises(ValueError, match="no formulation registered"):
            Elem(**{**ELEM_KWARGS, "settings": settings})

    def test_rejects_unregistered_element_order(self):
        settings = _FakeSettings(beam_theory="timoshenko", shear_theory="cowper",
                                  integration_method="single_point", element_order="quadratic")
        with pytest.raises(ValueError, match="no formulation registered"):
            Elem(**{**ELEM_KWARGS, "settings": settings})

    def test_element_order_defaults_to_linear_when_absent_from_settings(self, settings_euler_bernoulli):
        """settings sem atributo element_order (ex.: BeamModelSettings
        hoje) tem de cair em 'linear' via getattr(..., 'linear')."""
        elem = Elem(**{**ELEM_KWARGS, "settings": settings_euler_bernoulli})
        assert elem.element_order == "linear"


# =====================================================================
# Elem -- find_node_index (staticmethod usado por from_mesh)
# =====================================================================

class TestElemFindNodeIndex:
    def test_finds_exact_match(self):
        assert Elem.find_node_index([0.0, 10.0, 20.0], 10.0) == 1

    def test_finds_match_within_tolerance(self):
        assert Elem.find_node_index([0.0, 10.0, 20.0], 10.0 + 1e-9, tol=1e-6) == 1

    def test_raises_when_no_node_within_tolerance(self):
        with pytest.raises(ValueError, match="No node found"):
            Elem.find_node_index([0.0, 10.0, 20.0], 15.0, tol=0.01)

    def test_returns_first_match_when_ambiguous(self):
        """Nós a 0.0 e 0.05 mm, tol=1.0 -- ambos batem; find_node_index
        devolve o primeiro na ordem de iteração, não o mais próximo."""
        assert Elem.find_node_index([0.0, 0.05, 20.0], 0.02, tol=1.0) == 0


# =====================================================================
# Elem -- repr
# =====================================================================

class TestElemRepr:
    def test_repr_contains_key_fields(self, timoshenko_elem):
        text = repr(timoshenko_elem)
        assert "timoshenko" in text
        assert "cowper" in text
        assert "nodes=(0, 1)" in text


# =====================================================================
# QuadraticTimoshenkoElem -- placeholder de design, não funcional
# =====================================================================

class TestQuadraticTimoshenkoElemPlaceholder:
    def test_init_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            QuadraticTimoshenkoElem()

    def test_node_indices_raises_not_implemented(self):
        """object.__new__ evita passar por __init__ (que já rebenta
        sozinho) só para conseguir chegar à property abstrata."""
        instance = object.__new__(QuadraticTimoshenkoElem)
        with pytest.raises(NotImplementedError):
            _ = instance.node_indices

    def test_validate_raises_not_implemented(self):
        instance = object.__new__(QuadraticTimoshenkoElem)
        with pytest.raises(NotImplementedError):
            instance.validate()
