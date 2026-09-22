# test_beam_model_settings.py
"""Testes de beam_model_settings.py -- BeamModelSettings valida a
combinação beam_theory/shear_theory/integration_method no
__post_init__ (dataclass frozen, sem defaults -- ver docstring da
própria classe)."""
import dataclasses
import pytest

from axisforge.mesh.shaft.beam_model_settings import (
    BeamModelSettings, VALID_SHEAR_THEORIES, VALID_INTEGRATION_METHODS,
)


class TestEulerBernoulli:
    def test_valid_construction(self):
        settings = BeamModelSettings(beam_theory="euler_bernoulli", shear_theory=None, integration_method=None)
        assert settings.beam_theory == "euler_bernoulli"
        assert settings.shear_theory is None
        assert settings.integration_method is None

    def test_rejects_shear_theory(self):
        with pytest.raises(ValueError, match="does not use shear_theory"):
            BeamModelSettings(beam_theory="euler_bernoulli", shear_theory="cowper", integration_method=None)

    def test_rejects_integration_method(self):
        with pytest.raises(ValueError, match="does not use"):
            BeamModelSettings(beam_theory="euler_bernoulli", shear_theory=None, integration_method="exact")


class TestTimoshenko:
    @pytest.mark.parametrize("shear_theory", VALID_SHEAR_THEORIES)
    @pytest.mark.parametrize("integration_method", VALID_INTEGRATION_METHODS)
    def test_valid_combinations_construct(self, shear_theory, integration_method):
        settings = BeamModelSettings(beam_theory="timoshenko", shear_theory=shear_theory,
                                      integration_method=integration_method)
        assert settings.shear_theory == shear_theory
        assert settings.integration_method == integration_method

    def test_rejects_invalid_shear_theory(self):
        with pytest.raises(ValueError, match="shear_theory must be one of"):
            BeamModelSettings(beam_theory="timoshenko", shear_theory="bogus", integration_method="exact")

    def test_rejects_missing_shear_theory(self):
        with pytest.raises(ValueError, match="shear_theory must be one of"):
            BeamModelSettings(beam_theory="timoshenko", shear_theory=None, integration_method="exact")

    def test_rejects_invalid_integration_method(self):
        with pytest.raises(ValueError, match="integration_method must be one of"):
            BeamModelSettings(beam_theory="timoshenko", shear_theory="cowper", integration_method="bogus")

    def test_rejects_missing_integration_method(self):
        with pytest.raises(ValueError, match="integration_method must be one of"):
            BeamModelSettings(beam_theory="timoshenko", shear_theory="cowper", integration_method=None)


class TestBeamTheoryValidation:
    def test_rejects_unknown_beam_theory(self):
        with pytest.raises(ValueError, match="beam_theory must be one of"):
            BeamModelSettings(beam_theory="bogus", shear_theory=None, integration_method=None)


class TestFrozenDataclass:
    def test_is_frozen(self):
        settings = BeamModelSettings(beam_theory="euler_bernoulli", shear_theory=None, integration_method=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.beam_theory = "timoshenko"
