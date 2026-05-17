"""
tests/test_core/test_loads.py
Unit tests for RadialLoad, AxialLoad, TorqueLoad, ExternalMoment, LoadPlane.
"""

import pytest
from core.loads import (
    RadialLoad, AxialLoad, TorqueLoad, ExternalMoment, LoadPlane
)


class TestLoadPlane:
    def test_enum_values_exist(self):
        assert LoadPlane.XZ
        assert LoadPlane.XY
        assert LoadPlane.AXIAL

    def test_xz_xy_are_different(self):
        assert LoadPlane.XZ != LoadPlane.XY


class TestRadialLoad:

    def test_valid_creation_xz(self):
        l = RadialLoad(position=100.0, magnitude=5000.0, plane=LoadPlane.XZ)
        assert l.position == 100.0
        assert l.magnitude == 5000.0
        assert l.plane == LoadPlane.XZ

    def test_valid_creation_xy(self):
        l = RadialLoad(position=200.0, magnitude=3000.0, plane=LoadPlane.XY, label="F1")
        assert l.label == "F1"

    def test_raises_negative_position(self):
        with pytest.raises(ValueError, match="negative"):
            RadialLoad(position=-1.0, magnitude=1000.0, plane=LoadPlane.XY)

    def test_raises_axial_plane(self):
        with pytest.raises(ValueError, match="AXIAL"):
            RadialLoad(position=100.0, magnitude=1000.0, plane=LoadPlane.AXIAL)

    def test_zero_position_valid(self):
        l = RadialLoad(position=0.0, magnitude=1000.0, plane=LoadPlane.XZ)
        assert l.position == 0.0

    def test_zero_magnitude_valid(self):
        """Zero load is valid — solver will sum it without effect."""
        l = RadialLoad(position=100.0, magnitude=0.0, plane=LoadPlane.XY)
        assert l.magnitude == 0.0

    def test_negative_magnitude_valid(self):
        """Negative magnitude is valid — direction is sign-encoded."""
        l = RadialLoad(position=100.0, magnitude=-500.0, plane=LoadPlane.XZ)
        assert l.magnitude == -500.0


class TestAxialLoad:

    def test_valid_creation(self):
        l = AxialLoad(position=100.0, magnitude=2000.0)
        assert l.position == 100.0
        assert l.magnitude == 2000.0

    def test_raises_negative_position(self):
        with pytest.raises(ValueError, match="negative"):
            AxialLoad(position=-10.0, magnitude=1000.0)

    def test_positive_negative_magnitude_valid(self):
        l = AxialLoad(position=50.0, magnitude=-1000.0)
        assert l.magnitude == -1000.0

    def test_default_label(self):
        l = AxialLoad(position=0.0, magnitude=0.0)
        assert l.label == ""


class TestTorqueLoad:

    def test_valid_creation(self):
        l = TorqueLoad(position=200.0, magnitude=150_000.0)
        assert l.magnitude == 150_000.0

    def test_raises_negative_position(self):
        with pytest.raises(ValueError, match="negative"):
            TorqueLoad(position=-5.0, magnitude=100.0)

    def test_negative_torque_valid(self):
        """Negative torque = opposite rotation direction."""
        l = TorqueLoad(position=100.0, magnitude=-50_000.0)
        assert l.magnitude == -50_000.0

    def test_zero_torque_valid(self):
        l = TorqueLoad(position=0.0, magnitude=0.0)
        assert l.magnitude == 0.0


class TestExternalMoment:

    def test_valid_creation_xz(self):
        m = ExternalMoment(position=50.0, magnitude=10_000.0, plane=LoadPlane.XZ)
        assert m.plane == LoadPlane.XZ

    def test_valid_creation_xy(self):
        m = ExternalMoment(position=100.0, magnitude=5_000.0, plane=LoadPlane.XY)
        assert m.magnitude == 5_000.0

    def test_raises_negative_position(self):
        with pytest.raises(ValueError, match="negative"):
            ExternalMoment(position=-1.0, magnitude=1000.0, plane=LoadPlane.XY)

    def test_raises_axial_plane(self):
        with pytest.raises(ValueError, match="AXIAL"):
            ExternalMoment(position=100.0, magnitude=1000.0, plane=LoadPlane.AXIAL)
