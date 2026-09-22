# conftest.py
"""Fixtures partilhadas pelos testes de mesh/shaft/."""
import pytest

from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings


# ---------------------------------------------------------------------
# BeamModelSettings -- uma instância por combinação usada nos testes de
# elem.py
# ---------------------------------------------------------------------

@pytest.fixture
def settings_euler_bernoulli():
    return BeamModelSettings(beam_theory="euler_bernoulli", shear_theory=None, integration_method=None)


@pytest.fixture
def settings_timoshenko_single_point():
    return BeamModelSettings(beam_theory="timoshenko", shear_theory="cowper", integration_method="single_point")


@pytest.fixture
def settings_timoshenko_exact():
    return BeamModelSettings(beam_theory="timoshenko", shear_theory="cowper", integration_method="exact")


@pytest.fixture
def settings_timoshenko_hutchinson():
    return BeamModelSettings(beam_theory="timoshenko", shear_theory="hutchinson", integration_method="single_point")


# ---------------------------------------------------------------------
# Elem -- geometria/material de referência plausível, não de catálogo
# real, só para exercitar o código.
# ---------------------------------------------------------------------

ELEM_KWARGS = dict(length=50.0, E=210_000.0, I=1.0e5, A=500.0, v=0.3,
                    idx_node_1=0, idx_node_2=1, x_a=0.0, x_b=50.0)


@pytest.fixture
def euler_elem(settings_euler_bernoulli):
    from axisforge.mesh.shaft.element_type.elem import Elem
    return Elem(**ELEM_KWARGS, settings=settings_euler_bernoulli)


@pytest.fixture
def timoshenko_elem(settings_timoshenko_single_point):
    from axisforge.mesh.shaft.element_type.elem import Elem
    return Elem(**ELEM_KWARGS, settings=settings_timoshenko_single_point)


@pytest.fixture
def timoshenko_elem_exact(settings_timoshenko_exact):
    from axisforge.mesh.shaft.element_type.elem import Elem
    return Elem(**ELEM_KWARGS, settings=settings_timoshenko_exact)


# ---------------------------------------------------------------------
# Mesh1D -- fakes leves para ShaftSystem/Shaft/bearings/gears/loads,
# evitando montar a hierarquia real (GearElement/ShaftSystem/materiais)
# só para testar a lógica de _mandatory_positions()/_create_mesh().
# Mesh1D só lê estes atributos por duck typing, nunca isinstance.
# ---------------------------------------------------------------------

class FakeShaft:
    """Substitui Shaft: só n_sections/axial_start/axial_end importam
    aqui. `boundaries` são as fronteiras de secção consecutivas, ex.
    [0.0, 40.0, 100.0] -> 2 secções."""

    def __init__(self, boundaries):
        self._boundaries = list(boundaries)

    @property
    def n_sections(self):
        return len(self._boundaries) - 1

    def axial_start(self, i):
        return self._boundaries[i]

    def axial_end(self, i):
        return self._boundaries[i + 1]


class _Positioned:
    """Fake genérico para RadialLoad/ExternalMoment/etc. -- só
    `.position` é lido por Mesh1D._mandatory_positions()."""

    def __init__(self, position):
        self.position = position


class FakeBearing(_Positioned):
    def __init__(self, position, extent=None):
        super().__init__(position)
        self.extent = extent if extent is not None else (position, position)


class FakeGear(_Positioned):
    def __init__(self, position, extent=None):
        super().__init__(position)
        self.extent = extent if extent is not None else (position, position)


class FakeDistributedLoad:
    def __init__(self, x_lo, x_hi, centroid):
        self.x_lo = x_lo
        self.x_hi = x_hi
        self._centroid = centroid

    def centroid(self, plane):
        # plane (LoadPlane.XY) é ignorado de propósito -- o fake não
        # precisa de saber o que é, só de responder algo plausível.
        return self._centroid


class FakeShaftSystem:
    """Duck type de ShaftSystem -- só a superfície lida por
    Mesh1D._mandatory_positions(): .shaft, .bearings, .gears,
    .radial_loads, .axial_loads, .torque_loads, .external_moments,
    .distributed_radial_loads, .bearing_extent(), .gear_extent()."""

    def __init__(self, shaft, bearings=(), gears=(), radial_loads=(),
                 axial_loads=(), torque_loads=(), external_moments=(),
                 distributed_radial_loads=()):
        self.shaft = shaft
        self.bearings = list(bearings)
        self.gears = list(gears)
        self.radial_loads = list(radial_loads)
        self.axial_loads = list(axial_loads)
        self.torque_loads = list(torque_loads)
        self.external_moments = list(external_moments)
        self.distributed_radial_loads = list(distributed_radial_loads)

    def bearing_extent(self, bearing):
        return bearing.extent

    def gear_extent(self, gear):
        return gear.extent


@pytest.fixture
def fake_shaft_system():
    """Um veio simples: um único troço, 0 a 100 mm, sem bearings, gears
    ou loads."""
    return FakeShaftSystem(shaft=FakeShaft([0.0, 100.0]))
