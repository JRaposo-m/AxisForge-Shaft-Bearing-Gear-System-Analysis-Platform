# test_validation.py
"""Testes de validation.py -- check_bearing_ready, o único check
contact-agnostico corrido antes de um solve ISO/TS 16281.

FA_FLOATING_EPS/warn_if_floating_loaded foram removidos do
validation.py atual -- essa validação (bearing floating mas com Fa
aplicado) não existe mais aqui. O gap correspondente (o solver não
avisa nem rejeita esse caso hoje) está documentado como xfail em
test_contact_solver.py::TestSolverBaseSolve::
test_warns_when_floating_bearing_is_axially_loaded -- por indicação
explícita, o fix correto é do lado do solver de FEM (não deveria ser
possível chegar aqui nesse estado), não um warning aqui."""
from types import SimpleNamespace

import pytest

from axisforge.core.machine_elements.bearings.families.family import (
    CylindricalRollerFamily, DeepGrooveBallFamily,
)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.validation import (
    check_bearing_ready,
)


class TestCheckBearingReady:
    def test_passes_when_family_matches_capability_and_analysis_done(self):
        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=True)
        check_bearing_ready(bearing, "b1", "point_contact")  # não deve levantar

    def test_line_contact_family_passes_line_contact_capability(self):
        bearing = SimpleNamespace(family=CylindricalRollerFamily(), iso16281_analysis=True)
        check_bearing_ready(bearing, "b2", "line_contact")  # não deve levantar

    def test_raises_when_family_does_not_match_capability(self):
        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=True)
        with pytest.raises(RuntimeError, match="is not a line_contact family"):
            check_bearing_ready(bearing, "b1", "line_contact")

    def test_raises_when_iso16281_analysis_is_false(self):
        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=False)
        with pytest.raises(RuntimeError, match="iso16281_analysis is not True"):
            check_bearing_ready(bearing, "b1", "point_contact")

    def test_raises_when_iso16281_analysis_attribute_absent(self):
        bearing = SimpleNamespace(family=DeepGrooveBallFamily())
        with pytest.raises(RuntimeError, match="iso16281_analysis is not True"):
            check_bearing_ready(bearing, "b1", "point_contact")

    def test_family_check_takes_priority_over_analysis_check(self):
        """Family errada E iso16281_analysis em falta ao mesmo tempo --
        deve reportar o erro de family, não o de analysis (é a primeira
        verificação no código)."""
        bearing = SimpleNamespace(family=DeepGrooveBallFamily())
        with pytest.raises(RuntimeError, match="is not a line_contact family"):
            check_bearing_ready(bearing, "b1", "line_contact")

    def test_error_message_includes_label(self):
        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=False)
        with pytest.raises(RuntimeError, match="b1"):
            check_bearing_ready(bearing, "b1", "point_contact")

    def test_unknown_capability_skips_family_check_but_still_checks_analysis(self):
        """capability desconhecida (typo, ou nova capability sem entrada em
        _CAPABILITY_CHECK) -- check_bearing_ready não recusa a capability
        em si (isso é papel de register_contact_solver, chamado no
        registo do solver, não aqui); mas iso16281_analysis continua a
        ser exigido."""
        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=False)
        with pytest.raises(RuntimeError, match="iso16281_analysis is not True"):
            check_bearing_ready(bearing, "b1", "torque_contact")

    def test_unknown_capability_with_analysis_done_does_not_raise(self):
        bearing = SimpleNamespace(family=DeepGrooveBallFamily(), iso16281_analysis=True)
        check_bearing_ready(bearing, "b1", "torque_contact")  # não deve levantar