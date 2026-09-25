# test_contact.py
"""Testes de iso16281_contact.py. LineContactStiffness -- verificado a
fundo, tenho o código-fonte. PointContactStiffness,
SelfAligningPointContactStiffness, contact_angle_and_clearance --
só testes de contrato (assinatura, sinais plausíveis), nunca vi o corpo
destas três nesta conversa, por isso não valido a fórmula, só que não
rebentam e devolvem algo fisicamente plausível.

Nota: `PointContactStiffness.stiffness`/`LineContactStiffness.stiffness`/
`.lamina_stiffness` foram renomeados para `.cp`/`.cl`/`.cs` -- não são a
stiffness montada do rolamento (isso é o solver ISO 16281 ou o Hertz via
slippy que calcula), mas a constante de carga-deformação Hertziana que
alimenta esse cálculo. Só o nome mudou; os testes abaixo continuam a
verificar exactamente o mesmo comportamento."""
import math
import pytest
import numpy as np

from axisforge.core.machine_elements.bearings.families import iso16281_contact as bc


class TestLineContactStiffness:
    def test_lamina_positions_shape_and_symmetry(self):
        stiff = bc.LineContactStiffness(Dwe=8.0, Dpw=72.5, alpha_0=0.0, Lwe=8.0, n_s=40)
        x_k = stiff.lamina_positions
        assert x_k.shape == (40,)
        assert np.all(x_k > -4.0) and np.all(x_k < 4.0)
        # simetria em torno de 0: soma tem de ser ~0 para n_s par
        assert math.isclose(x_k.sum(), 0.0, abs_tol=1e-9)

    def test_gamma_branch_90deg_uses_Dwe_over_Dpw(self):
        stiff = bc.LineContactStiffness(Dwe=8.0, Dpw=72.5, alpha_0=np.pi / 2, Lwe=8.0, n_s=40)
        assert math.isclose(stiff.gamma, 8.0 / 72.5)

    def test_gamma_branch_non_90deg_includes_cosine(self):
        alpha_0 = np.radians(30.0)
        stiff = bc.LineContactStiffness(Dwe=8.0, Dpw=72.5, alpha_0=alpha_0, Lwe=8.0, n_s=40)
        assert math.isclose(stiff.gamma, 8.0 * np.cos(alpha_0) / 72.5)

    def test_cs_splits_cl_by_n_s(self):
        stiff = bc.LineContactStiffness(Dwe=8.0, Dpw=72.5, alpha_0=0.0, Lwe=8.0, n_s=40)
        assert math.isclose(stiff.cs, stiff.cl / 40)

    def test_cl_scales_with_Lwe(self):
        short = bc.LineContactStiffness(Dwe=8.0, Dpw=72.5, alpha_0=0.0, Lwe=4.0, n_s=40)
        long_ = bc.LineContactStiffness(Dwe=8.0, Dpw=72.5, alpha_0=0.0, Lwe=8.0, n_s=40)
        assert long_.cl > short.cl


# =====================================================================
# contrato apenas -- não vi o corpo destas funções/classes nesta
# conversa; se os nomes/assinaturas abaixo não baterem certo com o
# ficheiro real, ajusta -- não inventei a física, só a forma da chamada
# como foi usada em family.py.
# =====================================================================

class TestPointContactStiffnessContract:
    def test_returns_plausible_gamma_Ri_cp(self):
        stiff = bc.PointContactStiffness(Dw=8.0, ri=4.3, re=4.3, E=210_000.0, nu=0.3,
                                          alpha_0=0.0, Dpw=40.0)
        assert 0.0 < stiff.gamma < 1.0
        assert stiff.raceway_contact_radius > 0.0
        assert stiff.cp > 0.0


class TestSelfAligningPointContactStiffnessContract:
    def test_KNOWN_GAP_outer_term_not_implemented(self):
        """SelfAligningPointContactStiffness.cp depende de _outer_term,
        que ainda não tem forma fechada para contacto circular (chi_e=1)
        -- ver docstring da classe e a nota em SelfAligningBallFamily.
        Substitui este teste por asserts reais quando isso for derivado."""
        stiff = bc.SelfAligningPointContactStiffness(Dw=8.0, ri=4.24, re=15.0, E=210_000.0,
                                                       nu=0.3, alpha_0=0.0, Dpw=40.0)
        assert 0.0 < stiff.gamma < 1.0
        assert stiff.raceway_contact_radius > 0.0
        with pytest.raises(NotImplementedError):
            _ = stiff.cp


class TestContactAngleAndClearanceContract:
    def test_with_s_returns_alpha_0_and_s(self):
        alpha_0, s = bc.contact_angle_and_clearance(A=0.5, s=0.02)
        assert 0.0 <= alpha_0 <= np.pi / 2
        assert math.isclose(s, 0.02)

    def test_with_alpha_0_deg_returns_consistent_pair(self):
        alpha_0, s = bc.contact_angle_and_clearance(A=0.5, alpha_0_deg=25.0)
        assert math.isclose(alpha_0, np.radians(25.0))
