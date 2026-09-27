# test_contact.py
"""Testes de contact_models/iso16281.py -- PointContactStiffness,
SelfAligningPointContactStiffness e LineContactStiffness tal como
existem hoje: matemática de Hertz pura a partir de curvaturas +
materiais, sem geometria/cinemática (isso vive em families/geometry.py).

A versão anterior deste ficheiro testava um módulo
`families.iso16281_contact` com uma `contact_angle_and_clearance` e uma
`LineContactStiffness(Dwe, Dpw, alpha_0, Lwe, n_s)` que já não existem --
essas responsabilidades foram separadas: contact_models/iso16281.py fica
só com o Hertz puro, families/geometry.py fica com as curvaturas e
cinemática por par rolling-element/raceway."""
import math
import pytest
import numpy as np

from axisforge.core.machine_elements.bearings.contact_models import iso16281 as iso


# Geometria de referência: mesma usada em DeepGrooveBallFamily/
# AngularContactFamily (Dw=8.0, Dpw=40.0, ri=re=0.52*Dw=4.16) -- não é
# um valor arbitrário, é o que a família realmente monta.
STEEL = dict(e1=210_000.0, e2=210_000.0, nu1=0.3, nu2=0.3)


class TestPointContactStiffness:
    def test_returns_plausible_Ri_cp(self):
        stiff = iso.PointContactStiffness(Dw=8.0, Dpw=40.0, gamma=0.2,
                                           ri=4.16, re=4.16, alpha_0=0.0, **STEEL)
        assert stiff.raceway_contact_radius > 0.0
        assert stiff.cp > 0.0

    def test_cp_positive_for_angular_contact_geometry(self):
        """alpha_0 != 0 (contacto angular) não pode partir o Hertz --
        curvature_sum/diff e chi_inner/outer continuam bem definidos
        para um ângulo de contacto não-nulo."""
        alpha_0 = np.radians(25.0)
        gamma = 8.0 * np.cos(alpha_0) / 40.0
        stiff = iso.PointContactStiffness(Dw=8.0, Dpw=40.0, gamma=gamma,
                                           ri=4.16, re=4.16, alpha_0=alpha_0, **STEEL)
        assert stiff.cp > 0.0

    def test_cp_increases_with_stiffer_material(self):
        """Mesma geometria, e1/e2 mais rígido -> cp maior (E* entra
        linearmente em cp, ISO/TS 16281 Sec 5 eq.(11))."""
        soft = iso.PointContactStiffness(Dw=8.0, Dpw=40.0, gamma=0.2, ri=4.16, re=4.16,
                                          alpha_0=0.0, e1=70_000.0, e2=70_000.0, nu1=0.3, nu2=0.3)
        steel = iso.PointContactStiffness(Dw=8.0, Dpw=40.0, gamma=0.2, ri=4.16, re=4.16,
                                           alpha_0=0.0, **STEEL)
        assert steel.cp > soft.cp


class TestSelfAligningPointContactStiffness:
    def test_outer_term_not_implemented(self):
        """Sem forma fechada ainda para contacto circular (chi_e=1) no
        anel exterior -- ver docstring da classe em
        contact_models/iso16281.py."""
        stiff = iso.SelfAligningPointContactStiffness(
            Dw=8.0, Dpw=40.0, gamma=0.2, ri=4.24, re=15.0, alpha_0=0.0, **STEEL)
        with pytest.raises(NotImplementedError):
            _ = stiff.cp

    def test_inner_term_unaffected(self):
        """Só _outer_term foi sobreposto -- o termo do anel interior
        (partilhado com PointContactStiffness) tem de continuar a
        calcular normalmente."""
        stiff = iso.SelfAligningPointContactStiffness(
            Dw=8.0, Dpw=40.0, gamma=0.2, ri=4.24, re=15.0, alpha_0=0.0, **STEEL)
        assert stiff._inner_term > 0.0


class TestLineContactStiffness:
    def test_cs_splits_cl_by_n_s(self):
        x_k = np.linspace(-4.0, 4.0, 40)
        stiff = iso.LineContactStiffness(Lwe=8.0, n_s=40, Dwe=8.0, x_k=x_k, _LOG_ARG_EPS=1e-12)
        assert math.isclose(stiff.cs, stiff.cl / 40)

    def test_cl_scales_with_Lwe(self):
        x_k = np.linspace(-4.0, 4.0, 40)
        short = iso.LineContactStiffness(Lwe=4.0, n_s=40, Dwe=8.0, x_k=x_k, _LOG_ARG_EPS=1e-12)
        long_ = iso.LineContactStiffness(Lwe=8.0, n_s=40, Dwe=8.0, x_k=x_k, _LOG_ARG_EPS=1e-12)
        assert long_.cl > short.cl

    def test_reference_roller_profile_shape_matches_x_k(self):
        x_k = np.linspace(-4.0, 4.0, 40)
        stiff = iso.LineContactStiffness(Lwe=8.0, n_s=40, Dwe=8.0, x_k=x_k, _LOG_ARG_EPS=1e-12)
        P = stiff.reference_roller_profile
        assert P.shape == x_k.shape
        assert np.all(P >= 0.0)

    def test_reference_roller_profile_zero_at_center(self):
        """P(x_k=0) -> log(1/1) = 0 no ramo de rolo curto (Lwe <= 2.5*Dwe)."""
        x_k = np.array([0.0])
        stiff = iso.LineContactStiffness(Lwe=8.0, n_s=1, Dwe=8.0, x_k=x_k, _LOG_ARG_EPS=1e-12)
        P = stiff.reference_roller_profile
        assert math.isclose(P[0], 0.0, abs_tol=1e-9)

    def test_long_roller_flat_edge_branch(self):
        """Lwe > 2.5*Dwe muda para o perfil com zona plana -- pontos
        dentro da zona plana (|x_k| <= half_flat) dão P=0, pontos na
        zona coroada da margem dão P>0."""
        Dwe, Lwe = 4.0, 20.0  # Lwe > 2.5*Dwe = 10.0
        half_flat = (Lwe - 2.5 * Dwe) / 2.0
        x_k = np.array([0.0, half_flat + 0.5])
        stiff = iso.LineContactStiffness(Lwe=Lwe, n_s=2, Dwe=Dwe, x_k=x_k, _LOG_ARG_EPS=1e-12)
        P = stiff.reference_roller_profile
        assert math.isclose(P[0], 0.0, abs_tol=1e-9)
        assert P[1] > 0.0