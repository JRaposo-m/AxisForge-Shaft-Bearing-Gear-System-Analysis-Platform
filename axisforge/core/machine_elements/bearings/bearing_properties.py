# axisforge/core/machine_elements/bearings/bearing_properties.py
"""
core/machine_elements/bearings/bearing_properties.py

Geometria pura -- sem material, sem qualquer maquinaria de resolucao
iterativa (chi, integrais elipticos, cp/cl/cs -- isso e ISO/TS 16281 e
vive em solvers/machine_elements/bearings/load_distribution/iso_16281/,
ver ADR-001 no vault/40_ADR/).

Conteudo:
  - SurfacePair            : os (no maximo 2) materiais de um rolamento.
  - gamma_point_contact /
    gamma_line_contact     : fator geometrico gamma -- usado tanto por
                              capacity.py (ISO 281) como pelas curvaturas
                              de Hertz abaixo. NAO depende de material.
  - raceway_contact_radius : R_C, raio circunferencial da via.
  - point_contact_radii /
    line_contact_radii     : r1 (elemento rolante) / r2 (via, inner e
                              outer) no formato (Rx, Ry) por corpo --
                              exatamente o input de slippy.hertz_full
                              (r1, r2), nada mais.
  - lamina_positions /
    reference_roller_profile: geometria do rolo cilindrico (laminas,
                              perfil logaritmico) -- partilhada entre
                              CylindricalRollerFamily e
                              ThrustCylindricalRollerFamily (era
                              duplicada nas duas antes).

NOTA (aberto, ver ADR-001): a convencao de sinal de R_C para a via
EXTERIOR nao esta validada -- replicada aqui por simetria com o sinal
usado em curvature_sum_outer do antigo iso16281_contact.py, mas fica
marcada como TODO ate seres confirmada contra o Hertz/slippy. Nao
resolvida aqui de proposito -- core so precisa de expor a propriedade
legivel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from axisforge.core.materials import Material


# =====================================================================
# ---- materiais -----------------------------------------------------
# =====================================================================

@dataclass(frozen=True)
class SurfacePair:
    rolling_element: Material
    raceway: Material


# =====================================================================
# ---- gamma (geometrico, sem material) --------------------------------
# =====================================================================

def gamma_point_contact(D: float, alpha_0: float, Dpw: float) -> float:
    """Fator geometrico gamma -- ISO/TS 16281 eq.(2)/(9). Pura geometria:
    usada tanto por capacity.py (ISO 281, sem material) como pelas
    curvaturas de Hertz abaixo (tambem sem material)."""
    if math.isclose(alpha_0, math.pi / 2, abs_tol=1e-9):
        return D / Dpw
    return D * math.cos(alpha_0) / Dpw


def gamma_line_contact(D: float, alpha_0: float, Dpw: float) -> float:
    """Mesma formula que gamma_point_contact -- mantida como funcao
    separada so para espelhar a distincao point/line usada no resto do
    pacote (mesma matematica, vocabulario fisico diferente)."""
    return gamma_point_contact(D, alpha_0, Dpw)


# =====================================================================
# ---- raios de curvatura -- prontos para Hertz -------------------------
# =====================================================================

def raceway_contact_radius(Dpw: float, r_groove: float, D: float, alpha_0: float,
                            *, race: Literal["inner", "outer"]) -> float:
    """R_C -- raio de curvatura circunferencial (direcao de rolamento) da
    via. race="inner": formula ja validada no codigo atual (era
    `PointContactStiffness.raceway_contact_radius`). race="outer":
    sinal espelhado por analogia com curvature_sum_outer -- TODO,
    nao validado (ver ADR-001)."""
    sign = +1.0 if race == "inner" else -1.0
    return Dpw / 2.0 + sign * (r_groove - D / 2.0) * math.cos(alpha_0)


def point_contact_radii(Dw: float, ri: float, re: float | None,
                         alpha_0: float, Dpw: float) -> dict[str, tuple[float, float] | None]:
    """r1 (elemento rolante) / r2_inner / r2_outer -- cada um (Rx, Ry),
    exatamente o par que slippy.hertz_full recebe como r1/r2. Elemento
    rolante esferico: Rx=Ry=Dw/2 (mesmo raio nos dois planos). Via:
    Ry=r_groove (perfil/transversal), Rx=raceway_contact_radius
    (circunferencial). re=None -> sem via exterior (ex.: aplicacoes
    magneticas) -> r2_outer=None."""
    r1 = (Dw / 2.0, Dw / 2.0)
    r2_inner = (raceway_contact_radius(Dpw, ri, Dw, alpha_0, race="inner"), ri)
    r2_outer = None
    if re is not None:
        r2_outer = (raceway_contact_radius(Dpw, re, Dw, alpha_0, race="outer"), re)
    return dict(r1=r1, r2_inner=r2_inner, r2_outer=r2_outer)


def line_contact_radii(Dwe: float, Dpw: float, alpha_0: float) -> dict[str, tuple[float, float]]:
    """Equivalente a point_contact_radii para rolo cilindrico: contacto
    linear -> raio infinito na direcao axial (Ry) para os dois corpos,
    so ha curvatura na direcao de rolamento (Rx).

    TODO: Rc aqui e so Dpw/2 -- rolo cilindrico nao tem groove (ao
    contrario da esfera), por isso nao ha termo (r_groove - D/2)*cos(a0)
    a somar. Revalidar esta simplificacao quando ligares ao Hertz/slippy
    (ver ADR-001, mesma nota de R_C em aberto)."""
    r1 = (Dwe / 2.0, math.inf)
    Rc = Dpw / 2.0
    r2 = (Rc, math.inf)
    return dict(r1=r1, r2_inner=r2, r2_outer=r2)


# =====================================================================
# ---- geometria do rolo cilindrico (partilhada radial/thrust) ---------
# =====================================================================

def lamina_positions(Lwe: float, n_s: int) -> np.ndarray:
    """x_k -- pontos medios das laminas, estritamente dentro de
    (-Lwe/2, Lwe/2). ISO/TS 16281 Sec 5.2.2. Pura geometria -- nao
    depende de material nem de carga."""
    lamina_length = Lwe / n_s
    return lamina_length * (np.arange(n_s) + 0.5) - Lwe / 2.0


def reference_roller_profile(x_k: np.ndarray, Dwe: float, Lwe: float,
                              log_arg_eps: float = 1e-12) -> np.ndarray:
    """P(x_k) [mm] -- perfil logaritmico do rolo, ISO/TS 16281 Sec 6.2
    eq.(42)-(44). Identico em CylindricalRollerFamily e
    ThrustCylindricalRollerFamily no codigo antigo (duplicado) -- e
    geometria pura do rolo, faz sentido partilhado aqui em vez de
    reescrito por tipo."""
    P = np.zeros_like(x_k)
    if Lwe <= 2.5 * Dwe:
        arg = 1.0 - (2.0 * x_k / Lwe) ** 2
        arg = np.maximum(arg, log_arg_eps)
        P = 0.000350 * Dwe * np.log(1.0 / arg)
    else:
        half_flat = (Lwe - 2.5 * Dwe) / 2.0
        edge = np.abs(x_k) > half_flat
        if np.any(edge):
            xe = x_k[edge]
            arg = 1.0 - ((2.0 * np.abs(xe) - (Lwe - 2.5 * Dwe)) / (2.5 * Dwe)) ** 2
            arg = np.maximum(arg, log_arg_eps)
            P[edge] = 0.000500 * Dwe * np.log(1.0 / arg)
    return P