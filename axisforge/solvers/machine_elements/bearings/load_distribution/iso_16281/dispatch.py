"""
axisforge/solvers/machine_elements/bearings/load_distribution/iso_16281/dispatch.py

Dispatch de solver ISO/TS 16281 por capability + required-attrs da
BearingFamily -- sem tabela por BearingType. Ver rolling_bearing_solver.py.

register_contact_solver() é uma decorator factory: CAPABILITY e
REQUIRED_ATTRS não são declarados soltos no corpo da classe -- são
argumentos do próprio decorator, que os escreve na classe. Isto torna
impossível esquecer de os declarar (a assinatura do decorator exige-os)
em vez de só validar depois com getattr(), e deixa tudo visível numa
única linha no ponto de registo.

family_matches_capability() vive aqui, não em family.py: "capability"
(point_contact/line_contact como strings) é vocabulário deste módulo --
nasce em register_contact_solver() -- e nunca deveria vazar para
family.py, que só conhece factos de geometria sobre si própria
(is_point_contact_family/is_line_contact_family). Este módulo é que
traduz esse vocabulário solver-side para os factos family-side.
validation.py importa family_matches_capability DAQUI, não de family.py
-- import lateral dentro do mesmo pacote iso_16281/, mesma direcção que
este módulo já usa para chegar a family.py (solver -> core).

resolve_solver_cls() faz três verificações em sequência, não uma só:
  1. capability -- a family da bearing tem de estar registada no
     registo (_POINT_CONTACT/_LINE_CONTACT em family.py) que corresponde
     à CAPABILITY do solver candidato. Isto é sobre GEOMETRIA de
     contacto, nunca muda por instância.
  2. iso16281_analysis -- a bearing tem de ter completado a montagem de
     rigidez Hertziana (setada pela family em assemble_geometry). Isto é
     sobre esta instância específica ter os dados prontos.
"""
from __future__ import annotations

from axisforge.core.machine_elements.bearings.families.family import (
    is_point_contact_family, is_line_contact_family,
)


class SolverDispatchError(RuntimeError):
    pass


_REGISTRY: list[type] = []

_CAPABILITY_CHECK = {
    "point_contact": is_point_contact_family,
    "line_contact": is_line_contact_family,
}


def family_matches_capability(family, capability: str) -> bool:
    """Traduz uma CAPABILITY string (ver register_contact_solver) para um
    facto sobre a family -- single source of truth partilhada por
    resolve_solver_cls() aqui e por check_bearing_ready() em
    validation.py. Uma capability desconhecida (typo, ou nova capability
    registada sem entrada aqui) devolve False em vez de KeyError -- falha
    como 'nenhum solver corresponde', não como crash."""
    check = _CAPABILITY_CHECK.get(capability)
    return check is not None and check(family)


def register_contact_solver(*, capability: str):
    """
    Class decorator factory: regista `cls` como solver ISO/TS 16281 para
    `capability`, exigindo `required_attrs` na bearing/family que o
    dispatch lhe encaminhar. Escreve CAPABILITY/REQUIRED_ATTRS na própria
    classe -- quem lê `cls.CAPABILITY` depois (resolve_solver_cls() etc.)
    continua a funcionar sem mudanças.
    """
    if not capability:
        raise TypeError("register_contact_solver: capability must be non-empty")
    if capability not in _CAPABILITY_CHECK:
        raise TypeError(
            f"register_contact_solver: unknown capability {capability!r} -- "
            f"expected one of {sorted(_CAPABILITY_CHECK)}."
        )
    def _decorator(cls: type) -> type:
        cls.CAPABILITY     = capability
        _REGISTRY.append(cls)
        return cls
    return _decorator


def resolve_solver_cls(bearing, *, label: str = "") -> type:
    family = bearing.family

    candidates = [c for c in _REGISTRY if family_matches_capability(family, c.CAPABILITY)]
    if not candidates:
        raise SolverDispatchError(
            f"'{label}': nenhuma capability registada corresponde à family "
            f"'{family.name}' ({type(family).__name__})."
        )

    if not getattr(bearing, "iso16281_analysis", False):
        raise SolverDispatchError(
            f"'{label}': iso16281_analysis nao esta True nesta bearing -- "
            f"a family nunca completou a montagem de rigidez ISO/TS 16281."
        )

    single_row = [c for c in candidates if c.MULTIROW_SOLVER is not None]
    if len(single_row) != 1:
        raise SolverDispatchError(
            f"'{label}': esperava exactamente 1 solver single-row registado "
            f"para esta capability, encontrei {[c.__name__ for c in single_row]}."
        )

    return single_row[0].MULTIROW_SOLVER if hasattr(bearing, "rows") else single_row[0]