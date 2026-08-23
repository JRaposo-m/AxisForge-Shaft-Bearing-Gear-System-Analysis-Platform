"""Dispatch de solver ISO/TS 16281 por capability + required-attrs da
BearingFamily -- sem tabela por BearingType. Ver rolling_bearing_solver.py."""
from __future__ import annotations


class SolverDispatchError(RuntimeError):
    pass


_REGISTRY: list[type] = []


def register_contact_solver(cls):
    if not getattr(cls, "CAPABILITY", ""):
        raise TypeError(f"{cls.__name__} must declare CAPABILITY")
    if not getattr(cls, "REQUIRED_ATTRS", None):
        raise TypeError(f"{cls.__name__} must declare REQUIRED_ATTRS")
    _REGISTRY.append(cls)
    return cls


def _most_specific(matches: list[type]) -> list[type]:
    if len(matches) <= 1:
        return matches
    sets = {c: frozenset(c.REQUIRED_ATTRS) for c in matches}
    return [c for c in matches
            if not any(sets[c] < sets[o] for o in matches if o is not c)]


def resolve_solver_cls_for_attrs(attrs, *, label: str = "") -> type:
    """Match por atributos crus -- usado para row views (dicts) que nunca
    passam por Bearing.assemble(), ex. rows de um MultiRowThrustBallFamily."""
    attrs = frozenset(attrs)
    matches = _most_specific(
        [c for c in _REGISTRY if frozenset(c.REQUIRED_ATTRS) <= attrs]
    )
    if not matches:
        raise SolverDispatchError(f"'{label}': nenhum solver cobre {sorted(attrs)}")
    if len(matches) > 1:
        raise SolverDispatchError(f"'{label}': ambiguo entre {[c.__name__ for c in matches]}")
    return matches[0]


def resolve_solver_cls(bearing, *, label: str = "") -> type:
    family = bearing.family
    candidates = [c for c in _REGISTRY if bearing.is_enabled(c.CAPABILITY)]
    matches = _most_specific([
        c for c in candidates
        if frozenset(c.REQUIRED_ATTRS) <= family.REQUIRED_FOR.get(c.CAPABILITY, frozenset())
    ])
    if not matches:
        raise SolverDispatchError(
            f"'{label}': nenhum solver registado cobre a family '{family.name}' "
            f"(CAPABILITIES={sorted(family.CAPABILITIES)})"
        )
    if len(matches) > 1:
        raise SolverDispatchError(f"'{label}': ambiguo entre {[c.__name__ for c in matches]}")
    return matches[0]