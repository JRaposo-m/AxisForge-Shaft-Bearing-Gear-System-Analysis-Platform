"""
axisforge/results/_base.py

Shared conventions for every result shape in axisforge.results.

DC -- dataclass options used by all result classes:
  frozen=True   results never change after the solver hands them out;
                "attaching" = dataclasses.replace() -> new instance.
  eq=False      generated __eq__ would do `ndarray == ndarray` -> ValueError.
                Compare in tests with np.allclose, not ==.
  kw_only=True  lets a subclass add required fields after a base with
                defaulted ones, and makes adding a defaulted field later
                non-breaking for every constructor call. (Python >= 3.10)

_ABSTRACT -- a class is abstract iff it declares `_ABSTRACT = True` in its
own body (vars(), not inherited). Needed because a dataclass ABC with no
@abstractmethod instantiates silently.
"""
from __future__ import annotations

import numpy as np

DC = dict(frozen=True, eq=False, kw_only=True)


def is_abstract(obj) -> bool:
    return vars(type(obj)).get("_ABSTRACT", False)


def reject_if_abstract(obj) -> None:
    if is_abstract(obj):
        raise TypeError(f"{type(obj).__name__} is abstract -- instantiate a concrete subclass.")


def check_shape(owner: str, name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{owner}: {name} must be np.ndarray; got {type(arr).__name__}.")
    if arr.shape != shape:
        raise ValueError(f"{owner}: {name} must be shape {shape}; got {arr.shape}.")


def check_finite(owner: str, **values: float) -> None:
    bad = {k: v for k, v in values.items() if not np.isfinite(v)}
    if bad:
        raise ValueError(f"{owner}: non-finite values {bad}.")