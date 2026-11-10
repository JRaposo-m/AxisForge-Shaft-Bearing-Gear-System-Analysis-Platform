"""The ISO 6336 and ISO/TR 14179 slots are placeholders: a banner, contracts, and no calculation."""
import inspect

import pytest

from axisforge.core.machine_elements.gears import cylindrical_capacity
from axisforge.solvers.machine_elements.gears import iso6336, thermal

MODULES = (cylindrical_capacity, iso6336, thermal)
FUNCTIONS = [(m, name, fn) for m in MODULES for name, fn in inspect.getmembers(m, inspect.isfunction)
             if fn.__module__ == m.__name__]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_module_is_marked_as_a_placeholder(module):
    assert module.__doc__.startswith("PLACEHOLDER - not reviewed by José.")
    assert [name for m, name, _ in FUNCTIONS if m is module]      # it states at least one contract


@pytest.mark.parametrize("fn", [f for _, _, f in FUNCTIONS], ids=[f"{m.__name__.rsplit('.', 1)[-1]}.{n}" for m, n, _ in FUNCTIONS])
def test_placeholder_function_raises(fn):
    """Every function states its contract in the docstring and computes nothing."""
    required = [p for p in inspect.signature(fn).parameters.values() if p.default is p.empty]
    assert fn.__doc__
    with pytest.raises(NotImplementedError, match="placeholder: not reviewed"):
        fn(*[None] * len(required))
