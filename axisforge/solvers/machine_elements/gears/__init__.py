"""Gear solvers: the contact inputs along the path of contact (``contact``, reviewed: Q31, José
2026-10-15) and two placeholders (``iso6336``, ``thermal``), not reviewed.

The placeholders are not re-exported: they hold contracts whose functions raise
``NotImplementedError``. A reviewed solver is added to this rollup with its explicit names.
"""
from __future__ import annotations

from axisforge.solvers.machine_elements.gears.contact import contact_inputs, load_sharing

__all__: list[str] = ["contact_inputs", "load_sharing"]
