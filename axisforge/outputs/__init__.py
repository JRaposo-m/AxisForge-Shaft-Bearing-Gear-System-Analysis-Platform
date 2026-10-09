"""
axisforge/outputs

Records, text and CSV of AxisForge's own objects (a built system, FEM results, bearing load
distribution and life results). Mirrors the structure of ``axisforge.results``:

- ``construction``: what was built (shaft system, gear system, bearings, loads).
- ``solvers``: what was computed (FEM bearing nodes, load distribution, life).

Sweeps, plots and verdicts are not part of this package: they belong to the studies that use
AxisForge. The package imports neither matplotlib nor ``axisforge.solvers``.

Usage
-----
The sub-packages are exposed as namespaces and nothing is flattened::

    import axisforge.outputs as af_o
    af_o.solvers.bearings.load_distribution.load_distribution_results.print_bearing_result(r)
"""
from __future__ import annotations

from axisforge.outputs import _format
from axisforge.outputs import construction
from axisforge.outputs import solvers

__all__ = ["construction", "solvers"]
