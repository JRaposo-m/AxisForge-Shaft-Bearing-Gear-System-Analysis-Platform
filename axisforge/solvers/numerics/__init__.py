"""Generic numerical methods shared by the AxisForge solvers and by design-studies studies:
the Newton-Raphson solver of nonlinear systems (``newton``). Eager re-exports, no lazy loading.
"""
from __future__ import annotations

from .newton import NewtonError, NewtonOptions, newton

__all__ = ["NewtonError", "NewtonOptions", "newton"]
