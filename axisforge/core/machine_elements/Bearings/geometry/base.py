"""
Bearings/geometry/base.py

Abstract base for bearing internal geometry.
One subclass per contact type (point / line).
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class BearingGeometry(ABC):

    @abstractmethod
    def setup(self, **kwargs) -> None:
        """Cache internal geometry parameters as instance attributes."""
        ...

    @abstractmethod
    def hertz_spring_constant(self) -> float:
        """
        Compute and return the Hertzian spring constant.
          Point contact : c_p  [N/mm^(3/2)]
          Line contact  : c_l  [N/mm^(10/9)]
        """
        ...

    @property
    @abstractmethod
    def load_deflection_exponent(self) -> float:
        """
        n in Q = c · δⁿ
          Ball bearing   : 3/2
          Roller bearing : 10/9
        """
        ...
