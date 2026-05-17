# models/statics_result.py
"""
StaticsResult — immutable output of StaticsSolver.

Note on frozen dataclass with numpy arrays:
  frozen=True prevents attribute reassignment but np.ndarray is mutable internally.
  We set eq=False on array fields so dataclass equality checks don't call
  np.ndarray.__eq__ (which returns an array, not a bool).
  The dataclass is treated as immutable by convention; solvers never mutate results.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


def _array_field() -> np.ndarray:
    return np.array([])


@dataclass
class StaticsResult:
    """
    Complete output of a static analysis on a two-support shaft.

    Coordinate convention:
      x     : axial coordinate from datum (left end) [mm]
      XZ    : horizontal plane
      XY    : vertical plane (gravity direction)
      Positive forces: downward (XY), forward (XZ) by external load convention.
      Reactions: sign determined mathematically by equilibrium.

    All arrays have shape (N,) where N = SOLVER_RESOLUTION.
    """
    x: np.ndarray = field(compare=False)            # axial axis [mm]
    V_xz: np.ndarray = field(compare=False)         # shear force, XZ plane [N]
    V_xy: np.ndarray = field(compare=False)         # shear force, XY plane [N]
    M_xz: np.ndarray = field(compare=False)         # bending moment, XZ plane [N·mm]
    M_xy: np.ndarray = field(compare=False)         # bending moment, XY plane [N·mm]
    M_res: np.ndarray = field(compare=False)        # resultant bending moment [N·mm]
    T: np.ndarray = field(compare=False)            # torsion [N·mm]
    axial_force: np.ndarray = field(compare=False)  # axial force Fa(x) [N]
    reactions: dict = field(default_factory=dict)   # bearing reactions

    # reactions dict keys:
    #   "A_xz"  : reaction at bearing A, XZ plane [N]
    #   "A_xy"  : reaction at bearing A, XY plane [N]
    #   "B_xz"  : reaction at bearing B, XZ plane [N]
    #   "B_xy"  : reaction at bearing B, XY plane [N]
    #   "axial" : axial reaction at fixed support [N]

    @property
    def M_xz_max(self) -> float:
        return float(np.max(np.abs(self.M_xz)))

    @property
    def M_xy_max(self) -> float:
        return float(np.max(np.abs(self.M_xy)))

    @property
    def M_res_max(self) -> float:
        return float(np.max(self.M_res))

    @property
    def T_max(self) -> float:
        return float(np.max(np.abs(self.T)))

    @property
    def x_at_M_res_max(self) -> float:
        """Axial position where M_res is maximum [mm]."""
        return float(self.x[np.argmax(self.M_res)])
