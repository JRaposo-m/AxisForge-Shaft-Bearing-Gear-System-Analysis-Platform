"""
mechanical_system/loads.py

Point load primitives applied to a shaft.

Transverse loads (RadialLoad, ExternalMoment) are defined by a magnitude and
an angular position theta around the shaft cross-section, then decomposed
into the two principal bending planes (XY, XZ) for the StaticsSolver.

Sign convention:
  - theta_deg measured from +Y toward +Z, right-hand rule about +X.
  - magnitude always >= 0; direction fully encoded by theta.
  - AxialLoad, TorqueLoad act along/about +X — no angular decomposition needed.
  - Fy/Fz (RadialLoad) and My/Mz (ExternalMoment) are signed vector
    components about +Y and +Z respectively, right-hand rule — the same
    convention TorqueLoad already uses about +X (positive = CCW viewed
    from the positive axis).
  - My/Mz are therefore work-conjugate with the FEM rotational DOF theta
    at each node: positive nodal rotation is right-hand-rule about the
    plane's normal axis (+Z for XY-plane bending, +Y for XZ-plane
    bending), so a positive applied moment and a positive nodal rotation
    point the same physical way. This is what element_theories/ assumes
    when it defines theta on the stiffness/shape-function side — see
    solvers/README.md.

Units: N for forces, N*mm for moments/torques, mm for position.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Union
import numpy as np
from scipy import integrate as _quad


class LoadPlane(Enum):
    """Principal bending plane — used only as a decomposition result key,
    never as user input."""
    XY = auto()
    XZ = auto()


class Load:
    """Base class — not instantiated directly."""

    def __init__(self, position: float, label: str = "", source: str = "user"):
        self.position = position
        self.label = label
        self.source = source   # "user" | "gear_mesh" | "bearing_reaction"

    def validate(self) -> list[str]:
        errors: list[str] = []
        tag = self.label or self.__class__.__name__
        if self.position < 0:
            errors.append(f"{tag}: position must be >= 0, got {self.position}")
        return errors

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


class RadialLoad(Load):
    """
    Transverse point force at arbitrary angular position theta.

    Parameters
    ----------
    position  : axial coordinate [mm]
    magnitude : force magnitude [N], >= 0
    theta_deg : angular position [deg], measured from +Y toward +Z
    source    : "user" | "gear_mesh" | "bearing_reaction"
    """

    def __init__(self, position: float, magnitude: float,
                 theta_deg: float = 0.0, label: str = "",
                 source: str = "user"):
        super().__init__(position, label, source)
        self.magnitude = magnitude
        self.theta_deg = theta_deg % 360.0
        self.theta = np.radians(self.theta_deg)

    @property
    def Fy(self) -> float:
        return self.magnitude * np.cos(self.theta)

    @property
    def Fz(self) -> float:
        return self.magnitude * np.sin(self.theta)

    def component(self, plane: LoadPlane) -> float:
        """Signed magnitude projected onto a principal plane."""
        return self.Fy if plane == LoadPlane.XY else self.Fz

    def validate(self) -> list[str]:
        errors = super().validate()
        tag = self.label or self.__class__.__name__
        if self.magnitude < 0:
            errors.append(f"{tag}: magnitude must be >= 0, got {self.magnitude}")
        return errors

    def __repr__(self) -> str:
        return (f"RadialLoad(position={self.position:.2f} mm, "
                f"F={self.magnitude:.2f} N, theta={self.theta_deg:.1f}°, "
                f"Fy={self.Fy:.2f} N, Fz={self.Fz:.2f} N, "
                f"source={self.source!r}, label={self.label!r})")


class AxialLoad(Load):
    """Force along shaft axis. Positive = tensile (+X)."""

    def __init__(self, position: float, magnitude: float, label: str = "",
                 source: str = "user"):
        super().__init__(position, label, source)
        self.magnitude = magnitude

    def __repr__(self) -> str:
        return (f"AxialLoad(position={self.position:.2f} mm, "
                f"Fa={self.magnitude:.2f} N, source={self.source!r}, "
                f"label={self.label!r})")


class TorqueLoad(Load):
    """Torque about shaft axis. Positive = CCW viewed from +X. [N*m]"""

    def __init__(self, position: float, magnitude: float, label: str = "",
                 source: str = "user"):
        super().__init__(position, label, source)
        self.magnitude = magnitude

    def __repr__(self) -> str:
        return (f"TorqueLoad(position={self.position:.2f} mm, "
                f"T={self.magnitude:.2f} N·m, source={self.source!r}, "
                f"label={self.label!r})")


class ExternalMoment(Load):
    """
    Applied bending moment at arbitrary angular orientation theta.
    Same decomposition convention as RadialLoad — theta defines the
    moment vector direction in the YZ cross-section.

    Sign: My is the right-hand-rule moment component about +Y, Mz about
    +Z — the same convention TorqueLoad uses about +X (positive = CCW
    viewed from the positive axis). This is what makes My/Mz
    work-conjugate with the FEM nodal rotation DOF theta: a positive
    applied moment and a positive nodal rotation point the same physical
    way in each plane (+Z-about for XY-plane bending, +Y-about for
    XZ-plane bending). See solvers/README.md / element_theories/ for the
    element-side definition of theta.
    """

    def __init__(self, position: float, magnitude: float,
                 theta_deg: float = 0.0, label: str = "",
                 source: str = "user"):
        super().__init__(position, label, source)
        self.magnitude = magnitude
        self.theta_deg = theta_deg % 360.0
        self.theta = np.radians(self.theta_deg)

    @property
    def My(self) -> float:
        return self.magnitude * np.cos(self.theta)

    @property
    def Mz(self) -> float:
        return self.magnitude * np.sin(self.theta)

    def component(self, plane: LoadPlane) -> float:
        return self.My if plane == LoadPlane.XY else self.Mz

    def __repr__(self) -> str:
        return (f"ExternalMoment(position={self.position:.2f} mm, "
                f"M={self.magnitude:.2f} N·m, theta={self.theta_deg:.1f}°, "
                f"source={self.source!r}, label={self.label!r})")


class DistributedRadialLoad(Load):
    """
    Transverse load distributed over [x_lo, x_hi] with arbitrary intensity
    and direction profiles.

    Parameters
    ----------
    x_lo, x_hi : axial extent [mm]; x_hi > x_lo
    magnitude   : float [N]                — uniform q(x) = F / b; signed —
                                             negative values are valid (e.g. Ft/Fr
                                             from gear mesh pointing downward).
                  Callable[[float], float] — intensity function q(x) [N/mm], signed;
                                             F = ∫ q(x) dx computed once at init.
    theta_deg   : float [deg]              — constant angular direction
                  Callable[[float], float] — theta(x) [deg]; direction varies
                                             along the span.

    When theta_deg is constant, Fy and Fz share the same projection factor
    and the existing closed-form paths are used.  When theta_deg is callable,
    q(x) and theta(x) are evaluated together, so the two plane components are
    integrated independently:

        Fy_resultant = ∫ q(x)·cos(theta(x)) dx
        Fz_resultant = ∫ q(x)·sin(theta(x)) dx

    magnitude then stores ‖F‖ = sqrt(Fy² + Fz²); the individual resultants
    are available as resultant_Fy and resultant_Fz.  as_point_load() is
    intentionally unavailable for the variable-theta case because there is no
    single representative theta.

    Solver interface
    ----------------
    q(x)                                  → intensity [N/mm], zero outside span
    theta_at(x)                           → direction [deg] at x
    qy(x), qz(x)                          → signed intensity components [N/mm]
    component_intensity(x, plane)         → signed intensity [N/mm] onto plane
    resultant_component(plane)            → signed resultant [N] onto plane
    bending_moment_contribution(x, plane) → M(x) contribution [N·mm]
    centroid(plane)                       → per-plane resultant application x̄ [mm]
    as_point_load()                       → equivalent RadialLoad (constant-theta only)
    """

    def __init__(self,
                 x_lo: float,
                 x_hi: float,
                 magnitude: Union[float, Callable[[float], float]],
                 theta_deg: Union[float, Callable[[float], float]] = 0.0,
                 label: str = "",
                 source: str = "user"):
        tag = label or "DistributedRadialLoad"
        if x_hi <= x_lo:
            raise ValueError(f"{tag}: x_hi ({x_hi}) must be > x_lo ({x_lo})")

        super().__init__((x_lo + x_hi) / 2.0, label, source)

        self.x_lo = x_lo
        self.x_hi = x_hi
        self.b    = x_hi - x_lo

        # ------------------------------------------------------------------
        # theta — constant or variable
        # ------------------------------------------------------------------
        self._theta_variable = callable(theta_deg)

        if self._theta_variable:
            self._theta_fn  = theta_deg                          # deg(x)
            # theta_deg / theta stored as sentinels for __repr__ only
            self.theta_deg  = None
            self.theta      = None
        else:
            _tdeg           = float(theta_deg) % 360.0
            self.theta_deg  = _tdeg
            self.theta      = np.radians(_tdeg)
            self._theta_fn  = lambda x, _t=self.theta: _t       # radians, constant

        # ------------------------------------------------------------------
        # magnitude — constant float or callable q(x)
        # ------------------------------------------------------------------
        self._q_variable = callable(magnitude)

        if self._q_variable:
            self._q_fn    = magnitude
            self._uniform = False
        else:
            _q0           = float(magnitude) / self.b
            self._q_fn    = lambda x, _q=_q0: _q
            self._uniform = not self._theta_variable   # uniform iff both are constant

        # ------------------------------------------------------------------
        # Precompute resultants (both planes) — one quad pass each
        # ------------------------------------------------------------------
        if self._theta_variable:
            # theta varies: project inside the integrand
            def _fy_integrand(x):
                th = np.radians(self._theta_fn(x))
                return self._q_fn(x) * np.cos(th)

            def _fz_integrand(x):
                th = np.radians(self._theta_fn(x))
                return self._q_fn(x) * np.sin(th)

            self._Fy = float(_quad.quad(_fy_integrand, x_lo, x_hi)[0])
            self._Fz = float(_quad.quad(_fz_integrand, x_lo, x_hi)[0])
            self._magnitude = float(np.hypot(self._Fy, self._Fz))
        else:
            # theta constant: single integral for total F (signed), then project
            if self._q_variable:
                F = float(_quad.quad(self._q_fn, x_lo, x_hi)[0])
            else:
                F = float(magnitude)
            self._magnitude = F          # signed scalar resultant [N]
            self._Fy        = F * np.cos(self.theta)
            self._Fz        = F * np.sin(self.theta)

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def magnitude(self) -> float:
        """
        Signed scalar resultant [N] for constant-theta loads (= ∫ q(x) dx).
        For variable-theta loads: ‖F‖ = sqrt(Fy² + Fz²) — always positive.
        Gear-mesh loads carry a meaningful sign — it is NOT stripped here.
        """
        return self._magnitude

    @property
    def resultant_Fy(self) -> float:
        """Resultant Y-component [N] = ∫ q(x)·cos(theta(x)) dx."""
        return self._Fy

    @property
    def resultant_Fz(self) -> float:
        """Resultant Z-component [N] = ∫ q(x)·sin(theta(x)) dx."""
        return self._Fz

    @property
    def is_uniform(self) -> bool:
        """True only when both q and theta are constant."""
        return self._uniform

    # ------------------------------------------------------------------
    # Point-wise interface
    # ------------------------------------------------------------------

    def q(self, x: float) -> float:
        """Intensity [N/mm] at x. Zero outside [x_lo, x_hi]."""
        if self.x_lo <= x <= self.x_hi:
            return float(self._q_fn(x))
        return 0.0

    def theta_at(self, x: float) -> float:
        """Direction [deg] at x."""
        if self._theta_variable:
            return float(self._theta_fn(x)) % 360.0
        return self.theta_deg

    def qy(self, x: float) -> float:
        """Signed Y-intensity [N/mm] at x."""
        if self.x_lo <= x <= self.x_hi:
            th = np.radians(self._theta_fn(x)) if self._theta_variable else self.theta
            return float(self._q_fn(x)) * np.cos(th)
        return 0.0

    def qz(self, x: float) -> float:
        """Signed Z-intensity [N/mm] at x."""
        if self.x_lo <= x <= self.x_hi:
            th = np.radians(self._theta_fn(x)) if self._theta_variable else self.theta
            return float(self._q_fn(x)) * np.sin(th)
        return 0.0

    def component_intensity(self, x: float, plane: LoadPlane) -> float:
        """Signed intensity [N/mm] at x projected onto a principal plane."""
        return self.qy(x) if plane == LoadPlane.XY else self.qz(x)

    def resultant_component(self, plane: LoadPlane) -> float:
        """Signed resultant [N] projected onto a principal plane."""
        return self._Fy if plane == LoadPlane.XY else self._Fz

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def centroid(self, plane: LoadPlane = LoadPlane.XY) -> float:
        """
        Per-plane resultant application point:

            x̄_plane = ∫ x · q_plane(x) dx / F_plane

        For constant theta both planes give the same x̄ (reduces to the
        scalar centroid). For variable theta they may differ.
        Closed form for the uniform constant-theta case; numerical otherwise.
        """
        F_plane = self.resultant_component(plane)
        if F_plane == 0.0:
            return self.position   # degenerate: fallback to midpoint

        if self._uniform:
            return (self.x_lo + self.x_hi) / 2.0

        integrand = (
            (lambda x: x * self.qy(x)) if plane == LoadPlane.XY
            else (lambda x: x * self.qz(x))
        )
        moment = float(_quad.quad(integrand, self.x_lo, self.x_hi)[0])
        return moment / F_plane

    # ------------------------------------------------------------------
    # Bending moment contribution — called by StaticsSolver
    # ------------------------------------------------------------------

    def bending_moment_contribution(self, x: float, plane: LoadPlane) -> float:
        """
        M(x) = ∫_{x_lo}^{min(x, x_hi)} q_plane(ξ) · (x − ξ) dξ

        Closed form for the uniform constant-theta case.
        Numerical (quad) for all other cases.
        Returns [N·mm].
        """
        if x <= self.x_lo:
            return 0.0

        upper = min(x, self.x_hi)

        if self._uniform:
            proj  = np.cos(self.theta) if plane == LoadPlane.XY else np.sin(self.theta)
            q0    = self._magnitude / self.b
            delta = upper - self.x_lo
            return proj * q0 * (delta * x - self.x_lo * delta - delta ** 2 / 2.0)

        q_plane = self.qy if plane == LoadPlane.XY else self.qz
        return float(
            _quad.quad(lambda xi: q_plane(xi) * (x - xi), self.x_lo, upper)[0]
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def as_point_load(self) -> RadialLoad:
        """
        Statically equivalent RadialLoad at centroid (XY-plane centroid).
        Only available for constant-theta loads; raises otherwise because
        a single theta cannot represent a variable-direction distribution.
        """
        if self._theta_variable:
            raise TypeError(
                f"{self.label or 'DistributedRadialLoad'}: as_point_load() is "
                "undefined for variable-theta loads — no single representative "
                "theta exists. Use resultant_Fy / resultant_Fz directly."
            )
        return RadialLoad(
            self.centroid(LoadPlane.XY), self._magnitude, self.theta_deg,
            label=self.label, source=self.source,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        errors = super().validate()
        tag = self.label or self.__class__.__name__
        if self.b <= 0:
            errors.append(f"{tag}: span b must be > 0, got {self.b}")
        return errors

    def __repr__(self) -> str:
        q_str     = "uniform" if self._uniform else "q(x)"
        theta_str = (f"{self.theta_deg:.1f}°" if not self._theta_variable
                     else "theta(x)")
        F_str = f"F={self._magnitude:.2f} N" if not self._theta_variable else f"‖F‖={self._magnitude:.2f} N"
        return (f"DistributedRadialLoad([{self.x_lo:.2f}, {self.x_hi:.2f}] mm, "
                f"{F_str}, q={q_str}, theta={theta_str}, "
                f"source={self.source!r}, label={self.label!r})")


Load_T = RadialLoad | AxialLoad | TorqueLoad | ExternalMoment | DistributedRadialLoad


# ===========================================================================
# LoadingProfile — fatigue cycle decomposition
# ===========================================================================

@dataclass(frozen=True)
class LoadingProfile:
    label: str = ""
    R: float = 1.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.R <= 1.0:
            raise ValueError(f"R must be in [-1.0, 1.0], got {self.R}")

    @property
    def sigma_mean_factor(self) -> float:
        return (1 + self.R) / 2

    @property
    def sigma_amplitude_factor(self) -> float:
        return abs(1 - self.R) / 2

    @property
    def is_static(self) -> bool:
        return self.R == 1.0

    @property
    def is_fully_reversed(self) -> bool:
        return self.R == -1.0