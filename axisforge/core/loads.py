"""
Load primitives applied to a shaft: point forces, torques, moments and distributed transverse loads.

Transverse loads (RadialLoad, ExternalMoment, DistributedRadialLoad) are defined by a magnitude and an
angular position theta around the shaft cross-section and are decomposed into the two principal
bending planes (XY, XZ) for the FEM shaft solver. Positions are axial coordinates x [mm] measured
from the left end of the shaft.

Conventions
-----------
- theta_deg is measured from +Y toward +Z, right-hand rule about +X (D-0010).
- RadialLoad magnitude is >= 0 (checked in validate()); the direction is fully encoded by theta.
- AxialLoad and TorqueLoad act along / about +X and need no angular decomposition.
- Fy/Fz (RadialLoad) are the components of the force along +Y and +Z. My/Mz (ExternalMoment) are
  NAMED as moment components about +Y and +Z, the right-handed convention of Cao2004, Eq. (14),
  (15) and (20) (m_z with theta_z = gamma_z + dv/dx in the x-y plane, m_y with
  theta_y = gamma_y - dw/dx in the x-z plane). The FEM does not use that convention: its nodal
  rotation is theta = dv/dx in the plane being assembled, i.e. right-hand about +Z in the XY plane
  and MINUS right-hand about +Y in the XZ plane (mesh/shaft/element_type/elem.py). See Warnings.

Unit statements in the code: forces [N], positions [mm], distributed intensity [N/mm]. For torque
and moment two statements exist and disagree (open review item U1, not decided): (A) the text of
this module at commit b3f624a declared N mm for moments and torques; (B) TorqueLoad is documented
and printed in N m (``TorqueLoad.__repr__``), as is ``ExternalMoment.__repr__``. See Warnings.

Producers and consumers of each load::

    load / field          produced by (unit)                     consumed by (unit)
    TorqueLoad.magnitude  SpurHelicalGearSystem.resolve:         torsion solver (torsion.py):
                          P/omega and +-T from forces()          treats it as [N m], x 1000 -> [N mm]
                          [N m]                                  outputs (shaft_results.py): stores
                                                                 ShaftResults.T under the key T_Nmm
                                                                 and scales it by 1e-3 (treats it as
                                                                 [N mm])
    ExternalMoment.       no producer in the repository          FEM load vector, rotation DOF,
      magnitude                                                  as [N mm]; its __repr__ prints N m
    RadialLoad.magnitude  resolve: Fr, Ft of forces() [N]        FEM force vector [N]
    AxialLoad.magnitude   resolve: +-Fa of forces() [N]          FEM force vector [N]
    DistributedRadial-    resolve (distribute_loads): Fr, Ft     FEM q(x) [N/mm] (component_intensity);
      Load.magnitude      as total force [N]                     centroid [mm] read by the mesh

Warnings
--------
Torque and moment units (open review item U1, not decided). Statement (A): this module declared N mm
for moments and torques (commit b3f624a). Statement (B): TorqueLoad is documented and printed in
N m. What the code path does: gear system -> TorqueLoad -> torsion solver carries N m (x 1000 inside
the solver); the FEM adds ExternalMoment.magnitude to a rotation DOF with E in MPa and lengths in
mm, i.e. as N mm, while ``ExternalMoment.__repr__`` prints N m; the FEM-results records
(outputs/solvers/fem_results/shaft_results.py) store ShaftResults.T [N m] under the key ``T_Nmm``
and scale it by 1e-3. Which unit is intended is open.

ExternalMoment.component(XY) returns My and component(XZ) returns Mz. As coded, ``component(XY)``
(= ``magnitude cos(theta)``, named My) is added to the rotation DOF of the XY plane, where it acts
as a moment about +Z; ``component(XZ)`` (= ``magnitude sin(theta)``, named Mz) is added to the
rotation DOF of the XZ plane, where it acts as a moment about -Y (theta = dw/dx). Neither matches
the names My/Mz nor the Conventions above (open review item). No producer of ExternalMoment exists
in the repository.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Union
import numpy as np
from scipy import integrate as _quad


class LoadPlane(Enum):
    """
    Principal bending plane, used only as a decomposition key, never as user input.

    XY is the vertical plane (takes the component Fy of every load); XZ is the horizontal plane
    (takes the component Fz), as in the AxisForge README "Conventions" (D-0010).

    Notes
    -----
    The README names the planes after the radial and the tangential gear force. That holds for a
    mesh whose line of centres is along Y (phi_deg = 0 or 180); for any other phi_deg both forces
    have components in both planes.
    """
    XY = auto()
    XZ = auto()


class Load:
    """
    Base class of all loads; not instantiated directly.

    Parameters
    ----------
    position : float
        Axial coordinate along the shaft [mm], >= 0 (checked in validate()).
    label : str
        Reporting tag.
    source : str
        Origin of the load: "user", "gear_mesh" (replaced on every gear-system resolve) or
        "bearing_reaction".
    """

    def __init__(self, position: float, label: str = "", source: str = "user"):
        """Store the arguments; see the class docstring."""
        self.position = position
        self.label = label
        self.source = source   # "user" | "gear_mesh" | "bearing_reaction"

    def validate(self) -> list[str]:
        """Return the list of error messages (position must be >= 0); empty if valid."""
        errors: list[str] = []
        tag = self.label or self.__class__.__name__
        if self.position < 0:
            errors.append(f"{tag}: position must be >= 0, got {self.position}")
        return errors

    def validate_or_raise(self) -> None:
        """Raise ValueError with all messages of validate() if there are any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


class RadialLoad(Load):
    """
    Transverse point force of magnitude F acting at the angle theta in the YZ cross-section.

    Used for gear tooth forces (radial Fr and tangential Ft), user loads and bearing reactions. The
    force is split into its Y and Z components, so it contributes to both bending planes.

    Parameters
    ----------
    position : float
        Axial coordinate [mm].
    magnitude : float
        Force magnitude [N], >= 0 (checked in validate(), not at construction).
    theta_deg : float
        Direction of the force [deg], measured from +Y toward +Z, right-hand about +X; stored
        modulo 360.
    label : str
        Reporting tag.
    source : str
        "user" | "gear_mesh" | "bearing_reaction".

    Notes
    -----
    ``Fy = magnitude * cos(theta)`` and ``Fz = magnitude * sin(theta)``, with ``theta`` =
    ``radians(theta_deg)``. The XY plane takes Fy and the XZ plane takes Fz (D-0010).
    """

    def __init__(self, position: float, magnitude: float,
                 theta_deg: float = 0.0, label: str = "",
                 source: str = "user"):
        """Store the arguments; see the class docstring."""
        super().__init__(position, label, source)
        self.magnitude = magnitude
        self.theta_deg = theta_deg % 360.0
        self.theta = np.radians(self.theta_deg)

    @property
    def Fy(self) -> float:
        """Component of the force along +Y [N]."""
        return self.magnitude * np.cos(self.theta)

    @property
    def Fz(self) -> float:
        """Component of the force along +Z [N]."""
        return self.magnitude * np.sin(self.theta)

    def component(self, plane: LoadPlane) -> float:
        """Signed force component [N] in a principal plane: Fy for XY, Fz for XZ."""
        return self.Fy if plane == LoadPlane.XY else self.Fz

    def validate(self) -> list[str]:
        """Return the error messages of the base class plus magnitude >= 0; empty if valid."""
        errors = super().validate()
        tag = self.label or self.__class__.__name__
        if self.magnitude < 0:
            errors.append(f"{tag}: magnitude must be >= 0, got {self.magnitude}")
        return errors

    def __repr__(self) -> str:
        """
        Text of the load.

        Returns
        -------
        str
            Position [mm], magnitude [N], theta [deg], Fy and Fz [N], source and label.
        """
        return (f"RadialLoad(position={self.position:.2f} mm, "
                f"F={self.magnitude:.2f} N, theta={self.theta_deg:.1f}°, "
                f"Fy={self.Fy:.2f} N, Fz={self.Fz:.2f} N, "
                f"source={self.source!r}, label={self.label!r})")


class AxialLoad(Load):
    """
    Force along the shaft axis, signed; positive = directed along +X.

    Used for the axial thrust of helical gear meshes and for user axial loads. The code applies it
    as a point force on the axis, so it creates no bending moment (see Warnings of the gear system).

    Parameters
    ----------
    position : float
        Axial coordinate [mm].
    magnitude : float
        Axial force [N], signed (no restriction on the sign).
    label : str
        Reporting tag.
    source : str
        "user" | "gear_mesh" | "bearing_reaction".
    """

    def __init__(self, position: float, magnitude: float, label: str = "",
                 source: str = "user"):
        """Store the arguments; see the class docstring."""
        super().__init__(position, label, source)
        self.magnitude = magnitude

    def __repr__(self) -> str:
        """
        Text of the load.

        Returns
        -------
        str
            Position [mm], axial force Fa [N], source and label.
        """
        return (f"AxialLoad(position={self.position:.2f} mm, "
                f"Fa={self.magnitude:.2f} N, source={self.source!r}, "
                f"label={self.label!r})")


class TorqueLoad(Load):
    """
    Torque about the shaft axis, signed; stated as positive = counter-clockwise viewed from +X.

    Produced by the gear system (mesh torque, +T on the driver gear, -T at the entry of a driven
    gear and at the power input) and consumed by the torsion solver, which accumulates it from left
    to right. The gear system assigns the sign by power flow, whatever the sense of rotation (see
    SpurHelicalGearSystem.resolve, Warnings).

    Parameters
    ----------
    position : float
        Axial coordinate [mm].
    magnitude : float
        Torque, signed. The code path gear system -> torsion solver carries [N m] (this class
        docstring and ``__repr__`` say N m).
    label : str
        Reporting tag.
    source : str
        "user" | "gear_mesh" | "bearing_reaction".

    Warnings
    --------
    Statement (A) of the module docstring gives N mm for torques; this class gives N m (open review
    item U1).
    """

    def __init__(self, position: float, magnitude: float, label: str = "",
                 source: str = "user"):
        """Store the arguments; see the class docstring."""
        super().__init__(position, label, source)
        self.magnitude = magnitude

    def __repr__(self) -> str:
        """
        Text of the load.

        Returns
        -------
        str
            Position [mm], torque with the unit label "N m" (printed with a middle dot), source and label (open review item U1).
        """
        return (f"TorqueLoad(position={self.position:.2f} mm, "
                f"T={self.magnitude:.2f} N·m, source={self.source!r}, "
                f"label={self.label!r})")


class ExternalMoment(Load):
    """
    Applied bending moment defined by a magnitude and an angle theta in the YZ cross-section
    (meaning of theta: see theta_deg and Warnings).

    Same decomposition as RadialLoad: theta is NAMED as the direction of the moment vector, whose
    components about +Y and +Z are My and Mz (see Warnings for how the FEM consumes them). No producer of this class exists in the repository
    (no gear builds one); the FEM load assembly reads it.

    Parameters
    ----------
    position : float
        Axial coordinate [mm].
    magnitude : float
        Moment magnitude. The FEM load vector adds it on the rotation DOF as [N mm]; ``__repr__``
        prints N m.
    theta_deg : float
        Angle [deg], from +Y toward +Z; stored modulo 360. NAMED as the direction of the moment
        vector. As consumed by the FEM the moment acts in the bending plane that contains X and the
        direction theta (rotation dv/dx > 0 for the deflection v along theta), i.e. its vector lies
        at theta + 90 deg (see Warnings).
    label : str
        Reporting tag.
    source : str
        "user" | "gear_mesh" | "bearing_reaction".

    Warnings
    --------
    As coded, ``component(XY)`` (= ``magnitude cos(theta)``, named My) is added to the rotation DOF
    of the XY plane, where it acts as a moment about +Z; ``component(XZ)`` (= ``magnitude
    sin(theta)``, named Mz) is added to the rotation DOF of the XZ plane, where it acts as a moment
    about -Y (theta = dw/dx). Neither matches the names My/Mz nor the module Conventions (open
    review item). The unit statements (module docstring N mm, ``__repr__`` N m) also disagree (open
    review item U1).

    Notes
    -----
    ``My = magnitude * cos(theta)`` and ``Mz = magnitude * sin(theta)``; the names follow the
    right-hand rule about +Y and +Z (see the module Conventions for the FEM rotation actually used).
    """

    def __init__(self, position: float, magnitude: float,
                 theta_deg: float = 0.0, label: str = "",
                 source: str = "user"):
        """Store the arguments; see the class docstring."""
        super().__init__(position, label, source)
        self.magnitude = magnitude
        self.theta_deg = theta_deg % 360.0
        self.theta = np.radians(self.theta_deg)

    @property
    def My(self) -> float:
        """Component ``magnitude * cos(theta)``, NAMED as the moment about +Y (same unit as
        magnitude); the FEM applies it in the XY plane, where it acts about +Z (see the class
        Warnings)."""
        return self.magnitude * np.cos(self.theta)

    @property
    def Mz(self) -> float:
        """Component ``magnitude * sin(theta)``, NAMED as the moment about +Z (same unit as
        magnitude); the FEM applies it in the XZ plane, where it acts about -Y (see the class
        Warnings)."""
        return self.magnitude * np.sin(self.theta)

    def component(self, plane: LoadPlane) -> float:
        """Moment component handed to a bending plane: My for XY, Mz for XZ (see Warnings of the class)."""
        return self.My if plane == LoadPlane.XY else self.Mz

    def __repr__(self) -> str:
        """
        Text of the load.

        Returns
        -------
        str
            Position [mm], magnitude with the unit label "N m" (printed with a middle dot), theta [deg],
            source and label (open review item U1: the FEM load vector reads the magnitude as N mm).
        """
        return (f"ExternalMoment(position={self.position:.2f} mm, "
                f"M={self.magnitude:.2f} N·m, theta={self.theta_deg:.1f}°, "
                f"source={self.source!r}, label={self.label!r})")


class DistributedRadialLoad(Load):
    """
    Transverse load distributed over [x_lo, x_hi] with arbitrary intensity
    and direction profiles.

    Used for the tooth forces spread over the gear face width (SpurHelicalMeshLink with
    distribute_loads) and for user line loads. ``position`` is the midpoint of the span.

    Parameters
    ----------
    x_lo, x_hi : float
        Axial extent of the load [mm]; x_hi > x_lo (otherwise ValueError).
    magnitude : float or callable
        A float is the TOTAL force F [N], spread uniformly (q0 = F / b with b = x_hi - x_lo); it is
        signed and negative values are accepted (e.g. Ft/Fr from a gear mesh pointing downward). A
        callable is the signed intensity function q(x) [N/mm]; F = integral of q(x) dx is computed
        once at init.
    theta_deg : float or callable
        A float is the constant angular direction [deg]; a callable is theta(x) [deg], so the
        direction varies along the span.
    label, source : str
        Reporting tag and origin, as in Load.

    Warnings
    --------
    ``magnitude`` means a total force [N] when a float is given and an intensity [N/mm] when a
    callable is given. Unlike RadialLoad, a negative ``magnitude`` is accepted.

    Notes
    -----
    When theta_deg is constant, Fy and Fz share the same projection factor and the existing
    closed-form paths are used. When theta_deg is callable, q(x) and theta(x) are evaluated
    together, so the two plane components are integrated independently::

        Fy_resultant = integral of q(x) cos(theta(x)) dx
        Fz_resultant = integral of q(x) sin(theta(x)) dx

    ``magnitude`` then stores ||F|| = sqrt(Fy^2 + Fz^2); the individual resultants are available as
    resultant_Fy and resultant_Fz. as_point_load() is intentionally unavailable for the
    variable-theta case because there is no single representative theta.

    Interface used by the FEM and the mesh::

        q(x)                                  intensity [N/mm], zero outside the span
        theta_at(x)                           direction [deg] at x
        qy(x), qz(x)                          signed intensity components [N/mm]
        component_intensity(x, plane)         signed intensity [N/mm] onto plane
        resultant_component(plane)            signed resultant [N] onto plane
        bending_moment_contribution(x, plane) M(x) contribution [N mm]
        centroid(plane)                       per-plane resultant application x [mm]
        as_point_load()                       equivalent RadialLoad (constant-theta only)
    """

    def __init__(self,
                 x_lo: float,
                 x_hi: float,
                 magnitude: Union[float, Callable[[float], float]],
                 theta_deg: Union[float, Callable[[float], float]] = 0.0,
                 label: str = "",
                 source: str = "user"):
        """
        Store the arguments and integrate the resultants; see the class docstring.

        Raises
        ------
        ValueError
            If ``x_hi <= x_lo``.
        """
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
                """Y-projected intensity ``q(x) cos(theta(x))`` [N/mm] at x [mm]."""
                th = np.radians(self._theta_fn(x))
                return self._q_fn(x) * np.cos(th)

            def _fz_integrand(x):
                """Z-projected intensity ``q(x) sin(theta(x))`` [N/mm] at x [mm]."""
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
    # Bending moment contribution [N mm]; no caller in axisforge
    # ------------------------------------------------------------------

    def bending_moment_contribution(self, x: float, plane: LoadPlane) -> float:
        """
        M(x) = ∫_{x_lo}^{min(x, x_hi)} q_plane(ξ) · (x − ξ) dξ

        Closed form for the uniform constant-theta case.
        Numerical (quad) for all other cases.
        Returns [N·mm]; x is the evaluation section [mm]. Zero for x <= x_lo.
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
        """Return the error messages of the base class plus span b > 0; empty if valid."""
        errors = super().validate()
        tag = self.label or self.__class__.__name__
        if self.b <= 0:
            errors.append(f"{tag}: span b must be > 0, got {self.b}")
        return errors

    def __repr__(self) -> str:
        """
        Text of the load.

        Returns
        -------
        str
            Span [mm], resultant force [N] (norm for a variable theta), intensity profile kind, theta,
            source and label.
        """
        q_str     = "uniform" if self._uniform else "q(x)"
        theta_str = (f"{self.theta_deg:.1f}°" if not self._theta_variable
                     else "theta(x)")
        F_str = f"F={self._magnitude:.2f} N" if not self._theta_variable else f"‖F‖={self._magnitude:.2f} N"
        return (f"DistributedRadialLoad([{self.x_lo:.2f}, {self.x_hi:.2f}] mm, "
                f"{F_str}, q={q_str}, theta={theta_str}, "
                f"source={self.source!r}, label={self.label!r})")


# Type alias: any of the load classes above.
Load_T = RadialLoad | AxialLoad | TorqueLoad | ExternalMoment | DistributedRadialLoad


# ===========================================================================
# LoadingProfile — fatigue cycle decomposition
# ===========================================================================

@dataclass(frozen=True)
class LoadingProfile:
    """
    Fatigue stress cycle described by the stress ratio R, split into mean and amplitude factors.

    Parameters
    ----------
    label : str
        Reporting tag.
    R : float
        Stress ratio [-], in [-1, 1] (ValueError otherwise). Default 1.0 (static load).

    Notes
    -----
    Not read by any module of the package.

    With R read as sigma_min / sigma_max (the code does not state it), the factors multiply the
    maximum stress: ``sigma_m = (1 + R) / 2 * sigma_max`` and ``sigma_a = |1 - R| / 2 * sigma_max``.
    Static: R = 1 (amplitude factor 0); fully reversed: R = -1 (mean factor 0).

    References
    ----------
    (source to find) Not found in the library; a shaft fatigue standard (DIN 743 or the FKM
    guideline) is not in the library either.
    """
    label: str = ""
    R: float = 1.0

    def __post_init__(self) -> None:
        """Validate the fields; see the class docstring for what is checked."""
        if not -1.0 <= self.R <= 1.0:
            raise ValueError(f"R must be in [-1.0, 1.0], got {self.R}")

    @property
    def sigma_mean_factor(self) -> float:
        """Mean-stress factor ``(1 + R) / 2`` [-], multiplying the maximum stress."""
        return (1 + self.R) / 2

    @property
    def sigma_amplitude_factor(self) -> float:
        """Stress-amplitude factor ``|1 - R| / 2`` [-], multiplying the maximum stress."""
        return abs(1 - self.R) / 2

    @property
    def is_static(self) -> bool:
        """True when R equals 1 (constant stress)."""
        return self.R == 1.0

    @property
    def is_fully_reversed(self) -> bool:
        """True when R equals -1 (zero mean stress)."""
        return self.R == -1.0

# Public names of this module.

__all__ = [
    "LoadPlane",
    "Load",
    "RadialLoad",
    "AxialLoad",
    "TorqueLoad",
    "ExternalMoment",
    "DistributedRadialLoad",
    "Load_T",
    "LoadingProfile",
]