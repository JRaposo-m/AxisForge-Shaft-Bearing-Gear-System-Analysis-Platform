"""Assembled rolling bearing: catalogue data plus the geometry and constants of its family.

``Bearing`` combines the catalogue data with the attributes computed by a
``BearingFamily`` and is immutable once assembled with ``Bearing.assemble()``.
It holds no load case, life or other analysis result; those belong to the
solvers.
"""
from __future__ import annotations
from typing import Any, Iterable
import numpy as np

from axisforge.core.machine_elements.bearings.base import BearingCatalog
from axisforge.core.machine_elements.bearings.base import BearingFamily
from axisforge.core.machine_elements.bearings.families.family import (
    is_thrust_family, is_point_contact_family, is_line_contact_family,
)

# attribute names of the ISO/TS 16281 constants a family stores on the bearing
_ISO16281_MATERIAL = ("e1", "e2", "nu1", "nu2")
_ISO16281_POINT    = ("cp", "Ri")
_ISO16281_LINE     = ("cL", "cs", "P_xk")

def _fmt_value(value) -> str:
    """Format a value for the summary: scalar as a number, short vector as a list, long vector as shape, min and max."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return f"{float(arr):.4g}"
    if arr.size <= 4:
        return "[" + ", ".join(f"{v:.4g}" for v in arr.ravel()) + "]"
    return f"array{arr.shape} min={arr.min():.4g} max={arr.max():.4g}"

def _validate_thrust_arrangement(family: BearingFamily, catalog: BearingCatalog) -> None:
    """Check that the family and the catalogue arrangement agree on thrust duty.

    A family registered as ``@thrust`` must be assembled with
    ``arrangement='thrust'``, and a non-thrust family must not be given it.

    Raises
    ------
    ValueError
        If exactly one of the two is thrust.
    """
    thrust_family = is_thrust_family(family)
    thrust_arrangement = catalog.arrangement == "thrust"
    if thrust_family and not thrust_arrangement:
        raise ValueError(
            f"{catalog.label or catalog.designation}: family {family.name!r} is a "
            f"thrust family but arrangement={catalog.arrangement!r} -- expected 'thrust'.")
    if thrust_arrangement and not thrust_family:
        raise ValueError(
            f"{catalog.label or catalog.designation}: arrangement='thrust' but family "
            f"{family.name!r} is not registered as a thrust family.")

# TODO: move this validation into family.py, as is done for the cylindrical roller

class Bearing:
    """Rolling bearing: catalogue data and family-derived geometry, immutable once assembled.

    Build it with ``Bearing.assemble()``, not directly: the catalogue values
    are copied, every key of the dictionary returned by
    ``family.assemble_geometry`` is set as an attribute, and afterwards the
    instance refuses attribute writes.

    Parameters
    ----------
    catalog : BearingCatalog
        Catalogue data; d, D, b, position [mm] and the identifying fields are
        copied.
    family : BearingFamily
        Family that provides the geometry and the load ratings.

    Attributes
    ----------
    d, D, b : float
        Bore diameter, outer diameter and width [mm].
    position : float
        Axial coordinate along the shaft [mm].
    designation, label : str
        Designation and identifier of the bearing.
    arrangement : str
        ``"locating"``, ``"floating"``, ``"non-locating"`` or ``"thrust"``.
    dm : float
        Mean of bore and outer diameter, 0.5 (d + D) [mm].
    surfaces : None
        Placeholder for a ``BearingSurfaces`` object, set if the family
        provides one; no AxisForge family does, so it stays None.

    Notes
    -----
    ``dm = 0.5 (d + D)`` is the approximation to the pitch diameter of
    Harris2007_vol1, 2.2, Eq. 2.1 (d_m ~ (1/2)(bore + O.D.)); the families
    carry their own pitch diameter ``Dpw``. The family attributes (``Z``,
    ``alpha_0`` [rad], ``gamma``, ...) are those returned by its
    ``assemble_geometry``.
    """

    def __init__(self, catalog: BearingCatalog, family: BearingFamily):
        """Copy the catalogue data and set the family attributes; see the class docstring."""
        self.d = catalog.d
        self.D = catalog.D
        self.b = catalog.b
        self.designation = catalog.designation
        self.label = catalog.label
        self.position = catalog.position
        self.arrangement = catalog.arrangement
        self.dm = 0.5 * (catalog.d + catalog.D)
        self.surfaces = None   # BearingSurfaces; set by assemble() if the family provides it

        self._family = family
        self._enabled_analyses: frozenset[str] = frozenset()
        self._assembled = False   # __setattr__ refuses further writes once True

    @classmethod
    def assemble(cls,
                 family: BearingFamily,
                 catalog: BearingCatalog,
                 geometry: dict[str, Any]) -> "Bearing":
        """Build an assembled, immutable bearing.

        Parameters
        ----------
        family : BearingFamily
            Family instance, passed directly.
        catalog : BearingCatalog
            Catalogue data.
        geometry : dict[str, Any]
            Raw keyword arguments forwarded to
            ``family.assemble_geometry(catalog, **geometry)``.

        Returns
        -------
        Bearing
            Bearing with the family attributes set, flagged as assembled. The
            ISO/TS 16281 analysis is enabled if the family returned
            ``iso16281_analysis = True``.

        Raises
        ------
        ValueError
            If the catalogue is invalid, if the family and the arrangement
            disagree on thrust duty, or if the family rejects its inputs.
        """
        catalog.validate_or_raise()
        _validate_thrust_arrangement(family, catalog)
        bearing = cls(catalog, family)

        computed = family.assemble_geometry(catalog, **geometry)
        for attr, value in computed.items():
            setattr(bearing, attr, value)

        enabled = set()
        if computed.get("iso16281_analysis", False):
            enabled.add("iso16281")
        bearing._enabled_analyses = frozenset(enabled)

        bearing._assembled = True
        return bearing

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute; refused once the bearing is assembled.

        Raises
        ------
        AttributeError
            If ``assemble()`` has completed.
        """
        if getattr(self, "_assembled", False):
            raise AttributeError(
                f"Bearing(label={getattr(self, 'label', '')!r}) is immutable "
                f"after assemble() -- cannot set '{name}'. Analysis results "
                f"belong in the solver's own result object, not on the Bearing."
            )
        super().__setattr__(name, value)

    def has_internal_geometry(self) -> bool:
        """Tell whether ``assemble()`` has completed.

        Returns
        -------
        bool
            True once the family attributes are set.
        """
        return self._assembled

    def is_locating(self) -> bool:
        """Tell whether the bearing is the locating bearing of the arrangement.

        Returns
        -------
        bool
            True if ``arrangement == "locating"``.
        """
        return self.arrangement == "locating"

    @property
    def family(self) -> BearingFamily:
        """
        The ``BearingFamily`` instance this ``Bearing`` was assembled
        with. Exposed publicly so that callers (scripts, solvers) can
        dispatch family-specific methods (e.g.
        ``per_element_dynamic_capacity()``) off the bearing itself,
        without having to re-import or track the concrete subtype
        class: ``bearing.family.per_element_dynamic_capacity(bearing)``
        rather than
        ``DeepGrooveBallFamily.per_element_dynamic_capacity(bearing)``.
        """
        return self._family

    @property
    def C(self) -> float:
        """
        Basic dynamic load rating (``Cr`` for radial duty, ``Ca`` for
        thrust duty) [N], computed by the family from the assembled
        geometry. Read by the solvers' ``_Cr_Ca()``. Requires
        ``assemble()`` to have completed.
        """
        if not self._assembled:
            raise RuntimeError(
                f"Bearing '{self.label or self.designation}': C requires "
                f"Bearing.assemble() to have completed.")
        return self._family.dynamic_capacity(self)

    def validate(self) -> list[str]:
        """Check the stored catalogue values and the assembled state.

        Called by ``ShaftSystem.validate()`` for every bearing on the shaft.
        The catalogue checks of ``BearingCatalog.validate`` are repeated on
        the copied attributes as defence in depth: the catalogue has already
        been validated by ``assemble()`` and the instance is immutable, so in
        normal use only a bypass of ``__setattr__`` can make them fail.

        Returns
        -------
        list[str]
            One message per violated condition (not assembled, position < 0,
            unknown arrangement, d <= 0, D <= 0, D <= d, b < 0); empty if
            valid.
        """
        errors: list[str] = []
        tag = self.label or self.designation or self.__class__.__name__

        if not self._assembled:
            errors.append(f"{tag}: not assembled -- Bearing.assemble() never completed")

        if self.position < 0:
            errors.append(f"{tag}: position must be >= 0, got {self.position}")
        if self.arrangement not in ("locating", "floating", "non-locating", "thrust"):
            errors.append(
                f"{tag}: arrangement must be 'locating', 'floating', "
                f"'non-locating', or 'thrust', got '{self.arrangement}'"
            )
        if self.d <= 0:
            errors.append(f"{tag}: d must be > 0, got {self.d}")
        if self.D <= 0:
            errors.append(f"{tag}: D must be > 0, got {self.D}")
        if self.D <= self.d:
            errors.append(f"{tag}: D must be > d, got D={self.D}, d={self.d}")
        if self.b < 0:
            errors.append(f"{tag}: b must be >= 0, got {self.b}")

        return errors

    def has_iso16281_analysis(self) -> bool:
        """Tell whether the family stored the ISO/TS 16281 contact constants.

        Returns
        -------
        bool
            True if the attribute ``iso16281_analysis`` is set and true.
        """
        return bool(getattr(self, "iso16281_analysis", False))

    def iso16281_attributes(self) -> dict[str, Any]:
        """Attributes associated with the ISO/TS 16281 analysis ({} if it is not active).

        Returns
        -------
        dict[str, Any]
            Materials ``e1``, ``e2``, ``nu1``, ``nu2`` plus, for a point-contact
            family, ``cp`` [N/mm^1.5] and ``Ri`` [mm], or, for a line-contact
            family, ``cL``, ``cs`` [N/mm^(10/9)] and ``P_xk`` [mm]; only those
            present on the bearing. Empty if the analysis is not active.
        """
        if not self.has_iso16281_analysis():
            return {}
        if is_point_contact_family(self._family):
            keys = _ISO16281_MATERIAL + _ISO16281_POINT
        elif is_line_contact_family(self._family):
            keys = _ISO16281_MATERIAL + _ISO16281_LINE
        else:
            keys = _ISO16281_MATERIAL
        return {k: getattr(self, k) for k in keys if hasattr(self, k)}

    def summary(self) -> str:
        """Text summary of the bearing: family, position [mm], arrangement, d, D, b [mm], computed C [N] and the ISO/TS 16281 constants.

        Returns
        -------
        str
            Multi-line text.
        """
        tag =self.label or self.designation or "Bearing"
        header = f"-- {tag} "
        c_txt = f"{self.C:.0f} N" if self._assembled else "n/a"
        lines = [
            header + "-" * max(0, 44 - len(header)),
            f"  family      : {self._family.name}",
            f"  position    : {self.position:.2f} mm",
            f"  arrangement : {self.arrangement}",
            f"  d / D       : {self.d:.1f} mm / {self.D:.1f} mm",
            f"  b           : {self.b:.1f} mm",
            f"  C (computed): {c_txt}",
            f"  geometry    : {'assembled' if self._assembled else 'not assembled'}",
        ]
        if self.has_iso16281_analysis():
            lines.append("  ISO 16281   : on")
            lines += [f"    {k:<4}      : {_fmt_value(v)}"
                      for k, v in self.iso16281_attributes().items()]
        else:
            lines.append("  ISO 16281   : off")
        lines.append("-" * 44)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Short representation: class, family name, designation, position [mm] and arrangement."""
        return (
            f"{self.__class__.__name__}("
            f"family={self._family.name!r}, "
            f"designation={self.designation!r}, "
            f"position={self.position:.2f} mm, "
            f"arrangement={self.arrangement!r})"
        )