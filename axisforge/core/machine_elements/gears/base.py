"""Contract of a gear family, and the two assembled, immutable objects every family returns.

A ``GearFamily`` turns input data into the geometry of one gear and of one gear pair; ``Gear`` and
``GearPair`` hold that geometry and refuse attribute writes once assembled, as ``Bearing`` does.
Nothing physical is computed here: the equations belong to the family (``cylindrical.py``).
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any

# class name -> class; filled by the decorator below at class definition
_GEAR_FAMILY_REGISTRY: dict[str, type["GearFamily"]] = {}


def register_gear_family(cls: type["GearFamily"]) -> type["GearFamily"]:
    """Class decorator: add the family to the registry of all gear families (by class name)."""
    _GEAR_FAMILY_REGISTRY[cls.__name__] = cls
    return cls


class GearFamily(ABC):
    """Contract that every gear family implements to be used by ``Gear.assemble`` and ``GearPair.assemble``.

    A family maps the input data of one gear type to the attributes stored on the ``Gear`` and on
    the ``GearPair``, and holds the checks and the texts that depend on the type. It is a plain
    instance contract with no state; registration is done with ``@register_gear_family``.

    Notes
    -----
    ``assemble_gear`` is concrete and raises by default, because a family may define its gears
    only through pair data (a bevel gear: shaft angle, both tooth numbers, one spiral angle); such
    a family returns its two gears from ``assemble_pair``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier of the family, e.g. ``'cylindrical'``."""

    @property
    @abstractmethod
    def axes(self) -> str:
        """Relative position of the two axes of a pair: ``"parallel"``, ``"intersecting"`` or ``"crossed"``.

        ``"intersecting"`` (bevel) and ``"crossed"`` (worm, crossed helical) are values only: no
        family with these axes exists yet.
        """

    def assemble_gear(self, data: Any) -> dict[str, Any]:
        """Map the input record of one gear to the attributes stored on the ``Gear``.

        Parameters
        ----------
        data : object
            Input record of the family (a frozen dataclass).

        Returns
        -------
        dict
            Flat dictionary of attributes, each set on the ``Gear`` by ``Gear.assemble``. The key
            ``"warnings"`` (tuple of str), if present, becomes ``Gear.warnings``.

        Raises
        ------
        NotImplementedError
            By default: the family does not define a gear alone.
        """
        raise NotImplementedError(f"{type(self).__name__}: a gear of this family is defined by pair data; "
                                  f"use GearPair.assemble")

    @abstractmethod
    def assemble_pair(self, **pair_inputs: Any) -> tuple["Gear", "Gear", dict[str, Any]]:
        """Map the pair inputs to the two gears and the attributes stored on the ``GearPair``.

        Returns
        -------
        tuple
            ``(gear1, gear2, attributes)``: the two assembled gears and a flat dictionary of the
            mesh geometry (key ``"warnings"`` as in ``assemble_gear``).
        """

    def validate_gear(self, gear: "Gear") -> list[str]:
        """Messages of the checks of one gear of this family; empty if valid (default: no check)."""
        return []

    @abstractmethod
    def validate_pair(self, pair: "GearPair") -> list[str]:
        """Messages of the checks of one pair of this family; empty if valid."""

    @abstractmethod
    def pair_summary(self, pair: "GearPair") -> str:
        """Multi-line text with the main data of the pair."""

    @abstractmethod
    def pair_repr(self, pair: "GearPair") -> str:
        """One-line representation of the pair."""


class _Assembled:
    """Behaviour shared by ``Gear`` and ``GearPair``: immutability after ``assemble``, checks, warnings."""

    _family: GearFamily
    _assembled: bool
    label: str
    warnings: tuple[str, ...]

    def _freeze(self, computed: dict[str, Any]) -> None:
        """Set the attributes computed by the family, then refuse further writes."""
        self.warnings = tuple(computed.pop("warnings", ()))
        for attr, value in computed.items():
            setattr(self, attr, value)
        self._assembled = True

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute; refused once the object is assembled.

        Raises
        ------
        AttributeError
            If ``assemble()`` has completed.
        """
        if getattr(self, "_assembled", False):
            raise AttributeError(
                f"{type(self).__name__}(label={getattr(self, 'label', '')!r}) is immutable "
                f"after assemble() -- cannot set '{name}'. Operating quantities (torque, driver, "
                f"speed) are arguments of the calls that need them, not stored geometry."
            )
        super().__setattr__(name, value)

    @property
    def family(self) -> GearFamily:
        """The ``GearFamily`` instance this object was assembled with."""
        return self._family

    def validate_or_raise(self) -> None:
        """Raise ``ValueError`` with all messages of ``validate()`` if any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


class Gear(_Assembled):
    """One gear: its input data and the geometry derived by its family, immutable once assembled.

    Build it with ``Gear.assemble()``, not directly: every field of the input record and every key
    of the dictionary returned by ``family.assemble_gear`` is set as an attribute, and afterwards
    the instance refuses attribute writes.

    Parameters
    ----------
    family : GearFamily
        Family that provides the geometry and the checks.
    data : object
        Input record of the family (a frozen dataclass); its fields are copied as attributes.

    Attributes
    ----------
    data : object
        The input record as given.
    warnings : tuple of str
        Advisory messages of the family (checks with no library source or advisory limits); they
        never raise. No family produces one yet: the tuple is empty.

    Notes
    -----
    The other attributes (diameters, pitches, angles and their units) are those documented by the
    family (``CylindricalFamily.assemble_gear``).
    """

    def __init__(self, family: GearFamily, data: Any):
        """Copy the fields of the input record; see the class docstring."""
        self.data = data
        for field in dataclasses.fields(data):
            setattr(self, field.name, getattr(data, field.name))
        self._family = family
        self._assembled = False   # __setattr__ refuses further writes once True

    @classmethod
    def assemble(cls, family: GearFamily, data: Any) -> "Gear":
        """Build an assembled, immutable gear.

        Parameters
        ----------
        family : GearFamily
            Family instance, passed directly.
        data : object
            Input record of the family.

        Returns
        -------
        Gear
            Gear with the family attributes set, flagged as assembled.

        Raises
        ------
        ValueError
            If the family cannot compute the gear from ``data``.
        NotImplementedError
            If the family defines its gears through pair data only.
        """
        gear = cls(family, data)
        # QUESTION(Q51): no validate() here, as in today's constructors (validation stays a separate call); the alternative of D-0026 item 9 raises at assembly.
        gear._freeze(family.assemble_gear(data))
        return gear

    def validate(self) -> list[str]:
        """Check the gear.

        Returns
        -------
        list of str
            One message per violation, from ``family.validate_gear``; empty if valid.
        """
        return self._family.validate_gear(self)

    def summary(self) -> str:
        """Text summary of the gear: family, then every scalar attribute as stored, then the warnings.

        Returns
        -------
        str
            Multi-line text; the units are those documented by the family.
        """
        tag = self.label or "Gear"
        header = f"-- {tag} "
        lines = [header + "-" * max(0, 44 - len(header)), f"  family : {self._family.name}"]
        lines += [f"  {name:<12}: {value}" for name, value in vars(self).items()
                  if not name.startswith("_") and name not in ("data", "label", "warnings")]
        lines += [f"  warning: {w}" for w in self.warnings]
        lines.append("-" * 44)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Short representation: class, family name and label."""
        return f"{self.__class__.__name__}(family={self._family.name!r}, label={self.label!r})"


class GearPair(_Assembled):
    """Two gears in mesh and the geometry of their mesh, immutable once assembled.

    The pair is the link between two gears. It holds the two gears and the mesh geometry computed
    by the family; it holds no shaft, no driver and no torque: those are operating quantities,
    given as arguments to the calls that need them.

    Parameters
    ----------
    family : GearFamily
        Family that provides the mesh geometry and the checks.
    gear1, gear2 : Gear
        The two gears of the pair, as returned by ``family.assemble_pair``.

    Attributes
    ----------
    gear1, gear2 : Gear
        The two gears.
    label : str
        Name used in the summary.
    warnings : tuple of str
        Advisory messages of the family; they never raise. No family produces one yet.

    Notes
    -----
    The other attributes (centre distances, working pressure angle, contact ratios, path of
    contact and their units) are those documented by the family
    (``CylindricalFamily.assemble_pair``).
    """

    def __init__(self, family: GearFamily, gear1: Gear, gear2: Gear):
        """Store the two gears; see the class docstring."""
        self.gear1 = gear1
        self.gear2 = gear2
        self._family = family
        self._assembled = False   # __setattr__ refuses further writes once True

    @classmethod
    def assemble(cls, family: GearFamily, **pair_inputs: Any) -> "GearPair":
        """Build an assembled, immutable pair.

        Parameters
        ----------
        family : GearFamily
            Family instance, passed directly.
        **pair_inputs
            Keyword arguments forwarded to ``family.assemble_pair`` (for the cylindrical family:
            ``gear1``, ``gear2``, ``label``, ``al``, ``equalise_gs``, ``addendum_reduction``).

        Returns
        -------
        GearPair
            Pair with the family attributes set, flagged as assembled.

        Raises
        ------
        ValueError
            If the family rejects the two gears (they cannot mesh).
        """
        gear1, gear2, computed = family.assemble_pair(**pair_inputs)
        pair = cls(family, gear1, gear2)
        # QUESTION(Q51): no validate() here, as in today's constructors: every internal pair fails validate_or_raise() today (Q24), so raising at assembly would refuse baseline cases 4, 5, 6, 6b.
        pair._freeze(computed)
        return pair

    def validate(self) -> list[str]:
        """Check the pair.

        Returns
        -------
        list of str
            One message per violation, from ``family.validate_pair``; empty if valid.
        """
        return self._family.validate_pair(self)

    def summary(self) -> str:
        """Multi-line text with the main data of the pair, followed by the warnings if any."""
        return "\n".join([self._family.pair_summary(self), *(f"  warning: {w}" for w in self.warnings)])

    def __repr__(self) -> str:
        """One-line representation of the pair, as written by the family."""
        return self._family.pair_repr(self)
