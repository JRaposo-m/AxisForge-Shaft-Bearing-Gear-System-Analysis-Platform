"""Contract of a gear family, the two assembled, immutable objects every family returns, and the
record of the forces of one mesh.

A ``GearFamily`` turns input data into the geometry of one gear and of one gear pair; ``Gear`` and
``GearPair`` hold that geometry and refuse attribute writes once assembled, as ``Bearing`` does.
``MeshForces`` is what a family returns for a torque on a pair; its docstring holds the mesh frame
and the sign table. Nothing physical is computed here: the equations belong to the family
(``cylindrical.py``).
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class MeshForces:
    """Forces of one gear mesh on its two gears, for one torque: magnitudes and signed components.

    One normal force F_bn acts between the flanks, in the plane tangent to both base cylinders
    and inclined by the base helix angle to the transverse plane. This record holds its
    components on the pitch cylinder (what the shafts and bearings receive), on the base cylinder
    (what the tooth contact receives) and the nominal tangential force at the reference cylinder
    (the input of the ISO 6336 rating formulae), then the same force as a signed vector on each
    gear.

    Mesh frame (D-0010 item 14), right-handed, origin at the pitch point C at mid face width:

    - ``e_c``: along the line of centres, from the axis of gear 1 to the axis of gear 2;
    - ``e_x``: the shaft axis of gear 1 (for parallel shafts, global +X);
    - ``e_t = e_x x e_c``: tangent to both pitch circles at C.

    A force here is the force exerted BY THE MESH ON a gear (D-0010 item 15), so ``on_2`` is the
    reaction of ``on_1``: ``on_2 = -on_1``, both applied at C. The shaft loads of a transmission are
    built from this record by ``mesh_forces_to_shaft_loads`` (Q05, José 2026-10-15).

    Sign table (D-0010 item 16), with T_1 the torque applied to gear 1 by its own shaft about
    e_x [N m], rho_i the signed distance of C from the axis of gear i along e_c [mm] and h_1 the
    hand of gear 1 (+1 right, -1 left). "Right-hand" is taken here geometrically: the tooth trace
    turns in the right-hand sense about e_x when followed along +e_x. The definition of the
    flank direction in ISO_21771_2007 was not transcribed, so this one has no library source::

        F_1,t = -1000 T_1 / rho_1                    the mesh balances the shaft torque
        F_1,c = -sign(z_1 z_2) F_r                   the flanks only push: an external gear is pushed
                                                     toward its own axis, an internal gear outward
        F_1,x = -h_1 sign(rho_1) tan(beta_w) F_1,t   normal to the tooth trace at C; equal to
                                                     h rot F_a on the DRIVING gear along its axis

    External pair, gear 1 driving with the sense rot (+1 or -1 about e_x), h the hand of gear 1::

        on gear 1:  (-F_r, -rot F_tw, +h rot F_a)        on gear 2:  (+F_r, +rot F_tw, -h rot F_a)

    Internal pair (gear 2 the ring), the pinion driving::

        on gear 1:  (+F_r, +rot F_tw, +h rot F_a)        on gear 2:  (-F_r, -rot F_tw, -h rot F_a)

    Attributes
    ----------
    F_tw_N : float
        Tangential force at the working pitch cylinder, 2000 abs(T_1) / d_w1 [N], >= 0.
    F_r_N : float
        Radial (separating) force at the pitch point, F_tw tan(alpha_wt) [N], >= 0.
    F_a_N : float
        Axial force, F_bt tan(beta_b) = F_tw tan(beta_w) [N], >= 0 (0 for a spur pair).
    F_bt_N : float
        Transverse force along the line of action, 2000 abs(T_1) / d_b1 [N], >= 0.
    F_bn_N : float
        Force normal to the flanks, F_bt / cos(beta_b) [N], >= 0: the load of the tooth contact
        (``axisforge.solvers.machine_elements.gears.contact_inputs(F_bn_N=...)``).
    F_t_nominal_N : float
        Nominal tangential force at the REFERENCE cylinder, 2000 abs(T_1) / d_1 [N], >= 0: the
        input of the rating formulae, never a shaft load.
    on_1, on_2 : tuple of float
        (F_c, F_t, F_x) [N]: force of the mesh on gear 1 and on gear 2 in the mesh frame, signed.
    rho_mm : tuple of float
        (rho_1, rho_2) [mm]: signed distance of C from the axis of each gear along e_c;
        rho_2 = rho_1 - a_w. External pair: (+r_w1, -r_w2); internal pair: (-r_w1, -r_w2).
    x_offset_mm : tuple of float
        Axial offset of C from the position of each gear on its shaft [mm]; (0, 0) for a
        cylindrical pair (C at mid face width). It exists for bevel pairs.

    Notes
    -----
    Units: forces [N] from a torque in [N m] and diameters in [mm], hence the factor 2000
    (D-0010 item 5). Assumptions: ideal mesh (no friction, no losses), rigid involute flanks,
    steady rotation.

    The three sets (pitch, base, reference cylinder) describe one force: F_tw**2 + F_r**2 +
    F_a**2 = F_bn**2.

    References
    ----------
    .. [1] ISO_6336-1_2019, 4.2.2, Formula (1), p. 19 (F_t nominal, reference diameter; checked).
    .. [2] D-0010 items 14-18; vault note ``units_and_conventions.md``, C.4 and C.5. F_tw, F_r,
           F_a, F_bt, F_bn and every direction: no library source (statics, compressive contact,
           geometry of the involute helicoid with ISO_21771_2007 Eq. (5), (19)).
    """

    F_tw_N: float
    F_r_N: float
    F_a_N: float
    F_bt_N: float
    F_bn_N: float
    F_t_nominal_N: float
    on_1: tuple[float, float, float]
    on_2: tuple[float, float, float]
    rho_mm: tuple[float, float]
    x_offset_mm: tuple[float, float] = (0.0, 0.0)


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

    @abstractmethod
    def mesh_forces(self, pair: "GearPair", T1_Nm: float, rot1: int) -> MeshForces:
        """Forces of the mesh on the two gears of the pair for a torque on gear 1.

        Parameters
        ----------
        pair : GearPair
            Pair of this family.
        T1_Nm : float
            Torque applied to gear 1 by its own shaft, about the axis of gear 1 [N m], signed.
        rot1 : int
            Sense of rotation of gear 1 about its axis, +1 or -1 [-]. Gear 1 drives when
            ``T1_Nm * rot1 > 0`` (D-0010 items 11-13).

        Returns
        -------
        MeshForces
            Magnitudes and signed components in the mesh frame (see ``MeshForces``).
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
    def assemble(cls, family: GearFamily, data: Any, *, check: bool = True) -> "Gear":
        """Build an assembled, immutable gear.

        Parameters
        ----------
        family : GearFamily
            Family instance, passed directly.
        data : object
            Input record of the family.
        check : bool, optional
            If True (default), run ``validate()`` and refuse an invalid gear. False only for the
            equivalence records, the planetary module until step 6 (Q52), and tests of the messages
            of invalid objects (D-0026 item 9; José, 2026-10-15, answer 3).

        Returns
        -------
        Gear
            Gear with the family attributes set, flagged as assembled.

        Raises
        ------
        ValueError
            If the family cannot compute the gear from ``data``, or (``check``) with every message
            of ``validate()`` in one error.
        NotImplementedError
            If the family defines its gears through pair data only.
        """
        gear = cls(family, data)
        gear._freeze(family.assemble_gear(data))
        if check:   # Q51 (José, 2026-10-15): validation at assembly, one error with every message
            gear.validate_or_raise()
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
        Advisory messages of the family; they never raise (cylindrical family: an imposed
        centre distance that leaves a backlash, see ``CylindricalFamily.assemble_pair``).

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
    def assemble(cls, family: GearFamily, *, check: bool = True, **pair_inputs: Any) -> "GearPair":
        """Build an assembled, immutable pair.

        Parameters
        ----------
        family : GearFamily
            Family instance, passed directly.
        check : bool, optional
            If True (default), run ``validate()`` (which includes the messages of the two gears)
            and refuse an invalid pair. False only for the equivalence records, the planetary module
            until step 6 (Q52), and tests of the messages of invalid objects (D-0026 item 9; José,
            2026-10-15, answer 3).
        **pair_inputs
            Keyword arguments forwarded to ``family.assemble_pair`` (for the cylindrical family:
            ``gear1``, ``gear2``, ``label``, ``al``, ``addendum_reduction``).

        Returns
        -------
        GearPair
            Pair with the family attributes set, flagged as assembled. Advisory messages are in
            ``warnings``.

        Raises
        ------
        ValueError
            If the family rejects the two gears (they cannot mesh), or (``check``) with every
            message of ``validate()`` in one error.
        """
        gear1, gear2, computed = family.assemble_pair(**pair_inputs)
        pair = cls(family, gear1, gear2)
        pair._freeze(computed)
        if check:   # Q51 (José, 2026-10-15): validation at assembly, one error with every message
            pair.validate_or_raise()
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
