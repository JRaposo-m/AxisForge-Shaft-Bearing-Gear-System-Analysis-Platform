"""
tests/core/mechanical_system/test_internal_gear_meshing.py

PORT of the legacy tests/test_core/mechanical_system/gear_meshing/
test_internal_gear_meshing.py, which no longer collects.

What changed and why
--------------------
1. SpurGear / HelicalGear no longer exist -- both were merged into the single
   SpurHelicalGear class (beta_n_deg=0 gives a spur gear). The old file
   imported spur_gear.py and helical_gear.py, which is what killed collection.
2. The meshing package moved:
       core.mechanical_system.gear_meshing
    -> core.mechanical_system.Parallel_Axis_systems.gear_meshing
3. InternalGear did NOT move -- that import was already correct.

Everything else -- the fixtures, the assertions, the round-trip check -- is
the original author's, unchanged. The point of a port is to find out whether
the old expectations still hold, not to quietly rewrite them.

SIGN CONVENTION -- read this before touching the fixtures
---------------------------------------------------------
The ring gear is built with NEGATIVE z (z2 = -60 here). That is the
convention InternalGearMeshing's working geometry actually implements:

    InternalGear.d  = mt * abs(z)        -> diameter always positive
    z_sum = z2 + z1 = -60 + 20 = -40     -> |z_sum| IS the tooth difference
    __init__ calls _set_al_from_x_diff(gear1.x + gear2.x)  -> a SUM

i.e. the plain external-pair formulae, which is precisely what the
negative-z convention exists to allow.

internal_gear_meshing.py's MODULE DOCSTRING says the opposite -- twice --
claiming z2 is "always taken here as a positive magnitude". It is wrong.
Building the ring with z=+60 yields al = 80.58 mm against a = 40.00 mm,
which is geometrically impossible (the pinion would sit entirely outside
the ring), and nothing raises. See tests/README.md, finding I.

`u` is therefore NEGATIVE (z2/z1 = -3): the sign encodes that an internal
pair rotates in the same sense, unlike an external pair.

ASCII only in console output.

Covers the two things the `driver` change introduced:
  1. The `driver` parameter ("gear1" / "gear2") -- validation and default.
  2. forces() switching input torque, working radius, helix angle and gear
     ratio depending on which gear drives.

gear1 = external pinion (SpurHelicalGear), gear2 = InternalGear (ring).
"""

import numpy as np
import pytest

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_helical_gear import (
    SpurHelicalGear,
)
from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.internal_gear import (
    InternalGear,
)
from axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.internal_gear_meshing import (
    InternalGearMeshing,
)


# ----------------------------------------------------------------------
# Fixtures -- a geometrically reasonable spur/internal pair and a
# helical/internal pair (same tooth counts, mn, x -- only beta differs).
# ----------------------------------------------------------------------

Z1 = 20      # external pinion
Z2 = -60     # ring -- NEGATIVE, see the sign-convention note above


@pytest.fixture
def spur_pair():
    """beta_n_deg=0 is what used to be SpurGear."""
    pinion = SpurHelicalGear(mn=2.0, z=Z1, x=0.3, b=25, alpha_n_deg=20.0,
                             beta_n_deg=0.0, label="pinion")
    ring = InternalGear(mn=2.0, z=Z2, x=0.0, b=25, alpha_n_deg=20.0,
                        beta_n_deg=0.0, label="ring")
    return pinion, ring


@pytest.fixture
def helical_pair():
    pinion = SpurHelicalGear(
        mn=2.0, z=Z1, x=0.3, b=25, alpha_n_deg=20.0, beta_n_deg=15.0,
        label="pinion-helical",
    )
    ring = InternalGear(
        mn=2.0, z=Z2, x=0.0, b=25, alpha_n_deg=20.0, beta_n_deg=15.0,
        label="ring-helical",
    )
    return pinion, ring


# ----------------------------------------------------------------------
# driver parameter -- validation & default
# ----------------------------------------------------------------------

def test_default_driver_is_gear1(spur_pair):
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring)
    assert mesh.driver == "gear1"


def test_driven_is_the_other_gear(spur_pair):
    """Added on port: `driven` is derived, not passed -- worth pinning."""
    pinion, ring = spur_pair
    assert InternalGearMeshing(pinion, ring, driver="gear1").driven == "gear2"
    assert InternalGearMeshing(pinion, ring, driver="gear2").driven == "gear1"


@pytest.mark.parametrize("driver", ["gear1", "gear2"])
def test_valid_driver_accepted(spur_pair, driver):
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring, driver=driver)
    assert mesh.driver == driver


@pytest.mark.parametrize("bad_driver", ["gear3", "GEAR1", "", None, 1])
def test_invalid_driver_raises(spur_pair, bad_driver):
    pinion, ring = spur_pair
    with pytest.raises(ValueError, match="driver must be"):
        InternalGearMeshing(gear1=pinion, gear2=ring, driver=bad_driver)


# ----------------------------------------------------------------------
# Pair construction guards
# ----------------------------------------------------------------------

def test_ring_must_have_more_teeth_than_the_pinion():
    """The constructor enforces abs(z2) > z1 -- magnitude, so sign-agnostic."""
    pinion = SpurHelicalGear(mn=2.0, z=60, b=25, label="too_big")
    ring = InternalGear(mn=2.0, z=-60, b=25, label="ring")
    with pytest.raises(ValueError, match="more teeth"):
        InternalGearMeshing(gear1=pinion, gear2=ring)


@pytest.mark.parametrize("field,value,message", [
    ("mn", 3.0, "Module mismatch"),
    ("alpha_n_deg", 25.0, "Pressure angle mismatch"),
    ("beta_n_deg", 10.0, "Helix angle mismatch"),
])
def test_incompatible_gears_rejected(field, value, message):
    """Added on port: _check_compatibility runs before anything else."""
    pinion = SpurHelicalGear(mn=2.0, z=Z1, x=0.3, b=25, alpha_n_deg=20.0,
                             beta_n_deg=0.0)
    kwargs = dict(mn=2.0, z=Z2, x=0.0, b=25, alpha_n_deg=20.0, beta_n_deg=0.0)
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        InternalGearMeshing(gear1=pinion, gear2=InternalGear(**kwargs))


@pytest.mark.xfail(
    strict=True,
    reason="A ring built with positive z silently produces al = 80.58 mm "
           "against a = 40.00 mm -- geometrically impossible, and nothing "
           "raises. The module docstring actively recommends positive z. "
           "See tests/README.md, finding I.",
)
def test_positive_z_ring_is_rejected_or_gives_sane_geometry():
    """
    The working geometry is written for the negative-z convention
    (z_sum = z2 + z1 IS the tooth difference). A positive-z ring should
    either be refused outright or produce al near a -- not garbage.
    """
    pinion = SpurHelicalGear(mn=2.0, z=Z1, x=0.3, b=25, label="pinion")
    ring = InternalGear(mn=2.0, z=abs(Z2), x=0.0, b=25, label="ring_positive_z")

    mesh = InternalGearMeshing(gear1=pinion, gear2=ring)
    assert mesh.al == pytest.approx(mesh.a, rel=0.15)


# ----------------------------------------------------------------------
# forces() -- driver = gear1 (legacy behaviour)
# ----------------------------------------------------------------------

def test_forces_driver_gear1_matches_manual_formula(spur_pair):
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear1")

    T_in = 50.0  # N.m
    result = mesh.forces(T_in)

    Ft_expected = T_in / (mesh.gear1_w.rl / 1000)
    Fr_expected = Ft_expected * np.tan(mesh.alphatw)
    Fa_expected = Ft_expected * np.tan(pinion.beta)  # beta = 0 for spur
    Fn_expected = Ft_expected / np.cos(mesh.alphatw)
    T_out_expected = T_in * mesh.u

    assert result["Ft"] == pytest.approx(Ft_expected)
    assert result["Fr"] == pytest.approx(Fr_expected)
    assert result["Fa"] == pytest.approx(Fa_expected)
    assert result["Fn"] == pytest.approx(Fn_expected)
    assert result["T_out"] == pytest.approx(T_out_expected)


def test_forces_fa_zero_for_spur_driver(spur_pair):
    """Spur external gear (beta=0) driving -> zero axial force."""
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear1")
    result = mesh.forces(T_in=50.0)
    assert result["Fa"] == pytest.approx(0.0, abs=1e-9)


def test_forces_fa_nonzero_for_helical_driver(helical_pair):
    """Helical external gear (beta!=0) driving -> nonzero axial force."""
    pinion, ring = helical_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear1")
    result = mesh.forces(T_in=50.0)
    assert result["Fa"] != pytest.approx(0.0, abs=1e-9)
    assert result["Fa"] == pytest.approx(result["Ft"] * np.tan(pinion.beta))


# ----------------------------------------------------------------------
# forces() -- driver = gear2 (ring is the input)
# ----------------------------------------------------------------------

def test_forces_driver_gear2_uses_gear2_radius_and_inverse_ratio(spur_pair):
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear2")

    T_in = 200.0  # N.m applied on the ring
    result = mesh.forces(T_in)

    Ft_expected = T_in / (mesh.gear2_w.rl / 1000)
    Fa_expected = Ft_expected * np.tan(ring.beta)
    T_out_expected = T_in * (1.0 / mesh.u)

    assert result["Ft"] == pytest.approx(Ft_expected)
    assert result["Fa"] == pytest.approx(Fa_expected)
    assert result["T_out"] == pytest.approx(T_out_expected)


def test_driver_gear1_vs_gear2_are_consistent_round_trip(spur_pair):
    """
    Driving gear1 with T1 and reading T_out (torque delivered to gear2)
    should be numerically consistent with driving gear2 with that same
    T_out and getting back (approximately) T1 as T_out, since
    u_eff(gear1) * u_eff(gear2) == 1.
    """
    pinion, ring = spur_pair
    mesh1 = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear1")
    mesh2 = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear2")

    T1 = 50.0
    T2 = mesh1.forces(T1)["T_out"]
    T1_roundtrip = mesh2.forces(T2)["T_out"]

    assert T1_roundtrip == pytest.approx(T1, rel=1e-9)


def test_ft_independent_of_driver_choice_definition(spur_pair):
    """
    Ft is defined relative to the *driving* gear's working radius, so
    Ft for a given T_in will differ between driver='gear1' and
    driver='gear2' unless T_in is scaled by the radius ratio. This test
    just documents/locks that relationship rather than asserting equality.
    """
    pinion, ring = spur_pair
    mesh1 = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear1")
    mesh2 = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear2")

    T_in = 50.0
    Ft1 = mesh1.forces(T_in)["Ft"]
    Ft2 = mesh2.forces(T_in)["Ft"]

    ratio = mesh1.gear2_w.rl / mesh1.gear1_w.rl
    assert Ft1 / Ft2 == pytest.approx(ratio, rel=1e-9)


# ----------------------------------------------------------------------
# Working geometry sanity (added on port)
# ----------------------------------------------------------------------

def test_gear_ratio_carries_the_sign_of_the_ring(spur_pair):
    """
    u = z2/z1 is NEGATIVE for an internal pair. The sign is meaningful: it
    encodes that driver and driven turn the same way, unlike an external
    pair. Magnitude is the plain tooth ratio.
    """
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring)
    assert mesh.u == pytest.approx(Z2 / Z1)
    assert mesh.u < 0.0
    assert abs(mesh.u) == pytest.approx(3.0)


def test_ring_diameter_is_positive_despite_the_negative_z(spur_pair):
    """InternalGear.d = mt*abs(z) -- the sign never reaches the diameter."""
    _, ring = spur_pair
    assert ring.d == pytest.approx(120.0)
    assert ring.db > 0.0


def test_reference_centre_distance_is_the_difference_not_the_sum(spur_pair):
    """
    The defining property of an internal pair: a = (d2 - d1)/2, not (d1+d2)/2.
    Works out because both diameters are positive magnitudes.
    """
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring)
    assert mesh.a == pytest.approx((ring.d - pinion.d) / 2.0)
    assert mesh.a == pytest.approx(40.0)
    assert mesh.a < (ring.d + pinion.d) / 2.0


def test_working_centre_distance_is_positive_and_near_the_reference(spur_pair):
    """
    The independent check: al must be confronted with a, computed from the
    nominal geometry, not with quantities derived from al itself. Every
    force assertion in this file compares against gear1_w.rl, which comes
    from al -- self-consistent, and blind to al being wrong.
    """
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring)
    assert mesh.al > 0.0
    assert mesh.al == pytest.approx(mesh.a, rel=0.15)


def test_driver_choice_does_not_change_the_geometry(spur_pair):
    """`driver` is a load-path choice; it must not touch working geometry."""
    pinion, ring = spur_pair
    m1 = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear1")
    m2 = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear2")

    assert m1.al == pytest.approx(m2.al)
    assert m1.alphatw == pytest.approx(m2.alphatw)
    assert m1.u == pytest.approx(m2.u)
    assert m1.epslon_alpha == pytest.approx(m2.epslon_alpha)


def test_reference_gears_are_not_mutated(spur_pair):
    """gear_geometry() works on copies -- gear1/gear2 must survive intact."""
    pinion, ring = spur_pair
    da1_before, da2_before = pinion.da, ring.da

    InternalGearMeshing(gear1=pinion, gear2=ring)

    assert pinion.da == pytest.approx(da1_before)
    assert ring.da == pytest.approx(da2_before)


# ----------------------------------------------------------------------
# Sanity: validate() runs and returns a list for both driver choices
# ----------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["gear1", "gear2"])
def test_validate_returns_list(spur_pair, driver):
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring, driver=driver)
    errors = mesh.validate()
    assert isinstance(errors, list)
    assert all(isinstance(e, str) for e in errors)


@pytest.mark.xfail(
    strict=True,
    reason="internal_gear.py's interference checks use (self.z - z1), which "
           "assumes a POSITIVE-z ring, while the meshing module's working "
           "geometry assumes negative z. With z2=-60, z1=20 the tooth "
           "difference evaluates to -80 instead of -40, so "
           "trimming_interference() fires on every internal pair. "
           "See tests/README.md, finding I.",
)
def test_trimming_interference_not_reported_for_a_40_tooth_difference(spur_pair):
    """|z2| - z1 = 40 teeth apart is nowhere near the 2-tooth trimming limit."""
    pinion, ring = spur_pair
    assert ring.trimming_interference(pinion.z) is False


@pytest.mark.xfail(
    strict=True,
    reason="Same root cause as the trimming check: validate_mesh() delegates "
           "to the positive-z interference formulae. See finding I.",
)
def test_a_healthy_pair_reports_no_mesh_errors(spur_pair):
    pinion, ring = spur_pair
    assert ring.validate_mesh(pinion.z, pinion.x) == []


def test_validate_or_raise_agrees_with_validate(spur_pair):
    """Added on port: the two must not disagree about whether the pair is ok."""
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring)

    if mesh.validate():
        with pytest.raises(ValueError):
            mesh.validate_or_raise()
    else:
        mesh.validate_or_raise()


# ----------------------------------------------------------------------
# Demonstracao manual (nao corre via pytest, so via `python este_ficheiro.py`)
# ----------------------------------------------------------------------

if __name__ == "__main__":

    def _print_forces(mesh, T_in, label):
        result = mesh.forces(T_in)
        print(f"  [{label}] driver={mesh.driver}, T_in={T_in} N.m")
        print(f"    Ft    = {result['Ft']:.3f} N")
        print(f"    Fr    = {result['Fr']:.3f} N")
        print(f"    Fa    = {result['Fa']:.3f} N")
        print(f"    Fn    = {result['Fn']:.3f} N")
        print(f"    T_out = {result['T_out']:.3f} N.m")

    print("=" * 70)
    print("EXEMPLO 1 -- Par spur (pinhao) + anel interno")
    print("=" * 70)

    # z=-80 on the ring, matching the original __main__ and the negative-z
    # convention the fixtures now use.
    pinion = SpurHelicalGear(mn=2.0, z=20, x=0.3, b=25, alpha_n_deg=20.0,
                             label="pinion")
    ring = InternalGear(mn=2.0, z=-80, x=0.0, b=25, alpha_n_deg=20.0,
                        label="ring")

    mesh_g1 = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear1",
                                  label="spur-ring (driver=gear1)")
    mesh_g2 = InternalGearMeshing(gear1=pinion, gear2=ring, driver="gear2",
                                  label="spur-ring (driver=gear2)")

    print(mesh_g1.summary())
    print()
    _print_forces(mesh_g1, T_in=50.0, label="spur, gear1 a acionar")
    _print_forces(mesh_g2, T_in=200.0, label="spur, gear2 (anel) a acionar")

    errors_g1 = mesh_g1.validate()
    print(f"\n  validate() [driver=gear1] -> {len(errors_g1)} erro(s):")
    for e in errors_g1:
        print(f"    - {e}")

    print("\n" + "=" * 70)
    print("EXEMPLO 2 -- Par helicoidal (pinhao) + anel interno")
    print("=" * 70)

    pinion_h = SpurHelicalGear(mn=2.0, z=20, x=0.3, b=25, alpha_n_deg=20.0,
                               beta_n_deg=15.0, label="pinion-helical")
    ring_h = InternalGear(mn=2.0, z=-80, x=0.0, b=25, alpha_n_deg=20.0,
                          beta_n_deg=15.0, label="ring-helical")

    mesh_h1 = InternalGearMeshing(gear1=pinion_h, gear2=ring_h, driver="gear1",
                                  label="helical-ring (driver=gear1)")
    mesh_h2 = InternalGearMeshing(gear1=pinion_h, gear2=ring_h, driver="gear2",
                                  label="helical-ring (driver=gear2)")

    print(mesh_h1.summary())
    print()
    _print_forces(mesh_h1, T_in=50.0, label="helical, gear1 a acionar (Fa != 0)")
    _print_forces(mesh_h2, T_in=200.0, label="helical, gear2 (anel) a acionar")

    errors_h1 = mesh_h1.validate()
    print(f"\n  validate() [driver=gear1] -> {len(errors_h1)} erro(s):")
    for e in errors_h1:
        print(f"    - {e}")

    print("\n" + "=" * 70)
    print("EXEMPLO 3 -- Round-trip de consistencia gear1 <-> gear2")
    print("=" * 70)

    T1 = 50.0
    T2 = mesh_g1.forces(T1)["T_out"]
    T1_roundtrip = mesh_g2.forces(T2)["T_out"]
    print(f"  T1 original      = {T1:.4f} N.m")
    print(f"  T2 (via gear1)   = {T2:.4f} N.m")
    print(f"  T1 round-trip    = {T1_roundtrip:.4f} N.m (deve ~= T1)")

    print("\n" + "=" * 70)
    print("repr():")
    print(mesh_g1)
    print(mesh_h1)