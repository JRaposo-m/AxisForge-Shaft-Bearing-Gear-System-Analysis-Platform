"""
tests/test_internal_gear_meshing.py

Covers the two things changed in InternalGearMeshing:
  1. The `driver` parameter ("gear1" / "gear2") — validation and default.
  2. forces() correctly switching input torque, working radius, helix
     angle and gear ratio depending on which gear is the driver.

Uses:
  - gear1 = external pinion (SpurGear for most cases, HelicalGear for the
    helix-angle-dependent axial force check)
  - gear2 = InternalGear (ring)
"""

import numpy as np
import pytest

from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.spur_gear import SpurGear
from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.helical_gear import HelicalGear
from axisforge.core.machine_elements.Gears.Parallel_Axis_gears.internal_gear import InternalGear
from axisforge.core.mechanical_system.gear_meshing.internal_gear_meshing import (
    InternalGearMeshing,
)



# ----------------------------------------------------------------------
# Fixtures — a geometrically reasonable spur/internal pair and a
# helical/internal pair (same tooth counts, mn, x — only beta differs).
# ----------------------------------------------------------------------

@pytest.fixture
def spur_pair():
    pinion = SpurGear(mn=2.0, z=20, x=0.3, b=25, alpha_n_deg=20.0, label="pinion")
    ring = InternalGear(mn=2.0, z=60, x=0.0, b=25, alpha_n_deg=20.0, label="ring")
    return pinion, ring


@pytest.fixture
def helical_pair():
    pinion = HelicalGear(
        mn=2.0, z=20, x=0.3, b=25, alpha_n_deg=20.0, beta_n_deg=15.0,
        label="pinion-helical",
    )
    ring = InternalGear(
        mn=2.0, z=60, x=0.0, b=25, alpha_n_deg=20.0, beta_n_deg=15.0,
        label="ring-helical",
    )
    return pinion, ring


# ----------------------------------------------------------------------
# driver parameter — validation & default
# ----------------------------------------------------------------------

def test_default_driver_is_gear1(spur_pair):
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring)
    assert mesh.driver == "gear1"


@pytest.mark.parametrize("driver", ["gear1", "gear2"])
def test_valid_driver_accepted(spur_pair, driver):
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring, driver=driver)
    assert mesh.driver == driver


@pytest.mark.parametrize("bad_driver", ["gear3", "GEAR1", "", None, 1])
def test_invalid_driver_raises(spur_pair, bad_driver):
    pinion, ring = spur_pair
    with pytest.raises(ValueError):
        InternalGearMeshing(gear1=pinion, gear2=ring, driver=bad_driver)


# ----------------------------------------------------------------------
# forces() — driver = gear1 (legacy behaviour)
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
# forces() — driver = gear2 (ring is the input)
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
# Sanity: validate() runs and returns a list for both driver choices
# ----------------------------------------------------------------------

@pytest.mark.parametrize("driver", ["gear1", "gear2"])
def test_validate_returns_list(spur_pair, driver):
    pinion, ring = spur_pair
    mesh = InternalGearMeshing(gear1=pinion, gear2=ring, driver=driver)
    errors = mesh.validate()
    assert isinstance(errors, list)
    assert all(isinstance(e, str) for e in errors)

 # ----------------------------------------------------------------------
# Demonstração manual (não corre via pytest, só via `python este_ficheiro.py`)
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
    print("EXEMPLO 1 — Par spur (pinhão) + anel interno")
    print("=" * 70)

    pinion = SpurGear(mn=2.0, z=20, x=0.3, b=25, alpha_n_deg=20.0, label="pinion")
    ring = InternalGear(mn=2.0, z=-80, x=0.0, b=25, alpha_n_deg=20.0, label="ring")

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
    print("EXEMPLO 2 — Par helicoidal (pinhão) + anel interno")
    print("=" * 70)

    pinion_h = HelicalGear(mn=2.0, z=20, x=0.3, b=25, alpha_n_deg=20.0,
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
    print("EXEMPLO 3 — Round-trip de consistência gear1 <-> gear2")
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