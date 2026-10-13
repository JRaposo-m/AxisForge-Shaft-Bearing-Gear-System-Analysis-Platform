"""Baseline of the gear code at commit 39c7133 (cases 1-7): the ONE place where the case objects are
built, re-recorded and checked.

Ported from the scripts that produced the baseline CSVs (vault note
``gears_mechsys_review_2026-10-12/baseline/_baseline.md``, scripts ``run_all.py`` and ``_record.py``):
same inputs, same order of calls, same row names. Only the columns that a re-recording cannot
reproduce are dropped (``unit``, ``source_file_line``, ``note``: text typed or looked up by the
recording script, not results of the gear code), and the GEARpie runner is left out.

A refactor edits the builders of this file (and, when attribute names change, a name map applied to
the recorded rows); the expected values in ``tests/data/gears_baseline/*.csv`` never change, except
for the rows that are ruled to change.

Since step 3 the gears and pairs are ``Gear`` and ``GearPair`` of ``CylindricalFamily``. The rows
were recorded on the classes that family replaces, so each object is recorded through the NAME MAP
below (``recorded_gear``, ``recorded_pair``): recorded row name -> accessor on the new objects, in
the recorded order. Every row is still compared; ``RENAMED_TEXT`` lists the few text rows that name
a removed class.

Units are those of AxisForge: lengths [mm], forces [N], torques [N m], angles [rad] in the gear
objects and [deg] for ``phi_deg`` and the load angles, speeds [rpm], power [W].
"""
from __future__ import annotations

import csv
import enum
import functools
import inspect
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from axisforge.core.loads import AxialLoad, RadialLoad, TorqueLoad
from axisforge.core.machine_elements.bearings import (
    Bearing,
    BearingCatalog,
    CylindricalRollerFamily,
    DeepGrooveBallFamily,
)
from axisforge.core.machine_elements.gears import cylindrical
from axisforge.core.machine_elements.gears.base import Gear, GearPair
from axisforge.core.machine_elements.gears.cylindrical import CylindricalFamily, CylindricalGearData
from axisforge.core.machine_elements.gears.parallel_axis.planetary_gear.planetary_gear_meshing import (
    PlanetaryGearTrainMeshing,
    PlanetaryMember,
)
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection, Shoulder
from axisforge.core.mechanical_system.gear_system import (
    SpurHelicalGearSystem,
    SpurHelicalMeshLink,
)
from axisforge.core.mechanical_system.shaft_system import (
    GearElement,
    ShaftSystem,
)

# <repo>/tests/data/gears_baseline: the case CSVs copied from the vault (question Q01)
DATA_DIR = Path(__file__).resolve().parents[5] / "tests" / "data" / "gears_baseline"

# QUESTION(Q02): equivalence tolerances. The rule of _baseline.md with its two amendments taken as
# defaults: relative 1e-12 on every float; absolute 1e-9 (in the unit of the quantity) whenever
# abs(baseline) < 1e-9, not only when the baseline is exactly 0 (several recorded values are
# rounding residue, e.g. k 8.9e-15, jbn 1.7e-14, coaxiality_error 4.3e-14); the brentq expression
# of alphatw is untouched, so 1e-12 also holds downstream of alphatw (the alternative of amendment
# 2 is 1e-10 there).
REL_TOL = 1e-12
ABS_TOL = 1e-9
ABS_TOL_BELOW = 1e-9

# ----------------------------------------------------------------------------------------------
# Inputs. GEARpie library gears: CLASSES/GEAR_LIBRARY.py (C14 line 52, H501 line 91); load stage
# CLASSES/LOAD_STAGES.py K4_05 (60.75 N m); speed chosen (GEARpie asks it on the console).
# ----------------------------------------------------------------------------------------------
T_NM = 60.75          # [N m] GEARpie FZG stage K4, 0.5 m arm; passed as the torque on gear 1
N_RPM = 1450.0        # [rpm] chosen
PHIS = (0.0, 90.0)    # [deg] line-of-centres angles of the force cases
ROTS = (1, -1)        # sense of rotation about +X
GEARPIE = {
    "C14": dict(alpha=20.0, beta=0.0, m=4.5, z=(16, 24), x=(0.1817, 0.1715), b=(14.0, 14.0), haP=1.0, hfP=1.25, rfP=0.38,
                Ra=(0.4, 0.31), Rq=(0.51, 0.4), Rz=(4.8, 4.8)),
    "H501": dict(alpha=20.0, beta=15.0, m=3.5, z=(20, 30), x=(0.1809, 0.0891), b=(23.0, 23.0), haP=1.0, hfP=1.25, rfP=0.38,
                 Ra=(0.6, 0.6), Rq=(0.7, 0.7), Rz=(4.8, 4.8)),
}
V3 = ("3a_gear1=pinion_gear2=wheel_driver=gear2", "3b_gear1=wheel_gear2=pinion_driver=gear1",
      "3c_gear1=wheel_gear2=pinion_driver=gear2")
# case 7: shaft and bearing inputs copied from AxisForge-Design-Studies
# validation/shafts/fem_stepped_distributed_gear/construction.py (SECTIONS :107-113, bearings :119-125)
SECTIONS = [(20.0, 20.0, "bearing_seat_A"), (65.0, 28.0, "body_left"), (30.0, 25.0, "gear_seat"),
            (65.0, 28.0, "body_right"), (20.0, 20.0, "bearing_seat_B")]   # (length, diameter) [mm]
PHI7 = 30.0           # [deg] line of centres a -> b, from +Y toward +Z
GEAR_POS = 100.0      # [mm] axial position of both gears from the left end of their shaft
RUNS7 = [("a_pinion_source_rot+1", "pinion", 1, False), ("b_pinion_source_rot-1", "pinion", -1, False),
         ("c_wheel_source_rot+1", "wheel", 1, False), ("d_wheel_source_rot-1", "wheel", -1, False),
         ("e_wheel_source_rot+1_ALT", "wheel", 1, True), ("f_wheel_source_rot-1_ALT", "wheel", -1, True)]
OMEGA = N_RPM * 2.0 * math.pi / 60.0   # [rad/s]

R: dict = {}        # objects and results kept for the physics checks (filled by the case functions)
CHECKS: list = []   # physics checks, in the order of the baseline checks.csv (291)
EXTRA_CHECKS: list = []   # checks added by the review of step 0, outside the recorded 291


# ----------------------------------------------------------------------------------------------
# Recorder: flattens objects and return values into rows (object, name, value)
# ----------------------------------------------------------------------------------------------
def fmt(v):
    """Text of a value as written in the baseline CSV (``repr`` of floats and strings, never rounded)."""
    if v is None:
        return "None"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return repr(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, enum.Enum):
        return repr(v.name)
    return repr(v)     # strings, complex numbers, lists


def is_scalar(v):
    return v is None or isinstance(v, (bool, int, float, str, complex, np.generic, enum.Enum))


class Recorder:
    """Rows ``(object, name, value)`` of one case; ``value`` is a float, or the CSV text of anything else."""

    def __init__(self, case):
        self.case = case
        self.rows = []

    def add(self, obj, name, value):
        # floats are kept as numbers (compared within the tolerances); everything else as its CSV text
        # (booleans, integers, strings, exception text: compared for equality)
        is_float = isinstance(value, (float, np.floating)) and not isinstance(value, (bool, np.bool_))
        self.rows.append((obj, name, float(value) if is_float else fmt(value)))

    def add_value(self, obj, name, value):
        """Flatten any value (scalar, sequence, dict, small ndarray, other object)."""
        if is_scalar(value):
            self.add(obj, name, value)
        elif isinstance(value, np.ndarray):
            for idx, x in np.ndenumerate(value):
                self.add(obj, f"{name}[{','.join(map(str, idx))}]", x)
        elif isinstance(value, dict):
            for k, v in value.items():
                self.add_value(obj, f"{name}.{k}", v)
        elif isinstance(value, (list, tuple)):
            self.add(obj, f"{name}.len", len(value))
            for i, v in enumerate(value):
                self.add_value(obj, f"{name}[{i}]", v)
        else:
            self.add(obj, name, f"<{type(value).__name__} object, recorded separately or not numeric>")

    def object_(self, objname, obj, skip=()):
        """Record every public instance attribute and property of obj (vars() + class properties)."""
        names = [n for n in vars(obj).keys() if not n.startswith("_")]
        for n in dir(type(obj)):
            if not n.startswith("_") and isinstance(inspect.getattr_static(type(obj), n), property) and n not in names:
                names.append(n)
        for n in names:
            if n in skip:
                continue
            try:
                v = getattr(obj, n)
            except Exception as e:
                self.add(objname, n, f"{type(e).__name__}: {e}")
                continue
            if callable(v) and not is_scalar(v):
                continue
            self.add_value(objname, n, v)

    def call(self, objname, obj, meth, args=(), label=None):
        """Call obj.meth(*args) and record the return value, or the exception type and message."""
        sig = ", ".join(fmt(a) if is_scalar(a) else type(a).__name__ for a in args)
        nm = label or f"{meth}({sig})"
        try:
            r = getattr(obj, meth)(*args)
        except Exception as e:
            self.add(objname, nm + "!exception", f"{type(e).__name__}: {e}")
            return None
        if isinstance(r, dict):
            for k, v in r.items():
                self.add_value(objname, f"{nm}.{k}", v)
        elif is_scalar(r) or isinstance(r, (list, tuple, np.ndarray)):
            self.add_value(objname, nm, r)
        elif hasattr(r, "__dataclass_fields__"):
            for k in r.__dataclass_fields__:
                self.add_value(objname, f"{nm}.{k}", getattr(r, k))
            for n in dir(type(r)):
                if not n.startswith("_") and isinstance(inspect.getattr_static(type(r), n), property):
                    self.add_value(objname, f"{nm}.{n}", getattr(r, n))
        else:
            self.add(objname, nm, f"<{type(r).__name__}>")
        return r


# ----------------------------------------------------------------------------------------------
# NAME MAP: recorded row name of the classes of commit 39c7133 -> accessor on Gear / GearPair
# ----------------------------------------------------------------------------------------------
FAMILY = CylindricalFamily()

_GEAR_INPUTS = ("label", "material_id", "position", "mn", "z", "Ca", "Cf", "x", "b", "alpha_n_deg", "alpha", "beta_n_deg",
                "beta", "haP", "cP", "hfP", "rfP", "Ra", "Rq", "Rz")
# recorded attribute -> attribute of Gear, in the recorded order (the order of assignment in the old
# constructors). The ring class had no Ca, Cf, betab, pb, rhoF, zn, de and named two quantities otherwise.
GEAR_ROWS = {
    "external": {n: n for n in (*_GEAR_INPUTS, "mt", "alphat", "betab", "rhoF", "pb", "pt", "pbt", "d", "r", "db", "rb",
                                "da", "ra", "df", "rf", "zn", "de")},
    "internal": {**{n: n for n in _GEAR_INPUTS if n not in ("Ca", "Cf")},
                 **{n: n for n in ("mt", "mx", "pn", "pt", "pz", "px", "d", "r")}, "beta_b": "betab",
                 **{n: n for n in ("db", "rb", "alphat", "tau", "pbt")}, "pbn": "pb",
                 **{n: n for n in ("dv", "rv", "da", "ra", "df", "rf")}},
}
# rows added to the working copies ``gear1_w`` / ``gear2_w`` (i = 1, 2) after the gear rows: the working
# data, all now stored on the pair (recorded name -> attribute of GearPair)
WORKING_ROWS = {"k": "k", "al": "al", "alphatw": "alphatw", "dl": "dl{i}", "rl": "rl{i}", "alpha_a": "alpha_a{i}",
                "epslon_a{i}": "epslon_a{i}", "galpha": "galpha{i}"}
# pair rows from ``u`` on, same names and order for both kinds and for every constructor option
_PAIR_TAIL = ("u", "k", "epslon_alpha", "b", "epslon_beta", "epslon_gamma", "galpha", "ReqI", "T1T2", "T1E", "T2A", "T1A",
              "T1B", "T1C", "T1D", "T2E", "T2B", "T2C", "T2D", "AE", "AB", "AC", "AD", "rA1", "rB1", "rD1", "rA2", "rB2",
              "rD2", "HV", "Ram", "Rrms", "RzS")

# QUESTION(Q50): text rows of the baseline that name a class removed in step 3 and that the new code
# cannot print: baseline value -> (value given now, number of such rows in cases 1-7). Every other
# presentation row (__repr__ 'HelicalGearMeshing(', 'InternalGearMeshing(', summary()) is reproduced
# verbatim by the family. The alternative is to rule new texts and replace the rows.
RENAMED_TEXT = {
    "\"AttributeError: 'InternalGear' object has no attribute 'zn'\"":
        ("\"AttributeError: 'Gear' object has no attribute 'zn'\"", 6),
}


# QUESTION(Q03) / QUESTION(Q50): the pair stores no driver (D-0026 item 4), so the recorded rows `driver`
# and `driven` of the 29 recorded pairs are an echo of the case input (they pass by construction):
# (row names, number of rows in cases 1-7). What the driver CHANGES is compared in the forces() rows.
ECHOED_ROWS = (("driver", "driven"), 58)


class Recorded:
    """One object of the old classes as the baseline recorded it, rebuilt from the new objects.

    ``rows`` become the public attributes, in the recorded order (``Recorder.object_`` reads
    ``vars()``); ``calls`` are the recorded methods, by name (``Recorder.call``); ``text`` gives the
    recorded ``__repr__``.
    """

    def __init__(self, rows, text=None, **calls):
        vars(self).update(rows)
        vars(self).update(calls)    # callables are skipped by Recorder.object_
        self._text = text

    def __repr__(self):
        return self._text() if self._text else object.__repr__(self)


def recorded_gear(g):
    """``Gear`` under the names of ``SpurHelicalGear`` (z > 0) or ``InternalGear`` (z < 0).

    The checks that were methods of the old gear classes are now ``CylindricalFamily.undercutting``,
    ``CylindricalFamily.values_at`` and, for the interference tests of a ring that are under
    question Q24, private functions of ``cylindrical`` (entries of its ``_LEGACY`` table).
    """
    kind = cylindrical._kind(g.z)
    rows = {name: getattr(g, attr) for name, attr in GEAR_ROWS[kind].items()}
    calls = dict(validate=g.validate, validate_or_raise=g.validate_or_raise)
    if kind == "external":
        calls.update(undercutting=lambda: FAMILY.undercutting(g))
    else:
        calls.update(tip_below_base_circle=lambda: g.da <= g.db,     # inlined in the ring limits of validate()
                     involute_interference=lambda z1, x1: cylindrical._involute_interference(g, z1, x1),
                     trochoid_interference=lambda z1, x1: cylindrical._trochoid_interference(g, z1, x1),
                     trimming_interference=lambda z1: cylindrical._trimming_interference(g, z1),
                     validate_mesh=lambda z1, x1: cylindrical._validate_mesh(g, z1, x1),
                     values_at=lambda dy: FAMILY.values_at(g, dy))
    return Recorded(rows, **calls)


def _recorded_working_gear(pair, i):
    """Rows of the shallow copy ``gear{i}_w`` of the old pair classes: the gear with its working tip
    diameter (now ``pair.da{i}_w``; the radius was da/2) followed by the working data of the pair."""
    g = getattr(pair, f"gear{i}")
    rows = {name: getattr(g, attr) for name, attr in GEAR_ROWS[cylindrical._kind(g.z)].items()}
    rows["da"] = getattr(pair, f"da{i}_w")
    rows["ra"] = rows["da"] / 2.0
    rows.update((name.format(i=i), getattr(pair, attr.format(i=i))) for name, attr in WORKING_ROWS.items())
    return Recorded(rows)


def recorded_pair(pair, driver="gear1"):
    """``GearPair`` under the names of ``SpurHelicalGearMeshing`` or ``InternalGearMeshing``.

    ``driver`` is the driver the case declares. The pair no longer stores it (it is an argument of
    ``CylindricalFamily.forces``), so the recorded rows ``driver`` / ``driven`` are the declared
    value itself and ``forces`` passes it on. The order of the first rows is that of the old
    constructors: with an imposed ``al`` the centre distance comes before the working pressure
    angle, and an internal pair then has the backlash ``jbn``. ``new`` is the ``GearPair``.
    """
    head = ("al", "alphatw", "jbn", "x1", "x2", "inv_alphatw") if pair.al_imposed else ("inv_alphatw", "alphatw", "al", "x1", "x2")
    rows = dict(driver=driver, driven="gear2" if driver == "gear1" else "gear1", label=pair.label, a=pair.a)
    rows.update((n, getattr(pair, n)) for n in (*head, *_PAIR_TAIL) if not (n == "jbn" and pair.jbn is None))
    # the consistency check of validate(): recorded as x_sum_* (external) or x_diff_* (internal)
    what = "diff" if cylindrical._kind(pair.gear2.z) == "internal" else "sum"
    _, rows[f"x_{what}_from_al"], rows[f"x_{what}_declared"] = pair.profile_shift_check
    return Recorded(
        dict(rows, gear1=pair.gear1, gear2=pair.gear2, gear1_w=_recorded_working_gear(pair, 1),
             gear2_w=_recorded_working_gear(pair, 2), new=pair),
        text=lambda: repr(pair), validate=pair.validate, validate_or_raise=pair.validate_or_raise, summary=pair.summary,
        forces=lambda T, phi_deg=0.0, rotation_dir=1: FAMILY.forces(pair, T, phi_deg, rotation_dir, driver=driver),
        correction_for_gs_equilibrium=lambda: FAMILY.correction_for_gs_equilibrium(pair.gear1, pair.gear2))


def assemble_pair(g1, g2, driver="gear1", **kw):
    """Pair of the cylindrical family for two gears, recorded with the declared ``driver``."""
    return recorded_pair(GearPair.assemble(FAMILY, gear1=g1, gear2=g2, **kw), driver)


# ----------------------------------------------------------------------------------------------
# builders of the case objects
# ----------------------------------------------------------------------------------------------
def gear(mn, z, **kw):
    """Assembled gear of the cylindrical family (z < 0: internal gear)."""
    return Gear.assemble(FAMILY, CylindricalGearData(mn, z, **kw))


def ext_gear(name, i, label, position=0.0):
    g = GEARPIE[name]
    # GEARpie gives the dedendum hfP; AxisForge takes the clearance cP = hfP - haP [multiples of mn]
    return gear(mn=g["m"], z=g["z"][i], x=g["x"][i], b=g["b"][i], alpha_n_deg=g["alpha"], beta_n_deg=g["beta"],
                haP=g["haP"], cP=g["hfP"] - g["haP"], rfP=g["rfP"], Ra=g["Ra"][i], Rq=g["Rq"][i], Rz=g["Rz"][i],
                label=label, position=position)


def pinion_wheel(name, position=0.0):
    return ext_gear(name, 0, f"{name}_pinion", position), ext_gear(name, 1, f"{name}_wheel", position)


def make_internal(beta, mn=2.0, b=20.0):
    pin = gear(mn=mn, z=20, x=0.0, b=b, alpha_n_deg=20.0, beta_n_deg=beta, label="pinion")
    # internal gear: negative number of teeth (sign convention of the family)
    ring = gear(mn=mn, z=-60, x=0.0, b=b, alpha_n_deg=20.0, beta_n_deg=beta, label="ring")
    return pin, ring


def _shaft(name):
    sh = Shaft(label=name)
    for L, d, lab in SECTIONS:
        sh.add_section(ShaftSection(length=L, diameter=d, label=f"{name}_{lab}"))
    for i in range(len(SECTIONS) - 1):
        dl, dr = SECTIONS[i][1], SECTIONS[i + 1][1]
        sh.set_transition(i, Shoulder(1.0, max(dl, dr), min(dl, dr)))   # fillet radius 1 mm
    return sh


def _shaft_system(name, ball_arrangement="locating"):
    ss = ShaftSystem(_shaft(name), name=name)
    ball = Bearing.assemble(family=DeepGrooveBallFamily(),
                            catalog=BearingCatalog(d=20.0, D=42.0, b=12.0, designation="6204", position=10.0,
                                                   arrangement=ball_arrangement, label=f"{name}_ball_locating"),
                            geometry=dict(Dw=7.0, Dpw=31.0, Z=9, s=0.02))
    roller = Bearing.assemble(family=CylindricalRollerFamily(),
                              catalog=BearingCatalog(d=20.0, D=47.0, b=14.0, designation="NU204", position=190.0,
                                                     arrangement="non-locating", label=f"{name}_roller_nonlocating"),
                              geometry=dict(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=40, i=1))
    ss.add_bearing(ball)
    ss.add_bearing(roller)
    return ss


class Case7(NamedTuple):
    """Objects of one case-7 run: two shafts joined by the H501 pair.

    ``P`` is the power [W] and ``rot`` the sense of rotation of the source shaft about +X (+1 or
    -1) passed to ``resolve``; ``phi`` the angle of the line of centres a -> b [deg], from +Y toward
    +Z. ``sP``, ``sW`` are the pinion and wheel ``ShaftSystem`` (after ``resolve_case7``: the loaded
    copies returned by ``resolve``; the shafts given to the system stay in ``system.shafts``),
    ``geP``, ``geW`` their ``GearElement``, ``p``, ``w`` the gears. ``mesh`` is the pair as recorded
    (``recorded_pair``); the link holds the ``GearPair`` itself (``mesh.new``).
    """
    system: SpurHelicalGearSystem
    mesh: Recorded
    link: SpurHelicalMeshLink
    sP: ShaftSystem
    sW: ShaftSystem
    p: Gear
    w: Gear
    geP: GearElement
    geW: GearElement
    P: float
    rot: int
    phi: float

    @property
    def driver_shaft(self):
        """Loaded shaft that receives the power (shaft a of the link)."""
        return (self.sP, self.sW)[self.system.shafts.index(self.link.shaft_a)]

    @property
    def driven_shaft(self):
        """Loaded shaft on the other side of the mesh (shaft b of the link)."""
        return (self.sP, self.sW)[self.system.shafts.index(self.link.shaft_b)]

    @property
    def T_driven_Nm(self):
        """Torque that ``resolve`` records for the driven shaft [N m]."""
        return self.system._resolved[id(self.link.shaft_b)]["T_out_Nm"]

    @property
    def rot_driven(self):
        """Sense of rotation that ``resolve`` records for the driven shaft, +1 or -1 [-]."""
        return self.system._resolved[id(self.link.shaft_b)]["rotation_dir"]


def build_case7(source, rot, alt=False, label=None, distribute_loads=False, gear_pos=GEAR_POS,
                ball_arrangement="locating"):
    """Two shafts of case 7 joined by the H501 pair, NOT yet resolved.

    Parameters
    ----------
    source : str
        ``"pinion"`` or ``"wheel"``: the shaft that receives the power ``T_NM * OMEGA`` [W].
    rot : int
        Sense of rotation of the source shaft about +X, +1 or -1 [-].
    alt : bool
        Wheel source only. False: pair built as ``(wheel, pinion)``, declared driver gear 1. True: built
        as ``(pinion, wheel)``, declared driver gear 2, with ``gear_a`` = wheel: the external pair still
        takes the torque on its gear 1, the pinion (cases 7e, 7f; question Q03).
    label : str or None
        Label of the system.
    distribute_loads : bool
        Forwarded to ``SpurHelicalMeshLink`` (False in every baseline case).
    gear_pos : float
        Axial position of the centre of both gears on their shafts [mm] (100 in every baseline case).
    ball_arrangement : str
        Arrangement of the ball bearing of both shafts ("locating" in every baseline case: it takes
        the axial thrust of the helical mesh).

    Returns
    -------
    Case7
    """
    p, w = pinion_wheel("H501", position=gear_pos)
    sP, sW = _shaft_system("shaft_pinion", ball_arrangement), _shaft_system("shaft_wheel", ball_arrangement)
    geP = GearElement(p, role="driver" if source == "pinion" else "driven", label="g_pinion")
    geW = GearElement(w, role="driver" if source == "wheel" else "driven", label="g_wheel")
    sP.add_gear(geP)
    sW.add_gear(geW)
    kw = dict(phi_deg=PHI7, label="stage", distribute_loads=distribute_loads)
    if source == "pinion":
        mesh = assemble_pair(p, w, label="stage", driver="gear1")
        link = SpurHelicalMeshLink(sP, geP, sW, geW, mesh.new, **kw)
    elif not alt:
        mesh = assemble_pair(w, p, label="stage", driver="gear1")
        link = SpurHelicalMeshLink(sW, geW, sP, geP, mesh.new, **kw)
    else:
        mesh = assemble_pair(p, w, label="stage", driver="gear2")
        link = SpurHelicalMeshLink(sW, geW, sP, geP, mesh.new, **kw)
    system = SpurHelicalGearSystem([sP, sW], [link], label=label)
    return Case7(system, mesh, link, sP, sW, p, w, geP, geW, T_NM * OMEGA, rot, PHI7)


def resolve_case7(c, sinks=()):
    """The ONE call of ``SpurHelicalGearSystem.resolve``: returns the ``Case7`` whose ``sP``, ``sW``
    are the loaded copies (``system.shafts`` is ``[sP, sW]``, and the copies come in that order).
    ``sinks``: ``(shaft of system.shafts, x [mm])`` pairs, forwarded (none in the baseline cases)."""
    sP, sW = c.system.resolve(c.P, N_RPM, c.rot, sinks=sinks)
    return c._replace(sP=sP, sW=sW)


def resolved_case7(source, rot, alt=False, distribute_loads=False):
    """``build_case7`` followed by ``resolve_case7``."""
    return resolve_case7(build_case7(source, rot, alt, distribute_loads=distribute_loads))


# ----------------------------------------------------------------------------------------------
# recording of the pair cases (1-5)
# ----------------------------------------------------------------------------------------------
def record_gear(rec, name, g):
    g = recorded_gear(g)
    rec.object_(name, g)
    for m in ("undercutting", "validate", "validate_or_raise"):
        rec.call(name, g, m)


def record_forces(rec, name, mesh, T, key):
    for phi in PHIS:
        for rot in ROTS:
            d = rec.call(name, mesh, "forces", (T, phi, rot))
            if d is not None:
                R.setdefault("forces", {})[(key, phi, rot)] = d


def record_pair(rec, name, mesh, T, key, forces=True):
    rec.object_(name, mesh, skip=("gear1", "gear2", "gear1_w", "gear2_w", "new"))
    for w in ("gear1_w", "gear2_w"):
        rec.object_(f"{name}.{w}", getattr(mesh, w))
    rec.add(name, "gear1.label", mesh.gear1.label)
    rec.add(name, "gear2.label", mesh.gear2.label)
    rec.call(name, mesh, "validate")
    rec.call(name, mesh, "validate_or_raise")
    rec.call(name, mesh, "summary")
    rec.add(name, "__repr__()", repr(mesh))
    if forces:
        record_forces(rec, name, mesh, T, key)


def run_external(case, gname):
    rec = Recorder(case)
    p, w = pinion_wheel(gname)
    rec.add("inputs", "GEARpie library gear", gname)
    rec.add("inputs", "T1 torque on gear1 / speed", f"{T_NM!r} N m / {N_RPM!r} rpm")
    record_gear(rec, "gear1(pinion)", p)
    record_gear(rec, "gear2(wheel)", w)
    mesh = assemble_pair(p, w, label=case, driver="gear1")
    record_pair(rec, "mesh", mesh, T_NM, (case, "base"))
    R[(case, "mesh")] = mesh
    rec.call("mesh", mesh, "correction_for_gs_equilibrium")
    # variants (all assembly options of the pair)
    for vname, kw in (("mesh_addendum_reduction", dict(addendum_reduction=True)),
                      ("mesh_equalise_gs", dict(equalise_gs=True)),
                      ("mesh_al_imposed_equal_a", dict(al=mesh.a))):
        try:
            mv = assemble_pair(p, w, label=vname, driver="gear1", **kw)
        except Exception as e:
            rec.add(vname, "constructor!exception", f"{type(e).__name__}: {e}")
            continue
        record_pair(rec, vname, mv, T_NM, (case, vname), forces=False)
        rec.call(vname, mv, "forces", (T_NM, 0.0, 1))
        R[(case, vname)] = mv
    if case.startswith("case1"):
        # extra pair for the z1 + z2 > 60 branch of correction_for_gs_equilibrium (the GEARpie pair has 40)
        g = GEARPIE[gname]
        pe = gear(mn=g["m"], z=35, b=g["b"][0], label="extra_z35")
        we = gear(mn=g["m"], z=45, b=g["b"][1], label="extra_z45")
        rec.add("inputs", "extra pair for equalise_gs branch z1+z2 > 60", "mn=4.5, z=35/45, x=0, b=14, default rack")
        try:
            me = assemble_pair(pe, we, label="extra_z80", equalise_gs=True)
            record_pair(rec, "mesh_extra_z35_z45_equalise_gs", me, T_NM, (case, "extra80"), forces=False)
            R[(case, "extra80")] = me
        except Exception as e:
            rec.add("mesh_extra_z35_z45_equalise_gs", "constructor!exception", f"{type(e).__name__}: {e}")
    return rec


def run_case3():
    """Same pair as case 2, wheel as driver. Records every way the API lets the driver be expressed."""
    case = "case3_helical_wheel_driver"
    rec = Recorder(case)
    p, w = pinion_wheel("H501")
    rec.add("inputs", "intent", "wheel delivers the input torque T_wheel = 60.75 N m")
    variants = {V3[0]: (p, w, "gear2"), V3[1]: (w, p, "gear1"), V3[2]: (w, p, "gear2")}
    for vname, (g1, g2, drv) in variants.items():
        mesh = assemble_pair(g1, g2, label=vname, driver=drv)
        record_pair(rec, vname, mesh, T_NM, (case, vname))
        R[(case, vname)] = mesh
    return rec


def record_internal_gear(rec, name, ring, z1, x1):
    ring = recorded_gear(ring)
    rec.object_(name, ring)
    for m in ("tip_below_base_circle", "validate", "validate_or_raise"):
        rec.call(name, ring, m)
    rec.call(name, ring, "involute_interference", (z1, x1))
    rec.call(name, ring, "trochoid_interference", (z1, x1))
    rec.call(name, ring, "trimming_interference", (z1,))
    rec.call(name, ring, "validate_mesh", (z1, x1))
    for dy in (ring.d, 0.5 * (ring.db + ring.d)):   # [mm] reference diameter and a diameter above the base circle
        rec.call(name, ring, "values_at", (dy,))


def run_internal(case, beta):
    rec = Recorder(case)
    pin, ring = make_internal(beta)
    rec.add("inputs", "chosen values", f"mn=2, z1=20, z2=-60, x=0, b=20, alpha_n=20 deg, beta={beta} deg, default basic rack")
    record_gear(rec, "gear1(pinion)", pin)
    record_internal_gear(rec, "gear2(ring)", ring, pin.z, pin.x)
    for drv in ("gear1", "gear2"):
        mesh = assemble_pair(pin, ring, label=case, driver=drv)
        record_pair(rec, f"mesh_driver_{drv}", mesh, T_NM, (case, drv))
        R[(case, drv)] = mesh
    base = R[(case, "gear1")]
    rec.call("mesh_driver_gear1", base, "correction_for_gs_equilibrium")
    for vname, kw in (("mesh_addendum_reduction", dict(addendum_reduction=True)),
                      ("mesh_equalise_gs", dict(equalise_gs=True)),
                      ("mesh_al_imposed_equal_al", dict(al=base.al)),
                      ("mesh_al_imposed_equal_a", dict(al=base.a))):
        try:
            mv = assemble_pair(pin, ring, label=vname, driver="gear1", **kw)
        except Exception as e:
            rec.add(vname, "constructor!exception", f"{type(e).__name__}: {e}")
            continue
        record_pair(rec, vname, mv, T_NM, (case, vname), forces=False)
        rec.call(vname, mv, "forces", (T_NM, 0.0, 1))
        R[(case, vname)] = mv
    return rec


# ----------------------------------------------------------------------------------------------
# planetary train (cases 6, 6b)
# ----------------------------------------------------------------------------------------------
def run_planetary(case, beta):
    rec = Recorder(case)
    mn, b = 2.0, 20.0
    sun = gear(mn=mn, z=20, x=0.0, b=b, beta_n_deg=beta, label="sun")
    planet = gear(mn=mn, z=20, x=0.0, b=b, beta_n_deg=beta, label="planet")
    ring = gear(mn=mn, z=-60, x=0.0, b=b, beta_n_deg=beta, label="ring")
    rec.add("inputs", "chosen values", f"mn=2, sun z=20, planet z=20, ring z=-60, k=4 (k=3 violates the assembly condition, see variant k3), x=0, b=20, alpha_n=20 deg, beta={beta} deg, default rack, phi0=0")
    record_gear(rec, "sun", sun)
    record_gear(rec, "planet", planet)
    record_internal_gear(rec, "ring", ring, planet.z, planet.x)
    pg = PlanetaryGearTrainMeshing(sun, planet, ring, k=4, label=case)
    R[(case, "train")] = pg
    rec.object_("planetary", pg, skip=("sun_gear", "planet_gear", "ring_gear", "meshing_12", "meshing_23"))
    m12, m23 = recorded_pair(pg.meshing_12), recorded_pair(pg.meshing_23)   # the train takes the torque on gear 1 of each
    record_pair(rec, "planetary.meshing_12", m12, T_NM, (case, "m12"), forces=False)
    record_pair(rec, "planetary.meshing_23", m23, T_NM, (case, "m23"), forces=False)
    M = PlanetaryMember
    for j in range(pg.k):
        rec.call("planetary", pg, "planet_phi_deg", (j,))
        rec.call("planetary", pg, "planet_position", (j,))
    rec.call("planetary", pg, "willis_residual", (N_RPM, 0.0, 0.0))
    # prescribed speeds [rpm]; the last two are the documented error paths
    for desc, pres in (("RING=0,SUN=n", {M.RING: 0.0, M.SUN: N_RPM}), ("SUN=0,RING=n", {M.SUN: 0.0, M.RING: N_RPM}),
                       ("CARRIER=0,SUN=n", {M.CARRIER: 0.0, M.SUN: N_RPM}), ("SUN=n,CARRIER=100", {M.SUN: N_RPM, M.CARRIER: 100.0}),
                       ("SUN=n,RING=-300", {M.SUN: N_RPM, M.RING: -300.0}), ("SUN=n only (expected error)", {M.SUN: N_RPM}),
                       ("PLANET=n (expected error)", {M.PLANET: N_RPM, M.SUN: 0.0})):
        r = rec.call("planetary", pg, "solve_kinematics", (pres,), label=f"solve_kinematics({{{desc}}})")
        if r is not None:
            R[(case, "kin", desc)] = r
    perms = [(M.SUN, M.CARRIER, M.RING), (M.CARRIER, M.SUN, M.RING), (M.RING, M.CARRIER, M.SUN), (M.CARRIER, M.RING, M.SUN),
             (M.SUN, M.RING, M.CARRIER), (M.RING, M.SUN, M.CARRIER), (M.SUN, M.PLANET, M.RING)]
    for A, B, C in perms:
        lab = f"({A.name},{B.name},{C.name})"
        v = rec.call("planetary", pg, "speed_ratio", (A, B, C), label=f"speed_ratio{lab}")
        rec.call("planetary", pg, "speed_ratio_equation", (A, B, C), label=f"speed_ratio_equation{lab}")
        rec.call("planetary", pg, "ratio", (A, B, C), label=f"ratio{lab}")
        if v is not None:
            rec.call("planetary", pg, "mode_of_work", (v,), label=f"mode_of_work(speed_ratio{lab})")
        rec.call("planetary", pg, "is_pseudo_pgt", (C,), label=f"is_pseudo_pgt({C.name})")
    for mem in (M.SUN, M.RING, M.CARRIER, M.PLANET):
        tq = rec.call("planetary", pg, "torques", (T_NM, mem), label=f"torques(T_in={T_NM!r}, {mem.name})")
        if tq is not None:
            R[(case, "torques", mem.name)] = tq
            rec.call("planetary", tq, "satisfies_torque_ordering", label=f"torques({mem.name}).satisfies_torque_ordering()")
    tq = R[(case, "torques", "SUN")]
    # docstring of torques(): T_mesh_sun / T_mesh_planet are the torques to pass to the forces() of each pair
    rec.call("planetary.meshing_12", m12, "forces", (tq.T_mesh_sun, 0.0, 1), label="forces(T_mesh_sun, phi=0, rot=+1)")
    rec.call("planetary.meshing_23", m23, "forces", (tq.T_mesh_planet, 0.0, 1), label="forces(T_mesh_planet, phi=0, rot=+1)")
    for m in ("coaxiality_error", "assembly_condition", "neighbouring_condition", "validate", "validate_or_raise", "summary"):
        rec.call("planetary", pg, m)
    rec.call("planetary", pg, "neighbouring_condition", (2.0,))
    rec.add("planetary", "t (property)", pg.t)
    rec.add("planetary", "__repr__()", repr(pg))
    for vname, kw in (("equalise_gs_23", dict(equalise_gs_23=True)), ("equalise_gs_12", dict(equalise_gs_12=True)),
                      ("addendum_reduction_12_and_23", dict(addendum_reduction_12=True, addendum_reduction_23=True)),
                      ("k3_assembly_violation", dict(k=3))):
        try:
            pv = PlanetaryGearTrainMeshing(sun, planet, ring, **{"k": 4, "label": vname, **kw})
            rec.add(vname, "constructor ok: al_12, al_23", f"{float(pv.al_12)!r}, {float(pv.al_23)!r}")
            rec.call(vname, pv, "validate")
        except Exception as e:
            rec.add(vname, "constructor!exception", f"{type(e).__name__}: {e}")
    return rec


# ----------------------------------------------------------------------------------------------
# case 7: SpurHelicalGearSystem.resolve on two shafts
# ----------------------------------------------------------------------------------------------
def load_row(rec, obj, i, ld):
    p = f"loads[{i}]"
    rec.add(obj, f"{p}.type", type(ld).__name__)
    rec.add(obj, f"{p}.label", ld.label)
    rec.add(obj, f"{p}.source", ld.source)
    rec.add(obj, f"{p}.position", ld.position)     # [mm] axial coordinate from the left end of the shaft
    rec.add(obj, f"{p}.magnitude", ld.magnitude)   # [N] or [N m], signed as stored
    if hasattr(ld, "theta_deg"):
        rec.add(obj, f"{p}.theta_deg", ld.theta_deg)   # [deg] from +Y toward +Z, right-hand about +X
        rec.add(obj, f"{p}.Fy", ld.Fy)
        rec.add(obj, f"{p}.Fz", ld.Fz)


def run_case7(tag, source, rot, alt=False):
    case = f"case7{tag}"
    rec = Recorder(case)
    c = build_case7(source, rot, alt, label=case)
    rec.add("inputs", "resolve args", f"P={c.P!r} W, rpm={N_RPM!r}, rotation_dir_source={rot}, source_position=(0.0, 0.0), source_torque_x=0.0")
    rec.call("system", c.system, "validate", label="validate() BEFORE resolve")
    c = R[(case, "system")] = resolve_case7(c)
    system, mesh, link, sP, sW, geP, geW = c.system, c.mesh, c.link, c.sP, c.sW, c.geP, c.geW
    rec.call("system", system, "validate", label="validate() AFTER resolve")
    rec.call("system", system, "summary")
    rec.add("system", "__repr__()", repr(system))
    rec.add("system", "label", system.label)
    for s in system.shafts:
        for k, v in system._resolved[id(s)].items():   # private record kept by resolve
            rec.add_value("system", f"_resolved[{s.name}].{k}", v)
    # `driver` of the link was added in step 3 (it was an attribute of the pair): recorded in the rows of the pair
    rec.object_("link", link, skip=("shaft_a", "shaft_b", "gear_a", "gear_b", "meshing", "driver"))
    rec.add("link", "shaft_a.name", link.shaft_a.name)
    rec.add("link", "shaft_b.name", link.shaft_b.name)
    rec.call("link", link, "validate")
    rec.add("link", "__repr__()", repr(link))
    record_pair(rec, "mesh", mesh, T_NM, (case, "mesh"))
    for nm, s, ge in (("shaft_pinion", sP, geP), ("shaft_wheel", sW, geW)):
        rec.object_(nm, s, skip=("shaft",))
        rec.add(nm, "shaft.total_length", s.shaft.total_length)
        rec.call(nm, s, "validate")
        rec.call(nm, s, "summary")
        rec.add(nm, "__repr__()", repr(s))
        rec.add(nm, "support_positions", list(s.support_positions))
        cnt = (len(s.radial_loads), len(s.axial_loads), len(s.torque_loads), len(s.external_moments), len(s.distributed_radial_loads))
        rec.add(nm, "n_radial/axial/torque/external_moment/distributed", str(cnt))
        for i, ld in enumerate(s.loads):
            load_row(rec, nm, i, ld)
        rec.object_(f"{nm}.gear_element", ge, skip=("gear",))
        rec.call(f"{nm}.gear_element", ge, "validate")
        rec.call(nm, s, "gear_extent", (ge,), label="gear_extent(gear_element)")
        for b in s.bearings:
            rec.call(nm, s, "bearing_extent", (b,), label=f"bearing_extent({b.label})")
        rec.add(nm, "snapshot() keys", ",".join(s.snapshot().keys()))
        rec.call(nm, s, "validate_or_raise")
    return rec


# ----------------------------------------------------------------------------------------------
# physics checks that need no standard. Frame for all vectors: the global YZ frame of the loads
# (theta from +Y toward +Z, right-hand about +X). Both shaft axes are parallel to X, so one frame
# serves both shafts and no rotation is needed; the positions of the mesh points come from phi (line
# of centres a -> b). Source: _baseline.md, "Physics checks without a standard" (no library source:
# statics and kinematics only).
# ----------------------------------------------------------------------------------------------
class Check(NamedTuple):
    """One physics check: ``result`` is "PASS", "FAIL" or "INFO" (a value recorded, no verdict)."""
    id: str
    case: str
    desc: str
    got: float
    expected: float | str
    result: str
    note: str


def dirv(theta_deg):
    """Unit vector (Y, Z) at the angle theta [deg] measured from +Y toward +Z."""
    t = math.radians(theta_deg)
    return np.array([math.cos(t), math.sin(t)])


def check(cid, case, desc, got, expected, scale=None, tol=1e-9, note="", extra=False):
    """Append a check: PASS when ``abs(got - expected) <= tol * scale`` (relative residual 1e-9).
    ``extra``: a check added by the review of step 0, kept outside the recorded 291."""
    got = float(got)
    expected = float(expected)
    sc = scale if scale is not None else max(abs(got), abs(expected), 1e-300)
    resid = abs(got - expected)
    ok = resid <= tol * sc if sc > 0 else resid == 0.0
    (EXTRA_CHECKS if extra else CHECKS).append(Check(cid, case, desc, got, expected, "PASS" if ok else "FAIL", note))


def force_vectors(F):
    """Force vectors (Y, Z) [N] on the driver-side and the driven-side gear, from a ``forces()`` dict."""
    Ft, Fr = F["Ft"], F["Fr"]
    Fr_a = Fr * dirv(F["theta_Fr_driver"])
    Ft_a = Ft * dirv(F["theta_Ft_driver"])
    Fr_b = Fr * dirv(F["theta_Fr_driven"])
    Ft_b = Ft * dirv(F["theta_Ft_driven"])
    return Fr_a, Ft_a, Fr_b, Ft_b


def physics_pair(cid_prefix, key, mesh, internal, driver):
    """Checks a, b, d of one pair, for every (phi, rot) of the force cases.

    a: power balance of an ideal mesh, ``abs(T_out) n_out = T_in n_in`` with ``n_out = n_in z_in /
    z_out`` [W]. b: action = reaction, the four transverse force vectors of the two gears sum to zero
    [N]. d: the radial force separates the gears: on an external gear it points from the mesh toward
    its own centre; on a ring it points outward from the ring centre (flank forces are compressive).
    For an internal pair the mesh point lies at ``-phi_hat`` from both centres when the pinion is on
    shaft a (``+phi_hat`` when the ring is): phi is READ HERE as the line of centres driver ->
    driven, whereas the docstring of ``InternalGearMeshing.forces`` says gear1 -> gear2; the two
    readings differ when the ring drives (question Q20). No library source (statics, kinematics).

    Check a takes T on the gear named by the argument ``driver``, which the callers set to "gear1"
    for every external pair, because ``SpurHelicalGearMeshing.forces`` ignores the declared driver
    (case 3, question Q03).

    Check e (internal pairs only, added by the review of step 0, outside the recorded 291): the
    tangential force on the driver gear, taken at the mesh point of the same reading, gives the
    moment ``-rot T_in`` about the axis [N m] (the mesh takes out the torque that the source puts
    in with the sense of rotation).
    """
    case = f"{key[0]}/{key[1]}"
    z1, z2 = abs(mesh.gear1.z), abs(mesh.gear2.z)
    zin, zout = (z1, z2) if driver == "gear1" else (z2, z1)
    for phi in PHIS:
        for rot in ROTS:
            F = R["forces"][(key, phi, rot)]
            tag = f"phi={phi:g},rot={rot:+d}"
            # a. power balance, ideal mesh, speeds from the tooth numbers
            n_out = N_RPM * zin / zout
            P_in = T_NM * N_RPM * 2 * math.pi / 60
            P_out = abs(F["T_out"]) * n_out * 2 * math.pi / 60
            if (phi, rot) == (0.0, 1):
                ignored = "" if mesh.driver == driver else f"; T taken on gear 1, declared driver ({mesh.driver}) ignored"
                check(f"{cid_prefix}.a", case, f"power balance |T_in n_in| vs |T_out n_out|, n_out=n_in*z_in/z_out [W]{ignored}", P_out, P_in, scale=P_in,
                      note=f"T_in={T_NM} N m at {N_RPM} rpm; T_out={F['T_out']!r}; z_in={zin}, z_out={zout}")
            # b. action = reaction
            Fr_a, Ft_a, Fr_b, Ft_b = force_vectors(F)
            tot = Fr_a + Ft_a + Fr_b + Ft_b
            check(f"{cid_prefix}.b", case, f"action=reaction: |F_a + F_b| / Ft, both gears, {tag} [-]", np.linalg.norm(tot) / F["Ft"], 0.0,
                  scale=1.0, note=f"radial pair sum {float(np.linalg.norm(Fr_a + Fr_b))!r}, tangential pair sum {float(np.linalg.norm(Ft_a + Ft_b))!r}")
            # d. radial separating, mesh direction from own centre
            ph = dirv(phi)
            if not internal:
                md_a, md_b, sg_a, sg_b = ph, -ph, -1, -1
            else:
                pin_is_a = (driver == "gear1")
                md_a = -ph if pin_is_a else ph
                md_b = md_a
                sg_a = -1 if pin_is_a else +1          # sign of the expected separating force along mesh_dir: pinion -1 (toward centre), ring +1 (outward)
                sg_b = +1 if pin_is_a else -1
                tag += ", phi read as driver -> driven"
                r_drv = (mesh.gear1_w if pin_is_a else mesh.gear2_w).rl      # [mm] working pitch radius of the driver
                Mx = mx_of(Ft_a, r_drv * md_a) / 1000.0                      # N mm -> N m
                check(f"{cid_prefix}.e", case, f"driver gear: Mx of Ft at the mesh point = -rot*T_in [N m], {tag}", Mx, -rot * T_NM, scale=T_NM, extra=True)
            for nm, dv, md, sg in (("driver-side gear", dirv(F["theta_Fr_driver"]), md_a, sg_a), ("driven-side gear", dirv(F["theta_Fr_driven"]), md_b, sg_b)):
                dot = float(np.dot(dv, md)) * sg
                check(f"{cid_prefix}.d", case, f"radial force separating, {nm}, {tag}: dot(F_r_dir, expected separating dir) = +1 [-]", dot, 1.0, scale=1.0,
                      note="expected dir: external gear -> away from the mesh (toward own centre); ring -> outward from its centre")


def physics_planetary(case):
    """Check a for the train: the external torques of sun, ring and carrier sum to zero [N m], and so
    does ``sum T_i omega_i`` [W] (ideal, no losses), for three input members and five speed pairs."""
    for mem in ("SUN", "RING", "CARRIER"):
        tq = R[(case, "torques", mem)]
        check("6.a1", case, f"sum of external torques T1+T3+TH = 0, input at {mem} [N m]", tq.T1 + tq.T3 + tq.TH, 0.0, scale=max(abs(tq.T1), abs(tq.T3), abs(tq.TH)), note="")
        for desc in ("RING=0,SUN=n", "SUN=0,RING=n", "CARRIER=0,SUN=n", "SUN=n,CARRIER=100", "SUN=n,RING=-300"):
            kin = R[(case, "kin", desc)]
            P = (tq.T1 * kin.omega_1 + tq.T3 * kin.omega_3 + tq.TH * kin.omega_H) * 2 * math.pi / 60.0   # speeds in rpm -> rad/s
            ref = max(abs(tq.T1 * kin.omega_1), abs(tq.T3 * kin.omega_3), abs(tq.TH * kin.omega_H), 1e-300) * 2 * math.pi / 60.0
            check("6.a2", case, f"power balance sum T_i*omega_i = 0 [W], input torque at {mem}, speeds {desc}", P, 0.0, scale=ref, note="")


def shaft_sums(s):
    """Sums of the stored gear-mesh loads of one ShaftSystem.

    Returns
    -------
    dict
        ``F`` lateral force (Y, Z) [N]; ``Fa`` axial force [N]; ``Tnet`` sum of the TorqueLoads
        [N m]; ``Ft_vec``, ``Fr_vec`` the loads labelled ``:Ft`` and ``:Fr`` [N]; ``T_gear`` the
        TorqueLoad labelled ``:T`` [N m].
    """
    Fy = Fz = Fa = Tn = 0.0
    Ft_vec = np.zeros(2)
    Fr_vec = np.zeros(2)
    Tg = None
    for ld in s.loads:
        if isinstance(ld, RadialLoad):
            v = np.array([ld.Fy, ld.Fz])
            Fy += ld.Fy
            Fz += ld.Fz
            if ld.label.endswith(":Ft"):
                Ft_vec += v
            if ld.label.endswith(":Fr"):
                Fr_vec += v
        elif isinstance(ld, AxialLoad):
            Fa += ld.magnitude
        elif isinstance(ld, TorqueLoad):
            Tn += ld.magnitude
            if ld.label.endswith(":T"):
                Tg = ld.magnitude
    return dict(F=np.array([Fy, Fz]), Fa=Fa, Tnet=Tn, Ft_vec=Ft_vec, Fr_vec=Fr_vec, T_gear=Tg)


def mx_of(vec, r_pos):
    """Moment about +X [N mm] of the force vec (Fy, Fz) [N] applied at r_pos (y, z) [mm] from the shaft axis."""
    return r_pos[0] * vec[1] - r_pos[1] * vec[0]


def physics_system(case):
    """Checks 7.b-7.d on the loads that ``resolve`` stored on the two shafts of one case-7 run.

    The mesh point of gear a lies at ``shaft_a + r phi_hat`` and of gear b at ``shaft_b - r phi_hat``
    (phi = line of centres a -> b). Sign convention of 7.c1 / 7.c2: a shaft turning with sense
    ``rot`` receives from the source a torque ``+rot T`` about X, so the tooth force on the driver
    gear must give a moment ``-rot T_in`` about the axis, and on the driven gear ``+rot_b T_out`` with
    ``T_out = T_in z_b / z_a``. Each moment is taken with the radius the code uses (``gear1_w.rl``,
    ``gear2_w.rl``) and with the geometric one ``al z / (z1 + z2)`` [mm]. No library source.

    Returns
    -------
    dict
        ``A``, ``B``: ``shaft_sums`` of the driver and the driven shaft of the link.
    """
    c = R[(case, "system")]
    mesh, link, P, rot, phi = c.mesh, c.link, c.P, c.rot, c.phi
    sa, sb = c.driver_shaft, c.driven_shaft
    ga, gb = link.gear_a.gear, link.gear_b.gear
    rot_a = rot
    rot_b = c.rot_driven
    T_src = P / OMEGA
    ph = dirv(phi)
    A, B = shaft_sums(sa), shaft_sums(sb)
    # radii: the ones the code used (mesh.gear1_w.rl is the SMALLER working radius) and the geometric ones from the tooth numbers
    r_code_a, r_code_b = mesh.gear1_w.rl, mesh.gear2_w.rl
    zsum = ga.z + gb.z
    r_geo_a, r_geo_b = mesh.al * ga.z / zsum, mesh.al * gb.z / zsum
    # b. action = reaction on the stored loads
    F = mesh.forces(T_src, phi, rot)
    tot = A["F"] + B["F"]
    check("7.b", case, "action=reaction on stored loads: |sum F_a + sum F_b| / Ft [-]", np.linalg.norm(tot) / F["Ft"], 0.0, scale=1.0)
    check("7.b2", case, "action=reaction on tangential loads only: |Ft_a + Ft_b| / Ft [-]", np.linalg.norm(A["Ft_vec"] + B["Ft_vec"]) / F["Ft"], 0.0, scale=1.0)
    check("7.b3", case, "action=reaction on radial loads only: |Fr_a + Fr_b| / Fr [-]", np.linalg.norm(A["Fr_vec"] + B["Fr_vec"]) / F["Fr"], 0.0, scale=1.0)
    check("7.b4", case, "axial loads: sum Fa_a + sum Fa_b = 0 [N]", A["Fa"] + B["Fa"], 0.0, scale=max(abs(A["Fa"]), 1.0), note=f"Fa_a={float(A['Fa'])!r}, Fa_b={float(B['Fa'])!r}")
    # c. torque direction: moment about the shaft axis of the stored Ft at the mesh point
    for lab, r_a, r_b in (("code radii (gear1_w.rl, gear2_w.rl)", r_code_a, r_code_b), ("geometric radii (al*z/(z1+z2))", r_geo_a, r_geo_b)):
        Ma = mx_of(A["Ft_vec"], r_a * ph) / 1000.0          # N mm -> N m
        Mb = mx_of(B["Ft_vec"], -r_b * ph) / 1000.0
        Tin = float(T_src)
        Tout_expected = Tin * gb.z / ga.z
        check("7.c1", case, f"driver shaft: Mx of stored Ft at the mesh point = -rot_a*T_in [N m], {lab}", Ma, -rot_a * Tin, scale=Tin,
              note=f"rot_a={rot_a}; T_in={Tin!r}; stored TorqueLoad at gear = {A['T_gear']!r} (power-flow sign, independent of rot)")
        check("7.c2", case, f"driven shaft: Mx of stored Ft at the mesh point = +rot_b*T_out_physical [N m], {lab}", Mb, rot_b * Tout_expected, scale=Tout_expected,
              note=f"rot_b={rot_b}; T_out_physical=T_in*z_b/z_a={Tout_expected!r}; stored TorqueLoad at gear = {B['T_gear']!r}")
        check("7.c3", case, "sign of Mx(Ft) equals sign of stored TorqueLoad, driver shaft [-]", math.copysign(1, Ma) if Ma else 0.0, math.copysign(1, A["T_gear"]),
              scale=1.0, note=f"{lab}")
        check("7.c4", case, "sign of Mx(Ft) equals sign of stored TorqueLoad, driven shaft [-]", math.copysign(1, Mb) if Mb else 0.0, math.copysign(1, B["T_gear"]),
              scale=1.0, note=f"{lab}")
        # statics: a force at the mesh point equals the same force on the axis plus its moment, so
        # the TorqueLoad of a gear IS Mx(Ft), in value and not only in sign (units_and_conventions.md, C.7)
        check("7.c12", case, f"driver shaft: Mx of stored Ft at the mesh point = stored TorqueLoad [N m], {lab}", Ma, A["T_gear"], scale=abs(A["T_gear"]), extra=True)
        check("7.c13", case, f"driven shaft: Mx of stored Ft at the mesh point = stored TorqueLoad [N m], {lab}", Mb, B["T_gear"], scale=abs(B["T_gear"]), extra=True)
    # torque along the shaft
    check("7.c5", case, "driver shaft: sum of stored TorqueLoads = 0 [N m]", A["Tnet"], 0.0, scale=T_src)
    CHECKS.append(Check("7.c6", case, "driven shaft: sum of stored TorqueLoads [N m] (open end; closed only by a user load)", B["Tnet"], 0.0, "INFO", ""))
    Tstore_b = c.T_driven_Nm
    check("7.c7", case, "driven-shaft torque T_out (resolved) = T_in*z_b/z_a [N m]", abs(Tstore_b), T_src * gb.z / ga.z, scale=T_src * gb.z / ga.z,
          note=f"_resolved T_out_Nm={Tstore_b!r}; z_a={ga.z}, z_b={gb.z}")
    nb_phys = N_RPM * ga.z / gb.z
    check("7.c8", case, "driven-shaft power |T_out|*n_b = P_in [W], n_b=n*z_a/z_b", abs(Tstore_b) * nb_phys * 2 * math.pi / 60.0, P, scale=P, note=f"n_b={nb_phys!r} rpm")
    check("7.c9", case, "driven shaft speed_rpm stored by resolve = n*z_a/z_b [rpm]", sb.speed_rpm, nb_phys, scale=nb_phys, note=f"ShaftSystem.speed_rpm={sb.speed_rpm!r}; source shaft speed_rpm={sa.speed_rpm!r} (given {N_RPM})")
    check("7.c14", case, "source shaft speed_rpm stored by resolve = n [rpm]", sa.speed_rpm, N_RPM, scale=N_RPM, extra=True)
    check("7.c10", case, "rotation sense of driven shaft = -rot_source (external pair) [-]", rot_b, -rot_a, scale=1.0)
    CHECKS.append(Check("7.c11", case, "tilting moment |Fa|*r absent from the loads [N mm] (required by Harris sec. 4.3 per the code docstring)",
                        float(abs(A["Fa"]) * r_geo_a), "stored: none", "INFO", f"Fa={float(A['Fa'])!r} N, r_geo_a={float(r_geo_a)!r} mm"))
    # d. on the stored loads
    for nm, S, md in (("driver shaft", A, ph), ("driven shaft", B, -ph)):
        v = S["Fr_vec"]
        dot = float(np.dot(v / np.linalg.norm(v), md))
        check("7.d", case, f"stored radial force on {nm}: dot(F_r dir, mesh dir from own centre) = -1 (separating) [-]", dot, -1.0, scale=1.0)
    return dict(A=A, B=B)


def physics_rotation_reversal(sysres):
    """Checks 7.x: reversing the sense of rotation of the source (same gears, so the same helix hand)
    reverses Ft and the axial thrust and leaves Fr unchanged. No library source (symmetry)."""
    for (ta, tb) in (("a_pinion_source_rot+1", "b_pinion_source_rot-1"), ("c_wheel_source_rot+1", "d_wheel_source_rot-1")):
        a, b = sysres[ta], sysres[tb]
        for sh in ("A", "B"):
            check("7.x1", f"{ta} vs {tb}", f"{sh}: Ft vector reverses when the sense of rotation reverses (|Ft(+) + Ft(-)|/|Ft|) [-]",
                  np.linalg.norm(a[sh]["Ft_vec"] + b[sh]["Ft_vec"]) / np.linalg.norm(a[sh]["Ft_vec"]), 0.0, scale=1.0)
            check("7.x2", f"{ta} vs {tb}", f"{sh}: Fr vector unchanged by the sense of rotation (|Fr(+) - Fr(-)|/|Fr|) [-]",
                  np.linalg.norm(a[sh]["Fr_vec"] - b[sh]["Fr_vec"]) / np.linalg.norm(a[sh]["Fr_vec"]), 0.0, scale=1.0)
            check("7.x3", f"{ta} vs {tb}", f"{sh}: axial thrust reverses with the sense of rotation for a fixed helix hand, Fa(+) + Fa(-) = 0 [N]",
                  a[sh]["Fa"] + b[sh]["Fa"], 0.0, scale=max(abs(a[sh]["Fa"]), 1.0), note=f"Fa(+)={float(a[sh]['Fa'])!r}, Fa(-)={float(b[sh]['Fa'])!r}; the code has no hand of helix")


# ----------------------------------------------------------------------------------------------
# one run of everything, and the comparison with the CSVs
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def run_baseline():
    """Build and record cases 1-7 once, then run the physics checks.

    Returns
    -------
    tuple
        ``(rows, checks, extra_checks)``: ``rows[case]`` is the list of recorded ``(object, name,
        value)`` in the order of the baseline CSV of that case; ``checks`` is the list of ``Check``
        in the order of the baseline ``checks.csv`` (291 checks); ``extra_checks`` those added by
        the review of step 0.
    """
    R.clear()
    CHECKS.clear()
    EXTRA_CHECKS.clear()
    recs = [run_external("case1_external_spur", "C14"), run_external("case2_external_helical", "H501"), run_case3(),
            run_internal("case4_internal_spur", 0.0), run_internal("case5_internal_helical", 15.0),
            run_planetary("case6_planetary_spur", 0.0), run_planetary("case6b_planetary_helical", 15.0)]
    recs += [run_case7(tag, src, rot, alt) for tag, src, rot, alt in RUNS7]
    physics_pair("1", ("case1_external_spur", "base"), R[("case1_external_spur", "mesh")], False, "gear1")
    physics_pair("2", ("case2_external_helical", "base"), R[("case2_external_helical", "mesh")], False, "gear1")
    for vname in V3:
        physics_pair("3", ("case3_helical_wheel_driver", vname), R[("case3_helical_wheel_driver", vname)], False, "gear1")
    for case, cid in (("case4_internal_spur", "4"), ("case5_internal_helical", "5")):
        for drv in ("gear1", "gear2"):
            physics_pair(cid, (case, drv), R[(case, drv)], True, drv)
    physics_planetary("case6_planetary_spur")
    physics_planetary("case6b_planetary_helical")
    physics_rotation_reversal({tag: physics_system(f"case7{tag}") for tag, _, _, _ in RUNS7})
    return {r.case: r.rows for r in recs}, tuple(CHECKS), tuple(EXTRA_CHECKS)


@functools.lru_cache(maxsize=None)
def baseline_rows(case):
    """Rows ``(object, name, value text)`` of ``tests/data/gears_baseline/<case>.csv``, in file order."""
    with open(DATA_DIR / f"{case}.csv", newline="", encoding="utf-8") as f:
        return [(r["object"], r["name"], r["value"]) for r in csv.DictReader(f)]


def baseline_objects(cases):
    """``(case, object)`` pairs of the given baseline CSVs, in file order (one golden test each)."""
    return [(case, obj) for case in cases for obj in dict.fromkeys(o for o, _, _ in baseline_rows(case))]


def equivalent(got, text):
    """Equivalence rule of the baseline for one value (see ``REL_TOL``).

    Parameters
    ----------
    got : float or str
        Recorded value: a float, or the CSV text of a boolean, integer, string or exception.
    text : str
        ``value`` cell of the baseline CSV.
    """
    if not isinstance(got, float):
        # QUESTION(Q02): text is compared exactly, including the numbers printed inside strings
        # (summary(), __repr__(), validate() messages, "constructor ok: al_12, al_23"); the
        # alternative is to compare the numbers found inside a string under the tolerances.
        if text in RENAMED_TEXT:
            return got == RENAMED_TEXT[text][0]     # QUESTION(Q50): the row names a removed class
        return got == text
    try:
        ref = float(text)
    except ValueError:
        return False
    if math.isinf(ref) or math.isnan(ref):
        return repr(got) == text     # inf (lead of a spur gear) and nan have no tolerance
    if abs(ref) < ABS_TOL_BELOW:
        return abs(got - ref) <= ABS_TOL
    return abs(got - ref) <= REL_TOL * abs(ref)


def mismatches(case, obj):
    """Rows of one object of one case that are not reproduced: ``(csv line, name, got, baseline)``.

    The comparison is by position: row i of the re-recording against row i of the CSV, which also
    pins the number and the order of the rows (and so the number of ``validate()`` messages).
    """
    got_rows = run_baseline()[0][case]
    bad = []
    for i, (o, name, text) in enumerate(baseline_rows(case)):
        if o != obj:
            continue
        got = got_rows[i] if i < len(got_rows) else (None, None, "<row missing>")
        if got[:2] != (o, name) or not equivalent(got[2], text):
            bad.append((i + 2, name, got, text))     # +2: header line, 1-based
    return bad


def check_params(ids, question):
    """Parameters of a physics-check test: one per ``Check`` whose id starts with one of ``ids``.

    ``question(check)`` returns the question id and review row of a check that FAILS today (it is
    then marked ``xfail(strict=True, raises=AssertionError)``, so that a change of today's behaviour,
    or an error other than the failed check, is reported), or None for a check that passes today.
    INFO checks carry no verdict and are left out.
    """
    params = []
    _, checks, extra = run_baseline()
    for i, c in enumerate(checks + extra):
        if c.id[0] not in ids or c.result == "INFO":
            continue
        q = question(c)
        params.append(pytest.param(c, id=f"{i:03d}-{c.id}-{c.case}",
                                   marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=q) if q else ()))
    return params
