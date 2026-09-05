"""
fixtures/studies/shafts/fem_studies/outputs/plots.py

CONTENT ONLY, single-solve Resolution figures for one shaft's
ShaftResults -- sibling of resolution_report.py in this same
fem_studies/outputs/ package, same domain (Resolution/shaft_fem), same
result SHAPE (ShaftResults), same "content only" discipline: every
function here BUILDS and RETURNS a matplotlib Figure, none of them call
savefig()/show()/close() themselves. write_resolution_plots() at the
bottom of this module is the one function that touches the filesystem
-- mirrors write_comparison_report()'s own writer role in
comparison_report.py, and text_report.py's own writer role for the text
side.

REVISION (component-pair + resultant/phase layout, replacing the
original "one figure per xz/xy/resultant" layout): for the three
quantities that have both an xz and an xy component -- bending moment
(M_xz/M_xy), shear (V_xz/V_xy), deflection (v_xz/v_xy) -- this module
now produces exactly TWO figures per quantity instead of three:

  1. <quantity>_components.png -- xz and xy plotted TOGETHER on one
     axis, both black, xz as a solid line and xy as a dashed line (a
     professional monochrome convention -- color was carrying no
     information here since xz/xy are just the two orthogonal planes,
     not two different physical quantities).
  2. <quantity>_resultant.png -- two stacked subplots sharing the x
     axis: the resultant magnitude on top (black, no governing-value
     marker -- the red-dot/crosshair convention this module used to
     draw here has been dropped for these three; still used by
     bending_stress_figure/shear_stress_figure was ALSO dropped, see
     below), and the PHASE ANGLE of the (xz, xy) vector underneath,
     via atan2(xy, xz) in degrees -- atan2, not a plain atan(xz/xy)
     ratio, specifically so that a pure-xz vector (xy = 0) reads as
     0 deg exactly, a pure-xy vector as +-90 deg, and the sign/quadrant
     stays correct everywhere without a division-by-zero at xy = 0.
     The phase axis is always drawn -180..180 deg (fixed y-limits,
     ticks every 45 deg: -180/-135/-90/-45/0/45/90/135/180, each with
     its own light dashed gridline so a reader can place a wiggle at a
     glance without eyeballing between only five marks), not
     autoscaled, so the SAME angle on two different plots (or two
     different shafts) always sits at the same height -- an autoscaled
     phase axis would be actively misleading (e.g. a shaft that only
     ever swings +-5 deg would get
     zoomed in enough to look as volatile as one that swings +-170 deg).

Single-component quantities (torsional shear stress tau, torque T)
keep the single-line-per-figure shape -- there is nothing to pair them
with, and torque has no xz/xy split at all (see the twist/axial note
below). Both also lost their red governing-value marker/crosshair/
annotation in this revision, for visual consistency with the black,
unmarked resultant plots -- black instead of tab:red, no marker.

bending_stress_figure() (sigma_b) is DIFFERENT from tau/T: sigma_b is
computed from the RESULTANT bending moment M (sigma_b = M / W, W being
isotropic for a circular section), so its own "direction" -- the phase
of the underlying (M_xz, M_xy) vector -- is mathematically identical to
bending_moment_resultant_figure()'s own phase curve; it is not a
separate quantity with its own phase to compute. So bending_stress_
figure() now uses the SAME two-subplot resultant+phase layout as
bending moment/shear/deflection: sigma_b(x) on top, and underneath the
identical M-phase curve (computed from result.M_xz/result.M_xy, not
from sigma_b itself, since sigma_b alone -- a scalar magnitude -- has
no phase of its own to recover). tau keeps a plain single-line figure:
it derives from torque T, which has no xz/xy pair in this model (a
single scalar per node, rotation about the shaft's own axis), so there
is no vector to take a phase of.

Two per-node geometry/section figures that used to live here --
diameter_profile_figure() (d(x)) and the two section-modulus figures
(section_modulus_bending_figure() for W(x), section_modulus_torsion_
figure() for Wt(x)) -- have been REMOVED entirely, on explicit
feedback that they are not of interest as deliverable figures. d, W,
and Wt remain in ShaftResults and in resolution_report.py's own
section_stress_table() (text report), only the standalone PNGs for
them are gone.

NOT INCLUDED, and deliberately not fabricated: a torsional deformation
(twist angle, phi) figure and an axial deformation (u along the shaft)
figure were asked for, but ShaftResults exposes neither as a per-node
array -- only T (torque, N.m) is stored per node for torsion, and u
(axial displacement) exists ONLY per bearing, on BearingNodeData
(bearing_nodes[i].u), not as a length-n array over the whole shaft. Two
values at two bearing locations is not a meaningful x-vs-u line plot.
Adding either would require RigidBearingFEMSolver/ShaftResultsReader to
compute and store a genuine per-node twist-angle and axial-displacement
array first (neither is fabricated here) -- flagged for the user to
decide whether that belongs in the solver/results layer before this
module gains those two figures.

Governing maxima ARE still marked where they always were and were not
touched by this revision: ShaftResults itself stores M_max/x_M_max,
v_max/x_v_max (used to be marked on the old single resultant figures,
now dropped along with the marker convention for M/V/v resultants --
see above) -- as of this revision, _line_figure()'s own `mark`
parameter is unused by every figure function in this module (kept on
_line_figure() itself only because a future single-component quantity
might still want it), since bending_stress_figure/shear_stress_figure
lost theirs too. If a marked single-line figure is wanted again later,
pass `mark=` back in at the call site -- the helper still supports it.

bearing_reactions_figure() -- the one non-per-node figure this module
used to have -- has been REMOVED entirely in this revision, on explicit
feedback that a bar chart of bearing_positions/R_xz/R_xy/R_axial ("são
só números") adds nothing over resolution_report.py's own
bearing_reactions_table() (still there, in the text report -- this was
only ever a redundant visual of the same table). No replacement figure
is planned for it.

Folder layout write_resolution_plots() creates, under a caller-supplied
base_dir (e.g. a check script's own HERE) -- 9 files per shaft now,
down from the original 16:

    <base_dir>/plots/<shaft_name>/bending_moment_components.png
    <base_dir>/plots/<shaft_name>/bending_moment_resultant.png
    <base_dir>/plots/<shaft_name>/shear_components.png
    <base_dir>/plots/<shaft_name>/shear_resultant.png
    <base_dir>/plots/<shaft_name>/deflection_components.png
    <base_dir>/plots/<shaft_name>/deflection_resultant.png
    <base_dir>/plots/<shaft_name>/torsion.png
    <base_dir>/plots/<shaft_name>/bending_stress.png
    <base_dir>/plots/<shaft_name>/shear_stress.png

One subfolder per shaft ("subdivided by shaft system", per the
request this module was originally built against) rather than one flat
folder with shaft name baked into every filename.

_FIGURE_REGISTRY at the bottom is still the single source of truth
mapping filename -> figure-building function -- write_resolution_
plots() and write_shaft_figures() both iterate it, so adding another
figure means adding one function plus one registry entry, not editing
the writer's own loop body.

No file I/O anywhere except write_resolution_plots()/
write_shaft_figures() -- every _figure() function is pure (ShaftResults
in, Figure out), matching resolution_report.py's own "no encoding, no
open()" discipline for its _table()-building functions.

Dependency (results/ and fixtures/ only, read-only access -- no core
modification):
  axisforge.results.fem_results.shaft_results
      ShaftResults -- the result SHAPE, same as resolution_report.py's
      own dependency, same TYPE_CHECKING-only reasoning (this module
      only ever reads a ShaftResults handed to it, never builds one).
  axisforge.fixtures.studies.shafts.fem_studies.results_library
      RigidBearingFEMResultsLibrary -- the registry write_resolution_
      plots() walks, same role it plays for
      fixtures/studies/text_report.py's own write_studies_report().
  axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system
      SpurHelicalGearSystem -- supplies shaft order/names, same
      reasoning as write_comparison_report()'s own `system` parameter.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.results.fem_results.shaft_results import ShaftResults
    from axisforge.fixtures.studies.shafts.fem_studies.results_library import (
        RigidBearingFEMResultsLibrary,
    )
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )

__all__ = [
    "bending_moment_components_figure", "bending_moment_resultant_figure",
    "shear_components_figure", "shear_resultant_figure",
    "deflection_components_figure", "deflection_resultant_figure",
    "torsion_figure",
    "bending_stress_figure", "shear_stress_figure",
    "write_shaft_figures",
    "write_resolution_plots",
]


# ---------------------------------------------------------------------------
# Shared figure builders.
# ---------------------------------------------------------------------------

def _line_figure(
    x,
    y,
    *,
    title: str,
    ylabel: str,
    xlabel: str = "x [mm]",
    color: str = "black",
    mark: tuple[float, float, str] | None = None,
) -> Figure:
    """
    One line, one axis, one figure. `mark`, if given, is
    (x_governing, y_governing, quantity_label) and draws a red marker
    plus a light dotted crosshair and an annotation -- kept as an
    option on this helper for a future single-component figure, but no
    call site in this module passes it any more (bending_stress_figure/
    shear_stress_figure dropped their marker in this revision, see this
    module's own top docstring).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, "-", color=color, linewidth=1.6)

    if mark is not None:
        x_m, y_m, label = mark
        ax.plot([x_m], [y_m], "o", color="tab:red", markersize=7, zorder=5)
        ax.axhline(y_m, color="tab:red", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(x_m, color="tab:red", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"{label} = {y_m:.4g}\n@ x = {x_m:.2f} mm",
            xy=(x_m, y_m), xytext=(10, 10), textcoords="offset points",
            fontsize=8, color="tab:red",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _components_figure(
    x,
    y_xz,
    y_xy,
    *,
    title: str,
    ylabel: str,
    xlabel: str = "x [mm]",
) -> Figure:
    """
    xz and xy plotted together on one axis, both black -- xz solid,
    xy dashed -- the "professional monochrome" pairing this module's
    top docstring describes. Used for bending moment, shear, and
    deflection; nothing else in ShaftResults has an xz/xy pair.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y_xz, "-", color="black", linewidth=1.6, label="xz")
    ax.plot(x, y_xy, "--", color="black", linewidth=1.6, label="xy")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _resultant_and_phase_figure(
    x,
    y_xz,
    y_xy,
    y_resultant,
    *,
    title: str,
    ylabel_resultant: str,
    xlabel: str = "x [mm]",
) -> Figure:
    """
    Two subplots sharing the x axis: resultant magnitude on top (black,
    no marker), phase angle underneath -- phase = degrees(atan2(y_xy,
    y_xz)), so a pure-xz vector (y_xy = 0) reads as 0 deg exactly (the
    convention explicitly requested), a pure-xy vector as +-90 deg, and
    atan2 (not a plain xz/xy ratio through atan()) keeps the sign and
    quadrant correct everywhere, including at y_xz = 0, without a
    division-by-zero. The phase axis is FIXED at -180..180 deg (not
    autoscaled) with ticks and a light dashed gridline every 45 deg
    (-180/-135/-90/-45/0/45/90/135/180) -- finer than a plain 5-tick
    axis, so a reader can place a wiggle in the curve against a nearby
    reference line instead of eyeballing between only 0/+-90/+-180 --
    so the same angle always sits at the same height across
    figures/shafts -- see this module's own top docstring for why that
    matters.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.3]},
    )

    ax_top.plot(x, y_resultant, "-", color="black", linewidth=1.6)
    ax_top.set_title(title)
    ax_top.set_ylabel(ylabel_resultant)
    ax_top.grid(True, alpha=0.3)

    phase_deg = np.degrees(np.arctan2(np.asarray(y_xy), np.asarray(y_xz)))
    phase_ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    ax_bot.plot(x, phase_deg, "-", color="black", linewidth=1.4, zorder=3)
    for t in phase_ticks:
        ax_bot.axhline(
            t, color="gray", linewidth=1.0 if t == 0 else 0.6,
            linestyle="-" if t == 0 else "--", alpha=0.7 if t == 0 else 0.4,
            zorder=1,
        )
    ax_bot.set_ylim(-180.0, 180.0)
    ax_bot.set_yticks(phase_ticks)
    ax_bot.set_ylabel("phase [deg]  (0 deg = xz plane)")
    ax_bot.set_xlabel(xlabel)
    ax_bot.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    return fig


def _shaft_title(shaft_name: str | None, quantity: str) -> str:
    return f"{shaft_name} -- {quantity}" if shaft_name else quantity


# ---------------------------------------------------------------------------
# BENDING MOMENT -- 2 figures (components + resultant/phase), replacing
# the former 3 (xz, xy, resultant).
# ---------------------------------------------------------------------------

def bending_moment_components_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _components_figure(
        result.x_nodes, result.M_xz, result.M_xy,
        title=_shaft_title(shaft_name, "Bending moment -- M_xz / M_xy(x)"),
        ylabel="M [N.mm]",
    )


def bending_moment_resultant_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _resultant_and_phase_figure(
        result.x_nodes, result.M_xz, result.M_xy, result.M,
        title=_shaft_title(shaft_name, "Resultant bending moment M(x)"),
        ylabel_resultant="M [N.mm]",
    )


# ---------------------------------------------------------------------------
# SHEAR -- 2 figures.
# ---------------------------------------------------------------------------

def shear_components_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _components_figure(
        result.x_nodes, result.V_xz, result.V_xy,
        title=_shaft_title(shaft_name, "Shear force -- V_xz / V_xy(x)"),
        ylabel="V [N]",
    )


def shear_resultant_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _resultant_and_phase_figure(
        result.x_nodes, result.V_xz, result.V_xy, result.V,
        title=_shaft_title(shaft_name, "Resultant shear force V(x)"),
        ylabel_resultant="V [N]",
    )


# ---------------------------------------------------------------------------
# DEFLECTION -- 2 figures.
# ---------------------------------------------------------------------------

def deflection_components_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _components_figure(
        result.x_nodes, result.v_xz, result.v_xy,
        title=_shaft_title(shaft_name, "Deflection -- v_xz / v_xy(x)"),
        ylabel="v [mm]",
    )


def deflection_resultant_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _resultant_and_phase_figure(
        result.x_nodes, result.v_xz, result.v_xy, result.v,
        title=_shaft_title(shaft_name, "Resultant deflection v(x)"),
        ylabel_resultant="v [mm]",
    )


# ---------------------------------------------------------------------------
# TORSION -- 1 figure. No xz/xy pair (torque is a single scalar per
# node), so no components/resultant split applies here.
# ---------------------------------------------------------------------------

def torsion_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    # T's own unit is N.m, not N.mm -- see resolution_report.py's own
    # deflection_torsion_table() docstring on this exact mismatch.
    return _line_figure(
        result.x_nodes, result.T,
        title=_shaft_title(shaft_name, "Torque T(x)"),
        ylabel="T [N.m]", color="black",
    )


# ---------------------------------------------------------------------------
# STRESS -- 2 figures.
#
# bending_stress_figure (sigma_b) reuses the resultant+phase layout:
# sigma_b = M / W is derived from the RESULTANT bending moment, so its
# phase IS the bending moment's own phase (see this module's own top
# docstring) -- computed here from result.M_xz/result.M_xy, not from
# sigma_b, which is a scalar magnitude with no phase of its own.
#
# shear_stress_figure (tau) stays a plain single line: tau derives from
# torque T, a single scalar per node with no xz/xy pair in this model,
# so there is no vector to take a phase of. No marker on either, for
# visual consistency with the black, unmarked resultant plots above.
# ---------------------------------------------------------------------------

def bending_stress_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _resultant_and_phase_figure(
        result.x_nodes, result.M_xz, result.M_xy, result.sigma_b,
        title=_shaft_title(shaft_name, "Bending stress sigma_b(x)"),
        ylabel_resultant="sigma_b [MPa]",
    )


def shear_stress_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _line_figure(
        result.x_nodes, result.tau,
        title=_shaft_title(shaft_name, "Torsional shear stress tau(x)"),
        ylabel="tau [MPa]", color="black",
    )


# ---------------------------------------------------------------------------
# Registry + writer -- see this module's own top docstring for the
# folder layout and why it is subdivided per shaft.
# ---------------------------------------------------------------------------

_FIGURE_REGISTRY: dict[str, Callable[["ShaftResults", str | None], Figure]] = {
    "bending_moment_components.png": bending_moment_components_figure,
    "bending_moment_resultant.png": bending_moment_resultant_figure,
    "shear_components.png": shear_components_figure,
    "shear_resultant.png": shear_resultant_figure,
    "deflection_components.png": deflection_components_figure,
    "deflection_resultant.png": deflection_resultant_figure,
    "torsion.png": torsion_figure,
    "bending_stress.png": bending_stress_figure,
    "shear_stress.png": shear_stress_figure,
}


def write_shaft_figures(
    result: "ShaftResults",
    shaft_name: str,
    base_dir: "str | Path",
    dpi: int = 150,
) -> list[Path]:
    """
    Build and save every figure in _FIGURE_REGISTRY for ONE shaft's
    ShaftResults, under <base_dir>/plots/<shaft_name>/. Creates the
    subfolder if missing. Each figure is closed (plt.close(fig)) right
    after saving -- generating 9 figures per shaft across a multi-shaft
    system without closing them would leave that many open matplotlib
    Figure objects in memory for the rest of the process.

    Returns the list of written file paths (same "return what was
    written" convention write_comparison_report() uses for its own
    text, so a caller can log/verify without re-listing the directory).
    """
    out_dir = Path(base_dir) / "plots" / shaft_name
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for filename, build_fn in _FIGURE_REGISTRY.items():
        fig = build_fn(result, shaft_name)
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        written.append(out_path)
    return written


def write_resolution_plots(
    library: "RigidBearingFEMResultsLibrary",
    system: "SpurHelicalGearSystem",
    base_dir: "str | Path",
    dpi: int = 150,
) -> dict[str, list[Path]]:
    """
    Walk every shaft in `system.shafts` (system's own order, same
    convention write_resolution_report()/write_comparison_report() use)
    and call write_shaft_figures() for each one that has a result in
    `library`. Shafts with no result (library.get_or_none() returns
    None) are skipped, not raised on -- same "report what's there"
    stance the text reports take for a missing shaft.

    Returns {shaft_name: [written PNG paths]} for every shaft that DID
    have a result -- shafts skipped for missing results are simply
    absent from the returned dict, not present with an empty list, so
    `len(result)` tells a caller how many shafts actually got plotted
    without having to inspect the lists.
    """
    out: dict[str, list[Path]] = {}
    for ss in system.shafts:
        result = library.get_or_none(ss.name)
        if result is None:
            continue
        out[ss.name] = write_shaft_figures(result, ss.name, base_dir, dpi=dpi)
    return out