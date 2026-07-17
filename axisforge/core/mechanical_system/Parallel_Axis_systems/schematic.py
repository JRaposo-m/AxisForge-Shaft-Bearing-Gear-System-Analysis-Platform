"""
axisforge/core/mechanical_system/Parallel_Axis_systems/schematic.py

Line schematic for the REAL AxisForge ShaftSystem / GearSystem
(systems/spur_helicoidal_system/{shaft_system,gear_system}.py).

This module owns NOTHING about the assembly. It only READS what is already
there (shaft.sections, shaft_position, shaft_origin_x, bearings, gears,
links) and draws lines. No ShaftSystem/GearSystem/Bearing/GearElement are
redefined here — they are imported from the real modules, so this file can
be dropped into the project without colliding with the engineering classes.

Purpose: VISUAL / TOPOLOGICAL check only — "is this the assembly I think it
is" (bearing count + arrangement, gear positions, mesh connectivity,
resolved shaft_position/shaft_origin_x). NOT a solver, and draws NO loads
(loads are always local to one shaft — if/when they need drawing, that
belongs on a future extension of draw_shaft_system(), reading
shaft_system.loads, never here on the multi-shaft side).

COORDINATE SYSTEM (changed from the earlier grid-column version):
    Both axes plot REAL mm coordinates directly — no column/row
    abstraction, no unit_mm scale factor. What you see IS the position.

    X axis : shaft_origin_x + local position [mm]  (always).
    Y axis : shaft_position[0] (if axis="y") or shaft_position[1]
             (if axis="z") [mm] — same real coordinate GearSystem.resolve()
             computed, not a symbolic row index.

    Both axes share the SAME scale: ax.set_aspect("equal") — 1 mm in x
    measures the same as 1 mm in y/z on screen. This is what makes the
    proportions "make sense" at a glance: a bearing block, a shaft
    diameter, and the distance between two shafts are all comparable
    directly, the way a real elevation drawing would be.

NOTE on `axis`: pick whichever of "y"/"z" actually separates your shafts.
A GearMeshLink with phi_deg=270 (as in design_load_lifting_shafts.py)
offsets shaft_position along z, not y — use axis="z" for that layout, or
every shaft lands on y=0 and overlaps.

Usage
-----
    from axisforge.core.mechanical_system.Parallel_Axis_systems.schematic import (
        draw_gear_system, draw_shaft_detail, recommended_figsize,
    )

    fig, ax = plt.subplots(figsize=recommended_figsize(gearbox, axis="z"))
    draw_gear_system(ax, gearbox, axis="z")
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# --- real AxisForge types — imports only, nothing redefined here -----------
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import (
    ShaftSystem, GearElement,
)
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.gear_system import (
    GearSystem, GearMeshLink,
)


# ===========================================================================
# STYLE — every visual choice lives here. All sizes are now real mm, since
# both axes plot real mm coordinates (see module docstring).
# ===========================================================================

INK = "black"

LW_SHAFT        = 1.0
LW_BEARING      = 1.1
LW_GEAR         = 1.1
LW_MESH_LINK    = 0.6

SHAFT_HEIGHT_MIN_MM = 3.0        # floor half-height, mm — keeps very thin shafts visible

BEARING_BLOCK_HEIGHT_MM   = 6.0  # how far the bearing block extends BEYOND the shaft edge, mm
BEARING_WIDTH_FALLBACK_MM = 10.0 # used when bearing.b is 0/undefined

GEAR_SIZE_FALLBACK_MM = 15.0     # used only if a gear object exposes neither .r nor .d
GEAR_TICK_MM        = 4.0

FONT_SIZE_LABEL = 7
FONT_SIZE_ROW   = 8
FONT_SIZE_TICK  = 6       # axis tick numbers (-40, -20, 0, ...) — was matplotlib's default (~10),
                           # too big for TICK_STEP_MM spacing on a compact figsize, causing overlap

TICK_STEP_MM = 20.0              # tick spacing, mm, BOTH axes (same scale -> same step reads naturally).
                                  # Was "every 5 grid columns" before; 5 mm would be too dense now that
                                  # axes are real mm on shafts that run to a few hundred mm — retune here.

# recommended_figsize() clamps width:height into this range — never more
# elongated than FIGSIZE_MAX_RATIO, never narrower than FIGSIZE_MIN_RATIO.
FIGSIZE_MIN_RATIO = 1.0
FIGSIZE_MAX_RATIO = 1.5


# ===========================================================================
# Symbol templates — pure data: lists of ((dx0,dy0),(dx1,dy1)) line segments
# in LOCAL mm offsets, centred on (0,0). Add a new element type by writing
# one more function like these and calling it from draw_shaft_system().
# ===========================================================================

def _symbol_bearing(inner_hh: float, block_height: float, half_width: float
                     ) -> list[tuple[tuple[float, float], tuple[float, float]]]:

    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for sign in (+1, -1):
        y_near = sign * inner_hh
        y_far = sign * (inner_hh + block_height)
        segs += [
            ((-half_width, y_near), (half_width, y_near)),  # inner edge — touches the shaft line
            ((-half_width, y_far), (half_width, y_far)),     # outer edge
            ((-half_width, y_near), (-half_width, y_far)),    # left edge
            ((half_width, y_near), (half_width, y_far)),      # right edge
        ]

        segs += [
            ((-half_width, y_near), (half_width, y_far)),
            ((-half_width, y_far), (half_width, y_near)),
        ]
    return segs


def _symbol_gear(pitch_r: float, tip_r: float, tick: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Spine perpendicular to the shaft, real size:
      - full spine out to tip_r (the gear's real outer/addendum extent —
        this is what two meshing gears' spines should just touch at, if
        shaft_position puts the shafts at the correct real centre distance)
      - a tick at pitch_r (the actual contact/meshing reference — for a
        standard external pair, centre_distance == r1 + r2, so it's the
        PITCH tick, not the tip, that lines up exactly between two shafts)
      - a tick at the tip, marking the outer boundary
    """
    return [
        ((0.0, -tip_r), (0.0, tip_r)),          # full spine, real outer extent
        ((-tick, tip_r), (tick, tip_r)),          # tip tick
        ((-tick, -tip_r), (tick, -tip_r)),
        ((-tick, pitch_r), (tick, pitch_r)),       # pitch tick — the real mesh/contact reference
        ((-tick, -pitch_r), (tick, -pitch_r)),
    ]


# ===========================================================================
# Small helpers
# ===========================================================================

def _axis_index(axis: str) -> int:
    if axis not in ("y", "z"):
        raise ValueError(f"axis must be 'y' or 'z', got {axis!r}")
    return 0 if axis == "y" else 1


def _shaft_half_height(diameter_mm: float) -> float:
    """Real half-height (radius), mm, with only a floor for visibility —
    no artificial slope/clamp: what you see is the real radius."""
    return max(SHAFT_HEIGHT_MIN_MM, diameter_mm / 2.0)


def _local_half_height(shaft_system: ShaftSystem, local_x: float, detail: bool) -> float:
    """0.0 in 1D (system) view — the shaft is a single line. In 2D (detail)
    view: the real local half-height, read off Shaft.diameter_at()."""
    if not detail:
        return 0.0
    return _shaft_half_height(shaft_system.shaft.diameter_at(local_x))


def gear_anchor(shaft_system: ShaftSystem, gear: GearElement, axis: str = "y") -> tuple[float, float]:
    """Real (x, y) mm coordinates of a gear's symbol centre — used for mesh connector lines."""
    idx = _axis_index(axis)
    return shaft_system.shaft_origin_x + gear.position, shaft_system.shaft_position[idx]


def _gear_radii(gear_element: GearElement) -> tuple[float, float]:
    """
    Real (pitch_radius, tip_radius) in mm, read off the underlying gear
    geometry object (gear_element.gear — SpurGear/HelicalGear/...).

    This is what makes the schematic a genuine check on shaft_position: for
    a standard external pair, centre_distance == r1 + r2, so if the layout
    is correct, the two gears' pitch ticks land exactly on top of each
    other. If they don't overlap (or overlap a lot), shaft_position/al is
    wrong somewhere upstream — GearSystem.resolve() or the meshing object.
    """
    gear = gear_element.gear
    pitch_r = getattr(gear, "r", None)
    if pitch_r is None:
        d = getattr(gear, "d", None)
        pitch_r = d / 2.0 if d else GEAR_SIZE_FALLBACK_MM
    tip_r = getattr(gear, "ra", pitch_r)
    return pitch_r, tip_r


def _max_gear_tip_radius(shafts: list[ShaftSystem]) -> float:
    """Largest gear tip radius across one or several shafts, mm — 0.0 if none have gears.
    Used only to size padding around the drawing, never the grid extent itself."""
    radii = [
        _gear_radii(g)[1]
        for s in shafts
        for g in s.gears
    ]
    return max(radii, default=0.0)


# ===========================================================================
# Drawing — free functions operating on the REAL ShaftSystem / GearSystem.
# Nothing here mutates the objects passed in.
# ===========================================================================

def draw_shaft_system(ax: Axes, shaft_system: ShaftSystem, detail: bool = False,
                       axis: str = "y", local: bool = False) -> None:

    idx = _axis_index(axis)
    x0 = 0.0 if local else shaft_system.shaft_origin_x
    x_lo = x0
    x_hi = x0 + shaft_system.shaft.total_length
    row = shaft_system.shaft_position[idx]   # REAL mm coordinate, not a symbolic index

    ax.annotate(f"{shaft_system.name}",
                (x_hi, row), xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=FONT_SIZE_ROW, color=INK)

    if not detail:
        # 1D: the shaft IS the line — top edge and bottom edge coincide.
        ax.add_line(Line2D([x_lo, x_hi], [row, row], color=INK, linewidth=LW_SHAFT))
    else:
        # 2D: stepped profile, one pair of lines per section, diagonal
        # transition at every boundary where the diameter actually changes
        # (a flat run — no diameter change — gets no boundary mark at all).
        sections = shaft_system.shaft.sections
        x_local = 0.0
        bounds = []  # (x0_mm, x1_mm, half_height_mm) per section
        for sec in sections:
            xa = x0 + x_local
            xb = x0 + x_local + sec.length
            bounds.append((xa, xb, _shaft_half_height(sec.diameter)))
            x_local += sec.length

        for xa, xb, hh in bounds:
            for sign in (+1, -1):
                ax.add_line(Line2D([xa, xb], [row + sign * hh, row + sign * hh],
                                    color=INK, linewidth=LW_SHAFT))

        # shoulder = a straight vertical step at the boundary (NOT a diagonal
        # chamfer) — only where the diameter actually changes; a flat run
        # gets no boundary mark at all.
        for (xa_a, xb_a, hh_a), (xa_b, xb_b, hh_b) in zip(bounds, bounds[1:]):
            if hh_a == hh_b:
                continue
            xb = xb_a  # == xa_b, the shared boundary position
            for sign in (+1, -1):
                ax.add_line(Line2D([xb, xb], [row + sign * hh_a, row + sign * hh_b],
                                    color=INK, linewidth=LW_SHAFT))

        # shaft ends — vertical caps
        hh_first, hh_last = bounds[0][2], bounds[-1][2]
        ax.add_line(Line2D([x_lo, x_lo], [row - hh_first, row + hh_first], color=INK, linewidth=LW_SHAFT))
        ax.add_line(Line2D([x_hi, x_hi], [row - hh_last, row + hh_last], color=INK, linewidth=LW_SHAFT))

    # bearings — inner edge flush with the local shaft edge; width from real b
    for b in shaft_system.bearings:
        x = x0 + b.position
        inner_hh = _local_half_height(shaft_system, b.position, detail)
        half_width = (getattr(b, "b", 0.0) or BEARING_WIDTH_FALLBACK_MM) / 2.0
        for (dx0, dy0), (dx1, dy1) in _symbol_bearing(inner_hh, BEARING_BLOCK_HEIGHT_MM, half_width):
            ax.add_line(Line2D([x + dx0, x + dx1], [row + dy0, row + dy1],
                                color=INK, linewidth=LW_BEARING))
        label = b.label or getattr(b, "designation", "") or "brg"
        ax.annotate(label, (x, row - inner_hh - BEARING_BLOCK_HEIGHT_MM),
                    xytext=(0, -6), textcoords="offset points",
                    ha="center", va="top", fontsize=FONT_SIZE_LABEL, color=INK)

    # gears — point symbols anchored at their position, sized from real geometry
    for g in shaft_system.gears:
        x = x0 + g.position
        pitch_r, tip_r = _gear_radii(g)
        for (dx0, dy0), (dx1, dy1) in _symbol_gear(pitch_r, tip_r, GEAR_TICK_MM):
            ax.add_line(Line2D([x + dx0, x + dx1], [row + dy0, row + dy1],
                                color=INK, linewidth=LW_GEAR))
        label = g.label or getattr(g.gear, "label", "") or "gear"
        ax.annotate(label, (x, row + tip_r),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FONT_SIZE_LABEL, color=INK)


def draw_gear_system(ax: Axes, gear_system: GearSystem, axis: str = "y") -> Axes:
    """
    Draw every shaft in `gear_system.shafts` as a 1D line (see
    draw_shaft_system(detail=False)) plus dotted connectors marking which
    gears mesh (from `gear_system.links`). Real mm coordinates, both axes
    to the same scale (aspect='equal').

    Call gear_system.resolve() BEFORE this — draw_gear_system() never calls
    it, so a layout bug never gets silently masked by a fresh re-resolve.
    Never creates a Figure — pass an `ax` you built yourself so you can
    place other axes (e.g. per-shaft stress) alongside it.
    """
    idx = _axis_index(axis)

    for s in gear_system.shafts:
        draw_shaft_system(ax, s, detail=False, axis=axis)

    for lk in gear_system.links:
        xa, ya = gear_anchor(lk.shaft_a, lk.gear_a, axis)
        xb, yb = gear_anchor(lk.shaft_b, lk.gear_b, axis)
        ax.add_line(Line2D([xa, xb], [ya, yb], color=INK,
                            linewidth=LW_MESH_LINK, linestyle=":"))

    xs = [s.shaft_origin_x for s in gear_system.shafts]
    xs += [s.shaft_origin_x + s.shaft.total_length for s in gear_system.shafts]
    ys = [s.shaft_position[idx] for s in gear_system.shafts]
    x_pad = max(10.0, 0.05 * (max(xs) - min(xs)))
    y_pad = _max_gear_tip_radius(gear_system.shafts) + 10.0  # room for gear symbols/labels above/below the outermost shaft

    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(TICK_STEP_MM))
    ax.yaxis.set_major_locator(MultipleLocator(TICK_STEP_MM))
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(f"{axis} [mm]")
    ax.set_title(gear_system.label or "System schematic")
    return ax


def draw_shaft_detail(ax: Axes, shaft_system: ShaftSystem, axis: str = "y",
                       local: bool = False) -> Axes:
    """
    2D detail view of ONE shaft: real stepped profile, shoulder transitions,
    bearings/gears flush against the local shaft edge. Real mm coordinates,
    both axes to the same scale (aspect='equal'). Never used by
    draw_gear_system(), which stays 1D.

    local : see draw_shaft_system() — pass True to align with a solver's
        0-based x_nodes (e.g. embedding this as a subplot above
        moment/shear/deflection panels that already use local coordinates).
    """
    draw_shaft_system(ax, shaft_system, detail=True, axis=axis, local=local)

    x0 = 0.0 if local else shaft_system.shaft_origin_x
    x_hi = x0 + shaft_system.shaft.total_length
    max_hh = max(_shaft_half_height(sec.diameter) for sec in shaft_system.shaft.sections)
    max_bearing_extent = max_hh + BEARING_BLOCK_HEIGHT_MM
    max_gear_extent = _max_gear_tip_radius([shaft_system])
    y_half_extent = max(max_bearing_extent, max_gear_extent) + 15.0  # room for labels

    x_pad = max(10.0, 0.05 * (x_hi - x0))
    ax.set_xlim(x0 - x_pad, x_hi + x_pad)
    ax.set_ylim(-y_half_extent, y_half_extent)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(TICK_STEP_MM))
    ax.yaxis.set_major_locator(MultipleLocator(TICK_STEP_MM))
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(f"{axis} [mm]")
    ax.set_title(shaft_system.name)
    return ax


# ===========================================================================
# Figure-size helper — still never creates a Figure, just picks numbers.
# ===========================================================================

def _clamped_ratio(x_span: float, y_span: float) -> float:
    return min(FIGSIZE_MAX_RATIO, max(FIGSIZE_MIN_RATIO, x_span / max(y_span, 1e-6)))


def recommended_figsize(target, axis: str = "y", height: float = 4.0) -> tuple[float, float]:
    """
    Suggests a (width, height) for plt.subplots(figsize=...), clamped to
    [FIGSIZE_MIN_RATIO, FIGSIZE_MAX_RATIO] width:height regardless of how
    stretched the real geometry is — same idea as before, just computed
    from real mm spans instead of grid columns.

    target : a GearSystem (spans computed like draw_gear_system) or a
             single ShaftSystem (spans computed like draw_shaft_detail).
    """
    if isinstance(target, GearSystem):
        idx = _axis_index(axis)
        xs = [s.shaft_origin_x for s in target.shafts]
        xs += [s.shaft_origin_x + s.shaft.total_length for s in target.shafts]
        ys = [s.shaft_position[idx] for s in target.shafts]
        x_span = (max(xs) - min(xs)) + 2 * max(10.0, 0.05 * (max(xs) - min(xs)))
        y_span = (max(ys) - min(ys)) + 2 * (_max_gear_tip_radius(target.shafts) + 10.0)
    else:
        x_span = target.shaft.total_length
        max_hh = max(_shaft_half_height(sec.diameter) for sec in target.shaft.sections)
        max_bearing_extent = max_hh + BEARING_BLOCK_HEIGHT_MM
        max_gear_extent = _max_gear_tip_radius([target])
        y_span = 2 * (max(max_bearing_extent, max_gear_extent) + 15.0)

    ratio = _clamped_ratio(x_span, y_span)
    return (height * ratio, height)