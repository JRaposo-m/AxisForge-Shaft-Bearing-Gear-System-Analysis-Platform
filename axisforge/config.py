"""
AxisForge — Global Configuration
All solver constants, unit conventions, and tuning parameters live here.
Units: SI throughout (N, mm, MPa = N/mm², rpm, h).
"""

# ---------------------------------------------------------------------------
# Solver discretisation
# ---------------------------------------------------------------------------
SOLVER_RESOLUTION: int = 1000          # Points along shaft axis [count]
SOLVER_TOLERANCE: float = 1e-6         # General numerical tolerance
BOUNDARY_MOMENT_TOLERANCE: float = 500.0  # |M| at supports ≤ this [N·mm]

# ---------------------------------------------------------------------------
# Bearing life defaults
# ---------------------------------------------------------------------------
DEFAULT_DESIGN_LIFE_HOURS: float = 20_000.0
STATIC_SAFETY_FACTOR_MIN: float = 1.0
STATIC_SAFETY_FACTOR_SHOCK: float = 1.5

# ---------------------------------------------------------------------------
# Fatigue / stress defaults
# ---------------------------------------------------------------------------
RELIABILITY_FACTOR_99: float = 0.868   # ke, 99% (Shigley Tab. 6-6)
RELIABILITY_FACTOR_90: float = 0.897   # ke, 90%
TEMPERATURE_FACTOR_DEFAULT: float = 1.0

# Se = 0.5*Sut valid only below this limit (Shigley §6-2)
SUT_ENDURANCE_CAP_MPa: float = 1400.0

MIN_FILLET_RADIUS_mm: float = 0.01    # Guard for notch sensitivity calc

TOL_GEOMETRY_mm: float = 1e-9   # Geometric tolerance for boundary checks [mm]

# ---------------------------------------------------------------------------
# Failure classification thresholds (FLA)
# ---------------------------------------------------------------------------
FLA_THRESHOLD_LOW: float = 0.25
FLA_THRESHOLD_HIGH: float = 0.60

# ---------------------------------------------------------------------------
# Numerical guards
# ---------------------------------------------------------------------------
MIN_LOAD_N: float = 1.0   # Minimum equivalent bearing load
