"""
AxisForge global configuration: solver tolerances, geometric guards and default factors.

Units are the AxisForge units, not SI base units: lengths [mm], forces [N], stresses
[MPa = N/mm^2], speeds [rpm], times [h]. BOUNDARY_MOMENT_TOLERANCE is a bending moment in [N mm].
The unit of torque loads is an open review item (U1, see core/loads.py): moments in [N mm] in the
FEM; torque in [N m] in the gear system and the torsion solver. Each constant below is annotated with its unit,
the modules that read it, or "not read by any module at this commit" (checked by text search of
``axisforge/`` outside ``tests/``). "Named in the code" sources (Shigley) are not in the thesis
library: (source to find).
"""

# ---------------------------------------------------------------------------
# Solver discretisation
# ---------------------------------------------------------------------------
# Points along the shaft axis [count]. Not read by any module at this commit.
SOLVER_RESOLUTION: int = 1000
# General numerical tolerance [-]. Used as a length tolerance [mm] by the shaft mesh grading, the
# submodel extraction, the torsion solver and the submodel Lagrange solver, and as the default
# tolerance of the ISO/TS 16281 contact solver.
SOLVER_TOLERANCE: float = 1e-6
# Largest |M| accepted at a support [N mm]. Not read by any module at this commit.
BOUNDARY_MOMENT_TOLERANCE: float = 500.0
# Minimum element length [mm]: extra (refinement) nodes closer than this are merged; mandatory
# nodes closer than this raise. Read by mesh/shaft/mesh_generation (mesh_1D.py, mesh_grade.py).
MESH_MIN_NODE_DIST_MM: float = 0.1
# Two positions within this distance are the same point [mm] (node lookup and coincident-node
# collapse). Read by mesh_1D.py and element_type/elem.py.
NODE_LOOKUP_TOL_MM: float = 1e-6
# Tolerance of the mesh-convergence study [-]. Not read by any module at this commit.
CONVERGENCE_TOLERANCE: float = 1e-6
# Iteration limit [count]. Not read by any module at this commit (named only in the docstring of
# axisforge/__init__.py).
MAX_ITER = 200

# ---------------------------------------------------------------------------
# Bearing life defaults
# ---------------------------------------------------------------------------
# Default design life of a shaft system [h]. Value not found in the library: (source to find).
# Not effectively read from here: shaft_system.py imports it with a bare ``from config import``
# and falls back to the same literal 20_000.0 when that import fails.
DEFAULT_DESIGN_LIFE_HOURS: float = 20_000.0
# Minimum static safety factor [-]. Not read by any module at this commit.
STATIC_SAFETY_FACTOR_MIN: float = 1.0
# Minimum static safety factor under shock loading [-]. Not read by any module at this commit.
STATIC_SAFETY_FACTOR_SHOCK: float = 1.5

# ---------------------------------------------------------------------------
# Fatigue / stress defaults  (RELIABILITY_FACTOR_99, RELIABILITY_FACTOR_90, TEMPERATURE_FACTOR_DEFAULT
# and SUT_ENDURANCE_CAP_MPa are not read by any module at this commit)
# ---------------------------------------------------------------------------
# Named in the code as the reliability factor k_e for 99 % reliability [-], Shigley Tab. 6-6
# (not in the library: (source to find)). Unchecked; with k_e = 1 - 0.08 z_a the value 0.868
# corresponds to 95 % (z_a = 1.645) and 99 % would be 0.814 (review item).
RELIABILITY_FACTOR_99: float = 0.868
# Reliability factor k_e for 90 % reliability [-]; no source named in the code.
RELIABILITY_FACTOR_90: float = 0.897
# Default temperature factor [-].
TEMPERATURE_FACTOR_DEFAULT: float = 1.0

# Upper limit of S_ut [MPa] for the rule Se = 0.5*Sut; named in the code as Shigley sec. 6-2
# (not in the library: (source to find)). Not read by any module at this commit.
SUT_ENDURANCE_CAP_MPa: float = 1400.0

# Smallest fillet radius used in the notch-sensitivity calculation [mm]. Read by
# solvers/machine_elements/shaft/utils.py through a bare ``from config import`` (resolves only
# when the ``axisforge/`` directory itself is on sys.path; ImportError otherwise).
MIN_FILLET_RADIUS_mm: float = 0.01

# Geometric tolerance for boundary and coincidence checks [mm]. Read by core/.../shaft/shaft.py and
# by SpurHelicalGearSystem._axial_alignment_errors (gear_system.py).
TOL_GEOMETRY_mm: float = 1e-9

# ---------------------------------------------------------------------------
# Failure classification thresholds (FLA; meaning to confirm). Not read at this commit.
# ---------------------------------------------------------------------------
# Lower and upper thresholds [-].
FLA_THRESHOLD_LOW: float = 0.25
FLA_THRESHOLD_HIGH: float = 0.60

# ---------------------------------------------------------------------------
# Numerical guards
# ---------------------------------------------------------------------------
# Minimum equivalent bearing load [N]. Not read by any module at this commit.
MIN_LOAD_N: float = 1.0
# Minimum separation between two supports [mm] (division-by-zero guard for the solver). Read by
# ShaftSystem (shaft_system.py).
MIN_BEARING_SEPARATION_MM = 1.0e-3
# Keep-out distance between a gear footprint and a shaft shoulder [mm]. Read by ShaftSystem.
MIN_SHOULDER_CLEARANCE_MM = 1.0
# Clearance [mm] that a floating bearing must keep to be considered "floating" (not abutting a
# shoulder). Read by ShaftSystem.
FLOATING_BEARING_CLEARANCE_MM = 0.1
# Face width [mm] below which an element is treated as a point load (no submodel study). Not read
# by any module at this commit.
MIN_FACE_WIDTH_FOR_CONVERGENCE_MM: float = 1.0
