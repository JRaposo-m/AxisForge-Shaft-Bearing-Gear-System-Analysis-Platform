"""
Refinement grades of a shaft subdomain by repeated bisection of the base mesh, for mesh-convergence studies.
"""
import numpy as np

from axisforge.config import SOLVER_TOLERANCE, MESH_MIN_NODE_DIST_MM


class RefinementFloorReached(ValueError):
    """Requested grade would create elements shorter than the mesh floor.

    Raised by :meth:`Grader.get_grade` before a bisection that would
    produce an interval of length <= ``MESH_MIN_NODE_DIST_MM``. Such
    midpoints would be merged away by :class:`Mesh1D`, so the grade would
    silently not be the mesh it claims to be (the refinement ratio r = 2
    assumed by Richardson extrapolation would be false; Richardson
    extrapolation: source to find).

    Subclasses ``ValueError`` so existing ``except ValueError`` handlers
    still catch it. The convergence dispatcher should catch it, stop the
    grade loop and report the interval as not converged at the floor.

    Attributes
    ----------
    grade : str
        Grade that was requested.
    h_min : float
        Shortest interval before the failing bisection [mm].
    floor : float
        ``MESH_MIN_NODE_DIST_MM`` [mm].
    """

    def __init__(self, grade: str, h_min: float, floor: float):
        """Store the arguments; see the class docstring."""
        self.grade = grade
        self.h_min = h_min
        self.floor = floor
        super().__init__(
            f"Grader: {grade!r} needs bisecting an interval of {h_min:.6g} mm, "
            f"which would give elements of {h_min / 2:.6g} mm <= "
            f"MESH_MIN_NODE_DIST_MM = {floor} mm. Refinement floor reached: "
            f"stop the grade loop at the previous grade."
        )


# ===========================================================================
# Submodel grader
# ===========================================================================

class Grader:
    """
    Generates standardised mesh grades for a subdomain [x_lo, x_hi]
    by successive elementwise bisection of the base mesh.

    Grade definition
    ----------------
    grade_0 : base mesh nodes already present in [x_lo, x_hi]
    grade_1 : grade_0 + midpoint of each element
    grade_2 : grade_1 + midpoint of each element  (4x grade_0 elements)
    grade_N : grade_{N-1} bisected elementwise

    The grade string is produced by RichardsonGCI and consumed
    by SubmodelSolver.


    Parameters
    ----------
    x_lo : float
        Lower bound of the subdomain [mm].
    x_hi : float
        Upper bound of the subdomain [mm].
    x_nodes : list of float
        Full mesh node positions [mm] (from ``Mesh1D.x_nodes``).

    Notes
    -----
    Every bisection halves each element (refinement ratio 2), which is the
    ratio that a Richardson-type convergence estimate assumes (source to
    find). A bisection is refused when it would create an interval of
    ``MESH_MIN_NODE_DIST_MM`` (0.1 mm) or less. The subdomain bounds are
    matched to the nodes with ``SOLVER_TOLERANCE`` (1e-6), used here as a
    length tolerance [mm]; Mesh1D uses ``NODE_LOOKUP_TOL_MM`` (also 1e-6 mm)
    for the same purpose. Both are values of this code, not from a source.
    """

    def __init__(
        self,
        x_lo: float,
        x_hi: float,
        x_nodes: list[float],
    ):
        """Store the arguments; see the class docstring."""
        self._x_lo    = x_lo
        self._x_hi    = x_hi
        self._x_nodes = x_nodes

    def get_grade(self, grade: str) -> list[float]:
        """
        Return node positions for the requested grade.

        Parameters
        ----------
        grade : str
            ``"grade_0"``, ``"grade_1"``, ..., ``"grade_N"`` [-].

        Returns
        -------
        list of float
            Ascending node positions within [x_lo, x_hi] [mm].

        Raises
        ------
        ValueError
            If the grade string is malformed or N is negative.
        RefinementFloorReached
            (subclass of ValueError) if a bisection would create an interval
            <= ``MESH_MIN_NODE_DIST_MM`` [mm].
        """
        n = self._parse_grade(grade)
        nodes = self._base_nodes()
        for _ in range(n):
            if len(nodes) >= 2:
                h_min = min(b - a for a, b in zip(nodes, nodes[1:]))
                if h_min / 2.0 <= MESH_MIN_NODE_DIST_MM:
                    raise RefinementFloorReached(grade, h_min, MESH_MIN_NODE_DIST_MM)
            nodes = self._bisect_once(nodes)
        return nodes

    def _parse_grade(self, grade: str) -> int:
        """
        Parse grade string into refinement level integer.

        ``"grade_0"`` -> 0, ``"grade_1"`` -> 1, etc.

        Raises
        ------
        ValueError
            If the string does not match the expected format.
        """
        prefix = "grade_"
        if not grade.startswith(prefix):
            raise ValueError(
                f"Invalid grade string: {grade!r}. "
                f"Expected format: 'grade_N' where N >= 0."
            )
        suffix = grade[len(prefix):]
        if not suffix.isdigit():
            raise ValueError(
                f"Invalid grade level: {suffix!r}. "
                f"Expected a non-negative integer after 'grade_'."
            )
        return int(suffix)

    def _base_nodes(self) -> list[float]:
        """
        Extract nodes from x_nodes that fall within [x_lo, x_hi] [mm].

        The bounds are widened by ``SOLVER_TOLERANCE`` (used as a length
        tolerance [mm]). These are grade_0, the natural mesh nodes in the subdomain,
        one per existing element boundary.
        """
        nodes = [
            x for x in self._x_nodes
            if self._x_lo - SOLVER_TOLERANCE <= x <= self._x_hi + SOLVER_TOLERANCE
        ]
        return sorted(nodes)

    def _bisect_once(self, nodes: list[float]) -> list[float]:
        """
        Insert the midpoint of every interval between consecutive nodes [mm].

        Applied iteratively to produce grade_1, grade_2, ... grade_N.
        """
        result = list(nodes)
        for a, b in zip(nodes, nodes[1:]):
            result.append((a + b) / 2.0)
        return sorted(result)