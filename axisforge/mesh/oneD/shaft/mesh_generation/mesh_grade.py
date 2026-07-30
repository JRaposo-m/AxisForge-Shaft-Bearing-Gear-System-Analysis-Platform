"""
axisforge/mesh/oneD/shaft/mesh_generation/mesh_grade.py
"""
import numpy as np

from axisforge.config import SOLVER_TOLERANCE


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
    x_lo    : lower bound of the subdomain
    x_hi    : upper bound of the subdomain
    x_nodes : full mesh node positions (from Mesh1D.x_nodes)
    """

    def __init__(
        self,
        x_lo: float,
        x_hi: float,
        x_nodes: list[float],
    ):
        self._x_lo    = x_lo
        self._x_hi    = x_hi
        self._x_nodes = x_nodes

    def get_grade(self, grade: str) -> list[float]:
        """
        Return node positions for the requested grade.

        Parameters
        ----------
        grade : "grade_0" | "grade_1" | ... | "grade_N"

        Returns
        -------
        Sorted list of node positions within [x_lo, x_hi].

        Raises
        ------
        ValueError if grade string is malformed or N is negative.
        """
        n = self._parse_grade(grade)
        nodes = self._base_nodes()
        for _ in range(n):
            nodes = self._bisect_once(nodes)
        return nodes

    def _parse_grade(self, grade: str) -> int:
        """
        Parse grade string into refinement level integer.

        "grade_0" -> 0, "grade_1" -> 1, etc.

        Raises
        ------
        ValueError if string does not match expected format.
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
        Extract nodes from x_nodes that fall within [x_lo, x_hi].
        
        These are grade_0 — the natural mesh nodes in the subdomain,
        one per existing element boundary.
        """
        nodes = [
            x for x in self._x_nodes
            if self._x_lo - SOLVER_TOLERANCE <= x <= self._x_hi + SOLVER_TOLERANCE
        ]
        return sorted(nodes)

    def _bisect_once(self, nodes: list[float]) -> list[float]:
        """
        Insert the midpoint of every interval between consecutive nodes.

        Applied iteratively to produce grade_1, grade_2, ... grade_N.
        """
        result = list(nodes)
        for a, b in zip(nodes, nodes[1:]):
            result.append((a + b) / 2.0)
        return sorted(result)