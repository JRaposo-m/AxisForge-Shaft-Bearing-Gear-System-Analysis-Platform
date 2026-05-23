"""
ui/runner.py
Cell execution engine for AxisForge UI.

Responsibilities:
  - Execute cell code via exec() in a shared namespace.
  - Capture stdout, stderr, and exceptions per cell.
  - Detect MechanicalSystem instances after execution.
  - Run validate() in background after every cell (non-raising).
  - Track cell dependency order for downstream invalidation.
  - NO PySide6 imports — this module is UI-framework-agnostic.
  - NO solver calls — solvers are called by user code inside cells.

Design decisions:
  - Shared namespace persists across cells (Jupyter-equivalent behaviour).
  - Each cell execution returns a CellResult dataclass.
  - Dependency tracking is name-based: records which names were written
    by each cell, enabling downstream invalidation in future phases.
  - Thread safety: namespace access is NOT thread-safe by design.
    QThread wrapper in code_editor.py is responsible for serialisation.

Usage:
    runner = CellRunner()
    result = runner.execute(cell_index=0, code="shaft = Shaft(...)")
    systems = runner.get_systems()
"""

from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    """
    Output of a single cell execution.

    Attributes
    ----------
    cell_index : int
        Index of the cell that produced this result.
    success : bool
        True if exec() completed without exception.
    stdout : str
        Captured stdout from the cell (print() calls).
    stderr : str
        Captured stderr from the cell.
    error_type : str
        Exception class name, empty string if success.
    error_message : str
        Exception message, empty string if success.
    traceback_lines : list[str]
        Filtered traceback lines relevant to user code.
        Internal runner frames are stripped.
    names_written : set[str]
        Names added or modified in the namespace by this cell.
    validation_errors : list[str]
        MechanicalSystem.validate() errors found after execution.
        Empty if no system present or system is valid.
    """
    cell_index: int
    success: bool
    stdout: str = ""
    stderr: str = ""
    error_type: str = ""
    error_message: str = ""
    traceback_lines: list[str] = field(default_factory=list)
    names_written: set[str] = field(default_factory=set)
    validation_errors: list[str] = field(default_factory=list)

    @property
    def has_output(self) -> bool:
        return bool(self.stdout or self.stderr or not self.success)

    @property
    def status_label(self) -> str:
        """Short status string for UI badge."""
        if self.success and not self.validation_errors:
            return "ok"
        if self.success and self.validation_errors:
            return "warning"
        return "error"


# ---------------------------------------------------------------------------
# Cell dependency record
# ---------------------------------------------------------------------------

@dataclass
class CellRecord:
    """Internal record of what each cell wrote to the namespace."""
    cell_index: int
    names_written: set[str] = field(default_factory=set)
    names_read: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class CellRunner:
    """
    Stateful cell execution engine.

    Maintains a shared namespace across all cells.
    Each call to execute() returns a CellResult.

    The namespace is seeded with AxisForge public imports so user
    code does not need to import anything — same as a Jupyter kernel
    with pre-loaded modules.
    """

    # Names that are always present — not counted as "written by cell"
    _SEEDED_NAMES: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self._namespace: dict[str, Any] = {}
        self._records: dict[int, CellRecord] = {}
        self._seed_namespace()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, cell_index: int, code: str) -> CellResult:
        """
        Execute code in the shared namespace.

        Parameters
        ----------
        cell_index : int
            Identifies the cell; used for dependency tracking.
        code : str
            Python source to execute.

        Returns
        -------
        CellResult
            Full execution result including output, errors, and validation.
        """
        code = code.strip()
        if not code:
            return CellResult(cell_index=cell_index, success=True)

        names_before = set(self._namespace.keys())

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, f"<cell {cell_index}>", "exec"), self._namespace)  # noqa: S102

            names_after = set(self._namespace.keys())
            names_written = names_after - names_before - self._SEEDED_NAMES

            record = CellRecord(
                cell_index=cell_index,
                names_written=names_written,
            )
            self._records[cell_index] = record

            validation_errors = self._run_validation()

            return CellResult(
                cell_index=cell_index,
                success=True,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                names_written=names_written,
                validation_errors=validation_errors,
            )

        except Exception as exc:  # noqa: BLE001
            tb_lines = self._filter_traceback(cell_index)
            return CellResult(
                cell_index=cell_index,
                success=False,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback_lines=tb_lines,
            )

    def reset(self) -> None:
        """Clear namespace and all cell records. Full restart."""
        self._namespace.clear()
        self._records.clear()
        self._seed_namespace()

    def reset_from(self, cell_index: int) -> None:
        """
        Invalidate cell_index and all cells recorded after it.

        Does NOT re-execute — just removes their name contributions
        from tracking. The namespace itself is NOT rolled back
        (full rollback requires re-execution from cell 0).

        Use this to mark downstream cells as stale in the UI.
        """
        stale = [i for i in self._records if i >= cell_index]
        for i in stale:
            del self._records[i]

    def get_systems(self) -> list[Any]:
        """
        Return all MechanicalSystem instances currently in the namespace.

        Used by canvas and component tree to discover what to render.
        Returns list to support multi-shaft future (Ano 2 roadmap).
        """
        try:
            from core.system import MechanicalSystem  # type: ignore[import]
        except ImportError:
            return []

        return [
            obj for obj in self._namespace.values()
            if isinstance(obj, MechanicalSystem)
        ]

    def get_namespace(self) -> dict[str, Any]:
        """Read-only view of the shared namespace. Do not mutate."""
        return dict(self._namespace)

    def downstream_cells(self, cell_index: int) -> list[int]:
        """
        Return sorted list of cell indices that depend on names written
        by cell_index.

        Dependency is name-based: cell B depends on cell A if B reads
        any name that A wrote.

        Note: read tracking is not yet implemented (Phase 2 scope).
        Returns empty list until read tracking is added.
        """
        if cell_index not in self._records:
            return []
        written = self._records[cell_index].names_written
        dependents: list[int] = []
        for idx, record in self._records.items():
            if idx <= cell_index:
                continue
            if record.names_read & written:
                dependents.append(idx)
        return sorted(dependents)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _seed_namespace(self) -> None:
        """
        Pre-populate namespace with AxisForge public API.

        User cells can use Shaft, Bearing, etc. without imports.
        Import failures are silently skipped — runner works even
        if solvers are partially unavailable.
        """

        import sys, os, pathlib
        # Ensure project root is on sys.path when running from ui/
        project_root = pathlib.Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        imports = [
            # core entities
            ("core.shaft", ["Shaft", "ShaftSection", "Shoulder"]),
            ("core.components", ["Bearing", "BearingType", "GearElement"]),
            ("core.loads", ["RadialLoad", "AxialLoad", "TorqueLoad",
                            "ExternalMoment", "LoadPlane"]),
            ("core.system", ["MechanicalSystem"]),
            ("core.materials", ["Material"]),
            # solvers
            ("solvers", ["StaticsSolver", "StaticFailureSolver",
                         "StressSolver", "BearingLifeSolver", "GearSolver"]),
        ]
        seeded: set[str] = set()
        for module_path, names in imports:
            try:
                import importlib
                mod = importlib.import_module(module_path)
                for name in names:
                    obj = getattr(mod, name, None)
                    if obj is not None:
                        self._namespace[name] = obj
                        seeded.add(name)
            except Exception:  # noqa: BLE001
                pass  # partial availability is acceptable

        # Also seed numpy for convenience
        try:
            import numpy as np
            self._namespace["np"] = np
            seeded.add("np")
        except ImportError:
            pass

        # Freeze seeded names so execute() doesn't count them as "written"
        object.__setattr__(self, "_SEEDED_NAMES", frozenset(seeded))

    def _run_validation(self) -> list[str]:
        """
        Run MechanicalSystem.validate() on every system in namespace.

        Returns aggregated error list. Does not raise.
        """
        errors: list[str] = []
        for system in self.get_systems():
            try:
                errors.extend(system.validate())
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Validation exception: {exc}")
        return errors

    @staticmethod
    def _filter_traceback(cell_index: int) -> list[str]:
        """
        Extract traceback lines, stripping internal runner frames.

        Keeps only frames where filename matches <cell N> pattern,
        plus the final exception line.
        """
        lines = traceback.format_exc().splitlines()
        cell_tag = f"<cell {cell_index}>"
        filtered: list[str] = []
        include_next = False
        for line in lines:
            if cell_tag in line:
                include_next = True
                filtered.append(line.strip())
            elif include_next:
                filtered.append(line.strip())
                include_next = False
        # Always include the final exception line
        if lines:
            last = lines[-1].strip()
            if last and last not in filtered:
                filtered.append(last)
        return filtered
