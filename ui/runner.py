"""
ui/runner.py
Section execution engine for AxisForge UI.

Replaces the cell-based CellRunner/CellResult model with a section-based
model that matches the # %% editor workflow.

Responsibilities:
  - Parse a .py file into sections delimited by # %%.
  - Execute individual sections or all sections in a shared namespace.
  - Capture stdout, stderr, and exceptions per section.
  - Detect MechanicalSystem instances after execution.
  - Run validate() in background after every section (non-raising).
  - Track which names each section wrote — enables stale detection in UI.
  - NO PySide6 imports — UI-framework-agnostic.
  - NO solver calls — solvers are called by user code in sections.

Section format:
  # %% Optional title text
  code line 1
  code line 2

  # %% Next section
  ...

A file with no # %% markers is treated as a single unnamed section.

Thread safety:
  Namespace access is NOT thread-safe by design.
  QThread wrapper in script_editor.py is responsible for serialisation.
  Never call execute() from two threads simultaneously.

Usage:
    runner = SectionRunner()
    sections = runner.parse_sections(source_code)
    result = runner.execute(sections[0])
    systems = runner.get_systems()
    ns = runner.get_namespace()
"""

from __future__ import annotations

import io
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Section — parsed unit from source file
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """
    A single # %% region parsed from a .py source file.

    Attributes
    ----------
    index : int
        Zero-based position in the file.
    title : str
        Text after '# %%' on the delimiter line. Empty string if no title.
    code : str
        Source code of the section (delimiter line excluded).
    line_start : int
        1-based line number of the # %% delimiter (or 1 for implicit first section).
    line_end : int
        1-based line number of the last line of code in this section.
    """
    index: int
    title: str
    code: str
    line_start: int
    line_end: int

    @property
    def display_title(self) -> str:
        """Human-readable title — falls back to 'Section N' if no title."""
        return self.title if self.title else f"Section {self.index + 1}"

    @property
    def is_empty(self) -> bool:
        return not self.code.strip()


# ---------------------------------------------------------------------------
# SectionResult — output of one section execution
# ---------------------------------------------------------------------------

@dataclass
class SectionResult:
    """
    Complete output of executing one Section.

    Attributes
    ----------
    section_index : int
    title : str
        Section title (from Section.display_title).
    success : bool
        True if exec() completed without raising.
    stdout : str
        Captured stdout (print() calls).
    stderr : str
        Captured stderr.
    error_type : str
        Exception class name. Empty if success.
    error_message : str
        Exception message. Empty if success.
    traceback_lines : list[str]
        Filtered traceback — only frames in user code.
    names_written : set[str]
        Names added or modified in the namespace by this section.
    validation_errors : list[str]
        MechanicalSystem.validate() errors found after execution.
        Empty if no system present or system is valid.
    execution_time_ms : float
        Wall-clock execution time in milliseconds.
    """
    section_index: int
    title: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    error_type: str = ""
    error_message: str = ""
    traceback_lines: list[str] = field(default_factory=list)
    names_written: set[str] = field(default_factory=set)
    validation_errors: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    @property
    def has_output(self) -> bool:
        return bool(self.stdout or self.stderr or not self.success)

    @property
    def status_label(self) -> str:
        """Short status string for UI indicators."""
        if not self.success:
            return "error"
        if self.validation_errors:
            return "warning"
        return "ok"

    def format_header(self) -> str:
        """
        Single-line header for OutputConsole display.

        Example: '─── § Statics — 0.12s ──────────────────'
        """
        t = f"{self.execution_time_ms / 1000:.2f}s"
        label = f" § {self.title} — {t} "
        pad = max(0, 80 - len(label) - 4)
        return f"{'─' * 4}{label}{'─' * pad}"


# ---------------------------------------------------------------------------
# Section dependency record (internal)
# ---------------------------------------------------------------------------

@dataclass
class _SectionRecord:
    section_index: int
    names_written: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# SectionParser
# ---------------------------------------------------------------------------

class SectionParser:
    """
    Parses Python source text into a list of Section objects.

    A line that starts with '# %%' (after stripping) begins a new section.
    Everything before the first '# %%' is treated as Section 0 with title=''.
    A file with no '# %%' markers produces a single Section covering the
    entire file.

    The delimiter line itself is NOT included in the section's code — only
    the body lines between consecutive delimiters are.
    """

    MARKER = "# %%"

    @classmethod
    def parse(cls, source: str) -> list[Section]:
        """
        Parse source text into Section list.

        Parameters
        ----------
        source : str
            Full text of a .py file.

        Returns
        -------
        list[Section]
            Ordered list of sections. Never empty — a blank file returns
            [Section(index=0, title='', code='', line_start=1, line_end=1)].
        """
        lines = source.splitlines(keepends=False)
        if not lines:
            return [Section(index=0, title="", code="", line_start=1, line_end=1)]

        # Locate delimiter lines
        delimiters: list[tuple[int, str]] = []   # (0-based line index, title)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(cls.MARKER):
                title = stripped[len(cls.MARKER):].strip()
                delimiters.append((i, title))

        if not delimiters:
            # Entire file is one section
            return [Section(
                index=0, title="", code=source,
                line_start=1, line_end=len(lines),
            )]

        sections: list[Section] = []

        # Preamble before first delimiter (may be empty)
        if delimiters[0][0] > 0:
            preamble_lines = lines[: delimiters[0][0]]
            preamble_code = "\n".join(preamble_lines)
            if preamble_code.strip():
                sections.append(Section(
                    index=0,
                    title="",
                    code=preamble_code,
                    line_start=1,
                    line_end=delimiters[0][0],
                ))

        # Sections defined by delimiters
        for i, (del_line, title) in enumerate(delimiters):
            body_start = del_line + 1  # 0-based, first body line
            if i + 1 < len(delimiters):
                body_end = delimiters[i + 1][0]  # 0-based, exclusive
            else:
                body_end = len(lines)

            code = "\n".join(lines[body_start:body_end])
            sections.append(Section(
                index=len(sections),
                title=title,
                code=code,
                line_start=del_line + 1,   # 1-based
                line_end=body_end,          # 1-based (last line)
            ))

        if not sections:
            return [Section(index=0, title="", code="", line_start=1, line_end=1)]

        return sections


# ---------------------------------------------------------------------------
# SectionRunner
# ---------------------------------------------------------------------------

class SectionRunner:
    """
    Stateful section execution engine.

    Maintains a shared namespace across all sections.
    Each call to execute() returns a SectionResult.

    The namespace is seeded with AxisForge public API so user code
    does not need to import anything — Shaft, Bearing, solvers, np
    are available immediately.

    Stale tracking
    --------------
    After executing section N, sections N+1..M are marked stale in
    the _records dict. The namespace is NOT rolled back — the user
    is responsible for re-running upstream sections when needed.
    The UI reads stale_after() to highlight affected sections.

    Thread safety
    -------------
    Not thread-safe. Callers must serialise execute() calls.
    Use SectionWorker (script_editor.py) for background execution.
    """

    _SEEDED_NAMES: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self._namespace: dict[str, Any] = {}
        self._records: dict[int, _SectionRecord] = {}
        self._seed_namespace()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def parse(source: str) -> list[Section]:
        """Convenience wrapper — parse source into sections."""
        return SectionParser.parse(source)

    def execute(self, section: Section) -> SectionResult:
        """
        Execute a Section in the shared namespace.

        Parameters
        ----------
        section : Section
            Parsed section to execute.

        Returns
        -------
        SectionResult
            Full result including stdout, errors, timing, and validation.
        """
        if section.is_empty:
            return SectionResult(
                section_index=section.index,
                title=section.display_title,
                success=True,
                execution_time_ms=0.0,
            )

        names_before = set(self._namespace.keys())
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        t0 = time.perf_counter()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(  # noqa: S102
                    compile(
                        section.code,
                        f"<§ {section.display_title}>",
                        "exec",
                    ),
                    self._namespace,
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            names_after = set(self._namespace.keys())
            names_written = names_after - names_before - self._SEEDED_NAMES - {"__builtins__"}

            record = _SectionRecord(
                section_index=section.index,
                names_written=names_written,
            )
            self._records[section.index] = record

            validation_errors = self._run_validation()

            return SectionResult(
                section_index=section.index,
                title=section.display_title,
                success=True,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                names_written=names_written,
                validation_errors=validation_errors,
                execution_time_ms=elapsed_ms,
            )

        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            tb_lines = self._filter_traceback(section.display_title)
            return SectionResult(
                section_index=section.index,
                title=section.display_title,
                success=False,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback_lines=tb_lines,
                execution_time_ms=elapsed_ms,
            )

    def execute_all(self, sections: list[Section]) -> list[SectionResult]:
        """
        Execute all sections in order.

        Stops on first error — downstream sections are not executed.
        Returns a result for each executed section.
        """
        results: list[SectionResult] = []
        for section in sections:
            result = self.execute(section)
            results.append(result)
            if not result.success:
                break
        return results

    def reset(self) -> None:
        """Clear namespace and all section records. Full restart."""
        self._namespace.clear()
        self._records.clear()
        self._seed_namespace()

    def stale_after(self, section_index: int) -> list[int]:
        """
        Return sorted list of section indices that are stale after
        section_index was modified.

        Currently returns all recorded indices > section_index.
        Read-tracking is deferred to Phase 3.

        Used by the UI to dim/mark sections that may be out of date.
        """
        return sorted(i for i in self._records if i > section_index)

    def get_systems(self) -> list[Any]:
        """
        Return all MechanicalSystem instances in the namespace.

        Used by canvas and workspace panel to discover what to render.
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
        """
        Snapshot of the shared namespace.

        Returns a shallow copy — do not mutate.
        Excludes seeded names so the UI shows only user-defined variables.
        """
        return {
            k: v for k, v in self._namespace.items()
            if k not in self._SEEDED_NAMES and not k.startswith("_")
        }

    def get_full_namespace(self) -> dict[str, Any]:
        """Full namespace including seeded names. For debugging."""
        return dict(self._namespace)

    def get_name_to_section_map(self) -> dict[str, int]:
        """
        Return mapping of variable name → section index that last wrote it.

        Used by MainWindow to determine staleness per variable in WorkspacePanel.
        Public replacement for accessing _records directly.
        """
        result: dict[str, int] = {}
        for idx, record in self._records.items():
            for name in record.names_written:
                result[name] = idx
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _seed_namespace(self) -> None:
        """
        Pre-populate namespace with AxisForge public API.

        User sections can use Shaft, Bearing, etc. without imports.
        Import failures are silently skipped — runner works even if
        solvers are partially unavailable (e.g. running tests without
        the full package installed).
        """
        import sys
        import os
        import pathlib

        # ui/ is a sibling of axisforge/ (the Python package)
        # repo_root = parent of ui/
        # axisforge/ = repo_root/axisforge
        ui_dir    = pathlib.Path(__file__).parent
        repo_root = ui_dir.parent
        axisforge = repo_root / "axisforge"

        for p in (str(repo_root), str(axisforge)):
            if p not in sys.path:
                sys.path.insert(0, p)

        imports = [
            ("core.shaft",                ["Shaft", "ShaftSection", "Shoulder"]),
            ("core.components",           ["Bearing", "BearingType", "GearElement"]),
            ("core.loads",                ["RadialLoad", "AxialLoad", "TorqueLoad",
                                           "ExternalMoment", "LoadPlane"]),
            ("core.system",               ["MechanicalSystem"]),
            ("core.gear_system",          ["GearStage", "GearSystem"]),
            ("core.materials",            ["Material", "get_material",
                                           "AISI_1045", "AISI_4340", "S355", "CrMo42"]),
            ("solvers.shaft.statics",     ["StaticsSolver"]),
            ("solvers.shaft.static_failure", ["StaticFailureSolver"]),
            ("solvers.shaft.stress",      ["StressSolver"]),
            ("solvers.bearings.life",     ["BearingLifeSolver"]),
            ("solvers.gears.geometry",    ["GearSolver"]),
        ]

        seeded: set[str] = set()
        import importlib
        for module_path, names in imports:
            try:
                mod = importlib.import_module(module_path)
                for name in names:
                    obj = getattr(mod, name, None)
                    if obj is not None:
                        self._namespace[name] = obj
                        seeded.add(name)
            except Exception:  # noqa: BLE001
                pass

        # numpy — always useful
        try:
            import numpy as np
            self._namespace["np"] = np
            seeded.add("np")
        except ImportError:
            pass

        # matplotlib — force Agg backend before any import of pyplot.
        # This prevents pyplot from trying to start an interactive GUI loop
        # (which would block the Qt event loop). Figures are rendered
        # by PlotWindow via FigureCanvasQTAgg when the user double-clicks
        # a Figure variable in the Workspace.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self._namespace["plt"] = plt
            seeded.add("plt")
        except ImportError:
            pass

        # exporters — ui/exporters/ is a sibling package of this file
        try:
            import importlib
            import sys as _sys
            # Ensure ui/ parent is on path so 'exporters' resolves as ui.exporters
            # but we seed flat names directly
            exporters_path = str(ui_dir)
            if exporters_path not in _sys.path:
                _sys.path.insert(0, exporters_path)
            exp = importlib.import_module("exporters")
            for name in [
                "export_statics", "export_stress", "export_static_failure",
                "export_bearing", "export_gear_geometry", "export_gear_forces",
            ]:
                obj = getattr(exp, name, None)
                if obj is not None:
                    self._namespace[name] = obj
                    seeded.add(name)
        except Exception:  # noqa: BLE001
            pass

        self._SEEDED_NAMES = frozenset(seeded)

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
    def _filter_traceback(section_title: str) -> list[str]:
        """
        Extract traceback lines, stripping internal runner frames.

        Keeps only frames where filename matches the section tag,
        plus the final exception line.
        """
        lines = traceback.format_exc().splitlines()
        tag = f"<§ {section_title}>"
        filtered: list[str] = []
        include_next = False
        for line in lines:
            if tag in line:
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