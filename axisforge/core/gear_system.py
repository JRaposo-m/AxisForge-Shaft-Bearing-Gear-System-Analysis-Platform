"""
core/gear_system.py
GearStage and GearSystem — multi-shaft gear transmission topology.

Responsibilities:
  - Model a directed graph of MechanicalSystem nodes connected by GearStages.
  - Validate geometric consistency (centre distance, X-contact alignment).
  - Propagate rotational speeds through the transmission chain.
  - Detect topological errors (cycles, orphaned shafts, missing stages).

This module contains ONLY data structures and validation logic.
No solvers, no GUI, no database calls.

Coordinate conventions:
  x_local  : axial position within a shaft (input to all solvers).
  x_global : shaft_origin_x + x_local — used for contact alignment only.
  (y, z)   : shaft_position — transverse position of shaft centreline [mm].

Units: mm, N, N·mm, rpm throughout.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.system import MechanicalSystem
from models.gear_result import GearGeometryResult, GearForceResult


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_TOL_DISTANCE_MM: float = 0.5    # centre-distance geometric tolerance [mm]
_TOL_X_CONTACT_MM: float = 0.5   # X-contact alignment tolerance [mm]
_TOL_SPEED_REL: float = 1e-4     # relative speed consistency tolerance [-]


# ---------------------------------------------------------------------------
# GearStage
# ---------------------------------------------------------------------------

@dataclass
class GearStage:
    """
    A single gear mesh — directed link between two MechanicalSystem shafts.

    The stage records which shaft is driver (input) and which is driven
    (output), together with the geometric and force results already
    computed by GearSolver.

    The X-contact positions (x_gear_driver_local, x_gear_driven_local)
    must be supplied explicitly so that validate() can check that both
    gears share the same global X coordinate.

    Parameters
    ----------
    label : str
        Human-readable identifier (e.g. "Stage_1").
    shaft_driver : MechanicalSystem
        Input shaft for this stage.
    shaft_driven : MechanicalSystem
        Output shaft for this stage.
    geometry : GearGeometryResult
        Output of GearSolver.compute_geometry(). Provides al, z1, z2.
    forces : GearForceResult
        Output of GearSolver.compute_forces(). Forces act on shaft_driver
        (pinion). Driven shaft receives reactive forces (same magnitude,
        opposite sign) — enforced by GearSystem conventions, not stored here.
    x_gear_driver_local : float
        Axial position of the driver gear in shaft_driver local coords [mm].
    x_gear_driven_local : float
        Axial position of the driven gear in shaft_driven local coords [mm].
    """

    label: str
    shaft_driver: MechanicalSystem
    shaft_driven: MechanicalSystem
    geometry: GearGeometryResult
    forces: GearForceResult
    x_gear_driver_local: float = 0.0
    x_gear_driven_local: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ratio(self) -> float:
        """Transmission ratio i = z2/z1 = n_driver / n_driven."""
        return self.geometry.z2 / self.geometry.z1

    @property
    def centre_distance_geometric(self) -> float:
        """
        Centre distance computed from shaft_position vectors [mm].
        Must equal geometry.al within _TOL_DISTANCE_MM.
        """
        dy = (self.shaft_driven.shaft_position[0]
              - self.shaft_driver.shaft_position[0])
        dz = (self.shaft_driven.shaft_position[1]
              - self.shaft_driver.shaft_position[1])
        return math.sqrt(dy**2 + dz**2)

    @property
    def x_gear_driver_global(self) -> float:
        """Global X position of the driver gear [mm]."""
        return self.shaft_driver.shaft_origin_x + self.x_gear_driver_local

    @property
    def x_gear_driven_global(self) -> float:
        """Global X position of the driven gear [mm]."""
        return self.shaft_driven.shaft_origin_x + self.x_gear_driven_local

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Return all consistency errors for this stage.

        Checks:
          1. Centre distance (Y,Z) vs geometry.al.
          2. X-contact alignment (driver and driven share same x_global).
          3. Speed ratio consistency (if both speeds are set and non-zero).
          4. Driver ≠ driven.
        """
        errors: list[str] = []

        if self.shaft_driver is self.shaft_driven:
            errors.append(
                f"[{self.label}] shaft_driver and shaft_driven are the same object."
            )
            return errors   # further checks are meaningless

        # 1. Centre distance
        al_geom = self.centre_distance_geometric
        al_spec = self.geometry.al
        if abs(al_geom - al_spec) > _TOL_DISTANCE_MM:
            errors.append(
                f"[{self.label}] Centre distance mismatch: "
                f"geometric = {al_geom:.3f} mm, "
                f"geometry.al = {al_spec:.3f} mm "
                f"(tolerance ±{_TOL_DISTANCE_MM} mm)."
            )

        # 2. X-contact alignment
        dx = abs(self.x_gear_driver_global - self.x_gear_driven_global)
        if dx > _TOL_X_CONTACT_MM:
            errors.append(
                f"[{self.label}] X-contact misalignment: "
                f"driver x_global = {self.x_gear_driver_global:.3f} mm, "
                f"driven x_global = {self.x_gear_driven_global:.3f} mm "
                f"(Δ = {dx:.3f} mm, tolerance ±{_TOL_X_CONTACT_MM} mm)."
            )

        # 3. Speed ratio consistency (both speeds must be set and > 0)
        n1 = self.shaft_driver.speed_rpm
        n2 = self.shaft_driven.speed_rpm
        if n1 > 0.0 and n2 > 0.0:
            expected_n2 = n1 / self.ratio
            rel_err = abs(n2 - expected_n2) / expected_n2
            if rel_err > _TOL_SPEED_REL:
                errors.append(
                    f"[{self.label}] Speed inconsistency: "
                    f"driver = {n1:.2f} rpm, driven = {n2:.2f} rpm, "
                    f"expected driven = {expected_n2:.2f} rpm (i = {self.ratio:.4f})."
                )

        return errors


# ---------------------------------------------------------------------------
# GearSystem
# ---------------------------------------------------------------------------

@dataclass
class GearSystem:
    """
    Complete gear transmission: N shafts, M gear stages.

    The system is modelled as a directed acyclic graph (DAG):
      nodes  = MechanicalSystem (shafts)
      edges  = GearStage (gear meshes)

    Supports:
      - Single-stage reducer (2 shafts, 1 stage).
      - Multi-stage reducer (N shafts, N-1 stages in series).
      - Compound gear trains (shared intermediate shafts, multiple meshes).

    Constraints (Phase 1):
      - Each shaft must have exactly 2 bearing supports.
      - Graph must be acyclic (DAG).
      - At most one input shaft (in-degree 0).
      - At most one output shaft (out-degree 0).

    Speed propagation:
      - Input shaft speed_rpm must be set before calling propagate_speeds().
      - propagate_speeds() performs BFS from input_shaft and writes
        speed_rpm to all reachable driven shafts.
      - Shafts not reachable from the input are flagged in validate().
    """

    name: str
    shafts: list[MechanicalSystem] = field(default_factory=list)
    stages: list[GearStage] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_shaft(self, shaft: MechanicalSystem) -> None:
        """Register a shaft. Duplicate objects (by identity) are silently ignored."""
        if not any(sh is shaft for sh in self.shafts):
            self.shafts.append(shaft)

    def add_stage(self, stage: GearStage) -> None:
        """
        Register a gear stage.

        The driver and driven shafts are auto-registered if not already
        present — this prevents silent topology errors from missing add_shaft
        calls.
        """
        self.add_shaft(stage.shaft_driver)
        self.add_shaft(stage.shaft_driven)
        self.stages.append(stage)

    # ------------------------------------------------------------------
    # Topology queries
    # ------------------------------------------------------------------

    def input_shaft(self) -> Optional[MechanicalSystem]:
        """
        Return the shaft that appears as driver but never as driven.
        Returns None if no unique input exists (cycle or disconnected graph).
        """
        driven_ids = {id(s.shaft_driven) for s in self.stages}
        candidates = [sh for sh in self.shafts if id(sh) not in driven_ids]
        return candidates[0] if len(candidates) == 1 else None

    def output_shaft(self) -> Optional[MechanicalSystem]:
        """
        Return the shaft that appears as driven but never as driver.
        Returns None if no unique output exists.
        """
        driver_ids = {id(s.shaft_driver) for s in self.stages}
        candidates = [sh for sh in self.shafts if id(sh) not in driver_ids]
        return candidates[0] if len(candidates) == 1 else None

    def _adjacency(self) -> dict[int, list[tuple[MechanicalSystem, "GearStage"]]]:
        """
        Build adjacency list keyed by id(shaft).
        MechanicalSystem is a mutable dataclass with no __hash__ — cannot be
        used as a dict key or set element directly.
        """
        adj: dict[int, list[tuple[MechanicalSystem, GearStage]]] = {
            id(sh): [] for sh in self.shafts
        }
        for stage in self.stages:
            adj[id(stage.shaft_driver)].append((stage.shaft_driven, stage))
        return adj

    def _has_cycle(self) -> bool:
        """
        Kahn's algorithm — return True if the directed graph contains a cycle.
        Uses id()-keyed structures throughout.
        """
        in_degree: dict[int, int] = {id(sh): 0 for sh in self.shafts}
        adj = self._adjacency()
        for sh in self.shafts:
            for driven, _ in adj[id(sh)]:
                in_degree[id(driven)] += 1

        queue: deque[MechanicalSystem] = deque(
            sh for sh in self.shafts if in_degree[id(sh)] == 0
        )
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for driven, _ in adj[id(node)]:
                in_degree[id(driven)] -= 1
                if in_degree[id(driven)] == 0:
                    queue.append(driven)

        return visited != len(self.shafts)

    # ------------------------------------------------------------------
    # Speed propagation
    # ------------------------------------------------------------------

    def propagate_speeds(self) -> None:
        """
        BFS from input_shaft — write speed_rpm to all downstream shafts.

        n_driven = n_driver / ratio   (ratio = z2/z1)

        Raises
        ------
        ValueError
            If input_shaft() is None (cycle or disconnected graph).
            If input shaft speed_rpm is 0 or negative.
        """
        inp = self.input_shaft()
        if inp is None:
            raise ValueError(
                f"[{self.name}] Cannot propagate speeds: no unique input shaft. "
                f"Check for cycles or multiple roots."
            )
        if inp.speed_rpm <= 0.0:
            raise ValueError(
                f"[{self.name}] Input shaft '{inp.name}' speed_rpm must be > 0, "
                f"got {inp.speed_rpm}."
            )

        adj = self._adjacency()
        queue: deque[MechanicalSystem] = deque([inp])
        visited_ids: set[int] = {id(inp)}

        while queue:
            driver = queue.popleft()
            for driven, stage in adj[id(driver)]:
                driven.speed_rpm = driver.speed_rpm / stage.ratio
                if id(driven) not in visited_ids:
                    visited_ids.add(id(driven))
                    queue.append(driven)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Return all system-level consistency errors.

        Checks (in order):
          1. Minimum 2 shafts and 1 stage.
          2. All stage shafts registered in self.shafts.
          3. Each shaft has exactly 2 bearings (Phase 1 constraint).
          4. No cycles in the transmission graph.
          5. Unique input and output shaft.
          6. All shafts reachable from input.
          7. Per-stage validation (centre distance, X-contact, speeds).
        """
        errors: list[str] = []

        # 1. Minimum structure
        if len(self.shafts) < 2:
            errors.append(
                f"[{self.name}] GearSystem requires ≥ 2 shafts, "
                f"got {len(self.shafts)}."
            )
        if not self.stages:
            errors.append(f"[{self.name}] GearSystem has no gear stages.")

        if errors:
            return errors   # topology checks below require ≥ 2 shafts + ≥ 1 stage

        # 2. Stage shaft membership
        shaft_ids = {id(sh) for sh in self.shafts}
        for stage in self.stages:
            if id(stage.shaft_driver) not in shaft_ids:
                errors.append(
                    f"[{stage.label}] shaft_driver '{stage.shaft_driver.name}' "
                    f"is not registered in GearSystem.shafts."
                )
            if id(stage.shaft_driven) not in shaft_ids:
                errors.append(
                    f"[{stage.label}] shaft_driven '{stage.shaft_driven.name}' "
                    f"is not registered in GearSystem.shafts."
                )

        # 3. Each shaft must have exactly 2 bearings (Phase 1)
        for sh in self.shafts:
            n = len(sh.bearings)
            if n != 2:
                errors.append(
                    f"Shaft '{sh.name}' has {n} bearing(s); "
                    f"exactly 2 required in Phase 1."
                )

        # 4. Cycle detection
        if self._has_cycle():
            errors.append(
                f"[{self.name}] Transmission graph contains a cycle. "
                f"GearSystem must be a DAG."
            )
            return errors   # propagation-based checks below are invalid with cycles

        # 5. Unique input / output
        inp = self.input_shaft()
        if inp is None:
            errors.append(
                f"[{self.name}] No unique input shaft found "
                f"(shafts with in-degree 0: check topology)."
            )

        out = self.output_shaft()
        if out is None:
            errors.append(
                f"[{self.name}] No unique output shaft found "
                f"(shafts with out-degree 0: check topology)."
            )

        # 6. Reachability from input
        if inp is not None:
            adj = self._adjacency()
            visited_ids: set[int] = set()
            queue: deque[MechanicalSystem] = deque([inp])
            while queue:
                node = queue.popleft()
                visited_ids.add(id(node))
                for driven, _ in adj[id(node)]:
                    if id(driven) not in visited_ids:
                        queue.append(driven)
            unreachable = [sh for sh in self.shafts if id(sh) not in visited_ids]
            for sh in unreachable:
                errors.append(
                    f"Shaft '{sh.name}' is not reachable from input shaft "
                    f"'{inp.name}'."
                )

        # 7. Per-stage validation
        for stage in self.stages:
            errors.extend(stage.validate())

        return errors

    def validate_or_raise(self) -> None:
        """Raise ValueError if validate() returns any errors."""
        errors = self.validate()
        if errors:
            raise ValueError(
                f"GearSystem '{self.name}' validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """One-line summary for CLI / logging."""
        inp = self.input_shaft()
        out = self.output_shaft()
        total_ratio = 1.0
        for stage in self.stages:
            total_ratio *= stage.ratio
        return (
            f"{self.name}: {len(self.shafts)} shafts, "
            f"{len(self.stages)} stage(s), "
            f"i_total = {total_ratio:.4f}, "
            f"input = '{inp.name if inp else '?'}', "
            f"output = '{out.name if out else '?'}'"
        )

    def __repr__(self) -> str:
        return (
            f"GearSystem(name={self.name!r}, "
            f"shafts={len(self.shafts)}, "
            f"stages={len(self.stages)})"
        )