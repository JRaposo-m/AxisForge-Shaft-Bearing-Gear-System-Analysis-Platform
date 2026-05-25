"""
axisforge/core/project_gen.py
Project file generation — UI-agnostic.

Responsibilities:
  - Create project structure (project.axf + scripts/full_pipeline.py only).
  - generate_shaft_file()  — called by create_shaft() seeded in runner namespace.
  - generate_bearing_file(), generate_gear_file() — called by post-exec scan.
  - generate_stage_file(), generate_gear_pair_file() — called by create_gear_pair().
  - scan_project() — filesystem snapshot for duplicate detection.

Behaviour on collision:
  - All generate_*() methods raise FileExistsError (not return None).
  - Callers catch and log to OutputConsole. User must fix the source.

No PySide6. No solver calls. No global state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template


# ---------------------------------------------------------------------------
# ProjectState — filesystem snapshot
# ---------------------------------------------------------------------------

@dataclass
class ProjectState:
    """
    Snapshot of what exists on disk under project root.

    Populated by ProjectManager.scan_project().
    """
    root: Path
    project_name: str = ""
    shaft_dirs: dict[str, Path] = field(default_factory=dict)
    stage_dirs: dict[str, Path] = field(default_factory=dict)
    bearing_labels: dict[str, set[str]] = field(default_factory=dict)
    gear_labels: dict[str, set[str]] = field(default_factory=dict)
    # stage_name -> number of gear_pair_N.py files found directly in stage dir
    gear_pair_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Skeleton templates
# ---------------------------------------------------------------------------

_TPL_FULL_PIPELINE = Template('''\
# scripts/full_pipeline.py
# $project_name — Pipeline Completo
# Gerado automaticamente por AxisForge ProjectManager
#
# Orquestrador puro: cria veios e executa ficheiros em ordem.
# _run() executa cada ficheiro no namespace partilhado do runner.
#
# Fluxo de desenvolvimento:
#   1. Adicionar create_shaft() para cada veio
#   2. Preencher cada shaft_N.py com geometria, rolamentos e engrenagens
#   3. Chamar create_gear_pair() para cada estágio
#   4. Preencher cada gear_pair_N.py com parâmetros da engrenagem
#   5. Adicionar _run() em ordem e o sumário final

import pathlib as _pathlib

def _run(_rel: str) -> None:
    """Executa um ficheiro .py no namespace partilhado do runner."""
    _p = _pathlib.Path.cwd() / _rel
    exec(compile(_p.read_text(encoding="utf-8"), str(_p), "exec"), globals())  # noqa: S102

print("=" * 62)
print(f" AxisForge — $project_name — Pipeline Completo")
print("=" * 62)

# %% 1. Criar veios (gera pastas e skeletons se não existirem)

create_shaft("shaft_1")   # TODO: adicionar mais veios conforme necessário
# create_shaft("shaft_2")

# %% 2. Engrenagem — Stage 1
# Descomente após criar gear_pair_1.py via create_gear_pair()
#
# _run("stages/stage_1/gear_pair_1.py")
# print(f"[gear] Ft={forces_stage_1.Ft:.1f}N  Fr={forces_stage_1.Fr:.1f}N")

# %% 3. Veios — estática, falha estática, fadiga
# _run("shafts/shaft_1/shaft_1.py")
# print(f"[shaft_1] M_max={sr_shaft_1.M_res_max:.0f}N·mm  nf_Good={stress_shaft_1.most_critical.nf_goodman:.3f}")

# %% 4. Rolamentos
# _run("shafts/shaft_1/elements/bearing_A/bearing_A.py")
# _run("shafts/shaft_1/elements/bearing_B/bearing_B.py")

# %% 5. Stage 1 — GearStage + GearSystem
# _run("stages/stage_1/stage_1.py")

# %% 6. Sumário final
# print()
# print("=" * 62)
# print(" SUMÁRIO FINAL")
# print("=" * 62)
''')

_TPL_SHAFT = Template('''\
# shafts/$shaft_name/$shaft_name.py
# $project_name — $shaft_name
# Gerado automaticamente por AxisForge ProjectManager
#
# Preencher: secções, apoios, cargas, velocidade, material
# Executar via full_pipeline.py → _run("shafts/$shaft_name/$shaft_name.py")

# %% Setup — geometry, system, bearings, gear
${shaft_name}_shaft = Shaft(name="$shaft_name")
${shaft_name}_shaft.add_section(ShaftSection(
    length=100.0, diameter=40.0, material_id="42CrMo4",  # TODO: geometria real
))
# ${shaft_name}_shaft.add_section(ShaftSection(
#     length=200.0, diameter=50.0, material_id="42CrMo4", label="§2",
#     shoulder_left =Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0),
#     shoulder_right=Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0),
# ))

sys_$shaft_name = MechanicalSystem(
    shaft=${shaft_name}_shaft,
    name="$shaft_name",
    speed_rpm=1450.0,           # TODO: velocidade real [rpm]
    design_life_hours=20000,    # TODO: vida de projecto [h]
    shaft_position=(0.0, 0.0),  # TODO: (y, z) centreline global [mm]
    shaft_origin_x=0.0,         # TODO: X global offset [mm]
)

sys_${shaft_name}.add_bearing(Bearing(
    label="A", position=0.0,    # TODO: posição [mm]
    C=30000.0, C0=18000.0,      # TODO: capacidade dinâmica e estática [N]
    arrangement="fixed",
))
sys_${shaft_name}.add_bearing(Bearing(
    label="B", position=100.0,  # TODO: posição [mm]
    C=30000.0, C0=18000.0,
    arrangement="floating",
))

# Engrenagem — injectada via gs.to_gear_element() após resolver gear_pair_N.py
# sys_${shaft_name}.add_gear(gs.to_gear_element(
#     position=50.0,              # TODO: posição [mm]
#     forces=forces_stage_1,      # TODO: GearForceResult do par correspondente
#     geometry=geo_stage_1,       # TODO: GearGeometryResult do par correspondente
#     label="TODO_label",
# ))

# Cargas externas (opcional)
# sys_${shaft_name}.add_load(RadialLoad(position=50.0, magnitude=1000.0, plane=LoadPlane.XY))
# sys_${shaft_name}.add_load(TorqueLoad(position=50.0, magnitude=50000.0))

# %% Estática
sr_$shaft_name = StaticsSolver().solve(sys_$shaft_name)

r = sr_${shaft_name}.reactions
print(f"REACÇÕES $shaft_name")
print(f"  A: XZ={r['A_xz']:+.1f}N  XY={r['A_xy']:+.1f}N  Fr={np.hypot(r['A_xz'],r['A_xy']):.1f}N")
print(f"  B: XZ={r['B_xz']:+.1f}N  XY={r['B_xy']:+.1f}N  Fr={np.hypot(r['B_xz'],r['B_xy']):.1f}N")
print(f"  M_res_max={sr_${shaft_name}.M_res_max:.0f}N·mm @ x={sr_${shaft_name}.x_at_M_res_max:.1f}mm  T_max={sr_${shaft_name}.T_max:.0f}N·mm")

export_statics(
    sr_$shaft_name,
    "shafts/$shaft_name/results/statics.txt",
    project="$project_name", element="$shaft_name", script="${shaft_name}.py",
)

# %% Falha estática
mat_$shaft_name = CrMo42  # TODO: material real

sf_$shaft_name = StaticFailureSolver().solve(sys_$shaft_name, sr_$shaft_name, mat_$shaft_name)

print(f"FALHA ESTÁTICA $shaft_name — {mat_${shaft_name}.material_id}  Sy={mat_${shaft_name}.Sy}MPa")
for s in sf_${shaft_name}.sections:
    print(f"  x={s.x:.1f}mm  d={s.diameter:.1f}mm  n_DE={s.n_DE:.2f}  n_MSS={s.n_MSS:.2f}  {s.risk_label}")

export_static_failure(
    sf_$shaft_name,
    "shafts/$shaft_name/results/static_failure.txt",
    project="$project_name", element="$shaft_name", script="${shaft_name}.py",
)

# %% Fadiga
stress_$shaft_name = StressSolver().solve(
    sys_$shaft_name, sr_$shaft_name, mat_$shaft_name,
    finish="machined", reliability_percent=99.0,
)

mc = stress_${shaft_name}.most_critical
print(f"FADIGA $shaft_name — Se_base={mat_${shaft_name}.endurance_limit:.0f}MPa")
print(f"  Crítica: x={mc.x:.0f}mm  nf_Goodman={mc.nf_goodman:.3f}  nf_ASME={mc.nf_asme:.3f}  ny={mc.ny:.3f}")

export_stress(
    stress_$shaft_name,
    "shafts/$shaft_name/results/stress.txt",
    project="$project_name", element="$shaft_name", script="${shaft_name}.py",
    mat=mat_$shaft_name,
)
''')

_TPL_BEARING = Template('''\
# shafts/$shaft_name/elements/bearing_$label/bearing_$label.py
# $project_name — $shaft_name — Rolamento $label
# Gerado automaticamente por AxisForge ProjectManager
#
# Depende de: sr_$shaft_name, sys_$shaft_name (de ${shaft_name}.py)

# %% Bearing life
bl = BearingLifeSolver()
bf = bl.extract_bearing_forces(sr_$shaft_name, sys_$shaft_name)

bearing_$label = sys_${shaft_name}.bearings[0]   # TODO: verificar índice
Fr_$label, Fa_$label = bf[bearing_${label}.label]

res_bearing_${label}_$shaft_name = bl.solve_bearing(
    bearing_$label, Fr_$label, Fa_$label,
    sys_${shaft_name}.speed_rpm,
    sys_${shaft_name}.design_life_hours,
)

print(f"Rolamento $label — $shaft_name  L10h={res_bearing_${label}_${shaft_name}.L10h:.0f}h  S0={res_bearing_${label}_${shaft_name}.S0:.3f}")

export_bearing(
    res_bearing_${label}_$shaft_name,
    "shafts/$shaft_name/elements/bearing_$label/results/life.txt",
    project="$project_name", element="${shaft_name}_bearing_$label", script="bearing_${label}.py",
    C=bearing_${label}.C, C0=bearing_${label}.C0,
)
''')

_TPL_GEAR_ELEMENT = Template('''\
# shafts/$shaft_name/elements/gear_$label/gear_$label.py
# $project_name — $shaft_name — Engrenagem $label
# Gerado automaticamente por AxisForge ProjectManager
#
# Nota: a geometria e forças desta engrenagem são calculadas em gear_pair_N.py
# Este ficheiro é apenas de referência/documentação local.
# Para usar as forças no veio, usar gs.to_gear_element() em ${shaft_name}.py.

print(f"Engrenagem $label — $shaft_name  (ver gear_pair correspondente para geo e forças)")
''')

_TPL_GEAR_PAIR = Template('''\
# stages/$stage_name/gear_pair_$pair_index.py
# $project_name — $stage_name — Par de engrenagens $pair_index
# Gerado automaticamente por AxisForge ProjectManager
#
# Preencher: parâmetros geométricos e torque de entrada
# Executar via full_pipeline.py → _run("stages/$stage_name/gear_pair_$pair_index.py")
#
# Variáveis exportadas para o namespace:
#   geo_stage_$pair_index    — GearGeometryResult
#   forces_stage_$pair_index — GearForceResult

# %% Geometria e forças
gs = GearSolver()

geo_stage_$pair_index = gs.compute_geometry(
    mn=4.5,          # TODO: módulo normal [mm]
    z1=16,           # TODO: dentes do pinhão
    z2=24,           # TODO: dentes da roda
    alpha_n_deg=20.0,
    beta_deg=0.0,    # TODO: ângulo de hélice [°] (0 = spur)
    al=None,         # TODO: distância de centros de trabalho [mm] ou None (calculado)
    x1=0.0,          # TODO: coeficiente de adendenda pinhão
    x2=0.0,          # TODO: coeficiente de adendenda roda
    b=50.0,          # TODO: largura de face [mm]
)

forces_stage_$pair_index = gs.compute_forces(
    T1_Nm=0.0,       # TODO: binário de entrada [N·m]
    geometry=geo_stage_$pair_index,
)

print(f"Gear pair $pair_index: u={geo_stage_${pair_index}.u:.4f}  "
      f"al={geo_stage_${pair_index}.al:.3f}mm  "
      f"Ft={forces_stage_${pair_index}.Ft:.1f}N  Fr={forces_stage_${pair_index}.Fr:.1f}N")
''')

_TPL_STAGE = Template('''\
# stages/$stage_name/$stage_name.py
# $project_name — $stage_name — GearStage + GearSystem
# Gerado automaticamente por AxisForge ProjectManager
#
# Depende de: geo_stage_N, forces_stage_N (de gear_pair_N.py)
#             sys_shaft_driver, sys_shaft_driven (de shaft_N.py)
# Executar via full_pipeline.py → _run("stages/$stage_name/$stage_name.py")

# %% GearStage
stage_$stage_index = GearStage(
    label="$stage_name",
    shaft_driver=sys_TODO_driver,      # TODO: MechanicalSystem do veio condutor
    shaft_driven=sys_TODO_driven,      # TODO: MechanicalSystem do veio conduzido
    geometry=geo_stage_$stage_index,   # TODO: GearGeometryResult de gear_pair_N.py
    forces=forces_stage_$stage_index,  # TODO: GearForceResult de gear_pair_N.py
    x_gear_driver_local=0.0,           # TODO: posição da engrenagem no veio condutor [mm]
    x_gear_driven_local=0.0,           # TODO: posição da engrenagem no veio conduzido [mm]
)

# %% GearSystem
gear_system = GearSystem(name="$project_name")
gear_system.add_stage(stage_$stage_index)
# gear_system.add_stage(stage_2)  # TODO: adicionar mais estágios se necessário
gear_system.propagate_speeds()

errors = gear_system.validate()
if errors:
    print("⚠ GearSystem erros:")
    for e in errors:
        print(f"  - {e}")
else:
    print(f"✓ {gear_system.summary()}")
''')


# ---------------------------------------------------------------------------
# project.axf — JSON metadata
# ---------------------------------------------------------------------------

def _write_axf(project_root: Path, name: str) -> Path:
    """Write project.axf metadata file."""
    axf = {
        "name": name,
        "version": "1.0",
        "created": datetime.now().isoformat(timespec="seconds"),
        "axisforge_version": "phase1",
        "description": "",
    }
    axf_path = project_root / "project.axf"
    axf_path.write_text(json.dumps(axf, indent=2, ensure_ascii=False), encoding="utf-8")
    return axf_path


def _read_axf(project_root: Path) -> dict:
    """Read project.axf. Returns empty dict if not found or invalid."""
    axf_path = project_root / "project.axf"
    if not axf_path.exists():
        return {}
    try:
        return json.loads(axf_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# ProjectManager
# ---------------------------------------------------------------------------

class ProjectManager:
    """
    Filesystem-level project manager.

    Collision policy: generate_*() methods raise FileExistsError.
    Callers catch and log — user must fix their source code.

    No Qt, no solver imports.
    """

    # ------------------------------------------------------------------
    # Project creation
    # ------------------------------------------------------------------

    @staticmethod
    def create_project(name: str, root: Path) -> Path:
        """
        Create minimal project structure.

        Structure created:
          name/
            project.axf
            scripts/
              full_pipeline.py

        No shaft directories — those are created by create_shaft().

        Returns
        -------
        Path
            Project root directory.

        Raises
        ------
        FileExistsError
            If the project directory already exists.
        """
        project_root = root / name
        if project_root.exists():
            raise FileExistsError(f"Directory already exists: {project_root}")

        (project_root / "scripts").mkdir(parents=True)
        (project_root / "shafts").mkdir()
        (project_root / "stages").mkdir()
        (project_root / "results").mkdir()

        _write_axf(project_root, name)

        pipeline = project_root / "scripts" / "full_pipeline.py"
        pipeline.write_text(
            _TPL_FULL_PIPELINE.substitute(project_name=name),
            encoding="utf-8",
        )

        return project_root

    # ------------------------------------------------------------------
    # Shaft generation — called by create_shaft() in runner namespace
    # ------------------------------------------------------------------

    @staticmethod
    def generate_shaft_file(
        shaft_name: str,
        root: Path,
        project_name: str = "",
    ) -> Path:
        """
        Create shafts/{shaft_name}/{shaft_name}.py skeleton.

        Also creates shafts/{shaft_name}/elements/ and results/ directories.

        Returns
        -------
        Path
            Path to the created .py file.

        Raises
        ------
        FileExistsError
            If shaft directory already exists. Caller logs to OutputConsole.
        ValueError
            If shaft_name does not match pattern shaft_N (enforced convention).
        """
        import re
        if not re.match(r'^shaft_\w+$', shaft_name):
            raise ValueError(
                f"shaft_name must match 'shaft_<id>' pattern, got '{shaft_name}'. "
                f"Examples: shaft_1, shaft_2, shaft_input."
            )

        shaft_dir = root / "shafts" / shaft_name
        if shaft_dir.exists():
            raise FileExistsError(
                f"Shaft '{shaft_name}' already exists at {shaft_dir}. "
                f"Remove create_shaft('{shaft_name}') from full_pipeline.py."
            )

        shaft_dir.mkdir(parents=True)
        (shaft_dir / "elements").mkdir()
        (shaft_dir / "results").mkdir()

        shaft_py = shaft_dir / f"{shaft_name}.py"
        shaft_py.write_text(
            _TPL_SHAFT.substitute(
                project_name=project_name or root.name,
                shaft_name=shaft_name,
            ),
            encoding="utf-8",
        )
        return shaft_py

    # ------------------------------------------------------------------
    # Element file generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_bearing_file(
        shaft_name: str,
        bearing_label: str,
        root: Path,
        project_name: str = "",
    ) -> Path:
        """
        Generate shafts/{shaft_name}/elements/bearing_{label}/bearing_{label}.py.

        Raises
        ------
        FileExistsError
            If file already exists — label collision or manual file present.
        """
        target_dir = root / "shafts" / shaft_name / "elements" / f"bearing_{bearing_label}"
        target_file = target_dir / f"bearing_{bearing_label}.py"

        if target_file.exists():
            raise FileExistsError(
                f"bearing_{bearing_label}.py already exists in '{shaft_name}'. "
                f"Use a different label or delete the existing file."
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "results").mkdir(exist_ok=True)

        target_file.write_text(
            _TPL_BEARING.substitute(
                project_name=project_name or root.name,
                shaft_name=shaft_name,
                label=bearing_label,
            ),
            encoding="utf-8",
        )
        return target_file

    @staticmethod
    def generate_gear_file(
        shaft_name: str,
        gear_label: str,
        root: Path,
        project_name: str = "",
    ) -> Path:
        """
        Generate shafts/{shaft_name}/elements/gear_{label}/gear_{label}.py.

        Raises
        ------
        FileExistsError
            If file already exists.
        """
        target_dir = root / "shafts" / shaft_name / "elements" / f"gear_{gear_label}"
        target_file = target_dir / f"gear_{gear_label}.py"

        if target_file.exists():
            raise FileExistsError(
                f"gear_{gear_label}.py already exists in '{shaft_name}'. "
                f"Use a different label or delete the existing file."
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "results").mkdir(exist_ok=True)

        target_file.write_text(
            _TPL_GEAR_ELEMENT.substitute(
                project_name=project_name or root.name,
                shaft_name=shaft_name,
                label=gear_label,
            ),
            encoding="utf-8",
        )
        return target_file

    @staticmethod
    def generate_gear_pair_file(
        stage_name: str,
        pair_index: int,
        root: Path,
        project_name: str = "",
    ) -> Path:
        """
        Generate stages/{stage_name}/gear_pair_{N}.py directly in stage dir.

        A gear can belong to multiple pairs — no restriction on reuse.
        pair_index is 1-based and sequential per stage.

        Raises
        ------
        FileExistsError
            If file already exists.
        """
        stage_dir = root / "stages" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "results").mkdir(exist_ok=True)

        target_file = stage_dir / f"gear_pair_{pair_index}.py"
        if target_file.exists():
            raise FileExistsError(
                f"gear_pair_{pair_index}.py already exists in '{stage_name}'."
            )

        target_file.write_text(
            _TPL_GEAR_PAIR.substitute(
                project_name=project_name or root.name,
                stage_name=stage_name,
                pair_index=pair_index,
            ),
            encoding="utf-8",
        )
        return target_file

    @staticmethod
    def generate_stage_file(
        stage_name: str,
        stage_index: int,
        root: Path,
        project_name: str = "",
    ) -> Path:
        """
        Generate stages/{stage_name}/{stage_name}.py skeleton.

        Raises
        ------
        FileExistsError
            If file already exists.
        """
        stage_dir = root / "stages" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        target_file = stage_dir / f"{stage_name}.py"
        if target_file.exists():
            raise FileExistsError(
                f"{stage_name}.py already exists in stages/{stage_name}/."
            )

        target_file.write_text(
            _TPL_STAGE.substitute(
                project_name=project_name or root.name,
                stage_name=stage_name,
                stage_index=stage_index,
            ),
            encoding="utf-8",
        )
        return target_file

    # ------------------------------------------------------------------
    # Disk scan
    # ------------------------------------------------------------------

    @staticmethod
    def scan_project(root: Path) -> ProjectState:
        """
        Scan project root and build ProjectState from filesystem.

        Discovers shafts, stages, bearing labels, gear labels, gear pairs.
        gear_pair_counts: counts gear_pair_N.py files directly in stage dir.
        """
        axf = _read_axf(root)
        state = ProjectState(root=root, project_name=axf.get("name", root.name))

        shafts_dir = root / "shafts"
        if shafts_dir.is_dir():
            for shaft_dir in sorted(shafts_dir.iterdir()):
                if not shaft_dir.is_dir():
                    continue
                name = shaft_dir.name
                state.shaft_dirs[name] = shaft_dir
                state.bearing_labels[name] = set()
                state.gear_labels[name] = set()

                elements_dir = shaft_dir / "elements"
                if elements_dir.is_dir():
                    for el_dir in elements_dir.iterdir():
                        if not el_dir.is_dir():
                            continue
                        el_name = el_dir.name
                        if el_name.startswith("bearing_"):
                            state.bearing_labels[name].add(el_name[len("bearing_"):])
                        elif el_name.startswith("gear_"):
                            state.gear_labels[name].add(el_name[len("gear_"):])

        stages_dir = root / "stages"
        if stages_dir.is_dir():
            for stage_dir in sorted(stages_dir.iterdir()):
                if not stage_dir.is_dir():
                    continue
                s_name = stage_dir.name
                state.stage_dirs[s_name] = stage_dir
                # Count gear_pair_N.py files directly in stage dir
                count = sum(
                    1 for f in stage_dir.iterdir()
                    if f.is_file() and f.name.startswith("gear_pair_") and f.suffix == ".py"
                )
                state.gear_pair_counts[s_name] = count

        return state

    # ------------------------------------------------------------------
    # create_shaft — seeded into runner namespace
    # ------------------------------------------------------------------

    @staticmethod
    def make_create_shaft(root: Path, project_name: str, console_log_fn=None):
        """
        Factory returning a create_shaft() closure bound to project root.

        Seeded into SectionRunner namespace by runner.py.
        On collision: logs error to OutputConsole, does NOT raise in user code.
        """
        pm = ProjectManager()

        def _log(msg: str) -> None:
            if console_log_fn is not None:
                console_log_fn(msg)

        def create_shaft(shaft_name: str) -> None:
            """
            Create shafts/{shaft_name}/ with shaft_name.py skeleton.

            If shaft already exists: logs error, does nothing.
            Convention: shaft_name must match 'shaft_<id>' (e.g. shaft_1, shaft_2).
            """
            try:
                result = pm.generate_shaft_file(shaft_name, root, project_name)
                _log(f"[ProjectGen] Created: {result.relative_to(root)}")
            except FileExistsError as exc:
                _log(f"[ProjectGen] ERRO: {exc}")
            except ValueError as exc:
                _log(f"[ProjectGen] ERRO: {exc}")

        return create_shaft

    # ------------------------------------------------------------------
    # create_gear_pair — seeded into runner namespace
    # ------------------------------------------------------------------

    @staticmethod
    def make_create_gear_pair(root: Path, project_name: str, console_log_fn=None):
        """
        Factory returning a create_gear_pair() closure bound to project root.

        Seeded into SectionRunner namespace by runner.py.
        A gear can belong to multiple pairs — no restriction enforced here.
        """
        pm = ProjectManager()

        def _log(msg: str) -> None:
            if console_log_fn is not None:
                console_log_fn(msg)

        def create_gear_pair(
            sys_driver,
            gear_label_driver: str,
            sys_driven,
            gear_label_driven: str,
            stage_name: str = "",
        ) -> None:
            """
            Generate stages/{stage_name}/gear_pair_N.py and stage_name.py skeletons.

            Parameters
            ----------
            sys_driver : MechanicalSystem
                Driver shaft system.
            gear_label_driver : str
                Label of the driving gear on sys_driver.
            sys_driven : MechanicalSystem
                Driven shaft system.
            gear_label_driven : str
                Label of the driven gear on sys_driven.
            stage_name : str
                Stage directory name (e.g. "stage_1").
                If empty, auto-numbered from existing stages on disk.

            Side effects
            ------------
            Creates stages/{stage_name}/gear_pair_N.py and stage_name.py.
            Logs to OutputConsole. Does not raise.
            """
            # Auto-number stage
            if not stage_name:
                stages_dir = root / "stages"
                n = len([d for d in stages_dir.iterdir() if d.is_dir()]) if stages_dir.is_dir() else 0
                stage_name = f"stage_{n + 1}"

            state = pm.scan_project(root)
            pair_index = state.gear_pair_counts.get(stage_name, 0) + 1

            # gear_pair_N.py
            try:
                gp_py = pm.generate_gear_pair_file(stage_name, pair_index, root, project_name)
                _log(f"[ProjectGen] Created: {gp_py.relative_to(root)}")
            except FileExistsError as exc:
                _log(f"[ProjectGen] ERRO: {exc}")

            # stage_name.py
            try:
                st_py = pm.generate_stage_file(stage_name, pair_index, root, project_name)
                _log(f"[ProjectGen] Created: {st_py.relative_to(root)}")
            except FileExistsError:
                pass  # stage file already exists — normal on second pair call

        return create_gear_pair

    # ------------------------------------------------------------------
    # project.axf helpers
    # ------------------------------------------------------------------

    @staticmethod
    def read_project_name(root: Path) -> str:
        """Return project name from project.axf, or directory name as fallback."""
        return _read_axf(root).get("name", root.name)