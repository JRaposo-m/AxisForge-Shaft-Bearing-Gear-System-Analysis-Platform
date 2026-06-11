# shafts/shaft_1/shaft_1.py
# demo_reducer_2stages — shaft_1
# Gerado automaticamente por AxisForge ProjectManager
#
# Preencher: secções, apoios, cargas, velocidade, material
# Executar via full_pipeline.py → _run("shafts/shaft_1/shaft_1.py")

# %% Setup — geometry, system, bearings, gear
shaft_1_shaft = Shaft(name="shaft_1")
shaft_1_shaft.add_section(ShaftSection(
    length=100.0, diameter=40.0, material_id="42CrMo4",  # TODO: geometria real
))
shaft_1_shaft.add_section(ShaftSection(
    length=200.0, diameter=50.0, material_id="42CrMo4", label="§2",
    shoulder_left =Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0),
    shoulder_right=Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0),
))

shaft_1_shaft.add_section(ShaftSection(
    length=100.0, diameter=40.0, material_id="42CrMo4",
))

sys_shaft_1 = MechanicalSystem(
    shaft=shaft_1_shaft,
    name="shaft_1",
    speed_rpm=1450.0,           # TODO: velocidade real [rpm]
    design_life_hours=20000,    # TODO: vida de projecto [h]
    shaft_position=(0.0, 0.0),  # TODO: (y, z) centreline global [mm]
    shaft_origin_x=0.0,         # TODO: X global offset [mm]
)

sys_shaft_1.add_bearing(Bearing(
    label="A", position=0.0,    # TODO: posição [mm]
    C=30000.0, C0=18000.0,      # TODO: capacidade dinâmica e estática [N]
    arrangement="fixed",
))
sys_shaft_1.add_bearing(Bearing(
    label="B", position=400.0,  # TODO: posição [mm]
    C=30000.0, C0=18000.0,
    arrangement="floating",
))

# Engrenagem — injectada via gs.to_gear_element() após resolver gear_pair_N.py
sys_shaft_1.add_gear(gs.to_gear_element(
    position=150.0,              # TODO: posição [mm]
    forces=forces_stage_1,      # TODO: GearForceResult do par correspondente
    geometry=geo_stage_1,       # TODO: GearGeometryResult do par correspondente
    label="Pinion_S1",
))

# Cargas externas (opcional)
# sys_shaft_1.add_load(RadialLoad(position=50.0, magnitude=1000.0, plane=LoadPlane.XY))
# sys_shaft_1.add_load(TorqueLoad(position=50.0, magnitude=50000.0))

# %% Estática
sr_shaft_1 = StaticsSolver().solve(sys_shaft_1)

r = sr_shaft_1.reactions
print(f"REACÇÕES shaft_1")
print(f"  A: XZ={r['A_xz']:+.1f}N  XY={r['A_xy']:+.1f}N  Fr={np.hypot(r['A_xz'],r['A_xy']):.1f}N")
print(f"  B: XZ={r['B_xz']:+.1f}N  XY={r['B_xy']:+.1f}N  Fr={np.hypot(r['B_xz'],r['B_xy']):.1f}N")
print(f"  M_res_max={sr_shaft_1.M_res_max:.0f}N·mm @ x={sr_shaft_1.x_at_M_res_max:.1f}mm  T_max={sr_shaft_1.T_max:.0f}N·mm")

export_statics(
    sr_shaft_1,
    "shafts/shaft_1/results/statics.txt",
    project="demo_reducer_2stages", element="shaft_1", script="shaft_1.py",
)

# %% Falha estática
mat_shaft_1 = CrMo42  # TODO: material real

sf_shaft_1 = StaticFailureSolver().solve(sys_shaft_1, sr_shaft_1, mat_shaft_1)

print(f"FALHA ESTÁTICA shaft_1 — {mat_shaft_1.material_id}  Sy={mat_shaft_1.Sy}MPa")
for s in sf_shaft_1.sections:
    print(f"  x={s.x:.1f}mm  d={s.diameter:.1f}mm  n_DE={s.n_DE:.2f}  n_MSS={s.n_MSS:.2f}  {s.risk_label}")

export_static_failure(
    sf_shaft_1,
    "shafts/shaft_1/results/static_failure.txt",
    project="demo_reducer_2stages", element="shaft_1", script="shaft_1.py",
)

# %% Fadiga
stress_shaft_1 = StressSolver().solve(
    sys_shaft_1, sr_shaft_1, mat_shaft_1,
    finish="machined", reliability_percent=99.0,
)

mc = stress_shaft_1.most_critical
print(f"FADIGA shaft_1 — Se_base={mat_shaft_1.endurance_limit:.0f}MPa")
print(f"  Crítica: x={mc.x:.0f}mm  nf_Goodman={mc.nf_goodman:.3f}  nf_ASME={mc.nf_asme:.3f}  ny={mc.ny:.3f}")

export_stress(
    stress_shaft_1,
    "shafts/shaft_1/results/stress.txt",
    project="demo_reducer_2stages", element="shaft_1", script="shaft_1.py",
    mat=mat_shaft_1,
)
