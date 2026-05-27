# shafts/shaft_2/shaft_2.py
# demo_reducer_2stages — shaft_2
# Gerado automaticamente por AxisForge ProjectManager
#
# Preencher: secções, apoios, cargas, velocidade, material
# Executar via full_pipeline.py → _run("shafts/shaft_2/shaft_2.py")

# %% Setup — geometry, system, bearings, gear
shaft_2_shaft = Shaft(name="shaft_2")
shaft_2_shaft.add_section(ShaftSection(
    length=100.0, diameter=40.0, material_id="42CrMo4",  # TODO: geometria real
))
shaft_2_shaft.add_section(ShaftSection(
    length=200.0, diameter=50.0, material_id="42CrMo4", label="§2",
    shoulder_left =Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0),
    shoulder_right=Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0),
))
shaft_2_shaft.add_section(ShaftSection(
    length=100.0, diameter=40.0, material_id="42CrMo4",  # TODO: geometria real
))

sys_shaft_2 = MechanicalSystem(
    shaft=shaft_2_shaft,
    name="shaft_2",
    speed_rpm=1450.0,           # TODO: velocidade real [rpm]
    design_life_hours=20000,    # TODO: vida de projecto [h]
    shaft_position=(0.0, 90),  # TODO: (y, z) centreline global [mm]
    shaft_origin_x=0.0,         # TODO: X global offset [mm]
)

sys_shaft_2.add_bearing(Bearing(
    label="A", position=0.0,    # TODO: posição [mm]
    C=30000.0, C0=18000.0,      # TODO: capacidade dinâmica e estática [N]
    arrangement="fixed",
))
sys_shaft_2.add_bearing(Bearing(
    label="B", position=400.0,  # TODO: posição [mm]
    C=30000.0, C0=18000.0,
    arrangement="floating",
))

# Engrenagem — injectada via gs.to_gear_element() após resolver gear_pair_N.py
sys_shaft_2.add_gear(gs.to_gear_element(
    position=150.0,              # TODO: posição [mm]
    forces=forces_stage_1,      # TODO: GearForceResult do par correspondente
    geometry=geo_stage_1,       # TODO: GearGeometryResult do par correspondente
    label="Wheel_S1",
))

sys_shaft_2.add_gear(gs.to_gear_element(
	position=225.0,
	forces=forces_stage_2,
	geometry=geo_stage_2,
	label="Pinion_S2",
))

# Cargas externas (opcional)
# sys_shaft_2.add_load(RadialLoad(position=50.0, magnitude=1000.0, plane=LoadPlane.XY))
# sys_shaft_2.add_load(TorqueLoad(position=50.0, magnitude=50000.0))

# %% Estática
sr_shaft_2 = StaticsSolver().solve(sys_shaft_2)

r = sr_shaft_2.reactions
print(f"REACÇÕES shaft_2")
print(f"  A: XZ={r['A_xz']:+.1f}N  XY={r['A_xy']:+.1f}N  Fr={np.hypot(r['A_xz'],r['A_xy']):.1f}N")
print(f"  B: XZ={r['B_xz']:+.1f}N  XY={r['B_xy']:+.1f}N  Fr={np.hypot(r['B_xz'],r['B_xy']):.1f}N")
print(f"  M_res_max={sr_shaft_2.M_res_max:.0f}N·mm @ x={sr_shaft_2.x_at_M_res_max:.1f}mm  T_max={sr_shaft_2.T_max:.0f}N·mm")

export_statics(
    sr_shaft_2,
    "shafts/shaft_2/results/statics.txt",
    project="demo_reducer_2stages", element="shaft_2", script="shaft_2.py",
)

# %% Falha estática
mat_shaft_2 = CrMo42  # TODO: material real

sf_shaft_2 = StaticFailureSolver().solve(sys_shaft_2, sr_shaft_2, mat_shaft_2)

print(f"FALHA ESTÁTICA shaft_2 — {mat_shaft_2.material_id}  Sy={mat_shaft_2.Sy}MPa")
for s in sf_shaft_2.sections:
    print(f"  x={s.x:.1f}mm  d={s.diameter:.1f}mm  n_DE={s.n_DE:.2f}  n_MSS={s.n_MSS:.2f}  {s.risk_label}")

export_static_failure(
    sf_shaft_2,
    "shafts/shaft_2/results/static_failure.txt",
    project="demo_reducer_2stages", element="shaft_2", script="shaft_2.py",
)

# %% Fadiga
stress_shaft_2 = StressSolver().solve(
    sys_shaft_2, sr_shaft_2, mat_shaft_2,
    finish="machined", reliability_percent=99.0,
)

mc = stress_shaft_2.most_critical
print(f"FADIGA shaft_2 — Se_base={mat_shaft_2.endurance_limit:.0f}MPa")
print(f"  Crítica: x={mc.x:.0f}mm  nf_Goodman={mc.nf_goodman:.3f}  nf_ASME={mc.nf_asme:.3f}  ny={mc.ny:.3f}")

export_stress(
    stress_shaft_2,
    "shafts/shaft_2/results/stress.txt",
    project="demo_reducer_2stages", element="shaft_2", script="shaft_2.py",
    mat=mat_shaft_2,
)
