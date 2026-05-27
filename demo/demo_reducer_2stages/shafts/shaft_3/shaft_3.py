# shafts/shaft_3/shaft_3.py
# demo_reducer_2stages — shaft_3
#
# Executar via full_pipeline.py → _run("shafts/shaft_3/shaft_3.py")
# Depende de: geo_stage_2, forces_stage_2, sys_shaft_2

# %% Setup — geometry, system, bearings, gear

shaft_3_shaft = Shaft(name="shaft_3")
shaft_3_shaft.add_section(ShaftSection(
    length=100.0, diameter=40.0, material_id="42CrMo4", label="§1",  # TODO: geometria real
))
shaft_3_shaft.add_section(ShaftSection(
    length=200.0, diameter=50.0, material_id="42CrMo4", label="§2",
    shoulder_left =Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0),
    shoulder_right=Shoulder(fillet_radius=2.0, diameter_large=50.0, diameter_small=40.0),
))
shaft_3_shaft.add_section(ShaftSection(
    length=100.0, diameter=40.0, material_id="42CrMo4", label="§3",  # TODO: geometria real
))

# Posicionamento automático
_x_gear_driven_local_s3 = 50.0   # posição da engrenagem no shaft_3 [mm] — única entrada
_x_gear_global_s3       = sys_shaft_2.shaft_origin_x + sys_shaft_2.gears[1].position
_shaft_3_origin_x       = _x_gear_global_s3 - _x_gear_driven_local_s3
_shaft_3_z              = sys_shaft_2.shaft_position[1] + geo_stage_2.al

print(f"Shaft_3 — Posicionamento automático:")
print(f"  x_gear_global  = {_x_gear_global_s3:.1f} mm")
print(f"  shaft_origin_x = {_shaft_3_origin_x:.1f} mm")
print(f"  shaft_position = (0.0, {_shaft_3_z:.3f} mm)  [al = {geo_stage_2.al:.3f} mm]")

sys_shaft_3 = MechanicalSystem(
    shaft=shaft_3_shaft,
    name="shaft_3",
    speed_rpm=sys_shaft_2.speed_rpm / geo_stage_2.u,   # automático: n3 = n2 / i2
    design_life_hours=20000,
    shaft_position=(0.0, _shaft_3_z),
    shaft_origin_x=_shaft_3_origin_x,
)

sys_shaft_3.add_bearing(Bearing(
    position=20.0, C=30000.0, C0=18000.0,    # TODO: valores reais
    arrangement="fixed", label="A",
))
sys_shaft_3.add_bearing(Bearing(
    position=380.0, C=30000.0, C0=18000.0,   # TODO: valores reais
    arrangement="floating", label="B",
))

sys_shaft_3.add_gear(gs.to_gear_element(
    position=_x_gear_driven_local_s3,
    forces=forces_stage_2,
    geometry=geo_stage_2,
    label="Wheel_S2",
))

print(f"Shaft_3: {sys_shaft_3.name}  n={sys_shaft_3.speed_rpm:.1f} rpm  L={shaft_3_shaft.total_length:.0f}mm")
print(f"  Rolamentos: A@{sys_shaft_3.bearings[0].position:.0f}mm   B@{sys_shaft_3.bearings[1].position:.0f}mm")
print(f"  Engrenagem: @{_x_gear_driven_local_s3:.0f}mm local  →  x_global={_x_gear_global_s3:.1f}mm")

# %% Estática

sr_shaft_3 = StaticsSolver().solve(sys_shaft_3)

r = sr_shaft_3.reactions
print("REACÇÕES shaft_3")
print(f"  A: XZ={r['A_xz']:+.1f}N  XY={r['A_xy']:+.1f}N  Fr={np.hypot(r['A_xz'],r['A_xy']):.1f}N")
print(f"  B: XZ={r['B_xz']:+.1f}N  XY={r['B_xy']:+.1f}N  Fr={np.hypot(r['B_xz'],r['B_xy']):.1f}N")
print(f"  M_res_max={sr_shaft_3.M_res_max:.0f}N·mm @ x={sr_shaft_3.x_at_M_res_max:.1f}mm  T_max={sr_shaft_3.T_max:.0f}N·mm")

export_statics(
    sr_shaft_3,
    "shafts/shaft_3/results/statics.txt",
    project="demo_reducer_2stages", element="shaft_3", script="shaft_3.py",
)

# %% Falha estática

mat_shaft_3 = CrMo42

sf_shaft_3 = StaticFailureSolver().solve(sys_shaft_3, sr_shaft_3, mat_shaft_3)

print(f"FALHA ESTÁTICA shaft_3 — {mat_shaft_3.material_id}  Sy={mat_shaft_3.Sy}MPa")
for s in sf_shaft_3.sections:
    print(f"  x={s.x:.1f}mm  d={s.diameter:.1f}mm  n_DE={s.n_DE:.2f}  n_MSS={s.n_MSS:.2f}  {s.risk_label}")

export_static_failure(
    sf_shaft_3,
    "shafts/shaft_3/results/static_failure.txt",
    project="demo_reducer_2stages", element="shaft_3", script="shaft_3.py",
)

# %% Fadiga

stress_shaft_3 = StressSolver().solve(
    sys_shaft_3, sr_shaft_3, mat_shaft_3,
    finish="machined", reliability_percent=99.0,
)

mc = stress_shaft_3.most_critical
print(f"FADIGA shaft_3 — Se_base={mat_shaft_3.endurance_limit:.0f}MPa")
print(f"  Crítica: x={mc.x:.0f}mm  nf_Goodman={mc.nf_goodman:.3f}  nf_ASME={mc.nf_asme:.3f}  ny={mc.ny:.3f}")

export_stress(
    stress_shaft_3,
    "shafts/shaft_3/results/stress.txt",
    project="demo_reducer_2stages", element="shaft_3", script="shaft_3.py",
    mat=mat_shaft_3,
)