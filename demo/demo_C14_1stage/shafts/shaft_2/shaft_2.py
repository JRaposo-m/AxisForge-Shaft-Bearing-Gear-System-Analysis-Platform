# shafts/shaft_2/shaft_2.py
# Shaft_2 — Veio Conduzido (Wheel)
#
# Geometria : 3 secções, 42CrMo4
#   §1 : L=80mm   d=30mm
#   §2 : L=190mm  d=35mm  shoulder_left r=3.0mm  shoulder_right r=2.5mm
#   §3 : L=80mm   d=25mm
#   L_total = 350mm
#
# Rolamentos : A @ x=40mm  (fixo,     C=45kN, C0=28kN)
#              B @ x=310mm (flutuante, C=45kN, C0=28kN)
# Engrenagem : x=175mm  — forças reactivas ao stage_1 (sentido invertido)
# Velocidade : 1000 rpm  (= 1500 × 16/24)
# Material   : 42CrMo4  (Sut=1000MPa, Sy=800MPa)
#
# NOTA: forças no shaft_2 são reactivas — Ft, Fr, Fa de forces_stage_1
# com Ft e Fr invertidas de plano (shaft_driver → shaft_driven).
# T2 = T1 × (z2/z1) = 200 × 1.5 = 300 N·m

# %% Sistema Mecânico

shaft_2 = Shaft(name="Shaft_2")
shaft_2.add_section(ShaftSection(
    length=80.0, diameter=30.0, material_id="42CrMo4", label="§1",
))
shaft_2.add_section(ShaftSection(
    length=190.0, diameter=35.0, material_id="42CrMo4", label="§2",
    shoulder_left =Shoulder(fillet_radius=2.0, diameter_large=35.0, diameter_small=30.0),
    shoulder_right=Shoulder(fillet_radius=2.5, diameter_large=35.0, diameter_small=25.0),
))
shaft_2.add_section(ShaftSection(
    length=80.0, diameter=25.0, material_id="42CrMo4", label="§3",
))

# Forças reactivas: Ft e Fr mantêm magnitude, torque = T2
gear_elem_s2 = GearElement(
    position=175.0,
    tangential_force=forces_stage_1.Ft,   # mesma magnitude — plano invertido pelo solver
    radial_force=forces_stage_1.Fr,
    axial_force=forces_stage_1.Fa,        # spur → 0; helical: sentido invertido
    pitch_diameter=geo_stage_1.dl2,       # diâmetro da roda
    torque=forces_stage_1.T2_Nmm,
    label="Gear_S1_driven",
)

sys_shaft_2 = MechanicalSystem(
    shaft=shaft_2,
    name="Shaft_2",
    speed_rpm=1000.0,
    shaft_position=(0.0, 91.5),   # al = 91.5mm — afastamento em Y
    shaft_origin_x=25.0,          # x_gear_local=175 → x_global=200 ✓ contacto
)
sys_shaft_2.add_bearing(Bearing(
    position=40.0, C=45_000.0, C0=28_000.0,
    arrangement="fixed", label="A",
))
sys_shaft_2.add_bearing(Bearing(
    position=310.0, C=45_000.0, C0=28_000.0,
    arrangement="floating", label="B",
))
sys_shaft_2.add_gear(gear_elem_s2)

print(f"Shaft_2: {sys_shaft_2.name}  n={sys_shaft_2.speed_rpm:.0f} rpm  L={shaft_2.total_length:.0f}mm")
print(f"  Rolamentos: A@{sys_shaft_2.bearings[0].position:.0f}mm   B@{sys_shaft_2.bearings[1].position:.0f}mm")
print(f"  Engrenagem: @{sys_shaft_2.gears[0].position:.0f}mm  (x_global={sys_shaft_2.shaft_origin_x + 175.0:.0f}mm)")
print(f"  T2 = {forces_stage_1.T2_Nmm / 1000:.1f} N·m")

# %% Estática

sr_shaft_2 = StaticsSolver().solve(sys_shaft_2)

r = sr_shaft_2.reactions
print("REACÇÕES Shaft_2")
print(f"  A: XZ={r['A_xz']:+.1f} N   XY={r['A_xy']:+.1f} N   Fr_A={np.hypot(r['A_xz'], r['A_xy']):.1f} N")
print(f"  B: XZ={r['B_xz']:+.1f} N   XY={r['B_xy']:+.1f} N   Fr_B={np.hypot(r['B_xz'], r['B_xy']):.1f} N")
print(f"  M_res_max = {sr_shaft_2.M_res_max:.0f} N·mm  @ x={sr_shaft_2.x_at_M_res_max:.1f} mm")
print(f"  T_max     = {sr_shaft_2.T_max:.0f} N·mm")

export_statics(
    sr_shaft_2,
    "shafts/shaft_2/results/statics.txt",
    project="demo_C14_1stage", element="shaft_2", script="shaft_2.py",
)

# %% Falha Estática

mat_shaft_2 = CrMo42

sf_shaft_2 = StaticFailureSolver().solve(sys_shaft_2, sr_shaft_2, mat_shaft_2)

print(f"FALHA ESTÁTICA Shaft_2 — {mat_shaft_2.material_id}  Sy={mat_shaft_2.Sy} MPa")
print(f"{'x [mm]':>8} {'d [mm]':>7} {'σx [MPa]':>10} {'τxy [MPa]':>10} {'n_DE':>7} {'n_MSS':>7} {'Estado':>12}")
print("-" * 65)
for s in sf_shaft_2.sections:
    print(f"{s.x:>8.1f} {s.diameter:>7.1f} {s.sigma_x:>10.2f} {s.tau_xy:>10.2f} {s.n_DE:>7.2f} {s.n_MSS:>7.2f} {s.risk_label:>12}")

export_static_failure(
    sf_shaft_2,
    "shafts/shaft_2/results/static_failure.txt",
    project="demo_C14_1stage", element="shaft_2", script="shaft_2.py",
)

# %% Fadiga

stress_shaft_2 = StressSolver().solve(
    sys_shaft_2, sr_shaft_2, mat_shaft_2,
    finish="machined", reliability_percent=99.0,
)

print(f"FADIGA Shaft_2 — Se_base={mat_shaft_2.endurance_limit:.0f} MPa")
cs0 = stress_shaft_2.sections[0]
print(f"  Se' corrigido = {cs0.Se_prime:.1f} MPa  (ka={cs0.ka:.3f}, kb={cs0.kb:.3f}, ke={cs0.ke:.3f})")
print(f"{'x [mm]':>8} {'d':>5} {'Kt':>6} {'Kf':>6} {'σa [MPa]':>10} {'nf_Good':>9} {'nf_ASME':>9} {'ny':>7}")
print("-" * 65)
for cs in stress_shaft_2.sections:
    nfg = f"{cs.nf_goodman:.3f}" if cs.nf_goodman < 1e6 else "∞"
    nfa = f"{cs.nf_asme:.3f}"    if cs.nf_asme    < 1e6 else "∞"
    nyv = f"{cs.ny:.3f}"         if cs.ny          < 1e6 else "∞"
    print(f"{cs.x:>8.1f} {cs.diameter:>5.0f} {cs.Kt:>6.3f} {cs.Kf:>6.3f} {cs.sigma_a:>10.3f} {nfg:>9} {nfa:>9} {nyv:>7}")

mc = stress_shaft_2.most_critical
print(f"\n✓ Secção crítica: x={mc.x:.0f}mm  nf_Goodman={mc.nf_goodman:.3f}  nf_ASME={mc.nf_asme:.3f}")

export_stress(
    stress_shaft_2,
    "shafts/shaft_2/results/stress.txt",
    project="demo_C14_1stage", element="shaft_2", script="shaft_2.py",
    mat=mat_shaft_2,
)
