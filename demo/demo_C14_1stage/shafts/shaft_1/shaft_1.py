# shafts/shaft_1/shaft_1.py
# Shaft_1 — Veio Condutor (Pinion)
#
# Geometria : 3 secções, 42CrMo4
#   §1 : L=100mm  d=25mm
#   §2 : L=200mm  d=30mm  shoulder_left r=2.5mm  shoulder_right r=2.0mm
#   §3 : L=100mm  d=20mm
#   L_total = 400mm
#
# Rolamentos : A @ x=50mm (fixo, C=60kN, C0=38kN)
#              B @ x=350mm (flutuante, C=60kN, C0=38kN)
# Engrenagem : x=200mm  (importada de stages/stage_1/elements/gear/gear.py)
# Velocidade : 1500 rpm
# Material   : 42CrMo4  (Sut=1000MPa, Sy=800MPa)

# %% Sistema Mecânico

shaft_1 = Shaft(name="Shaft_1")
shaft_1.add_section(ShaftSection(
    length=100.0, diameter=25.0, material_id="42CrMo4", label="§1",
))
shaft_1.add_section(ShaftSection(
    length=200.0, diameter=30.0, material_id="42CrMo4", label="§2",
    shoulder_left =Shoulder(fillet_radius=2.5, diameter_large=30.0, diameter_small=25.0),
    shoulder_right=Shoulder(fillet_radius=2.0, diameter_large=30.0, diameter_small=20.0),
))
shaft_1.add_section(ShaftSection(
    length=100.0, diameter=20.0, material_id="42CrMo4", label="§3",
))

sys_shaft_1 = MechanicalSystem(
    shaft=shaft_1,
    name="Shaft_1",
    speed_rpm=1500.0,
    shaft_position=(0.0, 0.0),
    shaft_origin_x=0.0,
)
sys_shaft_1.add_bearing(Bearing(
    position=50.0, C=60_000.0, C0=38_000.0,
    arrangement="fixed", label="A",
))
sys_shaft_1.add_bearing(Bearing(
    position=350.0, C=60_000.0, C0=38_000.0,
    arrangement="floating", label="B",
))
sys_shaft_1.add_gear(gs.to_gear_element(200.0, forces_stage_1, geo_stage_1, label="Gear_S1"))

print(f"Shaft_1: {sys_shaft_1.name}  n={sys_shaft_1.speed_rpm:.0f} rpm  L={shaft_1.total_length:.0f}mm")
print(f"  Rolamentos: A@{sys_shaft_1.bearings[0].position:.0f}mm   B@{sys_shaft_1.bearings[1].position:.0f}mm")
print(f"  Engrenagem: @{sys_shaft_1.gears[0].position:.0f}mm")

# %% Estática

sr_shaft_1 = StaticsSolver().solve(sys_shaft_1)

r = sr_shaft_1.reactions
print("REACÇÕES Shaft_1")
print(f"  A: XZ={r['A_xz']:+.1f} N   XY={r['A_xy']:+.1f} N   Fr_A={np.hypot(r['A_xz'], r['A_xy']):.1f} N")
print(f"  B: XZ={r['B_xz']:+.1f} N   XY={r['B_xy']:+.1f} N   Fr_B={np.hypot(r['B_xz'], r['B_xy']):.1f} N")
print(f"  M_res_max = {sr_shaft_1.M_res_max:.0f} N·mm  @ x={sr_shaft_1.x_at_M_res_max:.1f} mm")
print(f"  T_max     = {sr_shaft_1.T_max:.0f} N·mm")

export_statics(
    sr_shaft_1,
    "shafts/shaft_1/results/statics.txt",
    project="demo_C14_1stage", element="shaft_1", script="shaft_1.py",
)

# %% Falha Estática

mat_shaft_1 = CrMo42

sf_shaft_1 = StaticFailureSolver().solve(sys_shaft_1, sr_shaft_1, mat_shaft_1)

print(f"FALHA ESTÁTICA Shaft_1 — {mat_shaft_1.material_id}  Sy={mat_shaft_1.Sy} MPa")
print(f"{'x [mm]':>8} {'d [mm]':>7} {'σx [MPa]':>10} {'τxy [MPa]':>10} {'n_DE':>7} {'n_MSS':>7} {'Estado':>12}")
print("-" * 65)
for s in sf_shaft_1.sections:
    print(f"{s.x:>8.1f} {s.diameter:>7.1f} {s.sigma_x:>10.2f} {s.tau_xy:>10.2f} {s.n_DE:>7.2f} {s.n_MSS:>7.2f} {s.risk_label:>12}")

export_static_failure(
    sf_shaft_1,
    "shafts/shaft_1/results/static_failure.txt",
    project="demo_C14_1stage", element="shaft_1", script="shaft_1.py",
)

# %% Fadiga

stress_shaft_1 = StressSolver().solve(
    sys_shaft_1, sr_shaft_1, mat_shaft_1,
    finish="machined", reliability_percent=99.0,
)

print(f"FADIGA Shaft_1 — Se_base={mat_shaft_1.endurance_limit:.0f} MPa")
cs0 = stress_shaft_1.sections[0]
print(f"  Se' corrigido = {cs0.Se_prime:.1f} MPa  (ka={cs0.ka:.3f}, kb={cs0.kb:.3f}, ke={cs0.ke:.3f})")
print(f"{'x [mm]':>8} {'d':>5} {'Kt':>6} {'Kf':>6} {'σa [MPa]':>10} {'nf_Good':>9} {'nf_ASME':>9} {'ny':>7}")
print("-" * 65)
for cs in stress_shaft_1.sections:
    nfg = f"{cs.nf_goodman:.3f}" if cs.nf_goodman < 1e6 else "∞"
    nfa = f"{cs.nf_asme:.3f}"    if cs.nf_asme    < 1e6 else "∞"
    nyv = f"{cs.ny:.3f}"         if cs.ny          < 1e6 else "∞"
    print(f"{cs.x:>8.1f} {cs.diameter:>5.0f} {cs.Kt:>6.3f} {cs.Kf:>6.3f} {cs.sigma_a:>10.3f} {nfg:>9} {nfa:>9} {nyv:>7}")

mc = stress_shaft_1.most_critical
print(f"\n✓ Secção crítica: x={mc.x:.0f}mm  nf_Goodman={mc.nf_goodman:.3f}  nf_ASME={mc.nf_asme:.3f}")

export_stress(
    stress_shaft_1,
    "shafts/shaft_1/results/stress.txt",
    project="demo_C14_1stage", element="shaft_1", script="shaft_1.py",
    mat=mat_shaft_1,
)
