# systems/C14_shaft/C14_shaft.py
# Caso C14 — Análise completa do veio
#
# Geometria: 3 secções, 42CrMo4, d=25/30/20 mm, L_total=400 mm
# Rolamentos: A @ x=50mm (fixo, C=60kN), B @ x=350mm (flutuante, C=60kN)
# Engrenagem: x=200mm, Ft=5464N, Fr=2257N, Fa=0N
# Material: 42CrMo4 (Sut=1000MPa, Sy=800MPa)
#
# NOTA: veio genérico compatível com forças C14; não é o veio real do report.

# %% Geometria e Forças (importar do elemento)

gs = GearSolver()

geo_c14 = gs.compute_geometry(
    mn=4.5, z1=16, z2=24,
    alpha_n_deg=20.0, beta_deg=0.0,
    al=91.5, x1=0.1817, x2=0.1715, b=20,
)
forces_c14 = gs.compute_forces(T1_Nm=200.0, geometry=geo_c14)

print(f"Forças C14: Ft={forces_c14.Ft:.1f} N  Fr={forces_c14.Fr:.1f} N  Fa={forces_c14.Fa:.1f} N")

# %% Sistema Mecânico

shaft_c14 = Shaft(name="C14_Shaft")
shaft_c14.add_section(ShaftSection(
    length=100.0, diameter=25.0, material_id="42CrMo4", label="§1",
))
shaft_c14.add_section(ShaftSection(
    length=200.0, diameter=30.0, material_id="42CrMo4", label="§2",
    shoulder_left =Shoulder(fillet_radius=2.5, diameter_large=30.0, diameter_small=25.0),
    shoulder_right=Shoulder(fillet_radius=2.0, diameter_large=30.0, diameter_small=20.0),
))
shaft_c14.add_section(ShaftSection(
    length=100.0, diameter=20.0, material_id="42CrMo4", label="§3",
))

sys_c14 = MechanicalSystem(shaft=shaft_c14, name="C14_System", speed_rpm=1500.0)
sys_c14.add_bearing(Bearing(position=50.0,  C=60_000.0, C0=38_000.0, arrangement="fixed",    label="A"))
sys_c14.add_bearing(Bearing(position=350.0, C=60_000.0, C0=38_000.0, arrangement="floating", label="B"))
sys_c14.add_gear(gs.to_gear_element(200.0, forces_c14, geo_c14, label="C14"))

print(f"Sistema: {sys_c14.name}  n={sys_c14.speed_rpm:.0f} rpm")
print(f"Rolamentos: A @ {sys_c14.bearings[0].position}mm   B @ {sys_c14.bearings[1].position}mm")
print(f"Engrenagem: @ {sys_c14.gears[0].position}mm")

# %% Estática

sr_c14 = StaticsSolver().solve(sys_c14)

r = sr_c14.reactions
print("REACÇÕES C14")
print(f"  A: XZ={r['A_xz']:+.1f} N   XY={r['A_xy']:+.1f} N   Fr_A={np.hypot(r['A_xz'], r['A_xy']):.1f} N")
print(f"  B: XZ={r['B_xz']:+.1f} N   XY={r['B_xy']:+.1f} N   Fr_B={np.hypot(r['B_xz'], r['B_xy']):.1f} N")
print(f"  M_res_max = {sr_c14.M_res_max:.0f} N·mm  @ x={sr_c14.x_at_M_res_max:.1f} mm")
print(f"  T_max     = {sr_c14.T_max:.0f} N·mm")

export_statics(
    sr_c14,
    "systems/C14_shaft/results/statics.txt",
    project="demo_C14_H501", element="C14_shaft", script="C14_shaft.py",
)

# %% Falha Estática

mat_c14 = CrMo42

sf_c14 = StaticFailureSolver().solve(sys_c14, sr_c14, mat_c14)

print(f"FALHA ESTÁTICA C14 — {mat_c14.material_id}  Sy={mat_c14.Sy} MPa")
print(f"{'x [mm]':>8} {'d [mm]':>7} {'σx [MPa]':>10} {'τxy [MPa]':>10} {'n_DE':>7} {'n_MSS':>7} {'Estado':>12}")
print("-" * 65)
for s in sf_c14.sections:
    print(f"{s.x:>8.1f} {s.diameter:>7.1f} {s.sigma_x:>10.2f} {s.tau_xy:>10.2f} {s.n_DE:>7.2f} {s.n_MSS:>7.2f} {s.risk_label:>12}")

export_static_failure(
    sf_c14,
    "systems/C14_shaft/results/static_failure.txt",
    project="demo_C14_H501", element="C14_shaft", script="C14_shaft.py",
)

# %% Fadiga

stress_c14 = StressSolver().solve(sys_c14, sr_c14, mat_c14, finish="machined", reliability_percent=99.0)

print(f"FADIGA C14 — Se_base={mat_c14.endurance_limit:.0f} MPa")
cs0 = stress_c14.sections[0]
print(f"  Se' corrigido = {cs0.Se_prime:.1f} MPa  (ka={cs0.ka:.3f}, kb={cs0.kb:.3f}, ke={cs0.ke:.3f})")
print(f"{'x [mm]':>8} {'d':>5} {'Kt':>6} {'Kf':>6} {'σa [MPa]':>10} {'nf_Good':>9} {'nf_ASME':>9} {'ny':>7}")
print("-" * 65)
for cs in stress_c14.sections:
    nfg = f"{cs.nf_goodman:.3f}" if cs.nf_goodman < 1e6 else "∞"
    nfa = f"{cs.nf_asme:.3f}"    if cs.nf_asme    < 1e6 else "∞"
    nyv = f"{cs.ny:.3f}"         if cs.ny          < 1e6 else "∞"
    print(f"{cs.x:>8.1f} {cs.diameter:>5.0f} {cs.Kt:>6.3f} {cs.Kf:>6.3f} {cs.sigma_a:>10.3f} {nfg:>9} {nfa:>9} {nyv:>7}")

mc = stress_c14.most_critical
print(f"\n✓ Secção crítica: x={mc.x:.0f}mm  nf_Goodman={mc.nf_goodman:.3f}  nf_ASME={mc.nf_asme:.3f}")

export_stress(
    stress_c14,
    "systems/C14_shaft/results/stress.txt",
    project="demo_C14_H501", element="C14_shaft", script="C14_shaft.py",
    mat=mat_c14,
)

# %% Rolamentos

bl = BearingLifeSolver()
bf_c14 = bl.extract_bearing_forces(sr_c14, sys_c14)

print(f"ROLAMENTOS C14  (n={sys_c14.speed_rpm:.0f} rpm, C=60kN, C0=38kN)")
print(f"{'':>4} {'Fr [N]':>8} {'Fa [N]':>8} {'P [N]':>8} {'L10h [h]':>10} {'S0':>6} {'Estado':>8}")
print("-" * 60)
for bearing in sys_c14.bearings:
    Fr, Fa = bf_c14[bearing.label]
    res = bl.solve_bearing(bearing, Fr, Fa, sys_c14.speed_rpm, sys_c14.design_life_hours)
    estado = "✓ OK" if res.is_safe else "✗ FALHA"
    print(f"  {bearing.label}  {res.Fr:>8.1f} {res.Fa:>8.1f} {res.P:>8.1f} {res.L10h:>10.0f} {res.S0:>6.2f} {estado:>8}")

    export_bearing(
        res,
        f"elements/bearing_C14_{bearing.label}/results/life.txt",
        project="demo_C14_H501",
        element=f"bearing_C14_{bearing.label}",
        script="C14_shaft.py",
        C=bearing.C, C0=bearing.C0,
    )
