# systems/H501_shaft/H501_shaft.py
# Caso H501 — Análise completa do veio
#
# Geometria: 3 secções, 42CrMo4, d=25/30/25 mm, L_total=400 mm
# Rolamentos: A @ x=50mm (fixo, C=35kN), B @ x=350mm (flutuante, C=35kN)
# Engrenagem: x=200mm, Ft=2732N, Fr=1110N, Fa=740N (helicoidal → Fa ≠ 0)
# Material: 42CrMo4 (Sut=1000MPa, Sy=800MPa)
#
# NOTA: veio genérico compatível com forças H501; não é o veio real do report.
# NOTA: rolamento A (fixo) absorve força axial Fa da engrenagem helicoidal.

# %% Geometria e Forças

gs = GearSolver()

geo_h501 = gs.compute_geometry(
    mn=3.5, z1=20, z2=30,
    alpha_n_deg=20.0, beta_deg=15.0,
    al=91.5, x1=0.1809, x2=0.0891, b=20,
)
forces_h501 = gs.compute_forces(T1_Nm=100.0, geometry=geo_h501)

print(f"Forças H501: Ft={forces_h501.Ft:.1f} N  Fr={forces_h501.Fr:.1f} N  Fa={forces_h501.Fa:.1f} N")
print(f"  (Fa ≠ 0 — engrenagem helicoidal, rolamento A absorve carga axial)")

# %% Sistema Mecânico

shaft_h501 = Shaft(name="H501_Shaft")
shaft_h501.add_section(ShaftSection(
    length=100.0, diameter=25.0, material_id="42CrMo4", label="§1",
))
shaft_h501.add_section(ShaftSection(
    length=200.0, diameter=30.0, material_id="42CrMo4", label="§2",
    shoulder_left =Shoulder(fillet_radius=2.5, diameter_large=30.0, diameter_small=25.0),
    shoulder_right=Shoulder(fillet_radius=2.5, diameter_large=30.0, diameter_small=25.0),
))
shaft_h501.add_section(ShaftSection(
    length=100.0, diameter=25.0, material_id="42CrMo4", label="§3",
))

sys_h501 = MechanicalSystem(shaft=shaft_h501, name="H501_System", speed_rpm=1000.0)
sys_h501.add_bearing(Bearing(position=50.0,  C=35_000.0, C0=22_000.0, arrangement="fixed",    label="A"))
sys_h501.add_bearing(Bearing(position=350.0, C=35_000.0, C0=22_000.0, arrangement="floating", label="B"))
sys_h501.add_gear(gs.to_gear_element(200.0, forces_h501, geo_h501, label="H501"))

print(f"Sistema: {sys_h501.name}  n={sys_h501.speed_rpm:.0f} rpm")

# %% Estática

sr_h501 = StaticsSolver().solve(sys_h501)

r = sr_h501.reactions
print("REACÇÕES H501")
print(f"  A: XZ={r['A_xz']:+.1f} N   XY={r['A_xy']:+.1f} N   Fr_A={np.hypot(r['A_xz'], r['A_xy']):.1f} N")
print(f"  B: XZ={r['B_xz']:+.1f} N   XY={r['B_xy']:+.1f} N   Fr_B={np.hypot(r['B_xz'], r['B_xy']):.1f} N")
print(f"  Axial (rolamento fixo A): {abs(r['axial']):.1f} N  (Fa engrenagem={forces_h501.Fa:.1f} N)")
print(f"  M_res_max = {sr_h501.M_res_max:.0f} N·mm  @ x={sr_h501.x_at_M_res_max:.1f} mm")
print(f"  T_max     = {sr_h501.T_max:.0f} N·mm")

export_statics(
    sr_h501,
    "systems/H501_shaft/results/statics.txt",
    project="demo_C14_H501", element="H501_shaft", script="H501_shaft.py",
)

# %% Falha Estática

mat_h501 = CrMo42

sf_h501 = StaticFailureSolver().solve(sys_h501, sr_h501, mat_h501)

print(f"FALHA ESTÁTICA H501 — {mat_h501.material_id}  Sy={mat_h501.Sy} MPa")
print(f"{'x [mm]':>8} {'d [mm]':>7} {'σx [MPa]':>10} {'τxy [MPa]':>10} {'n_DE':>7} {'n_MSS':>7} {'Estado':>12}")
print("-" * 65)
for s in sf_h501.sections:
    print(f"{s.x:>8.1f} {s.diameter:>7.1f} {s.sigma_x:>10.2f} {s.tau_xy:>10.2f} {s.n_DE:>7.2f} {s.n_MSS:>7.2f} {s.risk_label:>12}")

export_static_failure(
    sf_h501,
    "systems/H501_shaft/results/static_failure.txt",
    project="demo_C14_H501", element="H501_shaft", script="H501_shaft.py",
)

# %% Fadiga

stress_h501 = StressSolver().solve(sys_h501, sr_h501, mat_h501, finish="machined", reliability_percent=99.0)

print("FADIGA H501")
cs0 = stress_h501.sections[0]
print(f"  Se' corrigido = {cs0.Se_prime:.1f} MPa  (ka={cs0.ka:.3f}, kb={cs0.kb:.3f}, ke={cs0.ke:.3f})")
print(f"{'x [mm]':>8} {'d':>5} {'Kt':>6} {'Kf':>6} {'σa [MPa]':>10} {'nf_Good':>9} {'nf_ASME':>9} {'ny':>7}")
print("-" * 65)
for cs in stress_h501.sections:
    nfg = f"{cs.nf_goodman:.3f}" if cs.nf_goodman < 1e6 else "∞"
    nfa = f"{cs.nf_asme:.3f}"    if cs.nf_asme    < 1e6 else "∞"
    nyv = f"{cs.ny:.3f}"         if cs.ny          < 1e6 else "∞"
    print(f"{cs.x:>8.1f} {cs.diameter:>5.0f} {cs.Kt:>6.3f} {cs.Kf:>6.3f} {cs.sigma_a:>10.3f} {nfg:>9} {nfa:>9} {nyv:>7}")

mc = stress_h501.most_critical
print(f"\n✓ Secção crítica: x={mc.x:.0f}mm  nf_Goodman={mc.nf_goodman:.3f}  nf_ASME={mc.nf_asme:.3f}")

export_stress(
    stress_h501,
    "systems/H501_shaft/results/stress.txt",
    project="demo_C14_H501", element="H501_shaft", script="H501_shaft.py",
    mat=mat_h501,
)

# %% Rolamentos

bl = BearingLifeSolver()
bf_h501 = bl.extract_bearing_forces(sr_h501, sys_h501)

print(f"ROLAMENTOS H501  (n={sys_h501.speed_rpm:.0f} rpm, C=35kN, C0=22kN)")
print(f"  Nota: rolamento A (fixo) absorve Fa={forces_h501.Fa:.1f} N axial da engrenagem helicoidal")
print(f"{'':>4} {'Fr [N]':>8} {'Fa [N]':>8} {'P [N]':>8} {'L10h [h]':>10} {'S0':>6} {'Estado':>8}")
print("-" * 60)
for bearing in sys_h501.bearings:
    Fr, Fa = bf_h501[bearing.label]
    res = bl.solve_bearing(bearing, Fr, Fa, sys_h501.speed_rpm, sys_h501.design_life_hours)
    estado = "✓ OK" if res.is_safe else "✗ FALHA"
    print(f"  {bearing.label}  {res.Fr:>8.1f} {res.Fa:>8.1f} {res.P:>8.1f} {res.L10h:>10.0f} {res.S0:>6.2f} {estado:>8}")

    export_bearing(
        res,
        f"elements/bearing_H501_{bearing.label}/results/life.txt",
        project="demo_C14_H501",
        element=f"bearing_H501_{bearing.label}",
        script="H501_shaft.py",
        C=bearing.C, C0=bearing.C0,
    )
