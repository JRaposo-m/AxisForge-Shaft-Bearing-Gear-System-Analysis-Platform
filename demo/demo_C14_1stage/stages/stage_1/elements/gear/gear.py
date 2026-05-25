# stages/stage_1/elements/gear/gear.py
# Stage_1 — Engrenagem C14 (Reta com Desvio de Perfil)
#
# Referência GEARpie: mn=4.5, z1=16, z2=24, al=91.5mm, x1=0.1817, x2=0.1715
# Validado contra GEARpie: desvio < 0.5%
# T1 = 200 N·m  @  n1 = 1500 rpm  →  n2 = 1000 rpm  (i = 1.5)

# %% Geometria

gs = GearSolver()

geo_stage_1 = gs.compute_geometry(
    mn=4.5, z1=16, z2=24,
    alpha_n_deg=20.0, beta_deg=0.0,
    al=91.5,
    x1=0.1817, x2=0.1715,
    b=20,
)

print("GEOMETRIA Stage_1 (C14)")
print(f"  mn              = {geo_stage_1.mn} mm")
print(f"  z1/z2           = {geo_stage_1.z1}/{geo_stage_1.z2}   i = {geo_stage_1.u:.4f}")
print(f"  x1/x2           = {geo_stage_1.x1}/{geo_stage_1.x2}")
print(f"  a  (standard)   = {geo_stage_1.a:.3f} mm")
print(f"  al (working)    = {geo_stage_1.al:.3f} mm  (+{geo_stage_1.al - geo_stage_1.a:.3f} mm)")
print(f"  alpha_tw        = {geo_stage_1.alpha_tw_deg:.4f}°")
print(f"  d1/d2           = {geo_stage_1.d1:.3f}/{geo_stage_1.d2:.3f} mm")
print(f"  dl1/dl2         = {geo_stage_1.dl1:.3f}/{geo_stage_1.dl2:.3f} mm")
print(f"  da1/da2         = {geo_stage_1.da1:.3f}/{geo_stage_1.da2:.3f} mm")
print(f"  eps_alpha       = {geo_stage_1.eps_alpha:.4f}  (GEARpie ref: 1.46)")
print(f"  Spur            = {geo_stage_1.is_spur}")

export_gear_geometry(
    geo_stage_1,
    "stages/stage_1/elements/gear/results/geometry.txt",
    project="demo_C14_1stage", element="stage_1_gear", script="gear.py",
)

# %% Forças

forces_stage_1 = gs.compute_forces(T1_Nm=200.0, geometry=geo_stage_1)

print("FORÇAS Stage_1 (Shaft_1 — driver / pinion)")
print(f"  Ft = {forces_stage_1.Ft:.1f} N   (GEARpie ref: 5464.5 N)  desvio={abs(forces_stage_1.Ft - 5464.5)/5464.5*100:.2f}%")
print(f"  Fr = {forces_stage_1.Fr:.1f} N   (GEARpie ref: 2256.6 N)  desvio={abs(forces_stage_1.Fr - 2256.6)/2256.6*100:.2f}%")
print(f"  Fa = {forces_stage_1.Fa:.1f} N   (spur → 0 exacto)")
print(f"  T1 = {forces_stage_1.T1_Nmm/1000:.1f} N·m   T2 = {forces_stage_1.T2_Nmm/1000:.1f} N·m")
print(f"  Shaft_2 recebe forças reactivas (mesma magnitude, plano invertido)")

export_gear_forces(
    forces_stage_1,
    "stages/stage_1/elements/gear/results/forces.txt",
    project="demo_C14_1stage", element="stage_1_gear", script="gear.py",
    geometry=geo_stage_1,
)
