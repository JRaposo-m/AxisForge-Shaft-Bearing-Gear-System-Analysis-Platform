# elements/gear_C14/gear_C14.py
# Caso C14 — Engrenagem Reta com Desvio de Perfil
# Validado contra GEARpie: desvio < 0.5%
#
# Referência: mn=4.5, z1=16, z2=24, al=91.5mm, x1=0.1817, x2=0.1715
# T1=200 N·m, n1=1500 rpm

# %% Geometria C14

geo_c14 = GearSolver().compute_geometry(
    mn=4.5, z1=16, z2=24,
    alpha_n_deg=20.0, beta_deg=0.0,
    al=91.5,
    x1=0.1817, x2=0.1715,
    b=20,
)

print("GEOMETRIA C14")
print(f"  mn              = {geo_c14.mn} mm")
print(f"  z1/z2           = {geo_c14.z1}/{geo_c14.z2}")
print(f"  x1/x2           = {geo_c14.x1}/{geo_c14.x2}")
print(f"  a (standard)    = {geo_c14.a:.3f} mm")
print(f"  al (working)    = {geo_c14.al:.3f} mm  (+{geo_c14.al - geo_c14.a:.3f} mm)")
print(f"  alpha_tw        = {geo_c14.alpha_tw_deg:.4f}°")
print(f"  d1/d2           = {geo_c14.d1:.3f}/{geo_c14.d2:.3f} mm")
print(f"  dl1/dl2         = {geo_c14.dl1:.3f}/{geo_c14.dl2:.3f} mm")
print(f"  da1/da2         = {geo_c14.da1:.3f}/{geo_c14.da2:.3f} mm")
print(f"  eps_alpha       = {geo_c14.eps_alpha:.4f}  (GEARpie ref: 1.46)")
print(f"  Spur gear       = {geo_c14.is_spur}")

export_gear_geometry(
    geo_c14,
    "elements/gear_C14/results/geometry.txt",
    project="demo_C14_H501", element="gear_C14", script="gear_C14.py",
)

# %% Forças C14

forces_c14 = GearSolver().compute_forces(T1_Nm=200.0, geometry=geo_c14)

print("FORÇAS C14")
print(f"  Ft = {forces_c14.Ft:.1f} N   (GEARpie ref: 5464.5 N)  desvio={abs(forces_c14.Ft - 5464.5) / 5464.5 * 100:.2f}%")
print(f"  Fr = {forces_c14.Fr:.1f} N   (GEARpie ref: 2256.6 N)  desvio={abs(forces_c14.Fr - 2256.6) / 2256.6 * 100:.2f}%")
print(f"  Fa = {forces_c14.Fa:.1f} N   (spur → 0 exacto)")
print(f"  T1 = {forces_c14.T1_Nmm / 1000:.1f} N·m   T2 = {forces_c14.T2_Nmm / 1000:.1f} N·m")

export_gear_forces(
    forces_c14,
    "elements/gear_C14/results/forces.txt",
    project="demo_C14_H501", element="gear_C14", script="gear_C14.py",
    geometry=geo_c14,
)
