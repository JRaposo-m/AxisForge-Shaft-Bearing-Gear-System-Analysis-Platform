# %% Geometria e forças — C14 (Spur, x≠0)
# Caso de validação GEARpie C14: mn=4.5, z1=16, z2=24, spur, x≠0
# Referência: GEARpie Ft=5464.5 N, Fr=2256.6 N, desvio < 0.5%

gs = GearSolver()

geo = gs.compute_geometry(
    mn=4.5, z1=16, z2=24,
    alpha_n_deg=20.0, beta_deg=0.0,
    al=91.5, x1=0.1817, x2=0.1715, b=20,
)
forces = gs.compute_forces(T1_Nm=200.0, geometry=geo)

print(geo)
print()
print(forces)
print()
print(f"Validação GEARpie:")
print(f"  Ft = {forces.Ft:.1f} N   (ref: 5464.5 N)  desvio = {abs(forces.Ft - 5464.5) / 5464.5 * 100:.2f}%")
print(f"  Fr = {forces.Fr:.1f} N   (ref: 2256.6 N)  desvio = {abs(forces.Fr - 2256.6) / 2256.6 * 100:.2f}%")
print(f"  Fa = {forces.Fa:.1f} N   (spur → 0 exacto)")

# %% Sistema mecânico
# Veio 42CrMo4 — 3 secções, ombros em x=100mm e x=300mm
# Rolamentos SKF genérico C=60kN, C0=38kN
# Engrenagem na posição x=200mm (meio do veio)

shaft = Shaft(name="C14_Shaft")
shaft.add_section(ShaftSection(
    length=100.0, diameter=25.0, material_id="42CrMo4", label="§1",
))
shaft.add_section(ShaftSection(
    length=200.0, diameter=30.0, material_id="42CrMo4", label="§2",
    shoulder_left =Shoulder(fillet_radius=2.5, diameter_large=30.0, diameter_small=25.0),
    shoulder_right=Shoulder(fillet_radius=2.0, diameter_large=30.0, diameter_small=20.0),
))
shaft.add_section(ShaftSection(
    length=100.0, diameter=20.0, material_id="42CrMo4", label="§3",
))

system = MechanicalSystem(shaft=shaft, name="C14_System", speed_rpm=1500.0)
system.add_bearing(Bearing(position=50.0,  C=60_000.0, C0=38_000.0, arrangement="fixed",    label="A"))
system.add_bearing(Bearing(position=350.0, C=60_000.0, C0=38_000.0, arrangement="floating", label="B"))
system.add_gear(gs.to_gear_element(200.0, forces, geo, label="C14"))

print(system)

# %% Estática
statics = StaticsSolver().solve(system)

print(statics)
print()

r = statics.reactions
import numpy as np
Fr_A = np.hypot(r["A_xz"], r["A_xy"])
Fr_B = np.hypot(r["B_xz"], r["B_xy"])
print(f"Rolamento A:  Fr = {Fr_A:.1f} N")
print(f"Rolamento B:  Fr = {Fr_B:.1f} N")

# %% Diagramas — matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

x  = statics.x
xA = system.bearings[0].position
xB = system.bearings[1].position
xG = system.gears[0].position

fig = plt.figure(figsize=(14, 9))
gs_p = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

def marks(ax):
    ax.axvline(xA, color="steelblue", lw=1.2, ls="--", alpha=0.7)
    ax.axvline(xB, color="steelblue", lw=1.2, ls=":",  alpha=0.7)
    ax.axvline(xG, color="tomato",    lw=1.2, ls="-.", alpha=0.7)
    ax.axhline(0,  color="k",         lw=0.6)

ax1 = fig.add_subplot(gs_p[0, 0])
ax1.fill_between(x, statics.V_xz, alpha=0.25, color="royalblue")
ax1.plot(x, statics.V_xz, "royalblue", lw=1.5)
marks(ax1)
ax1.set_title("V(x) — Plano XZ"); ax1.set_ylabel("V [N]"); ax1.set_xlabel("x [mm]")

ax2 = fig.add_subplot(gs_p[0, 1])
ax2.fill_between(x, statics.V_xy, alpha=0.25, color="seagreen")
ax2.plot(x, statics.V_xy, "seagreen", lw=1.5)
marks(ax2)
ax2.set_title("V(x) — Plano XY"); ax2.set_ylabel("V [N]"); ax2.set_xlabel("x [mm]")

ax3 = fig.add_subplot(gs_p[1, 0])
ax3.plot(x, statics.M_xz / 1e3, "royalblue", lw=1.5, label="M_XZ")
ax3.plot(x, statics.M_xy / 1e3, "seagreen",  lw=1.5, label="M_XY")
marks(ax3); ax3.legend(fontsize=8)
ax3.set_title("M(x) — ambos os planos"); ax3.set_ylabel("M [N·m]"); ax3.set_xlabel("x [mm]")

ax4 = fig.add_subplot(gs_p[1, 1])
ax4.fill_between(x, statics.M_res / 1e3, alpha=0.2, color="darkorange")
ax4.plot(x, statics.M_res / 1e3, "darkorange", lw=2)
marks(ax4)
ax4.set_title("M_res(x)"); ax4.set_ylabel("M_res [N·m]"); ax4.set_xlabel("x [mm]")

ax5 = fig.add_subplot(gs_p[2, 0])
ax5.fill_between(x, statics.T / 1e3, alpha=0.2, color="mediumpurple")
ax5.plot(x, statics.T / 1e3, "mediumpurple", lw=2)
marks(ax5)
ax5.set_title("T(x)"); ax5.set_ylabel("T [N·m]"); ax5.set_xlabel("x [mm]")

ax6 = fig.add_subplot(gs_p[2, 1])
ax6.set_xlim(-20, shaft.total_length + 20)
ax6.set_ylim(-45, 45)
ax6.set_aspect("equal")
secs = [
    (0,   100, shaft.sections[0].diameter),
    (100, 300, shaft.sections[1].diameter),
    (300, 400, shaft.sections[2].diameter),
]
for x0, x1, d in secs:
    ax6.add_patch(plt.Rectangle((x0, -d / 2), x1 - x0, d,
                                color="#87ceeb", ec="navy", lw=1.2))
ax6.plot([xA, xA], [-38, 38], "v", ms=10, color="steelblue")
ax6.plot([xB, xB], [-38, 38], "v", ms=10, color="steelblue")
ax6.plot(xG, 0, "s", ms=12, color="tomato")
ax6.text(xA, -42, "A", ha="center", fontsize=9, color="steelblue")
ax6.text(xB, -42, "B", ha="center", fontsize=9, color="steelblue")
ax6.text(xG,  35, "C14", ha="center", fontsize=9, color="tomato")
ax6.set_title("Esquema do veio"); ax6.set_xlabel("x [mm]"); ax6.set_yticks([])
ax6.axhline(0, color="k", lw=0.5)

fig.suptitle("AxisForge — Análise Estática C14 (Spur, x≠0)", fontsize=12, fontweight="bold")
plt.show()

# %% Falha estática
mat = get_material("42CrMo4")

sf = StaticFailureSolver().solve(system, statics, mat)
print(sf)

# %% Fadiga
stress = StressSolver().solve(system, statics, mat, finish="machined", reliability_percent=99.0)
print(stress)

# %% Rolamentos
bl = BearingLifeSolver()
bf = bl.extract_bearing_forces(statics, system)

print(f"ROLAMENTOS C14  (n={system.speed_rpm:.0f} rpm)")
print(f"{'':>4} {'Fr [N]':>8} {'Fa [N]':>8} {'P [N]':>8} {'L10h [h]':>10} {'S0':>6} {'OK':>6}")
print("-" * 58)
for bearing in system.bearings:
    Fr, Fa = bf[bearing.label]
    res = bl.solve_bearing(bearing, Fr, Fa, system.speed_rpm)
    ok = "✓" if res.is_safe else "✗"
    print(f"  {bearing.label}  {res.Fr:>8.1f} {res.Fa:>8.1f} {res.P:>8.1f} "
          f"{res.L10h:>10.0f} {res.S0:>6.2f} {ok:>6}")

# %% Export resultados
# Descomenta para gerar os .txt por elemento:
# statics.export_txt("results/statics.txt", project="C14_Demo", element="C14_Shaft", script="C14.py")
# sf.export_txt("results/static_failure.txt", project="C14_Demo", element="C14_Shaft", script="C14.py")
# stress.export_txt("results/stress.txt", project="C14_Demo", element="C14_Shaft", script="C14.py",
#                   Sy=mat.Sy, Sut=mat.Sut, Se=mat.endurance_limit)
