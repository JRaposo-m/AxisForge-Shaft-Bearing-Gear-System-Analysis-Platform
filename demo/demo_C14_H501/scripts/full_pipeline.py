# scripts/full_pipeline.py
# Pipeline completo — C14 + H501
#
# Corre ambos os casos em sequência e exporta todos os resultados.
# Útil para verificação rápida de que nada quebrou após alterações ao axisforge/.
#
# Executa com: Ctrl+Shift+Enter (Run All) no ScriptEditor

# %% Setup

gs = GearSolver()
bl = BearingLifeSolver()
mat = CrMo42

print("=" * 58)
print(" AxisForge — Pipeline C14 + H501")
print("=" * 58)

# %% C14 — Geometria e Forças

geo_c14    = gs.compute_geometry(mn=4.5, z1=16, z2=24, alpha_n_deg=20.0, beta_deg=0.0,  al=91.5, x1=0.1817, x2=0.1715, b=20)
forces_c14 = gs.compute_forces(T1_Nm=200.0, geometry=geo_c14)

export_gear_geometry(geo_c14,    "elements/gear_C14/results/geometry.txt", project="demo_C14_H501", element="gear_C14",    script="full_pipeline.py")
export_gear_forces(forces_c14,   "elements/gear_C14/results/forces.txt",   project="demo_C14_H501", element="gear_C14",    script="full_pipeline.py", geometry=geo_c14)

print(f"C14 Forças: Ft={forces_c14.Ft:.1f} N  Fr={forces_c14.Fr:.1f} N  Fa={forces_c14.Fa:.1f} N")

# %% C14 — Sistema e Estática

shaft_c14 = Shaft(name="C14_Shaft")
shaft_c14.add_section(ShaftSection(length=100.0, diameter=15.0, material_id="42CrMo4", label="§1"))
shaft_c14.add_section(ShaftSection(length=200.0, diameter=20.0, material_id="42CrMo4", label="§2",
    shoulder_left =Shoulder(fillet_radius=1.0, diameter_large=20.0, diameter_small=15.0),
    shoulder_right=Shoulder(fillet_radius=1.0, diameter_large=20.0, diameter_small=15.0),
))
shaft_c14.add_section(ShaftSection(length=100.0, diameter=15.0, material_id="42CrMo4", label="§3"))

sys_c14 = MechanicalSystem(shaft=shaft_c14, name="C14_System", speed_rpm=1500.0)
sys_c14.add_bearing(Bearing(position=50.0,  C=60_000.0, C0=38_000.0, arrangement="fixed",    label="A"))
sys_c14.add_bearing(Bearing(position=350.0, C=60_000.0, C0=38_000.0, arrangement="floating", label="B"))
sys_c14.add_gear(gs.to_gear_element(200.0, forces_c14, geo_c14, label="C14"))

sr_c14 = StaticsSolver().solve(sys_c14)
export_statics(sr_c14, "systems/C14_shaft/results/statics.txt", project="demo_C14_H501", element="C14_shaft", script="full_pipeline.py")

# %% C14 — Stress e Falha Estática

sf_c14     = StaticFailureSolver().solve(sys_c14, sr_c14, mat)
stress_c14 = StressSolver().solve(sys_c14, sr_c14, mat, finish="machined", reliability_percent=99.0)

export_static_failure(sf_c14,    "systems/C14_shaft/results/static_failure.txt", project="demo_C14_H501", element="C14_shaft", script="full_pipeline.py")
export_stress(stress_c14,        "systems/C14_shaft/results/stress.txt",          project="demo_C14_H501", element="C14_shaft", script="full_pipeline.py", mat=mat)

mc_c14  = stress_c14.most_critical
sfc_c14 = sf_c14.critical_section
print(f"C14 Fadiga:   x={mc_c14.x:.0f}mm  nf_Goodman={mc_c14.nf_goodman:.3f}  nf_ASME={mc_c14.nf_asme:.3f}")
print(f"C14 Estática: x={sfc_c14.x:.0f}mm  n_gov={sfc_c14.governing_n:.3f} ({sfc_c14.governing_theory})")

# %% C14 — Rolamentos

bf_c14 = bl.extract_bearing_forces(sr_c14, sys_c14)
for bearing in sys_c14.bearings:
    Fr, Fa = bf_c14[bearing.label]
    res_c14 = bl.solve_bearing(bearing, Fr, Fa, sys_c14.speed_rpm, sys_c14.design_life_hours)
    export_bearing(res_c14, f"elements/bearing_C14_{bearing.label}/results/life.txt",
        project="demo_C14_H501", element=f"bearing_C14_{bearing.label}", script="full_pipeline.py",
        C=bearing.C, C0=bearing.C0)
    print(f"C14 Rolamento {bearing.label}: L10h={res_c14.L10h:.0f}h  S0={res_c14.S0:.3f}  {'✓' if res_c14.is_safe else '✗'}")

# %% H501 — Geometria e Forças

geo_h501    = gs.compute_geometry(mn=3.5, z1=20, z2=30, alpha_n_deg=20.0, beta_deg=15.0, al=91.5, x1=0.1809, x2=0.0891, b=20)
forces_h501 = gs.compute_forces(T1_Nm=100.0, geometry=geo_h501)

export_gear_geometry(geo_h501,   "elements/gear_H501/results/geometry.txt", project="demo_C14_H501", element="gear_H501", script="full_pipeline.py")
export_gear_forces(forces_h501,  "elements/gear_H501/results/forces.txt",   project="demo_C14_H501", element="gear_H501", script="full_pipeline.py", geometry=geo_h501)

print(f"H501 Forças: Ft={forces_h501.Ft:.1f} N  Fr={forces_h501.Fr:.1f} N  Fa={forces_h501.Fa:.1f} N")

# %% H501 — Sistema e Estática

shaft_h501 = Shaft(name="H501_Shaft")
shaft_h501.add_section(ShaftSection(length=100.0, diameter=15.0, material_id="42CrMo4", label="§1"))
shaft_h501.add_section(ShaftSection(length=200.0, diameter=20.0, material_id="42CrMo4", label="§2",
    shoulder_left =Shoulder(fillet_radius=1.0, diameter_large=20.0, diameter_small=15.0),
    shoulder_right=Shoulder(fillet_radius=1.0, diameter_large=20.0, diameter_small=15.0),
))
shaft_h501.add_section(ShaftSection(length=100.0, diameter=15.0, material_id="42CrMo4", label="§3"))

sys_h501 = MechanicalSystem(shaft=shaft_h501, name="H501_System", speed_rpm=1000.0)
sys_h501.add_bearing(Bearing(position=50.0,  C=35_000.0, C0=22_000.0, arrangement="fixed",    label="A"))
sys_h501.add_bearing(Bearing(position=350.0, C=35_000.0, C0=22_000.0, arrangement="floating", label="B"))
sys_h501.add_gear(gs.to_gear_element(200.0, forces_h501, geo_h501, label="H501"))

sr_h501 = StaticsSolver().solve(sys_h501)
export_statics(sr_h501, "systems/H501_shaft/results/statics.txt", project="demo_C14_H501", element="H501_shaft", script="full_pipeline.py")

# %% H501 — Stress e Falha Estática

sf_h501     = StaticFailureSolver().solve(sys_h501, sr_h501, mat)
stress_h501 = StressSolver().solve(sys_h501, sr_h501, mat, finish="machined", reliability_percent=99.0)

export_static_failure(sf_h501,   "systems/H501_shaft/results/static_failure.txt", project="demo_C14_H501", element="H501_shaft", script="full_pipeline.py")
export_stress(stress_h501,       "systems/H501_shaft/results/stress.txt",          project="demo_C14_H501", element="H501_shaft", script="full_pipeline.py", mat=mat)

mc_h501  = stress_h501.most_critical
sfc_h501 = sf_h501.critical_section
print(f"H501 Fadiga:   x={mc_h501.x:.0f}mm  nf_Goodman={mc_h501.nf_goodman:.3f}  nf_ASME={mc_h501.nf_asme:.3f}")
print(f"H501 Estática: x={sfc_h501.x:.0f}mm  n_gov={sfc_h501.governing_n:.3f} ({sfc_h501.governing_theory})")

# %% H501 — Rolamentos

bf_h501 = bl.extract_bearing_forces(sr_h501, sys_h501)
for bearing in sys_h501.bearings:
    Fr, Fa = bf_h501[bearing.label]
    res_h501 = bl.solve_bearing(bearing, Fr, Fa, sys_h501.speed_rpm, sys_h501.design_life_hours)
    export_bearing(res_h501, f"elements/bearing_H501_{bearing.label}/results/life.txt",
        project="demo_C14_H501", element=f"bearing_H501_{bearing.label}", script="full_pipeline.py",
        C=bearing.C, C0=bearing.C0)
    print(f"H501 Rolamento {bearing.label}: L10h={res_h501.L10h:.0f}h  S0={res_h501.S0:.3f}  {'✓' if res_h501.is_safe else '✗'}")

# %% Sumário

print()
print("=" * 58)
print(" SUMÁRIO FINAL")
print("=" * 58)
for nome, geo, forces, sr, stress, sf in [
    ("C14",  geo_c14,  forces_c14,  sr_c14,  stress_c14,  sf_c14),
    ("H501", geo_h501, forces_h501, sr_h501, stress_h501, sf_h501),
]:
    mc  = stress.most_critical
    sfc = sf.critical_section
    nfg = f"{mc.nf_goodman:.3f}" if mc and mc.nf_goodman < 1e6 else "∞"
    print(f"  {nome}  Ft={forces.Ft:.0f}N  Fa={forces.Fa:.0f}N  M_max={sr.M_res_max:.0f}N·mm  nf_Good={nfg}  n_gov={sfc.governing_n:.2f}")

print()
print("Ficheiros exportados para systems/ e elements/*/results/")
print("Abrir no WorkspacePanel → vista Outputs")

# %% Plots — Diagramas de Esforços

import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

def _plot_shaft_diagrams(sr, system, title: str):
    x    = sr.x
    bear = system.bearings

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(title, fontsize=11, fontweight="bold", color="#cdd6f4")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35)

    panels = [
        (gs[0, 0], sr.V_xz,  "Corte XZ",           "V  [N]"),
        (gs[0, 1], sr.V_xy,  "Corte XY",            "V  [N]"),
        (gs[1, 0], sr.M_res, "Momento Resultante",  "M  [N·mm]"),
        (gs[1, 1], sr.T,     "Torção",              "T  [N·mm]"),
    ]

    for spec, data, label, ylabel in panels:
        ax = fig.add_subplot(spec)
        ax.fill_between(x, data, alpha=0.15, color="#89b4fa")
        ax.plot(x, data, color="#89b4fa", lw=1.4)
        ax.axhline(0, color="#45475a", lw=0.8)

        for b in bear:
            ax.axvline(b.position, color="#585b70", lw=0.8, ls="--")

        i_max = int(np.argmax(np.abs(data)))
        ax.annotate(f"{data[i_max]:+.0f}",
                    xy=(x[i_max], data[i_max]), xytext=(6, 4),
                    textcoords="offset points", fontsize=7, color="#cdd6f4")

        ax.set_title(label, fontsize=9, color="#cdd6f4", pad=4)
        ax.set_xlabel("x  [mm]", fontsize=8, color="#6c7086")
        ax.set_ylabel(ylabel, fontsize=8, color="#6c7086")
        ax.tick_params(labelsize=7, colors="#6c7086")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v/1000:.1f}k" if abs(v) >= 1000 else f"{v:.0f}"
        ))
        ax.set_facecolor("#181825")
        for spine in ax.spines.values():
            spine.set_edgecolor("#313244")
        ax.grid(color="#313244", lw=0.4, zorder=0)

    fig.patch.set_facecolor("#1e1e2e")
    return fig


# %% Plots — Tensão Equivalente

def _plot_tensao_continua(sr, system, mat, title: str):
    x   = sr.x
    M   = sr.M_res
    T   = sr.T
    d   = np.array([system.shaft.diameter_at(float(xi)) for xi in x])

    sig_b  = np.where(d > 0, 32.0 * M / (np.pi * d**3), 0.0)
    tau_t  = np.where(d > 0, 16.0 * np.abs(T) / (np.pi * d**3), 0.0)
    sig_eq = np.sqrt(sig_b**2 + 3.0 * tau_t**2)

    bear = system.bearings

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=11, fontweight="bold", color="#cdd6f4")

    ax.fill_between(x, sig_eq, alpha=0.20, color="#89b4fa")
    ax.plot(x, sig_eq, color="#89b4fa", lw=1.6)

    for b in bear:
        ax.axvline(b.position, color="#585b70", lw=0.8, ls="--")

    i_max = int(np.argmax(sig_eq))
    ax.annotate(f"{sig_eq[i_max]:.1f} MPa",
                xy=(x[i_max], sig_eq[i_max]), xytext=(8, 6),
                textcoords="offset points", fontsize=8, color="#cdd6f4",
                arrowprops=dict(arrowstyle="-", color="#585b70", lw=0.6))

    ax.set_xlabel("x  [mm]", fontsize=9, color="#6c7086")
    ax.set_ylabel("σ'_eq  [MPa]", fontsize=9, color="#6c7086")
    ax.tick_params(labelsize=8, colors="#6c7086")
    ax.set_ylim(bottom=0)
    ax.grid(color="#313244", lw=0.5)
    ax.set_facecolor("#181825")
    fig.patch.set_facecolor("#1e1e2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#313244")

    return fig


fig_c14       = _plot_shaft_diagrams(sr_c14,  sys_c14,  "C14  — Diagramas de Esforços")
fig_h501      = _plot_shaft_diagrams(sr_h501, sys_h501, "H501 — Diagramas de Esforços")
fig_tensao_c14  = _plot_tensao_continua(sr_c14,  sys_c14,  mat, "C14  — Tensão Equivalente")
fig_tensao_h501 = _plot_tensao_continua(sr_h501, sys_h501, mat, "H501 — Tensão Equivalente")

print("Plots gerados.")