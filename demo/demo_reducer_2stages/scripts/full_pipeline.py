# scripts/full_pipeline.py
# demo_reducer_2stages — Pipeline Completo
# Gerado automaticamente por AxisForge ProjectManager
#
# Orquestrador puro: cria veios e executa ficheiros em ordem.
# _run() executa cada ficheiro no namespace partilhado do runner.
#
# Fluxo de desenvolvimento:
#   1. Adicionar create_shaft() para cada veio
#   2. Preencher cada shaft_N.py com geometria, rolamentos e engrenagens
#   3. Chamar create_gear_pair() para cada estágio
#   4. Preencher cada gear_pair_N.py com parâmetros da engrenagem
#   5. Adicionar _run() em ordem e o sumário final

import pathlib as _pathlib

def _run(_rel: str) -> None:
    """Executa um ficheiro .py no namespace partilhado do runner."""
    _p = _pathlib.Path.cwd() / _rel
    exec(compile(_p.read_text(encoding="utf-8"), str(_p), "exec"), globals())  # noqa: S102

print("=" * 62)
print(f" AxisForge — demo_reducer_2stages — Pipeline Completo")
print("=" * 62)

# %% 1. Criar veios (gera pastas e skeletons se não existirem)

# create_shaft("shaft_1")   # TODO: adicionar mais veios conforme necessário
# create_shaft("shaft_2")

# %% 2. Engrenagem — Stage 1
# create_gear_pair(sys_shaft_1, "Pinion_S1", sys_shaft_2, "Wheel_S1", stage_name="stage_1") tenho de ter o workspace com pelo menos 2 veios (os dois aqui identificados)
# Descomente após criar gear_pair_1.py via create_gear_pair()
#
_run("stages/stage_1/gear_pair_1.py")
print(f"[gear] Ft={forces_stage_1.Ft:.1f}N  Fr={forces_stage_1.Fr:.1f}N")

# %% 3. Veios — estática, falha estática, fadiga
_run("shafts/shaft_1/shaft_1.py")
print(f"[shaft_1] M_max={sr_shaft_1.M_res_max:.0f}N·mm  nf_Good={stress_shaft_1.most_critical.nf_goodman:.3f}")

# %% 4. Rolamentos
_run("shafts/shaft_1/elements/bearing_A/bearing_A.py")
_run("shafts/shaft_1/elements/bearing_B/bearing_B.py")

# %% 5. Stage 1 — GearStage + GearSystem
_run("stages/stage_1/stage_1.py")

# %% 6. Sumário final
print()
print("=" * 62)
print(" SUMÁRIO FINAL")
print("=" * 62)

# Veios
print(f"  {'Veio':<10} {'n [rpm]':>8} {'M_max [N·mm]':>14} {'nf_Good':>9} {'n_gov':>7} {'Gov.':>6}")
print(f"  {'-'*10} {'-'*8} {'-'*14} {'-'*9} {'-'*7} {'-'*6}")
for _nome, _sr, _stress, _sf, _spd in [
    ("shaft_1", sr_shaft_1, stress_shaft_1, sf_shaft_1, sys_shaft_1.speed_rpm),
    ("shaft_2", sr_shaft_2, stress_shaft_2, sf_shaft_2, sys_shaft_2.speed_rpm),
    ("shaft_3", sr_shaft_3, stress_shaft_3, sf_shaft_3, sys_shaft_3.speed_rpm),
]:
    _mc  = _stress.most_critical
    _sfc = _sf.critical_section
    _nfg = f"{_mc.nf_goodman:.3f}" if _mc.nf_goodman < 1e6 else "∞"
    print(f"  {_nome:<10} {_spd:>8.1f} {_sr.M_res_max:>14.0f} {_nfg:>9} {_sfc.governing_n:>7.3f} {_sfc.governing_theory:>6}")

# Estágios
print()
print(f"  {'Estágio':<10} {'i':>6} {'n_driver':>10} {'n_driven':>10} {'al [mm]':>9} {'Estado':>8}")
print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*10} {'-'*9} {'-'*8}")
for _stage in gear_system.stages:
    _ok = "✓" if not _stage.validate() else "✗"
    print(f"  {_stage.label:<10} {_stage.ratio:>6.4f} {_stage.shaft_driver.speed_rpm:>10.1f} {_stage.shaft_driven.speed_rpm:>10.1f} {_stage.geometry.al:>9.3f} {_ok:>8}")

# Sistema
print()
print(f"  {gear_system.summary()}")
print()
print("Ficheiros exportados para shafts/*/results/ e stages/*/elements/*/results/")

# %% 7. Diagramas — shaft_1
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

def _fmt(v, _):
    return f"{v/1e6:.2f}k" if abs(v) >= 1e6 else (f"{v/1e3:.1f}k" if abs(v) >= 1e3 else f"{v:.0f}")

def _panel(ax, x, data, title, ylabel, bear, gears, color="#89b4fa"):
    ax.fill_between(x, data, alpha=0.15, color=color)
    ax.plot(x, data, color=color, lw=1.4)
    ax.axhline(0, color="#45475a", lw=0.8)
    for b in bear:
        ax.axvline(b.position, color="#585b70", lw=0.8, ls="--",
                   label="Rolamento" if b == bear[0] else "")
    for g in gears:
        ax.axvline(g.position, color="#fab387", lw=0.8, ls=":",
                   label="Engrenagem" if g == gears[0] else "")
    i_max = int(np.argmax(np.abs(data)))
    ax.annotate(f"{data[i_max]:+.0f}",
                xy=(x[i_max], data[i_max]), xytext=(6, 4),
                textcoords="offset points", fontsize=7, color="#cdd6f4")
    ax.set_title(title, fontsize=9, color="#cdd6f4", pad=4)
    ax.set_xlabel("x  [mm]", fontsize=8, color="#6c7086")
    ax.set_ylabel(ylabel, fontsize=8, color="#6c7086")
    ax.tick_params(labelsize=7, colors="#6c7086")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    ax.set_facecolor("#181825")
    ax.grid(color="#313244", lw=0.4, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor("#313244")

# ── figura 1: esforços ────────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(13, 7))
fig1.suptitle("shaft_1 — Diagramas de Esforços", fontsize=11,
              fontweight="bold", color="#cdd6f4")
fig1.patch.set_facecolor("#1e1e2e")

gs1 = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.52, wspace=0.35)

x    = sr_shaft_1.x
bear = sys_shaft_1.bearings
gear = sys_shaft_1.gears

_panel(fig1.add_subplot(gs1[0, 0]), x, sr_shaft_1.V_xz,
       "Corte XZ",            "V  [N]",    bear, gear)
_panel(fig1.add_subplot(gs1[0, 1]), x, sr_shaft_1.V_xy,
       "Corte XY",            "V  [N]",    bear, gear, color="#a6e3a1")
_panel(fig1.add_subplot(gs1[1, 0]), x, sr_shaft_1.M_xz,
       "Momento Fletor XZ",   "M  [N·mm]", bear, gear)
_panel(fig1.add_subplot(gs1[1, 1]), x, sr_shaft_1.M_xy,
       "Momento Fletor XY",   "M  [N·mm]", bear, gear, color="#a6e3a1")

# ── figura 2: M_res + T + σ'_eq ──────────────────────────────────────────────
fig2 = plt.figure(figsize=(13, 7))
fig2.suptitle("shaft_1 — Momento Resultante · Torção · Tensão Equivalente",
              fontsize=11, fontweight="bold", color="#cdd6f4")
fig2.patch.set_facecolor("#1e1e2e")

gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.52, wspace=0.35)

# M_res
_panel(fig2.add_subplot(gs2[0, 0]), x, sr_shaft_1.M_res,
       "Momento Resultante",  "M  [N·mm]", bear, gear, color="#f38ba8")

# Torção
_panel(fig2.add_subplot(gs2[0, 1]), x, sr_shaft_1.T,
       "Torção",              "T  [N·mm]", bear, gear, color="#cba6f7")

# σ'_eq contínua (nominal, sem Kf)
d_arr   = np.array([sys_shaft_1.shaft.diameter_at(float(xi)) for xi in x])
sig_b   = np.where(d_arr > 0, 32.0 * sr_shaft_1.M_res / (np.pi * d_arr**3), 0.0)
tau_t   = np.where(d_arr > 0, 16.0 * np.abs(sr_shaft_1.T) / (np.pi * d_arr**3), 0.0)
sig_eq  = np.sqrt(sig_b**2 + 3.0 * tau_t**2)

ax_eq = fig2.add_subplot(gs2[1, 0])
ax_eq.fill_between(x, sig_eq, alpha=0.20, color="#89b4fa")
ax_eq.plot(x, sig_eq, color="#89b4fa", lw=1.6)
ax_eq.axhline(0, color="#45475a", lw=0.8)
for b in bear:
    ax_eq.axvline(b.position, color="#585b70", lw=0.8, ls="--")
for g in gear:
    ax_eq.axvline(g.position, color="#fab387", lw=0.8, ls=":")
i_max = int(np.argmax(sig_eq))
ax_eq.annotate(f"{sig_eq[i_max]:.1f} MPa",
               xy=(x[i_max], sig_eq[i_max]), xytext=(8, 6),
               textcoords="offset points", fontsize=8, color="#cdd6f4",
               arrowprops=dict(arrowstyle="-", color="#585b70", lw=0.6))
ax_eq.set_title("σ'_eq nominal  (sem Kf)", fontsize=9, color="#cdd6f4", pad=4)
ax_eq.set_xlabel("x  [mm]", fontsize=8, color="#6c7086")
ax_eq.set_ylabel("σ'_eq  [MPa]", fontsize=8, color="#6c7086")
ax_eq.set_ylim(bottom=0)
ax_eq.tick_params(labelsize=7, colors="#6c7086")
ax_eq.set_facecolor("#181825")
ax_eq.grid(color="#313244", lw=0.4)
for sp in ax_eq.spines.values():
    sp.set_edgecolor("#313244")

# nf_Goodman nos ombros
ax_nf = fig2.add_subplot(gs2[1, 1])
secs = [s for s in stress_shaft_1.sections if math.isfinite(s.nf_goodman)]
if secs:
    xs_  = [s.x          for s in secs]
    sa_  = [s.sigma_a    for s in secs]
    sm_  = [s.sigma_m    for s in secs]
    se_  = [s.Se_prime   for s in secs]
    nfg_ = [s.nf_goodman for s in secs]

    ax_nf.bar(xs_, sa_, width=6, color="#f38ba8", alpha=0.75, label="σ'_a")
    ax_nf.bar(xs_, sm_, width=6, bottom=sa_, color="#fab387", alpha=0.65, label="σ'_m")
    ax_nf.step(xs_, se_, where="mid", color="#a6e3a1", lw=1.2, ls="--", label="S'_e")

    for i, (_x, _nfg) in enumerate(zip(xs_, nfg_)):
        _clr = "#a6e3a1" if _nfg >= 1.5 else ("#f9e2af" if _nfg >= 1.0 else "#f38ba8")
        ax_nf.annotate(f"{_nfg:.2f}",
                       xy=(_x, sa_[i] + sm_[i] + 0.5),
                       ha="center", va="bottom", fontsize=7,
                       color=_clr, fontweight="bold")

ax_nf.set_title("Secções críticas — nf Goodman", fontsize=9, color="#cdd6f4", pad=4)
ax_nf.set_xlabel("x  [mm]", fontsize=8, color="#6c7086")
ax_nf.set_ylabel("Tensão  [MPa]", fontsize=8, color="#6c7086")
ax_nf.tick_params(labelsize=7, colors="#6c7086")
ax_nf.legend(fontsize=7, facecolor="#313244", edgecolor="#45475a",
             labelcolor="#cdd6f4")
ax_nf.set_facecolor("#181825")
ax_nf.grid(color="#313244", lw=0.4, axis="y")
for sp in ax_nf.spines.values():
    sp.set_edgecolor("#313244")

fig2.patch.set_facecolor("#1e1e2e")

plt.show()
print("[plot] Diagramas shaft_1 gerados.")