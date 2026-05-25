# scripts/plots.py
# demo_C14_1stage — Plots Técnicos
#
# Pré-requisito: full_pipeline.py executado no mesmo namespace.
# Variáveis necessárias:
#   sr_shaft_1, sr_shaft_2       — StaticsResult
#   sys_shaft_1, sys_shaft_2     — MechanicalSystem
#   stress_shaft_1, stress_shaft_2 — StressResult
#   mat                          — Material
#   gear_system                  — GearSystem
#
# Executa com: Ctrl+Shift+Enter após full_pipeline.py
# Ou via _run("scripts/plots.py") no final do full_pipeline.py

import matplotlib.gridspec as _gs
import matplotlib.ticker   as _tk

# ---------------------------------------------------------------------------
# Paleta — Catppuccin Mocha com contraste aumentado
# ---------------------------------------------------------------------------
_BG_FIG   = "#1e1e2e"   # fundo da figura
_BG_AX    = "#181825"   # fundo dos eixos
_GRID     = "#313244"   # grid e spines
_LINE     = "#89b4fa"   # linha principal (azul)
_FILL     = "#89b4fa"   # fill (mesmo tom, alpha baixo)
_BEARING  = "#6c7086"   # linha de rolamento (tracejado)
_TEXT     = "#cdd6f4"   # títulos, labels, ticks — claro
_ACCENT   = "#f38ba8"   # secção crítica de fadiga (vermelho suave)

def _apply_style(ax):
    """Estilo base comum a todos os eixos."""
    ax.set_facecolor(_BG_AX)
    ax.tick_params(labelsize=8, colors=_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.grid(color=_GRID, lw=0.4, zorder=0)
    ax.axhline(0, color=_GRID, lw=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)


def _mark_bearings(ax, system):
    """Linhas verticais tracejadas nos rolamentos."""
    for b in system.bearings:
        ax.axvline(b.position, color=_BEARING, lw=0.9, ls="--", alpha=0.8)


def _mark_critical(ax, stress_result, y_top):
    """Linha vertical na secção crítica de fadiga."""
    mc = stress_result.most_critical
    if mc is not None:
        ax.axvline(mc.x, color=_ACCENT, lw=1.0, ls=":", alpha=0.9)


# ---------------------------------------------------------------------------
# Plot 1 — Diagramas de Esforços (2×2)
# ---------------------------------------------------------------------------

def _plot_shaft_diagrams(sr, system, stress_result, title: str):
    """
    4 painéis: Corte XZ, Corte XY, Momento Resultante, Torção.
    Corte em N, Momento e Torção em N·m.
    Linha vermelha pontilhada na secção crítica de fadiga.
    """
    x    = sr.x
    bear = system.bearings

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(title, fontsize=12, fontweight="bold",
                 color=_TEXT, y=0.97)
    grid = _gs.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38)

    panels = [
        (grid[0, 0], sr.V_xz,         "Corte XZ",           "V  [N]",    False),
        (grid[0, 1], sr.V_xy,         "Corte XY",           "V  [N]",    False),
        (grid[1, 0], sr.M_res / 1e3,  "Momento Resultante", "M  [N·m]",  True),
        (grid[1, 1], sr.T    / 1e3,   "Torção",             "T  [N·m]",  True),
    ]

    for spec, data, label, ylabel, mark_crit in panels:
        ax = fig.add_subplot(spec)
        ax.fill_between(x, data, alpha=0.12, color=_FILL)
        ax.plot(x, data, color=_LINE, lw=1.5)
        _mark_bearings(ax, system)
        if mark_crit:
            _mark_critical(ax, stress_result, data.max())
        ax.set_title(label, fontsize=9, pad=5)
        ax.set_xlabel("x  [mm]", fontsize=8)
        ax.set_ylabel(ylabel,    fontsize=8)
        _apply_style(ax)

    fig.patch.set_facecolor(_BG_FIG)
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — Tensão Equivalente Von Mises ao longo do veio
# ---------------------------------------------------------------------------

def _plot_stress_envelope(sr, system, title: str):
    """
    Tensão equivalente σ'_eq = √(σ_b² + 3τ_t²) em MPa.
    Linha vermelha pontilhada na secção crítica.
    """
    x   = sr.x
    M   = sr.M_res   # N·mm
    T   = sr.T       # N·mm
    d   = np.array([system.shaft.diameter_at(float(xi)) for xi in x])

    sig_b  = np.where(d > 0, 32.0 * M / (np.pi * d**3), 0.0)
    tau_t  = np.where(d > 0, 16.0 * np.abs(T) / (np.pi * d**3), 0.0)
    sig_eq = np.sqrt(sig_b**2 + 3.0 * tau_t**2)   # MPa

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold",
                 color=_TEXT, y=1.01)

    ax.fill_between(x, sig_eq, alpha=0.15, color=_FILL)
    ax.plot(x, sig_eq, color=_LINE, lw=1.6)
    _mark_bearings(ax, system)

    # Secção crítica
    i_max = int(np.argmax(sig_eq))
    ax.axvline(x[i_max], color=_ACCENT, lw=1.0, ls=":", alpha=0.9)

    ax.set_xlabel("x  [mm]", fontsize=9)
    ax.set_ylabel("σ'eq  [MPa]", fontsize=9)
    ax.set_ylim(bottom=0)
    _apply_style(ax)
    fig.patch.set_facecolor(_BG_FIG)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3 — Posições Globais (diagrama de posição)
# ---------------------------------------------------------------------------

def _plot_global_layout(gear_system_obj, title: str = "Layout Global — Posições dos Veios"):
    """
    Vista em planta (X–Y) e perfil (X–Z) das posições dos veios e engrenagens.
    Cada veio é representado como um segmento horizontal.
    Rolamentos e engrenagens são marcados sobre o veio.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold",
                 color=_TEXT, y=0.98)

    view_labels = ["Vista XY (plano horizontal)", "Vista XZ (plano vertical)"]
    coord_idx   = [0, 1]   # 0=Y, 1=Z para os dois planos

    for ax, view_lbl, cidx in zip(axes, view_labels, coord_idx):
        for sh in gear_system_obj.shafts:
            L       = sh.shaft.total_length
            ox      = sh.shaft_origin_x
            pos_c   = sh.shaft_position[cidx]   # Y ou Z

            # Linha do veio
            ax.plot([ox, ox + L], [pos_c, pos_c],
                    color=_LINE, lw=3.0, solid_capstyle="round",
                    label=sh.name)
            ax.text(ox + L + 4, pos_c, sh.name,
                    color=_TEXT, fontsize=7.5, va="center")

            # Rolamentos
            for b in sh.bearings:
                xg = ox + b.position
                ax.plot(xg, pos_c, marker="^", ms=7,
                        color="#a6e3a1", zorder=5)   # verde suave

            # Engrenagens
            for g in sh.gears:
                xg = ox + g.position
                ax.plot(xg, pos_c, marker="o", ms=8,
                        color="#fab387", zorder=5)   # laranja suave

        ax.set_xlabel("x_global  [mm]", fontsize=9)
        ax.set_ylabel("Y  [mm]" if cidx == 0 else "Z  [mm]", fontsize=9)
        ax.set_title(view_lbl, fontsize=9, pad=6)
        _apply_style(ax)
        ax.set_ylim(-40, max(
            sh.shaft_position[cidx] for sh in gear_system_obj.shafts
        ) + 40)

    # Legenda manual
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=_LINE,     lw=3,  label="Veio"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#a6e3a1",
               ms=8, lw=0, label="Rolamento"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#fab387",
               ms=8, lw=0, label="Engrenagem"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               frameon=False, fontsize=8, labelcolor=_TEXT)

    fig.patch.set_facecolor(_BG_FIG)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Gerar todos os plots
# ---------------------------------------------------------------------------

# %% Posições Globais

print("POSIÇÕES GLOBAIS — demo_C14_1stage")
print(f"  {'Elemento':<20} {'Veio':<10} {'x_local':>9} {'x_global':>10} {'y':>8} {'z':>8}")
print(f"  {'-'*20} {'-'*10} {'-'*9} {'-'*10} {'-'*8} {'-'*8}")
for sh in gear_system.shafts:
    ox = sh.shaft_origin_x
    y  = sh.shaft_position[0]
    z  = sh.shaft_position[1]
    for b in sh.bearings:
        xg = ox + b.position
        print(f"  {'Bearing ' + b.label:<20} {sh.name:<10} {b.position:>7.1f}mm {xg:>8.1f}mm {y:>6.1f}mm {z:>6.1f}mm")
    for g in sh.gears:
        xg = ox + g.position
        print(f"  {'Gear (' + g.label + ')':<20} {sh.name:<10} {g.position:>7.1f}mm {xg:>8.1f}mm {y:>6.1f}mm {z:>6.1f}mm")

# %% Diagramas de Esforços

fig_diag_s1 = _plot_shaft_diagrams(
    sr_shaft_1, sys_shaft_1, stress_shaft_1,
    "Shaft_1 (Driver) — Diagramas de Esforços",
)
fig_diag_s2 = _plot_shaft_diagrams(
    sr_shaft_2, sys_shaft_2, stress_shaft_2,
    "Shaft_2 (Driven) — Diagramas de Esforços",
)

# %% Tensão Equivalente

fig_stress_s1 = _plot_stress_envelope(
    sr_shaft_1, sys_shaft_1,
    "Shaft_1 (Driver) — Tensão Equivalente Von Mises",
)
fig_stress_s2 = _plot_stress_envelope(
    sr_shaft_2, sys_shaft_2,
    "Shaft_2 (Driven) — Tensão Equivalente Von Mises",
)

# %% Layout Global

fig_layout = _plot_global_layout(
    gear_system,
    "demo_C14_1stage — Layout Global dos Veios",
)

print("Plots gerados: fig_diag_s1, fig_diag_s2, fig_stress_s1, fig_stress_s2, fig_layout")
