# scripts/full_pipeline.py
# demo_C14_1stage — Pipeline Completo
#
# Orquestrador puro: executa os ficheiros do projecto em ordem.
# Cada _run() lê o ficheiro e executa-o no namespace partilhado do runner —
# todas as variáveis ficam disponíveis nas secções seguintes.
#
# Ordem de execução:
#   1. gear.py      → geo_stage_1, forces_stage_1
#   2. shaft_1.py   → sys_shaft_1, sr_shaft_1, sf_shaft_1, stress_shaft_1
#   3. shaft_2.py   → sys_shaft_2, sr_shaft_2, sf_shaft_2, stress_shaft_2
#   4. bearing_A/B  → res_bearing_A_s1, res_bearing_B_s1, ...
#   5. stage_1.py   → stage_1, gear_system
#   6. sumário
#
# Pré-requisito: projecto aberto via File → Open Project (faz os.chdir para a raiz)
# Executa com: Ctrl+Shift+Enter (Run All) no ScriptEditor

import pathlib as _pathlib

def _run(_rel: str) -> None:
    """Executa um ficheiro .py no namespace partilhado do runner."""
    _p = _pathlib.Path.cwd() / _rel
    exec(compile(_p.read_text(encoding="utf-8"), str(_p), "exec"), globals())  # noqa: S102

print("=" * 62)
print(" AxisForge — demo_C14_1stage — Pipeline Completo")
print("=" * 62)

# %% 1. Engrenagem — Stage_1

_run("stages/stage_1/elements/gear/gear.py")
print(f"[1] Gear  Ft={forces_stage_1.Ft:.1f}N  Fr={forces_stage_1.Fr:.1f}N  "
      f"T1={forces_stage_1.T1_Nmm/1000:.1f}N·m  T2={forces_stage_1.T2_Nmm/1000:.1f}N·m")

# %% 2. Shaft_1 — Veio Condutor

_run("shafts/shaft_1/shaft_1.py")
mc1  = stress_shaft_1.most_critical
sfc1 = sf_shaft_1.critical_section
print(f"[2] Shaft_1  M_max={sr_shaft_1.M_res_max:.0f}N·mm  "
      f"nf_Good={mc1.nf_goodman:.3f}  n_gov={sfc1.governing_n:.2f}({sfc1.governing_theory})")

# %% 3. Shaft_2 — Veio Conduzido

_run("shafts/shaft_2/shaft_2.py")
mc2  = stress_shaft_2.most_critical
sfc2 = sf_shaft_2.critical_section
print(f"[3] Shaft_2  M_max={sr_shaft_2.M_res_max:.0f}N·mm  "
      f"nf_Good={mc2.nf_goodman:.3f}  n_gov={sfc2.governing_n:.2f}({sfc2.governing_theory})")

# %% 4. Rolamentos — Shaft_1

_run("shafts/shaft_1/elements/bearing_A/bearing_A.py")
print(f"[4] Shaft_1 Bearing A  L10h={res_bearing_A_s1.L10h:.0f}h  "
      f"S0={res_bearing_A_s1.S0:.3f}  {'✓' if res_bearing_A_s1.is_safe else '✗'}")

_run("shafts/shaft_1/elements/bearing_B/bearing_B.py")
print(f"[4] Shaft_1 Bearing B  L10h={res_bearing_B_s1.L10h:.0f}h  "
      f"S0={res_bearing_B_s1.S0:.3f}  {'✓' if res_bearing_B_s1.is_safe else '✗'}")

# %% 5. Rolamentos — Shaft_2

_run("shafts/shaft_2/elements/bearing_A/bearing_A.py")
print(f"[5] Shaft_2 Bearing A  L10h={res_bearing_A_s2.L10h:.0f}h  "
      f"S0={res_bearing_A_s2.S0:.3f}  {'✓' if res_bearing_A_s2.is_safe else '✗'}")

_run("shafts/shaft_2/elements/bearing_B/bearing_B.py")
print(f"[5] Shaft_2 Bearing B  L10h={res_bearing_B_s2.L10h:.0f}h  "
      f"S0={res_bearing_B_s2.S0:.3f}  {'✓' if res_bearing_B_s2.is_safe else '✗'}")

# %% 6. Stage_1 — GearStage + GearSystem

_run("stages/stage_1/stage_1.py")
print(f"[6] {gear_system.summary()}")

# %% 7. Sumário Final

print()
print("=" * 62)
print(" SUMÁRIO FINAL")
print("=" * 62)
print(f"  {'Veio':<10} {'n [rpm]':>8} {'M_max [N·mm]':>14} {'nf_Good':>9} {'n_gov':>7} {'Gov.':>6}")
print(f"  {'-'*10} {'-'*8} {'-'*14} {'-'*9} {'-'*7} {'-'*6}")
for _nome, _sr, _stress, _sf, _spd in [
    ("Shaft_1", sr_shaft_1, stress_shaft_1, sf_shaft_1, sys_shaft_1.speed_rpm),
    ("Shaft_2", sr_shaft_2, stress_shaft_2, sf_shaft_2, sys_shaft_2.speed_rpm),
]:
    _mc  = _stress.most_critical
    _sfc = _sf.critical_section
    _nfg = f"{_mc.nf_goodman:.3f}" if _mc.nf_goodman < 1e6 else "∞"
    print(f"  {_nome:<10} {_spd:>8.0f} {_sr.M_res_max:>14.0f} {_nfg:>9} {_sfc.governing_n:>7.3f} {_sfc.governing_theory:>6}")

print()
print("Ficheiros exportados para shafts/*/results/ e stages/*/elements/*/results/")