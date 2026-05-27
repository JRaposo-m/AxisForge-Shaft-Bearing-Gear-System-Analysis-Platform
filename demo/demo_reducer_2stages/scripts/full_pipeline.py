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