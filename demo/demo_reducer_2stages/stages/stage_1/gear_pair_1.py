# stages/stage_1/gear_pair_1.py
# demo_reducer_2stages — stage_1 — Par de engrenagens 1
# Gerado automaticamente por AxisForge ProjectManager
#
# Preencher: parâmetros geométricos e torque de entrada
# Executar via full_pipeline.py → _run("stages/stage_1/gear_pair_1.py")
#
# Variáveis exportadas para o namespace:
#   geo_stage_1    — GearGeometryResult
#   forces_stage_1 — GearForceResult

# %% Geometria e forças
gs = GearSolver()

geo_stage_1 = gs.compute_geometry(
    mn=4.5,          # TODO: módulo normal [mm]
    z1=16,           # TODO: dentes do pinhão
    z2=24,           # TODO: dentes da roda
    alpha_n_deg=20.0,
    beta_deg=0.0,    # TODO: ângulo de hélice [°] (0 = spur)
    al=None,         # TODO: distância de centros de trabalho [mm] ou None (calculado)
    x1=0.0,          # TODO: coeficiente de adendenda pinhão
    x2=0.0,          # TODO: coeficiente de adendenda roda
    b=50.0,          # TODO: largura de face [mm]
)

forces_stage_1 = gs.compute_forces(
    T1_Nm=200.0,       # TODO: binário de entrada [N·m]
    geometry=geo_stage_1,
)

print(f"Gear pair 1: u={geo_stage_1.u:.4f}  "
      f"al={geo_stage_1.al:.3f}mm  "
      f"Ft={forces_stage_1.Ft:.1f}N  Fr={forces_stage_1.Fr:.1f}N")
