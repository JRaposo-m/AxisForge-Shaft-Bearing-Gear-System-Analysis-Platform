# stages/stage_1/stage_1.py
# demo_reducer_2stages — stage_1 — GearStage + GearSystem
# Gerado automaticamente por AxisForge ProjectManager
#
# Depende de: geo_stage_N, forces_stage_N (de gear_pair_N.py)
#             sys_shaft_driver, sys_shaft_driven (de shaft_N.py)
# Executar via full_pipeline.py → _run("stages/stage_1/stage_1.py")

# %% GearStage
stage_1 = GearStage(
    label="stage_1",
    shaft_driver=sys_shaft_1,      # TODO: MechanicalSystem do veio condutor
    shaft_driven=sys_shaft_2,      # TODO: MechanicalSystem do veio conduzido
    geometry=geo_stage_1,   # TODO: GearGeometryResult de gear_pair_N.py
    forces=forces_stage_1,  # TODO: GearForceResult de gear_pair_N.py
    x_gear_driver_local=150.0,           # TODO: posição da engrenagem no veio condutor [mm]
    x_gear_driven_local=150.0,           # TODO: posição da engrenagem no veio conduzido [mm]
)

# %% GearSystem
gear_system = GearSystem(name="demo_reducer_2stages")
gear_system.add_stage(stage_1)
# gear_system.add_stage(stage_2)  # TODO: adicionar mais estágios se necessário
gear_system.propagate_speeds()

errors = gear_system.validate()
if errors:
    print("⚠ GearSystem erros:")
    for e in errors:
        print(f"  - {e}")
else:
    print(f"✓ {gear_system.summary()}")
