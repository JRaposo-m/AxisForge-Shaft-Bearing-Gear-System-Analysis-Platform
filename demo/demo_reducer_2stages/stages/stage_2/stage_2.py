# stages/stage_2/stage_2.py
# demo_reducer_2stages — stage_2 — GearStage + GearSystem
# Gerado automaticamente por AxisForge ProjectManager
#
# Depende de: geo_stage_N, forces_stage_N (de gear_pair_N.py)
#             sys_shaft_driver, sys_shaft_driven (de shaft_N.py)
# Executar via full_pipeline.py → _run("stages/stage_2/stage_2.py")
_x_gear_driver_local_s2 = sys_shaft_2.gears[1].position      # local no shaft_2
_x_gear_driven_local_s3 = sys_shaft_3.gears[0].position      # local no shaft_3

# %% GearStage
stage_2 = GearStage(
    label="stage_2",
    shaft_driver=sys_shaft_2,      # TODO: MechanicalSystem do veio condutor
    shaft_driven=sys_shaft_3,      # TODO: MechanicalSystem do veio conduzido
    geometry=geo_stage_2,   # TODO: GearGeometryResult de gear_pair_N.py
    forces=forces_stage_2,  # TODO: GearForceResult de gear_pair_N.py
    x_gear_driver_local=_x_gear_driver_local_s2,           # TODO: posição da engrenagem no veio condutor [mm]
    x_gear_driven_local=_x_gear_driven_local_s3,           # TODO: posição da engrenagem no veio conduzido [mm]
)

# %% GearSystem
gear_system = GearSystem(name="demo_reducer_2stages")
gear_system.add_stage(stage_2)
# gear_system.add_stage(stage_2)  # TODO: adicionar mais estágios se necessário
gear_system.propagate_speeds()

errors = gear_system.validate()
if errors:
    print("⚠ GearSystem erros:")
    for e in errors:
        print(f"  - {e}")
else:
    print(f"✓ {gear_system.summary()}")
