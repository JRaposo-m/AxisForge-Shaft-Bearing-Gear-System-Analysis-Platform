# stages/stage_1/stage_1.py
# Stage_1 — Redutor 1 Estágio C14
#
# Liga shaft_1 (driver, pinion) → shaft_2 (driven, wheel)
# Valida: distância entre centros, contacto X global, velocidades
#
# Depende de:
#   stages/stage_1/elements/gear/gear.py  → geo_stage_1, forces_stage_1
#   shafts/shaft_1/shaft_1.py             → sys_shaft_1
#   shafts/shaft_2/shaft_2.py             → sys_shaft_2

# %% GearStage

stage_1 = GearStage(
    label="Stage_1",
    shaft_driver=sys_shaft_1,
    shaft_driven=sys_shaft_2,
    geometry=geo_stage_1,
    forces=forces_stage_1,
    x_gear_driver_local=200.0,   # shaft_1: origin=0  → x_global=200mm
    x_gear_driven_local=175.0,   # shaft_2: origin=25 → x_global=200mm ✓
)

print("STAGE_1 — Validação de Estágio")
print(f"  i (z2/z1)              = {stage_1.ratio:.4f}")
print(f"  al geométrico          = {stage_1.centre_distance_geometric:.3f} mm")
print(f"  al especificado        = {geo_stage_1.al:.3f} mm")
print(f"  x_global driver        = {stage_1.x_gear_driver_global:.1f} mm")
print(f"  x_global driven        = {stage_1.x_gear_driven_global:.1f} mm")

stage_errors = stage_1.validate()
if stage_errors:
    print("  ERROS:")
    for e in stage_errors:
        print(f"    ✗ {e}")
else:
    print("  ✓ Estágio válido")

# %% GearSystem

gear_system = GearSystem(name="demo_C14_1stage")
gear_system.add_stage(stage_1)
gear_system.propagate_speeds()

print("\nGEARSYSTEM — Propagação de Velocidades")
for sh in gear_system.shafts:
    print(f"  {sh.name}: {sh.speed_rpm:.1f} rpm")

print(f"\n{gear_system.summary()}")

system_errors = gear_system.validate()
if system_errors:
    print("ERROS DE SISTEMA:")
    for e in system_errors:
        print(f"  ✗ {e}")
else:
    print("✓ Sistema válido")