# shafts/shaft_1/elements/bearing_B/bearing_B.py
# demo_reducer_2stages — shaft_1 — Rolamento B
# Gerado automaticamente por AxisForge ProjectManager
#
# Depende de: sr_shaft_1, sys_shaft_1 (de shaft_1.py)

# %% Bearing life
bl = BearingLifeSolver()
bf = bl.extract_bearing_forces(sr_shaft_1, sys_shaft_1)

bearing_B = sys_shaft_1.bearings[0]   # TODO: verificar índice
Fr_B, Fa_B = bf[bearing_B.label]

res_bearing_B_shaft_1 = bl.solve_bearing(
    bearing_B, Fr_B, Fa_B,
    sys_shaft_1.speed_rpm,
    sys_shaft_1.design_life_hours,
)

print(f"Rolamento B — shaft_1  L10h={res_bearing_B_shaft_1.L10h:.0f}h  S0={res_bearing_B_shaft_1.S0:.3f}")

export_bearing(
    res_bearing_B_shaft_1,
    "shafts/shaft_1/elements/bearing_B/results/life.txt",
    project="demo_reducer_2stages", element="shaft_1_bearing_B", script="bearing_B.py",
    C=bearing_B.C, C0=bearing_B.C0,
)
