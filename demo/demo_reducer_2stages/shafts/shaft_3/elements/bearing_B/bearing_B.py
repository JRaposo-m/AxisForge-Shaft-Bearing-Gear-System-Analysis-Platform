# shafts/shaft_3/elements/bearing_B/bearing_B.py
# demo_reducer_2stages — shaft_3 — Rolamento B
# Gerado automaticamente por AxisForge ProjectManager
#
# Depende de: sr_shaft_3, sys_shaft_3 (de shaft_3.py)

# %% Bearing life
bl = BearingLifeSolver()
bf = bl.extract_bearing_forces(sr_shaft_3, sys_shaft_3)

bearing_B = sys_shaft_3.bearings[0]   # TODO: verificar índice
Fr_B, Fa_B = bf[bearing_B.label]

res_bearing_B_shaft_3 = bl.solve_bearing(
    bearing_B, Fr_B, Fa_B,
    sys_shaft_3.speed_rpm,
    sys_shaft_3.design_life_hours,
)

print(f"Rolamento B — shaft_3  L10h={res_bearing_B_shaft_3.L10h:.0f}h  S0={res_bearing_B_shaft_3.S0:.3f}")

export_bearing(
    res_bearing_B_shaft_3,
    "shafts/shaft_3/elements/bearing_B/results/life.txt",
    project="demo_reducer_2stages", element="shaft_3_bearing_B", script="bearing_B.py",
    C=bearing_B.C, C0=bearing_B.C0,
)
