# shafts/shaft_2/elements/bearing_B/bearing_B.py
# demo_reducer_2stages — shaft_2 — Rolamento B
# Gerado automaticamente por AxisForge ProjectManager
#
# Depende de: sr_shaft_2, sys_shaft_2 (de shaft_2.py)

# %% Bearing life
bl = BearingLifeSolver()
bf = bl.extract_bearing_forces(sr_shaft_2, sys_shaft_2)

bearing_B = sys_shaft_2.bearings[0]   # TODO: verificar índice
Fr_B, Fa_B = bf[bearing_B.label]

res_bearing_B_shaft_2 = bl.solve_bearing(
    bearing_B, Fr_B, Fa_B,
    sys_shaft_2.speed_rpm,
    sys_shaft_2.design_life_hours,
)

print(f"Rolamento B — shaft_2  L10h={res_bearing_B_shaft_2.L10h:.0f}h  S0={res_bearing_B_shaft_2.S0:.3f}")

export_bearing(
    res_bearing_B_shaft_2,
    "shafts/shaft_2/elements/bearing_B/results/life.txt",
    project="demo_reducer_2stages", element="shaft_2_bearing_B", script="bearing_B.py",
    C=bearing_B.C, C0=bearing_B.C0,
)
