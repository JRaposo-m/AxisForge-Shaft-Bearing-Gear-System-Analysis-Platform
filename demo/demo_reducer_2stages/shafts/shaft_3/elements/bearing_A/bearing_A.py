# shafts/shaft_3/elements/bearing_A/bearing_A.py
# demo_reducer_2stages — shaft_3 — Rolamento A
# Gerado automaticamente por AxisForge ProjectManager
#
# Depende de: sr_shaft_3, sys_shaft_3 (de shaft_3.py)

# %% Bearing life
bl = BearingLifeSolver()
bf = bl.extract_bearing_forces(sr_shaft_3, sys_shaft_3)

bearing_A = sys_shaft_3.bearings[0]   # TODO: verificar índice
Fr_A, Fa_A = bf[bearing_A.label]

res_bearing_A_shaft_3 = bl.solve_bearing(
    bearing_A, Fr_A, Fa_A,
    sys_shaft_3.speed_rpm,
    sys_shaft_3.design_life_hours,
)

print(f"Rolamento A — shaft_3  L10h={res_bearing_A_shaft_3.L10h:.0f}h  S0={res_bearing_A_shaft_3.S0:.3f}")

export_bearing(
    res_bearing_A_shaft_3,
    "shafts/shaft_3/elements/bearing_A/results/life.txt",
    project="demo_reducer_2stages", element="shaft_3_bearing_A", script="bearing_A.py",
    C=bearing_A.C, C0=bearing_A.C0,
)
