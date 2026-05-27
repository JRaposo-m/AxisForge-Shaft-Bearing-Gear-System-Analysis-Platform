# shafts/shaft_2/elements/bearing_A/bearing_A.py
# demo_reducer_2stages — shaft_2 — Rolamento A
# Gerado automaticamente por AxisForge ProjectManager
#
# Depende de: sr_shaft_2, sys_shaft_2 (de shaft_2.py)

# %% Bearing life
bl = BearingLifeSolver()
bf = bl.extract_bearing_forces(sr_shaft_2, sys_shaft_2)

bearing_A = sys_shaft_2.bearings[0]   # TODO: verificar índice
Fr_A, Fa_A = bf[bearing_A.label]

res_bearing_A_shaft_2 = bl.solve_bearing(
    bearing_A, Fr_A, Fa_A,
    sys_shaft_2.speed_rpm,
    sys_shaft_2.design_life_hours,
)

print(f"Rolamento A — shaft_2  L10h={res_bearing_A_shaft_2.L10h:.0f}h  S0={res_bearing_A_shaft_2.S0:.3f}")

export_bearing(
    res_bearing_A_shaft_2,
    "shafts/shaft_2/elements/bearing_A/results/life.txt",
    project="demo_reducer_2stages", element="shaft_2_bearing_A", script="bearing_A.py",
    C=bearing_A.C, C0=bearing_A.C0,
)
