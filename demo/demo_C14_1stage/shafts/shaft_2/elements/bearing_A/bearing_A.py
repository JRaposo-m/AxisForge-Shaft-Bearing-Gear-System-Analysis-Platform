# shafts/shaft_2/elements/bearing_A/bearing_A.py
# Shaft_2 — Rolamento A (Fixo)
#
# Posição  : x = 40 mm
# Tipo     : fixo — suporta carga radial + axial
# C = 45 000 N   C0 = 28 000 N
#
# Depende de: sr_shaft_2 (StaticsResult de shaft_2.py)

# %% Rolamento A — Shaft_2

bl = BearingLifeSolver()
bf_shaft_2 = bl.extract_bearing_forces(sr_shaft_2, sys_shaft_2)

bearing_A_s2 = sys_shaft_2.bearings[0]   # posição 40mm, label="A"
Fr_A, Fa_A = bf_shaft_2[bearing_A_s2.label]

res_bearing_A_s2 = bl.solve_bearing(
    bearing_A_s2,
    Fr_A, Fa_A,
    sys_shaft_2.speed_rpm,
    sys_shaft_2.design_life_hours,
)

print(f"ROLAMENTO A — Shaft_2  (fixo @ x={bearing_A_s2.position:.0f}mm)")
print(f"  Fr = {res_bearing_A_s2.Fr:.1f} N   Fa = {res_bearing_A_s2.Fa:.1f} N")
print(f"  P  = {res_bearing_A_s2.P:.1f} N")
print(f"  L10h = {res_bearing_A_s2.L10h:.0f} h   S0 = {res_bearing_A_s2.S0:.3f}")
print(f"  Estado: {'✓ OK' if res_bearing_A_s2.is_safe else '✗ FALHA'}")

export_bearing(
    res_bearing_A_s2,
    "shafts/shaft_2/elements/bearing_A/results/life.txt",
    project="demo_C14_1stage",
    element="shaft_2_bearing_A",
    script="bearing_A.py",
    C=bearing_A_s2.C,
    C0=bearing_A_s2.C0,
)
