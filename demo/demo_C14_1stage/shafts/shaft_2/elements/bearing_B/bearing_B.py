# shafts/shaft_2/elements/bearing_B/bearing_B.py
# Shaft_2 — Rolamento B (Flutuante)
#
# Posição  : x = 310 mm
# Tipo     : flutuante — suporta apenas carga radial
# C = 45 000 N   C0 = 28 000 N
#
# Depende de: sr_shaft_2 (StaticsResult de shaft_2.py)

# %% Rolamento B — Shaft_2

bl = BearingLifeSolver()
bf_shaft_2 = bl.extract_bearing_forces(sr_shaft_2, sys_shaft_2)

bearing_B_s2 = sys_shaft_2.bearings[1]   # posição 310mm, label="B"
Fr_B, Fa_B = bf_shaft_2[bearing_B_s2.label]

res_bearing_B_s2 = bl.solve_bearing(
    bearing_B_s2,
    Fr_B, Fa_B,
    sys_shaft_2.speed_rpm,
    sys_shaft_2.design_life_hours,
)

print(f"ROLAMENTO B — Shaft_2  (flutuante @ x={bearing_B_s2.position:.0f}mm)")
print(f"  Fr = {res_bearing_B_s2.Fr:.1f} N   Fa = {res_bearing_B_s2.Fa:.1f} N")
print(f"  P  = {res_bearing_B_s2.P:.1f} N")
print(f"  L10h = {res_bearing_B_s2.L10h:.0f} h   S0 = {res_bearing_B_s2.S0:.3f}")
print(f"  Estado: {'✓ OK' if res_bearing_B_s2.is_safe else '✗ FALHA'}")

export_bearing(
    res_bearing_B_s2,
    "shafts/shaft_2/elements/bearing_B/results/life.txt",
    project="demo_C14_1stage",
    element="shaft_2_bearing_B",
    script="bearing_B.py",
    C=bearing_B_s2.C,
    C0=bearing_B_s2.C0,
)
