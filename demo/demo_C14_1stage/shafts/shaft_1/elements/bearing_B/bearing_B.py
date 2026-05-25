# shafts/shaft_1/elements/bearing_B/bearing_B.py
# Shaft_1 — Rolamento B (Flutuante)
#
# Posição  : x = 350 mm
# Tipo     : flutuante — suporta apenas carga radial
# C = 60 000 N   C0 = 38 000 N
#
# Depende de: sr_shaft_1 (StaticsResult de shaft_1.py)

# %% Rolamento B — Shaft_1

bl = BearingLifeSolver()
bf_shaft_1 = bl.extract_bearing_forces(sr_shaft_1, sys_shaft_1)

bearing_B_s1 = sys_shaft_1.bearings[1]   # posição 350mm, label="B"
Fr_B, Fa_B = bf_shaft_1[bearing_B_s1.label]

res_bearing_B_s1 = bl.solve_bearing(
    bearing_B_s1,
    Fr_B, Fa_B,
    sys_shaft_1.speed_rpm,
    sys_shaft_1.design_life_hours,
)

print(f"ROLAMENTO B — Shaft_1  (flutuante @ x={bearing_B_s1.position:.0f}mm)")
print(f"  Fr = {res_bearing_B_s1.Fr:.1f} N   Fa = {res_bearing_B_s1.Fa:.1f} N")
print(f"  P  = {res_bearing_B_s1.P:.1f} N")
print(f"  L10h = {res_bearing_B_s1.L10h:.0f} h   S0 = {res_bearing_B_s1.S0:.3f}")
print(f"  Estado: {'✓ OK' if res_bearing_B_s1.is_safe else '✗ FALHA'}")

export_bearing(
    res_bearing_B_s1,
    "shafts/shaft_1/elements/bearing_B/results/life.txt",
    project="demo_C14_1stage",
    element="shaft_1_bearing_B",
    script="bearing_B.py",
    C=bearing_B_s1.C,
    C0=bearing_B_s1.C0,
)
