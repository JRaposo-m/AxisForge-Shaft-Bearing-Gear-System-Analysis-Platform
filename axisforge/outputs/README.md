# axisforge.outputs

Records, text and CSV of AxisForge's own objects. It only reads; it never solves, plots or
judges.

```
outputs/
    _format.py        format_table, format_key_values, write_records_csv, read_records_csv, ...
    construction/     the built system (read through ShaftSystem and the existing summary()/repr)
        system.py     SpurHelicalGearSystem: stage, mesh links, resolved torques, every shaft
        shaft.py      ShaftSystem: shaft sections, shoulders, gears, bearings, loads
        bearings.py   assembled Bearing: family, geometry, contact constants, C
        loads.py      the loads of a shaft
    solvers/          one module per module of axisforge.results
        fem_results/shaft_results.py
        bearings/load_distribution/load_distribution_results.py
        bearings/life/basic_life_results.py
        bearings/bearing_analysis_result.py
```

Every module offers `*_records` (plain dicts), `*_text` (readable text) and `print_*` (the same
text on the console).

## Conventions

* The unit is part of the key of a number (`Fr_N`, `delta_r_mm`, `psi_rad`, `M_xz_Nmm`). A key
  without a unit suffix is dimensionless or text. Records keep the unit in which AxisForge
  stores the quantity: results and bearing angles in rad, inputs that AxisForge stores in
  degrees (`alpha_n_deg`, `beta_n_deg`, `phi_deg`, `theta_deg`) in degrees.
* Text may convert for reading and always names the unit in the header: the misalignment psi
  and the slopes are shown in rad and in mrad; element angles in degrees.
* NaN means "not available" (`nan` in CSV, `-` in text). `inf` is kept as `inf`.
* The system description uses the schema `{section, parameter, value, unit}` (it is the
  `system.csv` of the studies).
* No derived engineering quantity and no pass/fail verdict: equilibrium errors, convergence
  flags and residuals are reported as the result holds them.
* Element angles `phi_rad` are in the local frame of the row (0 on the load line), wrapped to
  [-pi, pi) with `(phi + pi) % (2 pi) - pi`.
* Only the standard library and NumPy are used; `axisforge.solvers` and matplotlib are not
  imported.

## BearingNodeData -> record key

| field | key | field | key |
|---|---|---|---|
| `label` | `label` | `Fr` | `Fr_N` |
| `position` | `x_mm` | `Fa` | `Fa_N` |
| `u` | `u_mm` | `M_xz` | `M_xz_Nmm` |
| `v_xz`, `v_xy` | `v_xz_mm`, `v_xy_mm` | `M_xy` | `M_xy_Nmm` |
| `theta_xz`, `theta_xy` | `theta_xz_rad`, `theta_xy_rad` | `psi_xz` | `psi_xz_rad` |
| `Fr_xz`, `Fr_xy` | `Fr_xz_N`, `Fr_xy_N` | `psi_xy` | `psi_xy_rad` |

## Open points

* `ShaftResults` does not state the units of its maxima (`M_max`, `sigma_b_max`, ...): N mm and
  MPa are assumed.
* `SpurHelicalGearSystem.resolve` does not keep the power, so `system_records` takes it as an
  argument; the resolved torques are read from the private `_resolved` dictionary.
* `ContactBearingStiffness` gives `inf` when a projected displacement is ~0, which includes the
  plain case `phi_Fr = 0` (Kr_xy). The secant stiffness does not exist in that direction in a
  stationary analysis, so `inf` means "not defined" and the record keeps it as `inf`.
* Multi-row bearings are reported structurally (`row` argument, `row_records`) because the
  result classes already hold `rows`; ISO/TS 16281 does not cover them and only single-row
  bearings are in use.
* The units of `e1` and `e2` are not stated by the bearing classes and are left empty.
