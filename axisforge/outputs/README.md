# axisforge.outputs

Records, text and CSV of AxisForge's own objects: the built system and the computed results.
The package reads AxisForge objects and writes only CSV files (`write_records_csv`). It does
not solve, plot or judge pass or fail; sweeps, plots and verdicts belong to the studies that
use AxisForge.

## Contents

```
outputs/
    _format.py        shared helpers: text tables, key-value text, CSV write/read, record utilities
    construction/     the built system
        system.py     SpurHelicalGearSystem: stage, mesh links, resolved torques, every shaft
        shaft.py      ShaftSystem: sections, shoulders, gears, bearings, loads
        bearings.py   assembled Bearing: family, geometry, contact constants
        loads.py      the loads of a shaft system
    solvers/          the computed results (mirrors part of axisforge.results)
        fem_results/shaft_results.py                 ShaftResults, BearingNodeData
        bearings/load_distribution/load_distribution_results.py   BearingResult (ISO/TS 16281)
        bearings/life/basic_life_results.py          basic reference rating life
        bearings/bearing_analysis_result.py          load distribution and life together
```

## Entry points

The sub-packages are namespaces; nothing is flattened. Call
`axisforge.outputs.<sub-package>...<module>.<function>`. The names follow the pattern
`*_record(s)` (dict or list of dicts), `*_text` (str) and `print_*` (console). See the module
and function docstrings for arguments and keys.

| Task | Call | Returns |
|---|---|---|
| Gear system description | `construction.system.system_records(system, power_W=None)` | list of `{section, parameter, value, unit}` |
| Gear system as text | `construction.system.system_text`, `print_system` | str, None |
| Shaft system description | `construction.shaft.shaft_system_records(shaft_system)` | list of dicts |
| Sections, shoulders, gears | `construction.shaft.section_records`, `shoulder_records`, `gear_element_records` | list of dicts |
| One bearing | `construction.bearings.bearing_records(bearing)`, `bearing_summary_record` | list of dicts, dict |
| Loads of a shaft system | `construction.loads.load_records(shaft_system)` | list of dicts |
| FEM bearing nodes | `solvers.fem_results.shaft_results.bearing_node_records(shaft_results)` | list of dicts |
| FEM mesh nodes, summary | `mesh_node_records`, `shaft_summary_record` (same module) | list of dicts, dict |
| Bearing load distribution | `solvers.bearings.load_distribution.load_distribution_results.bearing_result_record(result)` | dict |
| Rows, elements, laminae | `row_records`, `element_records(result, row=0)`, `lamina_records(result, row=0, element=None)` (same module) | list of dicts |
| Basic life | `solvers.bearings.life.basic_life_results.basic_life_record(life)`, `row_life_records` | dict, list of dicts |
| Bearing analysis (distribution and life) | `solvers.bearings.bearing_analysis_result.analysis_record(analysis)`, `analysis_records(analyses)` | dict, list of dicts |
| CSV | `_format.write_records_csv(path, records, fields=None)`, `read_records_csv(path, ...)` | None, list of dicts |
| Text table | `_format.format_table(columns, records, indent=0)`, `format_key_values(records)` | str |

## Conventions

* The unit is part of the key of a number (`Fr_N`, `delta_r_mm`, `psi_rad`, `M_xz_Nmm`,
  `sigma_b_MPa`). A key without a unit suffix is dimensionless or text, except `residual` (force norm in N). Records keep the unit in
  which AxisForge stores the quantity: results and bearing angles in rad, inputs stored in
  degrees (`alpha_n_deg`, `beta_n_deg`) in degrees.
* Exception: the key `T_Nmm` of the FEM node records holds the torque in N m, as
  `ShaftResults.T` does; `mesh_nodes_table_text` still applies 1e-3 to it (see its
  docstring). In the gear-system description the `unit` field is `N m` for torque, `W` for
  power and `deg` for angles, except `alphatw`, which is in `rad`.
* Text may convert for reading and always names the unit in the header: the misalignment psi
  and the slopes are shown in rad and in mrad; element angles in degrees.
* NaN means "not available" (`nan` in CSV, `-` in text). `inf` is kept as `inf`. In the
  load-distribution records `inf` stiffness means that the secant stiffness is not defined:
  the projected displacement is below a tolerance (for example Kr_xy when phi_Fr = 0) or the
  axial regime is `no_load` or `engaged`.
* The system description uses the schema `{section, parameter, value, unit}`.
* No engineering quantity is derived and no verdict is given: equilibrium errors, convergence
  flags and residuals are reported as the result holds them. The only computations are unit
  and angle conversions for display, the source torque of the gear system (power / angular
  speed) and the angle wrap below.
* Element angles `phi_rad` are in the local frame of the row (0 on the load line), wrapped to
  [-pi, pi) with `(phi + pi) % (2 pi) - pi`.
* Multi-row bearings are reported row by row (`row` argument, `row_records`); ISO/TS 16281
  gives equations for single-row bearings only.
* The bearing contact constants `e1` and `e2` are written without unit.
* Only the standard library and NumPy are imported; `axisforge.solvers` and matplotlib are
  not. Objects are read by attribute, so the records work with any object that has them.

## Standards

* ISO/TS 16281:2008: internal load distribution and basic reference rating life L10r of
  rolling bearings (not an ISO 281 catalogue life; no modification factor).
