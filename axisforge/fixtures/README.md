# AxisForge — Fixtures

Modular template library for rapid construction of analysis and validation scripts.

These files are **not pytest unit tests**. They are reusable building blocks
for assembling `design_xxx.py` scripts — functional analysis, pipeline
verification, and textbook validation runs.

---

## Proposed Structure

```
fixtures/
│
├── shafts/
│   ├── stepped_3section.py          # make_stepped_shaft (seat–body–seat geometry)
│   ├── uniform.py                   # single-section uniform shaft
│   └── hollow.py                    # hollow section (reserved for future extension)
│
├── bearings/
│   ├── dgbb_generic.py              # parametric DGBB factory
│   ├── crb_generic.py               # generic cylindrical roller bearing (e.g. N204)
│   └── angular_contact.py           # angular contact bearing (reserved for future extension)
│
├── systems/
│   └── linear_gear_chain.py         # build_systems(stages, shaft_specs, bearing_factory)
│                                    # n-stage linear chain → n+1 ShaftSystems, n MeshLinks
│                                    # external loads applied by caller after build
│
│   Interface
│   ---------
│   StageSpec(dataclass)
│     z_driver    : int              gear tooth count — driver
│     z_driven    : int              gear tooth count — driven
│     pos_driver  : float            axial position of driver gear on its shaft [mm]
│     pos_driven  : float            axial position of driven gear on its shaft [mm]
│     phi_deg     : float            line-of-centres angle in the shaft cross-section [deg]
│     mn          : float            normal module [mm]
│     alpha_n_deg : float            normal pressure angle [deg]
│     b           : float            face width [mm]
│     x_driver    : float = 0.0      profile shift coefficient — driver
│     x_driven    : float = 0.0      profile shift coefficient — driven
│     label       : str  = ""
│
│   ShaftSpec(dataclass)
│     total_length : float           shaft total length [mm]
│     d_seat       : float           bearing seat diameter [mm]
│     d_body       : float           body diameter under gears [mm]
│     l_seat_a     : float           length of seat section A [mm]
│     l_seat_b     : float           length of seat section B [mm]
│     fillet_r     : float           shoulder fillet radius [mm]
│     bearings     : list[tuple]     [(position, label, locating), ...]
│     speed_rpm    : float           nominal shaft speed [rpm]
│     bearing_type : str             bearing type key → resolved via bearing fixtures
│     bearing_subtype : str          bearing subtype key → resolved via bearing fixtures
│     material_id  : str = "AISI_1045"
│     name         : str = ""
│
│   build_systems(
│     stages          : list[StageSpec],
│     shaft_specs     : list[ShaftSpec],   # len == len(stages) + 1
│     bearing_factory : callable,          # (position, label, locating, type, subtype) → Bearing
│     P_W             : float,             # input power [W]
│     rpm_in          : float,             # input speed [rpm]
│     rotation_dir    : int = 1,
│   ) -> dict[str, ShaftSystem]
│
│   Dependencies (fixtures called internally)
│   ------------------------------------------
│   shafts/stepped_3section.py       make_stepped_shaft per ShaftSpec
│   bearings/dgbb_generic.py         if bearing_type == "DGBB"
│   bearings/crb_generic.py          if bearing_type == "CRB"
│   bearings/angular_contact.py      if bearing_type == "AC"  (future)
│
│   Notes
│   -----
│   - shaft_specs must have exactly len(stages) + 1 entries (one per shaft)
│   - shaft_specs[0] is the source shaft (receives motor input)
│   - shaft_origin_x is computed automatically from gear positions
│   - fan-out / power split: out of scope — handled by a dedicated solver
│     that pre-computes torque_split before passing to SpurHelicalGearSystem
│
├── solvers/
│   ├── fem_simple.py                # SimpleFEMSolver pipeline: solve → library
│   ├── fem_timoshenko.py            # Timoshenko variant with distribute_gear_labels
│   ├── iso16281_coupled.py          # IterativeBearingFEMSolver + BearingResult dataclass
│   └── static_analysis.py          # ShaftResultsReader standalone (no ISO 16281)
│
├── outputs/
│   ├── console_bearing.py           # print_bearing_result() — structured per-bearing table
│   ├── console_debug_Qci.py         # Q_ci/Q_ce step-by-step debug, §4.3.1.2 [remove post-validation]
│   ├── console_system_header.py     # analysis header (name, b, P, n)
│   └── console_shaft_summary.py     # per-shaft summary (FEM nodes, bearings, convergence status)
│
├── plots/
│   ├── deflection_3panel.py         # v_xz, v_xy, v_res — 3-subplot figure with bearing and gear markers
│   ├── bearing_polar.py             # Q_j vs φ_j (global frame) — polar plot per bearing
│   ├── convergence_gci.py           # GCI per plane (XZ, XY, resultant) per refinement interval
│   └── deflection_global.py         # single-plane global deflection diagram
│
├── convergence/
│   └── mesh_gci_study.py            # run_convergence_study() with automatic bearing seat exclusion
│
└── integration/
    ├── load_distribution_full.py    # full pipeline: build → FEM → ISO 16281 → console + plots
    └── convergence_study_full.py    # full pipeline: build → global FEM → GCI study → plots
```

---

## System Topology Constraints

- Each `SpurHelicalGearSystem` is always a **linear chain**: shaft₁ → shaft₂ → ... → shaftₙ
- `n` stages produce `n + 1` shafts and `n` mesh links
- Non-linear topologies are handled by running **multiple independent systems**
- Planetary and differential arrangements are out of scope

---

## Composition Pattern

A typical analysis script is assembled as follows:

```
systems/linear_gear_chain.py         ← system build + power flow resolution
  + shafts/stepped_3section.py       ← make_stepped_shaft (per shaft, independently)
  + bearings/dgbb_generic.py         ← bearing factory (injected into builder)
  + solvers/fem_simple.py            ← FEM solve → results library
  + solvers/iso16281_coupled.py      ← ISO 16281 → BearingResult per bearing
  + outputs/console_bearing.py       ← structured console output
  + plots/deflection_3panel.py       ← deflection figure
  + plots/bearing_polar.py           ← polar load distribution figure
```

The `integration/` pipelines combine all of the above. Copy and adjust
the `PARAMETERS` block to produce a new `design_xxx.py`.

---

## Output Formats

| Fixture | Output |
|---|---|
| `outputs/console_bearing.py` | Per-bearing table: reactions, solver status, Q_j distribution |
| `outputs/console_debug_Qci.py` | §4.3.1.2 intermediates for manual cross-check |
| `outputs/console_system_header.py` | Analysis header block |
| `outputs/console_shaft_summary.py` | FEM node count, bearing count, convergence status |
| `plots/deflection_3panel.py` | matplotlib — 3-subplot deflection (v_xz, v_xy, v_res) |
| `plots/bearing_polar.py` | matplotlib — polar Q_j per bearing |
| `plots/convergence_gci.py` | matplotlib — GCI per plane per refinement interval |
| `plots/deflection_global.py` | matplotlib — single-plane global deflection |

---

## Usage Rules

- Each fixture is self-contained and independently importable.
- `integration/` files are the only entry points that combine the full pipeline.
- Fixtures marked `[remove post-validation]` are inherently temporary.
- Do not modify a fixture for a specific case — duplicate and rename it instead.