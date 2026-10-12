# AxisForge

Deterministic, standards-based engine for shaft–bearing–gear systems. `README.md` is the entry
point; each package has its own README (`core/`, `mesh/`, `results/`, `solvers/`, `outputs/`).
Shared rules for all repositories: `../CLAUDE.md`.

## Layers (one-way; checked by the vault sync)
- `config` → nothing. `core` → `config`, `database`. `mesh` → `core`, `config`.
- `solvers` → `core`, `mesh`, `results`, `config`. **No solver imports another solver's internals.**
- `results` → nothing from axisforge at runtime (containers only, no computation).
- `outputs` → `core`, `results`, `mesh`, `config`; never `solvers`, never matplotlib (D-0008: AxisForge
  owns records/text/CSV, plots live in design-studies).
Before adding an import across layers, check this list; a new edge is a decision, not an edit.

## Patterns to follow (do not invent parallel ones)
- Public names are re-exported eagerly through each package `__init__.py` (explicit names, no
  `import *` chains, no lazy `__getattr__`). A new public name is added to its package rollup.
- Variants self-register: bearing families with `@register_family`, beam formulations with
  `@register_formulation`. Add a variant by registering it; never edit a dispatch table.
- Capacity and standard formulas belong to the element (`bearing.family....`), not to a solver.
- Every domain object: `validate()` returns, `validate_or_raise()` raises. Fail fast at construction.
- Records for reports: `outputs/` (`*_records`, `*_text`, `print_*`), units in the key (`Fr_N`).
  Reuse `outputs._format` (`format_table`, `write_records_csv`, `read_records_csv`, `union_fields`).
- Docstrings: numpydoc with units in brackets (D-0003, proposed).

## Commands
```bat
python -m pytest axisforge          :: tests live next to the code (pyproject testpaths "tests" is stale)
python -m pytest axisforge\solvers\machine_elements\bearings -q
```

## Known state (commit b3f624a, 2026-10-09; check before relying on it)
- Case-duplicated folders tracked by git: `core/machine_elements/Bearings` + `bearings`,
  `Shaft` + `shaft`. Works on Windows only; a Linux clone cannot import the package.
  See vault `80_Inbox/finding_axisforge_case_folders_2026-10-10.md`. Do not add files to the
  capitalised folders.
- `database/gears/.../Kv_methodB`, `Kv_methodC`: broken imports (old package path).
- `solvers/.../shaft/static_solvers/__init__.py`: lazy `__getattr__` to a missing module.
- Units of torque in `core/loads.py`: module docstring and `TorqueLoad` disagree (N·mm vs N·m); ask.

## Documentation
READMEs describe the code as it is now. Do not add "a previous pass of this document said…"
history; history is in git and in the vault.
