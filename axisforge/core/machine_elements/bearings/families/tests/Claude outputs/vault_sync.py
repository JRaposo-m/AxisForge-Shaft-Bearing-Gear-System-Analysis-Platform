"""
tools/vault_sync.py — v2.1. Derives the CODE AXIS of the vault from the repo by AST.

v2 changes (graph-quality + agent-readability):
  1. WIKILINKS everywhere the graph should show an edge. v1 wrote deps/consumers as
     plain text bullets, so Obsidian's graph view showed isolated dots (the
     fragmentation you saw). Now: deps, consumers, MOC children, MOC parent,
     standard_refs and validation_case are all [[wikilinks]] -> the graph view
     literally renders the import + traceability graph.
  2. DEP RESOLUTION to code_paths. v1 kept only top-level import names
     ("axisforge"), which resolves to nothing. v2 keeps full dotted paths and
     suffix-matches them against known code_paths, so deps: ["core/loads", ...]
     are real registry keys the orchestrator can resolve depth-1.
  3. FRONTMATTER PRESERVATION. v1 reset status to "planned" on every re-sync.
     v2 reads the existing note's frontmatter (status, standard_refs, concepts,
     validation_case, pair_type, verified_by) back into the REGISTRY and never
     overwrites it. Human/state data survives every sync.
  4. CONCEPT_INDEX.json. Concept-based second index built from the `concepts:`
     frontmatter field of Code/ and 20_Solvers/ notes. This is what the
     Researcher (K) queries first — it catches conceptual coupling the import
     graph can't see (axis convention, torque propagation, unit contracts).
  5. LAYER-LAW CHECK built in. ACTUAL edges are diffed against the ALLOWED law;
     violations are printed as warnings (Conformance agent still blocks; this is
     the early tripwire).

v2.1 fix (import_dotted / dep resolution):
  v2's import_dotted() only tracked `n.module` for a `from module import name`
  statement and threw away `n.names` entirely. That resolves fine when the
  module dotted path already IS the real file (`from pkg.mod import symbol`),
  but silently drops the dependency when the real file is only named as an
  IMPORTED NAME from a package (`from pkg import submodule` / `from pkg import
  submodule as alias`) -- e.g. `from ...iso_16281 import contact_postprocessing
  as pp`: the module dotted path is just the package `iso_16281` (no code_path
  of its own -- __init__.py is excluded from the registry), so the real
  target, contact_postprocessing.py, only showed up in n.names and vanished
  from the graph -- CASCADE.md then rendered contact_postprocessing as an
  isolated node with no edge into contact_solver, even though contact_solver.py
  genuinely imports it.

  Two changes fix this:
    a) import_dotted() now returns (plain_imports, from_imports) instead of one
       flattened set -- from_imports keeps module_dotted -> [names] paired, so
       pass 2 can try the module path first (the common case, unchanged
       behavior) and fall back to `module_dotted.name` per imported name only
       when the bare module doesn't resolve (the submodule-import case).
    b) pass 2 no longer buckets an unresolved import under `external` just
       because resolve_dep() failed -- if its top-level segment matches one of
       the repo's own registered top-level packages (internal_roots), it's a
       real gap (stale/typo'd import, or -- before this fix -- exactly this
       submodule-import case) and gets a printed "UNRESOLVED INTERNAL IMPORT"
       warning instead of being silently mislabeled as an external library
       (which is how v2 hid this bug: `ext.add("axisforge")`).

Usage:  python tools/vault_sync.py /path/to/axisforge_repo [--vault ./vault]
Only <!-- AUTO:BEGIN..END --> regions, REGISTRY.json and CONCEPT_INDEX.json are
written; frontmatter values and human prose are preserved.
"""
from __future__ import annotations
import ast
import json
import os
import re
import sys
import pathlib

AUTO_RE = re.compile(r"<!-- AUTO:BEGIN.*?-->.*?<!-- AUTO:END -->", re.S)
FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
SKIP_DIRS = {"__pycache__", ".git", "tests", ".venv", "venv", "build", "dist", "axisforge-agents"}
STDLIB = {"numpy", "np", "scipy", "math", "typing", "dataclasses", "enum", "copy",
          "collections", "functools", "itertools", "pathlib", "os", "sys", "re",
          "json", "abc", "warnings", "__future__", "matplotlib"}

# The layering law (mirror of 00_Master/WIRING.md ALLOWED). layer -> allowed layers.
ALLOWED = {
    "core":         set(),                       # only stdlib/numpy/scipy
    "models":       {"core"},
    "solvers":      {"core", "models"},
    "selectors":    {"core", "models", "solvers"},
    "integrations": {"core", "models", "solvers"},
    "database":     {"core", "models"},
    "ui":           {"solvers", "models", "selectors"},
}


# ---------------------------------------------------------------- helpers

def code_path(pyfile: str, root: str) -> str:
    rel = os.path.relpath(pyfile, root)
    return rel[:-3].replace(os.sep, "/")


def parse_frontmatter(text: str) -> dict:
    """Minimal YAML-ish frontmatter parser for our known key shapes:
    `key: value` and `key: [a, b, c]`. Good enough; no dependency."""
    m = FM_RE.match(text)
    fm: dict = {}
    if not m:
        return fm
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if v.startswith("[") and v.endswith("]"):
            items = [i.strip().strip("'\"") for i in v[1:-1].split(",") if i.strip()]
            fm[k] = items
        else:
            fm[k] = v.strip("'\"")
    return fm


def public_signatures(tree: ast.AST) -> list[str]:
    out = []
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
            init = next((m for m in n.body if isinstance(m, ast.FunctionDef)
                         and m.name == "__init__"), None)
            args = [a.arg for a in init.args.args[1:]] if init else []
            out.append(f"{n.name}({', '.join(args)})")
        elif isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            out.append(f"{n.name}({', '.join(a.arg for a in n.args.args)})")
    return out


def import_dotted(tree: ast.AST) -> tuple[set[str], dict[str, list[str]]]:
    """v2.1: split plain `import x.y` targets from `from module import name`
    statements, KEEPING module and names paired (module_dotted -> [names]).

    v2 only tracked n.module for ImportFrom and threw n.names away. That
    resolves fine when the module dotted path already IS the real file
    (`from pkg.mod import symbol`), but silently drops the dependency when
    the file is only named as an IMPORTED NAME from a package
    (`from pkg import submodule` / `from pkg import submodule as alias`) --
    e.g. `from ...iso_16281 import contact_postprocessing as pp`: the
    module dotted path is just the package `iso_16281` (no code_path of its
    own -- __init__.py is excluded from the registry), so the real target,
    contact_postprocessing.py, only shows up in n.names and got lost.

    Resolution (pass 2) tries module_dotted first (the common, already-
    working case) and falls back to `module_dotted.name` per name only if
    that fails -- so this never changes behavior for a plain symbol import,
    it just adds the missing path for the submodule-import case."""
    plain: set[str] = set()
    from_imports: dict[str, list[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                plain.add(a.name)
        elif isinstance(n, ast.ImportFrom) and n.module:
            from_imports.setdefault(n.module, [])
            from_imports[n.module].extend(a.name for a in n.names)
    return plain, from_imports


def resolve_dep(dotted: str, registry_keys: list[str]) -> str | None:
    """Map a dotted import to a known code_path by longest-suffix match.
    'axisforge.core.loads' -> 'core/loads' if that key exists."""
    parts = dotted.split(".")
    for take in range(len(parts), 0, -1):          # longest suffix first
        suffix = "/".join(parts[-take:])
        for cp in registry_keys:
            if cp == suffix or cp.endswith("/" + suffix):
                return cp
    return None


def layer_of(cp: str) -> str:
    return cp.split("/")[0]


def wl(cp: str) -> str:
    """Wikilink to a Code note."""
    return f"[[Code/{cp}|{cp}]]"


# ---------------------------------------------------------------- main

def main() -> int:
    if len(sys.argv) < 2:
        print("usage: vault_sync.py <repo_root> [--vault ./vault]")
        return 2
    repo = sys.argv[1]
    vault = pathlib.Path(sys.argv[sys.argv.index("--vault") + 1]
                         if "--vault" in sys.argv else "vault")
    code_dir = vault / "Code"
    code_dir.mkdir(parents=True, exist_ok=True)

    registry: dict[str, dict] = {}
    raw_imports: dict[str, tuple[set[str], dict[str, list[str]]]] = {}
    errors: list[str] = []
    warnings: list[str] = []

    # --- pass 1: parse every module ---
    for dirpath, dirnames, files in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in files:
            if not f.endswith(".py") or f == "__init__.py":
                continue
            py = os.path.join(dirpath, f)
            cp = code_path(py, repo)
            if cp in registry:
                errors.append(f"duplicate code_path: {cp}")
                continue
            try:
                tree = ast.parse(pathlib.Path(py).read_text(encoding="utf-8"))
            except SyntaxError as e:
                errors.append(f"parse error {cp}: {e}")
                continue
            raw_imports[cp] = import_dotted(tree)
            registry[cp] = {
                "base": f"Code/{cp}.md",
                "folder_moc": (f"Code/{os.path.dirname(cp)}/_MOC.md"
                               if "/" in cp else "Code/_MOC.md"),
                "layer": layer_of(cp),
                "exports": public_signatures(tree),
                "deps": [], "external": [],
                # state defaults — overwritten from existing frontmatter below
                "status": "planned", "standard_refs": [], "validated_by": [],
                "concepts": [], "pair_type": None, "verified_by": None,
            }

    keys = list(registry)
    internal_roots = {cp.split("/")[0] for cp in keys}

    # --- pass 2: resolve deps to code_paths; split external libs ---
    for cp, (plain_imports, from_imports) in raw_imports.items():
        deps, ext = set(), set()

        # `import x.y.z` style — unchanged from v2.
        for d in plain_imports:
            top = d.split(".")[0]
            if top in STDLIB:
                continue
            r = resolve_dep(d, keys)
            if r and r != cp:
                deps.add(r)
            elif top in internal_roots:
                warnings.append(f"UNRESOLVED INTERNAL IMPORT: {cp} -> {d}")
            else:
                ext.add(top)

        # `from module import name[, name2, ...]` style — v2.1 fix.
        for module_dotted, names in from_imports.items():
            top = module_dotted.split(".")[0]
            if top in STDLIB:
                continue
            r = resolve_dep(module_dotted, keys)
            if r and r != cp:
                deps.add(r)
                continue
            # module_dotted alone didn't resolve -- it may be a PACKAGE,
            # with the real dependency named only in `names` (a submodule
            # import, e.g. `from pkg import submodule`). Try each imported
            # name appended to the module path before giving up on it.
            resolved_any_name = False
            for name in names:
                if name == "*":
                    continue
                r2 = resolve_dep(f"{module_dotted}.{name}", keys)
                if r2 and r2 != cp:
                    deps.add(r2)
                    resolved_any_name = True
            if not resolved_any_name:
                if top in internal_roots:
                    # Shares a top-level package with our own code but never
                    # resolved -- a real gap (stale/typo'd import, or the
                    # submodule-import case this fix targets), NOT an
                    # external library. v2 silently mislabeled this as
                    # `ext.add("axisforge")`, which is exactly how the
                    # missing contact_postprocessing edge went unnoticed.
                    warnings.append(f"UNRESOLVED INTERNAL IMPORT: {cp} -> {module_dotted}")
                else:
                    ext.add(top)

        registry[cp]["deps"] = sorted(deps)
        registry[cp]["external"] = sorted(ext)

    # --- pass 3: preserve existing frontmatter state (v2: never reset) ---
    for cp, e in registry.items():
        note = code_dir / f"{cp}.md"
        if note.exists():
            fm = parse_frontmatter(note.read_text(encoding="utf-8"))
            for k in ("status", "pair_type", "verified_by", "validation_case"):
                if fm.get(k) not in (None, "", "null"):
                    e[k] = fm[k]
            for k in ("standard_refs", "concepts", "open_escalations"):
                if isinstance(fm.get(k), list) and fm[k]:
                    e[k] = fm[k]

    # --- reverse graph: consumers (from resolved deps — exact, not fuzzy) ---
    consumers: dict[str, list[str]] = {cp: [] for cp in registry}
    for cp, e in registry.items():
        for d in e["deps"]:
            consumers[d].append(cp)

    # --- pass 4: write interface notes (AUTO block only, with WIKILINKS) ---
    for cp, e in registry.items():
        note = code_dir / f"{cp}.md"
        note.parent.mkdir(parents=True, exist_ok=True)
        links = []
        for s in e.get("standard_refs", []):
            links.append(f"- implements: [[{s.strip('[]')}]]")
        vc = e.get("validation_case")
        if vc and vc != "null":
            links.append(f"- validated_by: [[30_Validation/{vc}|{vc}]]")
        auto = (
            "<!-- AUTO:BEGIN (vault_sync — do not edit) -->\n"
            "## Exports (public)\n"
            + "".join(f"- `{s}`\n" for s in e["exports"])
            + "## Depends on\n"
            + ("".join(f"- {wl(d)}\n" for d in e["deps"]) or "- (none internal)\n")
            + ("".join(f"- ext: `{x}`\n" for x in e["external"]))
            + "## Consumed by\n"
            + ("".join(f"- {wl(c)}\n" for c in sorted(consumers[cp])) or "- (none)\n")
            + "## Links\n"
            + ("\n".join(links) + "\n" if links else "- (add standard_refs in frontmatter)\n")
            + f"## Folder\n- [[{e['folder_moc'][:-3]}|_MOC]]\n"
            "<!-- AUTO:END -->"
        )
        if note.exists():
            body = note.read_text(encoding="utf-8")
            body = AUTO_RE.sub(auto, body) if AUTO_RE.search(body) else body + "\n" + auto
            note.write_text(body, encoding="utf-8")
        else:
            fm = (f"---\ncode_path: {cp}\nlayer: {e['layer']}\nstatus: planned\n"
                  f"standard_refs: []\nconcepts: []\nvalidation_case: null\n"
                  f"pair_type: null\nverified_by: null\nopen_escalations: []\n---\n\n")
            note.write_text(fm + auto + "\n\n## Contract (HUMAN)\n- Units:\n"
                            "- Invariants (axis / sign / units):\n## Edge cases\n",
                            encoding="utf-8")

    # --- folder MOCs (children + parent as wikilinks) ---
    by_dir: dict[str, list[str]] = {}
    for cp in registry:
        by_dir.setdefault(os.path.dirname(cp), []).append(cp)
    for d, members in by_dir.items():
        moc = code_dir / d / "_MOC.md" if d else code_dir / "_MOC.md"
        moc.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"---\nfolder: {d or '.'}\nkind: folder-moc\n---\n",
                 "## Modules here"]
        for m in sorted(members):
            lines.append(f"- {wl(m)} · `{registry[m]['status']}`")
        parent = os.path.dirname(d)
        if d:
            up = f"Code/{parent}/_MOC" if parent else "Code/_MOC"
            lines.append(f"\n## Up\n- [[{up}|_MOC]]")
        moc.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- CONCEPT_INDEX.json (v2): concepts -> code_paths / note paths ---
    concept_index: dict[str, list[str]] = {}
    for cp, e in registry.items():
        for c in e.get("concepts", []):
            concept_index.setdefault(c, []).append(cp)
    solvers_dir = vault / "20_Solvers"
    if solvers_dir.exists():
        for n in solvers_dir.rglob("*.md"):
            fm = parse_frontmatter(n.read_text(encoding="utf-8"))
            for c in fm.get("concepts", []) if isinstance(fm.get("concepts"), list) else []:
                concept_index.setdefault(c, []).append(
                    str(n.relative_to(vault)).replace(os.sep, "/"))
    (vault / "00_Master" / "CONCEPT_INDEX.json").write_text(
        json.dumps(concept_index, indent=2, sort_keys=True), encoding="utf-8")

    # --- REGISTRY.json ---
    (vault / "00_Master").mkdir(parents=True, exist_ok=True)
    (vault / "00_Master" / "REGISTRY.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8")

    # --- WIRING ACTUAL patch + layer-law check ---
    wiring = vault / "00_Master" / "WIRING.md"
    actual_lines = []
    for cp in sorted(registry):
        e = registry[cp]
        tgt = ", ".join(e["deps"] + [f"ext:{x}" for x in e["external"]]) or "(none)"
        actual_lines.append(f"- {cp} -> {tgt}")
        lay = e["layer"]
        if lay in ALLOWED:
            for d in e["deps"]:
                if layer_of(d) != lay and layer_of(d) not in ALLOWED[lay]:
                    warnings.append(f"LAYER VIOLATION: {cp} ({lay}) -> {d} ({layer_of(d)})")
            if lay == "core" and e["external"]:
                warnings.append(f"CORE PURITY: {cp} imports external {e['external']}")
    if wiring.exists():
        actual = ("<!-- AUTO:BEGIN -->\n" + "\n".join(actual_lines) + "\n<!-- AUTO:END -->")
        body = wiring.read_text(encoding="utf-8")
        body = AUTO_RE.sub(actual, body) if AUTO_RE.search(body) else body
        wiring.write_text(body, encoding="utf-8")

    # --- drift: orphan Code notes ---
    for note in code_dir.rglob("*.md"):
        if note.name == "_MOC.md":
            continue
        cp = str(note.relative_to(code_dir))[:-3].replace(os.sep, "/")
        if cp not in registry:
            errors.append(f"orphan Code note (source gone): {cp}")

    for w in warnings:
        print("WARN:", w)
    if errors:
        print("VAULT SYNC FAILED:")
        for e in errors:
            print("  -", e)
        return 1
    print(f"synced {len(registry)} modules, {len(by_dir)} MOCs, "
          f"{len(concept_index)} concepts -> {vault}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
