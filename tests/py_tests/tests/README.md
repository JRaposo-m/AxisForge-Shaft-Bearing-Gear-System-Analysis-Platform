# AxisForge — Test Suite

Primeira ronda de testes unitários **pytest** sobre o `core/`. Não substituem
os `fixtures/` (que são templates de análise, não testes) nem os scripts
exploratórios.

---

## Onde é que os ficheiros vão

```
<repo-root>/
├── pytest.ini                  <- NOVO, na raiz (não dentro de tests/)
├── axisforge/
└── tests/
    ├── conftest.py             <- substitui integralmente o antigo
    ├── README.md               <- este ficheiro
    └── core/
        ├── test_materials.py
        ├── test_loads.py
        ├── machine_elements/
        │   ├── test_shaft.py
        │   ├── test_bearings.py
        │   └── test_spur_helical_gear.py
        └── mechanical_system/
            └── test_shaft_system.py
```

Sem `__init__.py` em lado nenhum — o `pytest.ini` usa
`--import-mode=importlib`, que dispensa pacotes e permite nomes de módulo
repetidos.

## Correr

```
pytest                              # tudo
pytest tests/core/test_loads.py     # um ficheiro
pytest -k undercut                  # por nome
pytest -m validation                # só os validados contra referência
pytest --collect-only -q            # ver o inventário sem executar
```

O `axisforge` tem de ser importável — `pip install -e .` a partir da raiz,
ou correr `pytest` a partir da raiz do repositório.

---

## O que está coberto

| Ficheiro | Alvo | Testes |
|---|---|---|
| `test_shaft.py` | `Shoulder`, `ShaftSection`, `Keyway`, `Shaft` | geometria de secção fechada, endereçamento axial, `section_at` na fronteira, tabelas DIN 6885 / ISO 3912, `validate()` de shoulder mismatch |
| `test_materials.py` | `Material`, `GearMaterial`, lookups | limite de fadiga Shigley §6-2 e o tecto de 700 MPa, `equivalent_modulus`, bibliotecas embebidas |
| `test_loads.py` | todas as classes de `loads.py` | decomposição θ→Fy/Fz, `DistributedRadialLoad` uniforme / q(x) / θ(x), momento fletor contra a forma analítica, `LoadingProfile` |
| `test_bearings.py` | `BearingCatalog`, `Bearing`, `DeepGrooveBallFamily` | contrato do `assemble()`, imutabilidade após selagem, ISO 281 Tabela 1, guardas de geometria |
| `test_spur_helical_gear.py` | `SpurHelicalGear` | geometria de referência ISO 21771, limite spur↔helicoidal, undercut (limiar clássico dos 17 dentes), `validate()` |
| `test_shaft_system.py` | `GearElement`, `ShaftSystem` | guardas de colocação axial, acessores ordenados e filtrados por tipo, extents, idempotência de `set_gear_loads`, delegação de `validate()` |

**Fora do âmbito desta ronda** (propositadamente): solvers FEM, ISO/TS 16281,
malha/GCI, `SpurHelicalGearSystem.resolve()` e `SpurHelicalGearMeshing`.
Esses precisam de valores de referência de literatura e merecem uma suite
de *validação* separada, não unitária.

---

## Achados durante a revisão

Coisas que apareceram ao ler o código. Não corrigi nada — só documento e,
onde faz sentido, deixo um teste que as fixa.

### A. `Material.Su` não existe — o `shaft_post_processor` cai sempre no fallback

`solvers/.../static/shaft_post_processor.py`:

```python
mat = get_material(mat_id)
return float(mat.Su)          # <- Material expõe Sut, não Su
except Exception:
    return 700.0              # <- é sempre isto que acontece
```

O `except Exception:` engole o `AttributeError`, por isso **todas** as
secções usam 700 MPa em vez da resistência real do material. Um S355
(Sut=590) está a ser tratado como mais resistente do que é. Correção:
`mat.Sut`, e apertar o `except` para `KeyError`.

**Não escrevi teste** — é solver, fora do âmbito. Mas é o achado mais sério.

### B. `SpurHelicalGear.da` inclui a folga de topo `cP`

```python
self.da = self.d + 2 * mn * (haP + cP + x)     # implementado
self.df = self.d + 2 * mn * (x - self.hfP)     # hfP = haP + cP
```

ISO 21771 dá `da = d + 2·mn·(haP + x)`. A folga `cP` é o espaço entre o topo
de uma roda e a raiz da outra — já está contabilizada no `df` (via `hfP`).
Somá-la também no topo conta-a duas vezes: a altura total do dente sai
`2.5·mn` em vez dos `2.25·mn` normalizados, e para z=20/mn=2 o `da` dá
45 mm em vez de 44 mm.

Isto propaga-se: `gear_geometry()` no meshing recalcula `da` com a mesma
fórmula (`haP + cP + x - k`), e o `epsilon_alpha` depende do raio de topo.

Testes: `test_tip_diameter_as_implemented` (fixa o comportamento atual) +
`test_tip_diameter_iso21771` e `test_whole_depth_is_2_25_mn`, ambos
`xfail(strict=True)`. Quando corrigires, os xfail passam e o pytest avisa
que os marcadores já não fazem sentido — é esse o objetivo.

### C. Verificação de folga do rolamento *floating* é impossível de satisfazer

`shaft_system.py`:

```python
if not (lo - FLOATING_BEARING_CLEARANCE_MM == x_shoulder == hi + FLOATING_BEARING_CLEARANCE_MM):
```

Dois problemas independentes:

1. Pede que `x_shoulder` seja simultaneamente `lo - 0.1` **e** `hi + 0.1`.
   Como `hi > lo`, isto nunca é verdade — a condição é sempre falsa e o erro
   é sempre emitido para qualquer rolamento *floating* junto a um ressalto.
2. É igualdade exata entre floats. Mesmo que a lógica fosse a certa,
   precisava de `abs(...) <= TOL_GEOMETRY_mm`.

Provavelmente querias `x_shoulder <= lo - c or x_shoulder >= hi + c`.
Não testei — deixei o comentário `# falta o non-locating arrangement` que já
lá está a indicar que o bloco está em construção.

### D. `from config import DEFAULT_DESIGN_LIFE_HOURS` falha sempre

`shaft_system.py` importa `config`, não `axisforge.config`, dentro de um
`try/except Exception`. O import nunca resolve, por isso o valor usado é
sempre o fallback local de `20_000.0` — que por acaso coincide com o do
`config.py`, e é por isso que ninguém deu por nada. Se mudares o
`DEFAULT_DESIGN_LIFE_HOURS` no `config.py`, o `ShaftSystem` continua nos
20 000 h.

O teste `test_design_life_default_is_20000_hours` fixa o valor; se corrigires
o import e mudares a constante, ele avisa.

### E. `Shaft.shoulders()` só reporta `shoulder_right`

Documentado na docstring, portanto é intencional — mas significa que um
ressalto declarado apenas como `shoulder_left` da secção seguinte fica
invisível para quem consome `shoulders()` (o `_shoulder_errors()` do
`ShaftSystem`, por exemplo). No `stepped_shaft_400mm` do conftest declaro os
dois lados com o mesmo objeto `Shoulder`, que é o padrão que o `validate()`
espera. Vale a pena decidir se é contrato ou lacuna.

### F. `__repr__` com caracteres não-ASCII

`RadialLoad`, `TorqueLoad`, `ExternalMoment` e `ShaftSystem.summary()`
embebem `°`, `·` e caracteres de caixa (`──`). A regra do projeto é
ASCII-only por causa do cp1252 do PowerShell. O `°` e o `·` até sobrevivem
em cp1252; os `──` do `summary()` não.

Teste `test_repr_is_ascii`, `xfail(strict=True)`.

### G. `ShaftSystem.set_gear_loads()` não valida limites axiais

Escreve em `self._loads` diretamente, saltando o `_check_axial_bounds()` que
o `add_load()` aplica. É defensável (o `GearSystem` é confiável), mas é uma
assimetria que vale a pena ser deliberada. Fixei-a em
`test_set_gear_loads_bypasses_the_bounds_check`.

---

## Notas de manutenção

- Um teste toca em `_TABLES` (privado) do `iso3912.py`, para derivar um
  diâmetro válido a partir da própria tabela em vez de o cravar. Se
  reorganizares esse módulo, é o único ponto de acoplamento.
- Os `StubBearing` / `StubGear` do `conftest.py` implementam só a superfície
  que o `ShaftSystem` lê. Se o `ShaftSystem` passar a ler mais um atributo,
  os stubs têm de acompanhar — é de propósito, obriga a olhar para o
  contrato.
- `xfail_strict = true` está ligado no `pytest.ini`. Qualquer `xfail` que
  comece a passar torna-se falha.
- Convenção de nomes: as fixtures descrevem **o que são**
  (`stepped_shaft_400mm`), não o que testam.
