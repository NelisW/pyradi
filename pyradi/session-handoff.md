# PyRadi Refactor — Session Handoff & Progress Brief

_Last updated: 2026-04-25 | All 18 files refactored · F-string conversion complete · 784 tests green · URL cleanup complete (44 substitutions, 13 files)_

---

## Quick-start for next agent

### 1. Load the refactor skill

Open Cowork (or a new Claude session) and add the global skill:

```
C:\Users\nwillers\.claude\skills\python-refactor
```

This skill contains all established patterns, pitfalls, and phase definitions for this project. Read `SKILL.md` before making any code changes.

### 2. Open the project folder

Select (mount) the pyradi workspace folder in Cowork:

```
K:\WorkN\pyradi\pyradi\pyradi
```

Linux sandbox mount path (for bash tool): `/sessions/<session-id>/mnt/pyradi/`

### 3. Activate the conda environment (for test runs — Windows only)

```bash
conda activate pyradi
pytest test/ -v --tb=short
```

> **Critical**: tests cannot run in the Linux sandbox — scipy, numpy, and matplotlib are not installed there. Always run pytest in the `pyradi` conda environment on Windows.

### 4. Immediate next action

Run the full test suite to confirm the ~650 f-string conversions introduced no regressions:

```bash
conda run -n pyradi --no-capture-output pytest test/ -v --tb=short
```

Expected: **784 passed, 0 failed**.

---

## 1. Core Goal

Modernise every `.py` file in the `pyradi/` radiometry toolkit for Python 3.12+
compatibility, readability, and maintainability. The guiding skill is the python-refactor
skill at `C:\Users\nwillers\.claude\skills\python-refactor\SKILL.md`.

Each file goes through six phases:
**0. Understand → 1. Baseline tests → 2. Plan (user approval) → 3. Refactor →
4. Conda yaml → 5. README update → 6. Final verification (all tests green)**

**The cardinal rule: never change observable behaviour.** Pre-existing bugs are preserved,
documented in docstrings with a `Note:` section, and asserted in tests so any future
accidental fix is immediately caught.

---

## 2. Decisions Made

### Cardinal rules

- Never change observable behaviour — even buggy behaviour. Preserve, document, test.
- No new dependencies — only packages already in use.
- Public API signatures are frozen unless the user explicitly requests a change.
- Variable names stay as-is — ignore PEP-8 naming conventions.

### Technology / environment

| Item | Value |
|------|-------|
| Python target | 3.12+ |
| Conda environment | `pyradi` (Windows only) |
| Test runner | pytest, files in `test/test_<module>.py` |
| Source folder | `K:\WorkN\pyradi\pyradi\pyradi` |
| Linux mount | `/sessions/<session-id>/mnt/pyradi/` |
| Refactor skill | `C:\Users\nwillers\.claude\skills\python-refactor\SKILL.md` |

### Key architectural decisions

| Decision | Rationale |
|----------|-----------|
| Inline `_ryplanckql()` in rystare.py and rytarggen.py | Installed conda API (`planck(wl, T, stype='ql')`) differs from local source (`planckql(wl, T)`); inlining avoids version-sensitive breakage |
| Constants computed inside `_ryplanckql`, not at module level | Module-level constants computed from `scipy.constants` are corrupted when test stubs replace the module across test files |
| Inline `poissonarray` and `circ` helpers | Both absent from installed package; stubbing causes full-suite stub-corruption failures |
| `autouse` fixture to reinstall mocks in test_ryhdf5.py | pytest collects all files before running any test; module-level mock setup is overwritten during collection by other files' eviction code |
| Patch both `sys.modules['pyradi.ryplot']` AND `sys.modules['pyradi'].ryplot` | `from . import ryplot` resolves via the package object attribute, not just `sys.modules` |
| `os.path.abspath` instead of `os.path.relpath` | `relpath` raises `ValueError` across drive letters on Windows (C:\ vs K:\) |
| `np.savetxt` via `BytesIO` buffer | `np.savetxt` writes bytes; passing a text-mode file handle raises `TypeError` |
| `tarfile.extractall(filter='data')` | Required for Python 3.14 compatibility; strips absolute paths and `..` components |
| Three-pass f-string conversion | AST-based pass 1 (simple), pass 2 (quote switching + multi-line), pass 3 (dual-quote conflicts + constant simplifications) |
| Variable-template `.format()` calls left unconverted | Template is a runtime variable, not a string literal; AST cannot determine the value |
| Commented-out `.format()` calls left unconverted | No runtime effect; catalogued in README "F-string conversion" section |

### `__all__` audit pattern (applied to every file)

Every refactor audits `__all__` with `hasattr` to find phantom names (listed but non-existent) and missing names:

```python
import importlib
m = importlib.import_module('<module>')
phantoms = [n for n in m.__all__ if not hasattr(m, n)]
missing  = [n for n in dir(m) if callable(getattr(m, n))
              and not n.startswith('_') and n not in m.__all__]
```

`rylookup.py` and `ryscene.py` had completely wrong `__all__`; `rytarggen.py` was missing 5 public names; `ryptw.py` had 6 phantom names.

### Planck wavelength-units inconsistency

`ryplanck.planckql` (and all `*ql` / `*el` / `*en` variants) expects wavelength in **micrometres (µm)**, not SI metres. Functions that call `ryplanck.planck` and claim "all values in SI units" in their docstrings are incorrect. Affected functions (each has a `Note:` in its docstring):

- `rystare.nElecCntThermalScene`, `nEcntThermalOptics`, `nElecCntReflSun` — wavelength must be µm
- `rytarggen.calcTemperatureEquivalent` — wavelength must be µm
- `rytarggen.calcLuxEquivalent` — **exception**: this function uniquely expects SI metres; it multiplies by `1e6` internally

### Test isolation (multi-file pytest suite)

Three overlapping problems when running the full suite in one process:

1. **Stub vs real package ordering** — `test_ryhdf5.py` installs a `types.ModuleType` stub for pyradi; `test_rylookup.py` needs the real package. Fix: `test_rylookup.py` detects stubs by absence of `__path__` and evicts them.

2. **MagicMock wiped at collection time** — module-level mock setup in `test_ryhdf5.py` is overwritten during collection. Fix: `autouse` fixture reinstalls the mock before every test at run time.

3. **`from . import ryplot` bypasses `sys.modules`** — relative imports resolve via the package object attribute, not just `sys.modules['pyradi.ryplot']`. Both must be patched.

Correct pattern:
```python
_ryplot_mock = MagicMock()   # unconditional at module level
if 'pyradi' not in sys.modules:
    _stub = types.ModuleType('pyradi')
    _stub.ryplot = _ryplot_mock
    sys.modules['pyradi'] = _stub
sys.modules['pyradi.ryplot'] = _ryplot_mock
sys.modules['pyradi'].ryplot = _ryplot_mock

@pytest.fixture(autouse=True)
def _reinstall():
    sys.modules['pyradi.ryplot'] = _ryplot_mock
    sys.modules['pyradi'].ryplot = _ryplot_mock
    yield
```

### F-string conversion — three-pass strategy

| Pass | Conversions | Technique |
|------|-------------|-----------|
| Pass 1 | 479 | AST-based, single-line, no quote conflicts |
| Pass 2 | 146 | Quote switching + multi-line call support |
| Pass 3 + manual | ~25 | Dual-quote conflicts, LaTeX braces, constant simplifications, starred-args |
| **Total** | **~650** | All 17 `ry*.py` files; zero remaining detectable by AST scan |

Intentionally unconverted (catalogued in `README.md` "F-string conversion" section):
- `ryfiles.py:438` — `cmdline.format(filename)` (runtime-variable template)
- `ryplot.py:399` — `fstr.format(...)` (template built from caller-supplied format string)
- 14 lines in commented-out `#` code across 6 files

### Windows/Linux mount sync fix

The Edit tool writes to the Windows filesystem; the bash sandbox accesses the same files via a Linux mount that occasionally lags or pads with null bytes.

Fix pattern (apply after every Edit-tool write to a large file):
```bash
# 1. Strip null bytes
python3 -c "
with open('file.py','rb') as f: data=f.read()
data=data.rstrip(b'\x00')
with open('file.py','wb') as f: f.write(data)"

# 2. Syntax-check immediately
python3 -m py_compile file.py && echo "OK"
```

For large content additions (README sections, `__main__` blocks): write content as a Python string in a helper `.py` script using the Write tool, then execute via bash. **Never use heredocs (`cat << EOF`) or `cat >>` for multi-line content** — bash truncates them through the mount.

---

## 3. Progress — All 18 Files Complete

### Refactoring status

| File | Completed | Key changes |
|------|-----------|-------------|
| `ryutils.py` | 2026-04-19 | 4-space indent, docstrings, unused imports removed |
| `ryfiles.py` | 2026-04-19 | pathlib, with statements, tarfile filter, `os.path.relpath` → `abspath` |
| `ry3dnoise.py` | 2026-04-20 | numpy 2.x fixes, docstrings |
| `rychroma.py` | 2026-04-20 | docstrings, unused imports |
| `rydetector.py` | 2026-04-20 | pandas 2.x `DataFrame.append` → `pd.concat`, `np.savetxt` mode fix |
| `rylookup.py` | 2026-04-20 | `__all__` fixed, missing `self.` bug fixed, docstrings |
| `rymodtran.py` | 2026-04-20 | Python 2 branches removed, pathlib, docstrings |
| `ryhdf5.py` | 2026-04-20 | lazy imports kept, `__package__` trick for relative imports in tests |
| `rypflux.py` | 2026-04-20 | `collections.defaultdict`, docstrings |
| `ryplanck.py` | 2026-04-20 | `from __future__` removed, `xml.etree.cElementTree` simplified |
| `ryprob.py` | 2026-04-20 | docstrings, type hints updated |
| `ryplotspherical.py` | 2026-04-20 | `__all__` rebuilt, unused imports removed, Windows test fix |
| `ryptw.py` | 2026-04-21 | 6 phantom `__all__` names removed, Python 2 dead code purged, pre-existing bugs documented (`mylong <<32`, `terminateStrOnZero int/str`) |
| `ryscene.py` | 2026-04-21 | `__all__` completely rebuilt; `DataFrame.ix` → `.loc`; 7 unused imports removed; null bytes stripped; `transformations` stubbed in tests |
| `rystare.py` | 2026-04-21 | 7 unused imports removed; `__all__` duplicate fixed; `from __future__` removed; pre-existing bug in `get_summary_stats` documented; wavelength-units inconsistency documented; `_ryplanckql` + `poissonarray` inlined |
| `rytarggen.py` | 2026-04-21 | 7 unused imports removed; `ryplot` lazy-imported; `__all__` expanded 5→10; wavelength-units documented; `_ryplanckql`, `poissonarray`, `circ` inlined; `ryhdf5` import restored |
| `ryplot.py` | 2026-04-22 | VCS keywords stripped; 4 dead local `__all__` removed; `import os` removed; 46 long lines wrapped; docstrings added to `RGBToPyCmap` and `clippedcolorbar`; `shadoAllpe` bug documented; 86-test suite written and green |
| **F-strings** | 2026-04-22 | ~650 conversions across all 17 `ry*.py` files; all files pass `py_compile` |

### ryplot.py — documented pre-existing bugs

| Bug | Location | Description | Workaround in tests |
|-----|----------|-------------|---------------------|
| `shadoAllpe` | `turbo_colormap_data` block | Typo for `.shape[0]`; in an `else` branch that never runs (turbo already registered in mpl 3.x) | N/A — branch is dead |
| CDF normalisation | `ProcessImage.compressEqualizeImage` | Divides by `imgFlatSort[-1]` (max) instead of `imgFlatSort.cumsum()[-1]` (total); output not bounded to [0, 1] for random inputs | Test checks shape and tick type only |
| Multi-column `r` | `Plotter.polar` | `highlightNegative=True` raises `ValueError` on row-array comparison; body unconditionally overrides the parameter to `True` unless `rscale[0] < 0` | Pass BOTH `rscale=[-0.1, 1.0]` AND `highlightNegative=False` simultaneously |
| `labels=None` | `Plotter.plotArray` | Default `labels=None` raises `TypeError` at `labels[0]` | Always supply explicit `labels` list |

---

## 4. Test Suite

### Full suite command

```bash
conda run -n pyradi --no-capture-output pytest test/ -v --tb=short
```

Or explicitly:

```bash
conda run -n pyradi --no-capture-output pytest \
  test/test_ryutils.py test/test_ryfiles.py test/test_ry3dnoise.py \
  test/test_rychroma.py test/test_rydetector.py test/test_rylookup.py \
  test/test_rymodtran.py test/test_ryhdf5.py test/test_rypflux.py \
  test/test_ryplanck.py test/test_ryprob.py test/test_ryplotspherical.py \
  test/test_ryptw.py test/test_ryscene.py test/test_rystare.py \
  test/test_rytarggen.py test/test_ryplot.py -v --tb=short
```

### Test counts

| Milestone | Result | Date |
|-----------|--------|------|
| First 14 files baseline | 580 passed | 2026-04-20 |
| + rystare (wavelength-units fix) | 57 additional | 2026-04-21 |
| + rytarggen (API inlining) | 22 additional | 2026-04-21 |
| Full suite (16 files, cross-version API fixes) | 659 passed | 2026-04-21 |
| + ryplot (86 tests, polar bug workaround) | **784 passed, 0 failed** | 2026-04-22 |
| After f-string conversion (~650 changes) | **784 passed, 0 failed** | 2026-04-22 |

---

## 5. What Worked / What Didn't Work

### Worked

| Approach | Outcome |
|----------|---------|
| AST-based f-string conversion | Safe, deterministic — correctly identifies string literals vs variable templates |
| `py_compile` validation before every write | Caught several syntax errors before they reached the file |
| Three-pass f-string strategy | Maximised automation; manual pass 3 handled only ~25 hard cases |
| `autouse` fixture for mock reinstall | Solved full-suite mock-corruption race condition cleanly |
| Inlining `_ryplanckql` with constants inside the function | Prevented module-level constant corruption across test files |
| `rf"..."` raw f-strings for LaTeX multicolumn content | Preserves backslash literals while still interpolating expressions |
| Write tool + Python helper script for large content | Avoids null-byte truncation from bash heredocs |

### Didn't Work / Rejected

| Approach | Why rejected |
|----------|-------------|
| `getattr`-based shim for `ryplanck.planckql` vs `planck` | `test_ryplanck.py` stubs `pyradi.ryutils` at collection time; full suite failed with 12+ `AttributeError`s |
| `matplotlib.use('Agg')` after other imports | Must be the **first** import; calling it later is silently ignored → GUI/display errors in headless environments |
| `np.allclose` for asserting arrays are NOT equal | Photon noise (~3e-8 relative) falls within `allclose`'s default `rtol=1e-5`; use `np.array_equal` instead |
| `highlightNegative=False` alone as polar bug workaround | Method body overrides it back to `True` when `rscale is None` |
| `rscale=[-0.1, 1.0]` alone as polar bug workaround | `rscale[0] < 0` branch skips the override but keeps the default `True` value |
| `np.linspace(0.1, 1.0)` as log-compression test input | `np.log(1.0) = 0` → divide-by-zero → `RuntimeWarning` + NaN propagation |
| `pick_quote()` detection via `ast.unparse()` string for embedded-quote args | `ast.unparse('text with "inner" quotes')` renders the `"` visible in the expression string → false positive in quote-conflict detection |
| Heredocs (`cat << EOF`) for multi-line bash writes | The Windows/Linux mount truncates heredoc content through the virtual filesystem |
| `os.path.relpath` for path collection | Raises `ValueError` across drive letters on Windows |
| `MagicMock` as `sys.modules['pyradi']` stub | pytest scans modules for `pytest_plugins`; `MagicMock` auto-creates that attribute returning another `MagicMock` (not a list) → `UsageError` |

---

## 6. Pending / Next Actions

### ~~Step 1 — Reconfirm test suite after f-string conversion~~ ✓ Done

784 tests passed post-conversion. Confirmed green by user on 2026-04-22.

### ~~Step 2 — Delete REFACTOR_BRIEF.md~~ ✓ Done

`REFACTOR_BRIEF.md` deleted by user on 2026-04-22. This document is now the sole
progress record.

### ~~Step 3 — URL cleanup~~ ✓ Done (2026-04-25)

All URLs in `*.py`, `*.rst`, and `*.md` files tested and stale ones replaced.

- **`check_urls.py`** — scans all source files, tests each URL (HEAD then GET fallback), writes `staleURLs.md` with file, line, status, URL
- **`fix_urls.py`** — applies 38 high-confidence URL replacements; ordered longest→shortest to avoid prefix collisions; idempotent (safe to re-run)
- **Result**: 44 substitutions across 13 files; `staleURLs.md` has ✅/❌ per row

Families replaced: matplotlib (axes_grid removal, versioned paths, SourceForge), HDF Group site restructure, h5py docs migration, NumPy/SciPy versioned → stable docs, SciPy wiki/cookbook, nbviewer domain (ipython.org → org, jupyter.org → org), PyPI (pypi.python.org → pypi.org), GitHub Help → GitHub Docs, OSSIM Trac.

Not replaced (low confidence, 15 entries): Nabble forum threads (content unrecoverable), an astrometry SVN file (no direct GitHub equivalent found), specific journal PDF (only index page confirmed), 11 personal/deleted/specialty pages with no public equivalent.

To re-run URL check from `pyradi` conda env:
```bash
conda run -n pyradi python pyradi/check_urls.py
python pyradi/fix_urls.py
```

### Step 4 — Optional: convert commented-out `.format()` calls

14 lines of commented-out `.format()` calls remain in 6 files (catalogued in README).
These are inert. If desired, delete the comments or convert them for consistency.

### Step 5 — Optional: fix documented pre-existing bugs in ryplot.py

The four ryplot pre-existing bugs (see Section 3) are preserved by policy. If the user
decides to fix any of them, that is an explicit separate request — update both the
production code and remove the corresponding `Note:` / buggy-behaviour assertion from the
test suite.

### Step 6 — Optional: clean up stale ryplot section in SKILL.md

`SKILL.md` has a partial `ryplot.py — Active refactor progress` section near the bottom
that predates completion. Now that all work is done, this section can be removed or
replaced with a one-line "ryplot: complete 2026-04-22" note.

---

## 7. Supporting Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `K:\WorkN\pyradi\pyradi\pyradi\README.md` | Full API reference + f-string unconverted catalogue | Current |
| `K:\WorkN\pyradi\pyradi\pyradi\session-handoff.md` | **This file** — sole progress document | Current |
| `K:\WorkN\pyradi\pyradi\pyradi\REFACTOR_BRIEF.md` | Old progress brief — superseded | **Deleted 2026-04-22** |
| `C:\Users\nwillers\.claude\skills\python-refactor\SKILL.md` | Refactor skill (1388 lines) | Current |
| `K:\WorkN\pyradi\pyradi\pyradi\environment.yml` | Conda environment spec | Current |
| `K:\WorkN\pyradi\pyradi\pyradi\staleURLs.md` | URL audit: stale URLs, replacements found, ✅/❌ status | Current (2026-04-25) |
| `K:\WorkN\pyradi\pyradi\pyradi\check_urls.py` | Automated URL liveness checker; writes `staleURLs.md` | Created 2026-04-25 |
| `K:\WorkN\pyradi\pyradi\pyradi\fix_urls.py` | Applies high-confidence URL replacements to source files | Created 2026-04-25 |

---

## 8. Critical Gotchas for Next Agent

1. **Always run tests in the conda env on Windows** — `conda run -n pyradi ...`. The Linux sandbox cannot run the tests.

2. **`matplotlib.use('Agg')` must be the very first import** in any test file that touches `ryplot`. If it appears after any other matplotlib import (even indirect), it is silently ignored.

3. **The mount can inject null bytes** — after any Edit-tool write to a large file, strip null bytes and `py_compile` immediately (see Section 2, Windows/Linux mount sync fix).

4. **`ryplanck` wavelength units are µm, not SI metres** — every `planckql` / `planck` call expects micrometres. Functions claiming "all SI units" but calling ryplanck are documented with `Note:`.

5. **`test_ryplanck.py` stubs `pyradi.ryutils` at collection time** — any production code importing `ryutils.poissonarray` or `ryutils.circ` gets the stub in full-suite runs. Keep those helpers inlined in `rystare.py` and `rytarggen.py`.

6. **Variable-template `.format()` calls cannot be auto-converted** — `fstr.format(value)` where `fstr` is a runtime variable must stay as-is.

7. **`ast.unparse()` on string-literal args shows embedded quotes** — a string argument `'text with "inner" quotes'` unparses with the `"` visible, causing false positives in quote-conflict detection. This is why pass 3 was needed.

8. **`ryhdf5.erase_create_HDF` is in `pyradi.ryhdf5`, not `pyradi.ryfiles`** — the unused-import cleanup in `rytarggen.py` once accidentally dropped this import; it was restored. Always verify HDF5 call sites after any import cleanup.

9. **`Plotter.polar` workaround needs BOTH conditions** — `rscale=[-0.1, 1.0]` AND `highlightNegative=False` must be set simultaneously. Neither alone is sufficient because the function body unconditionally overrides `highlightNegative` in the `rscale is None` and `rscale[0] >= 0` branches.

10. **`transformations` package for ryscene** — third-party Gohlke wheel; stubbed in tests with `MagicMock` returning a known identity quaternion `[1, 0, 0, 0]`. Listed in `environment.yml` under `pip:`.
