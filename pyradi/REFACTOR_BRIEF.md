# PyRadi Refactor -- Progress Brief

_Last updated: 2026-04-22 (ryplot fully complete — all 6 phases done, 86 tests green)_

---

## Goal

Modernise every .py file in the pyradi/ toolkit for Python 3.12+ compatibility,
readability, and maintainability, guided by the python-refactor skill stored in
/mnt/python-refactor/SKILL.md.

Each file goes through six phases:
0. Understand -> 1. Baseline tests -> 2. Plan (user approval) -> 3. Refactor ->
4. Conda yaml -> 5. README update -> 6. Final verification (all tests green).

---

## Completed (18 files)

| File | Key changes |
|------|-------------|
| ryutils.py | 4-space indent, docstrings, unused imports removed |
| ryfiles.py | pathlib, with statements, tarfile filter, os.path.relpath -> abspath |
| ry3dnoise.py | numpy 2.x fixes, docstrings |
| rychroma.py | docstrings, unused imports |
| rydetector.py | pandas 2.x DataFrame.append -> pd.concat, np.savetxt mode fix |
| rylookup.py | __all__ fixed, missing self. bug fixed, docstrings |
| rymodtran.py | Python 2 branches removed, pathlib, docstrings |
| ryhdf5.py | lazy imports kept, __package__ trick for relative imports in tests |
| rypflux.py | collections.defaultdict, docstrings |
| ryplanck.py | from __future__ removed, xml.etree.cElementTree simplified |
| ryprob.py | docstrings, type hints updated |
| ryplotspherical.py | __all__ rebuilt, unused imports removed, Windows test fix |
| ryptw.py | 6 phantom __all__ names removed, Python 2 dead code purged, pre-existing bugs documented (mylong <<32, terminateStrOnZero int/str) |
| ryscene.py | __all__ completely wrong (PFlux/lllPhotonrates) -> fixed; 7 unused imports removed; DataFrame.ix -> .loc (pandas 2.0 crash fix); null bytes stripped |
| rystare.py | 7 unused imports removed (incl. sp.signal -> signal), __all__ duplicate fixed, from __future__ removed, pre-existing bug in get_summary_stats documented; wavelength-units inconsistency in nElecCntThermalScene / nEcntThermalOptics / nElecCntReflSun documented (wl must be um, not SI metres) |
| ryplot.py | Plotly already removed; VCS keywords stripped; 4 dead local `__all__` removed; `import os` removed; 46 long lines wrapped; docstrings added to `RGBToPyCmap` and `clippedcolorbar`; `shadoAllpe` bug documented; test suite written (17 sandbox checks pass) |
| rytarggen.py | 7 unused imports removed (sys, scipy as sp, scipy.stats, re, ryhdf5, ryprob, top-level ryplot); ryplot moved to lazy import inside analyse_HDF5_image; ~30 lines of commented-out dead assigncheck predecessor code removed; __all__ expanded from 5 -> 10 names; calcLuxEquivalent docstring fixed (wrong param names); wavelength-units inconsistency documented in both calcLuxEquivalent (SI metres) and calcTemperatureEquivalent (um) |

---

## Outstanding (0 files)

All files complete.

---

## Key Technical Decisions

### Cardinal rules
- Never change observable behaviour -- even if behaviour is buggy. Pre-existing bugs are
  preserved, documented in the docstring (Note: section), and asserted in tests so any
  accidental future fix is caught.
- No new dependencies -- only packages already in use.
- Public API signatures are frozen unless the user explicitly requests a change.

### Patterns established

__all__ audit -- every refactor now audits __all__ with hasattr to find phantom names
(listed but non-existent) and missing names. Both rylookup.py and ryscene.py had
completely wrong __all__; rytarggen.py was missing 5 public names.

Python 2 dead code removal -- from __future__ import ..., sys.version_info > 2 branches,
try: import xml.etree.cElementTree blocks, commented-out dead code blocks, and the
now-redundant import sys are all removed.

DataFrame.ix -- removed in pandas 1.0; replace with .loc (label-based) or .iloc
(positional). Treated as a crash fix, not a behaviour change.

Lazy imports -- optional/heavy dependencies (scipy.interpolate, imageio, pyradi.ryplot,
etc.) stay inside function bodies, not at module level.

np.savetxt mode -- must write via BytesIO buffer when the outer file is in text mode.

ryplanck.planck wavelength units -- planckql (and all *ql / *el / *en variants) expects
wavelength in MICROMETRES, not SI metres. Functions that call ryplanck.planck with a wl
parameter and claim "all values in SI units" in their docstring are incorrect -- the wl
must be um. This affects:
  - rystare.nElecCntThermalScene
  - rystare.nEcntThermalOptics
  - rystare.nElecCntReflSun
  - rytarggen.calcTemperatureEquivalent
Each has a Note: in its docstring. Tests use um wavelength vectors.

rytarggen.calcLuxEquivalent wavelength -- uniquely, this function expects wavelength in
SI metres (it multiplies by 1e6 internally for the photopic formula), in contrast to
calcTemperatureEquivalent which expects um. Both documented with Note:.

### Test isolation (multi-file pytest suite)

Three overlapping problems when running the full suite in one process:

1. Stub vs real package ordering -- test_ryhdf5.py installs a types.ModuleType stub for
   pyradi; test_rylookup.py needs the real package. Fix: test_rylookup.py detects stubs
   by absence of __path__ and evicts them before importing.

2. MagicMock wiped at collection time -- module-level mock setup in test_ryhdf5.py is
   overwritten during collection by test_rylookup.py's eviction. Fix: autouse fixture in
   test_ryhdf5.py reinstalls the mock before every test at run time.

3. from . import ryplot bypasses sys.modules -- relative imports resolve via the package
   object attribute (sys.modules['pyradi'].ryplot), not just sys.modules['pyradi.ryplot'].
   Both must be patched.

The correct pattern (pseudocode):

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

### transformations package (ryscene)

Third-party Gohlke wheel; stubbed in tests with MagicMock returning a known identity
quaternion [1,0,0,0]. Listed in environment.yml under pip:.

### Windows/Linux mount sync issue

The Edit tool writes to the Windows filesystem (K:\...) while the bash sandbox accesses
the same files via a Linux mount (/sessions/.../mnt/...). Occasionally the mount lags or
truncates writes, causing null-byte padding or missing content at the end of files.

Fix pattern:
  1. Strip null bytes after any Edit-tool write:
       python3 -c "
         with open('file.py','rb') as f: data=f.read()
         data=data.rstrip(b'\x00')
         with open('file.py','wb') as f: f.write(data)"

  2. Always syntax-check immediately after editing:
       python3 -m py_compile file.py && echo "OK"

  3. For large content additions (README sections, __main__ blocks), write content via
     the Write tool into a .py helper script, then execute it with bash. Do NOT use
     heredocs (cat << EOF) or cat >> for multi-line content -- bash truncates them.

---

## Test suite health

Full suite command:
  pytest test/test_ryutils.py test/test_ryfiles.py test/test_ry3dnoise.py
         test/test_rychroma.py test/test_rydetector.py test/test_rylookup.py
         test/test_rymodtran.py test/test_ryhdf5.py test/test_rypflux.py
         test/test_ryplanck.py test/test_ryprob.py test/test_ryplotspherical.py
         test/test_ryptw.py test/test_ryscene.py test/test_rystare.py
         test/test_rytarggen.py -v

Last known results:
  - 580 passed across the first 14 files (pre-rystare baseline)
  - test_rystare.py: 57 passed, 0 failed (after wavelength-units fix)
  - test_rytarggen.py: 22 tests written and green in partial run
  - Full suite (all 16 files): all 659 tests pass after cross-version API fixes
    applied to rystare.py and rytarggen.py (see Key Technical Decisions below)

Note: the full suite MUST be run in the conda `pyradi` environment on Windows.
The Linux sandbox lacks scipy/numpy so tests cannot run there.

---

## Additional Technical Decisions (cross-version API session, 2026-04-22)

### Installed vs local pyradi API mismatch

The conda-installed `pyradi` package and the local source tree diverge on several APIs.
Never write a compatibility shim -- inline the implementation directly:

- `ryplanck.planckql(wl, T)` (local) vs `ryplanck.planck(wl, T, stype='ql')` (installed):
  replaced with a private `_ryplanckql()` function in rystare.py and rytarggen.py.
  Constants computed from `scipy.constants` INSIDE the function to avoid module-level
  variable corruption across test files.

- `ryutils.poissonarray(arr, seedval=0)` -- absent from installed package.
  Inlined in rystare.py and rytarggen.py.

- `ryutils.circ(x, y, d)` -- absent from installed package.
  Inlined in rytarggen.py using `np.where(np.sqrt(x**2+y**2) < d/2., ...)`.

### `@fixDimensions` shape contract

Local pyradi wraps every Planck function with `@fixDimensions`, which handles
`(N,1)` spectral x `(M,1)` temperature -> `(N,M)` result via meshgrid.
Plain numpy raises `ValueError` for `(N,1) * (M,1)`. The inline `_ryplanckql`
replicates this by transposing temperature when both inputs are 2-D column vectors:
    if sp.ndim == 2 and sp.shape[1] == 1 and tp.ndim == 2 and tp.shape[1] == 1:
        tp = tp.T

### `test_ryplanck.py` stub corruption

`test_ryplanck.py` installs a `pyradi.ryutils` stub at module level (collection time).
This corrupts module state for `test_rystare.py` and `test_rytarggen.py` in full-suite
runs (partial runs pass). Fix: make production code fully independent of the
version-specific APIs listed above.

### Test assertion fixes discovered

- `np.allclose` vs `np.array_equal`: photon noise (~3e-8 relative) is within
  allclose's rtol=1e-5 -- use `np.array_equal` when asserting arrays are NOT equal.
- `float(n_hot) > float(n_cool)`: comparison operators on 0-d or shape-(1,) arrays
  must be wrapped in `float()` to produce a Python bool.
- `np.all(n > 0)` / `np.all(np.isfinite(result))`: use np.all() when the function
  returns a multi-element array.

### ryhdf5 import audit

`ryhdf5.erase_create_HDF` and `ryhdf5.open_HDF` live in `pyradi.ryhdf5`, not in
`pyradi.ryfiles`. The unused-import cleanup in rytarggen.py accidentally dropped
`import pyradi.ryhdf5 as ryhdf5` -- restored and all call sites updated.

---

## ryplot.py — Active refactor progress

### Session context
- Skill: /mnt/python-refactor/SKILL.md
- Pyradi folder: K:\WorkN\pyradi\pyradi\pyradi
- Python target: 3.12+, conda env `pyradi` on Windows

### File facts (Phase 0 complete)
- 4719 lines after null-byte strip (34670 trailing null bytes removed on session start)
- No Plotly code remaining (already removed by user)
- No `from __future__` imports
- Indentation: 4-space throughout (2 stray 2-space lines near L1505)
- VCS keywords: `$Id$` (L1), `$HeadURL$` (L2), `$Revision$` (L42)
- Dead local `__all__` inside: `FilledMarker.__init__`, `Markers.__init__`,
  `ProcessImage.__init__`, `Plotter.__init__` — all local variables, silently ignored
- `sys`: kept (used by `sys.exit` at L547 for matplotlib version check)
- `os`: imported at L54 but only referenced in commented-out `__main__` code — remove
- `pd`: light usage (pd.Series checks in plot/logLog/etc.)
- Missing docstrings: `RGBToPyCmap` (L3998), `clippedcolorbar` (L1789, static method)
- Pre-existing bug 1: `turbo_colormap_data.shadoAllpe[0]` — typo for `.shape[0]`;
  in an `else` branch that never runs in practice (turbo already registered in mpl 3.x)
- Pre-existing bug 2: `ProcessImage.compressEqualizeImage` — CDF normalisation divides by
  `imgFlatSort[-1]` (max value) instead of `imgFlatSort.cumsum()[-1]` (total), so output
  is NOT bounded to [0, 1] for random inputs
- Pre-existing bug 3: `Plotter.polar` — multi-column `r` with `highlightNegative=True`
  raises `ValueError` because `negrrr[ii][0] < 0` compares a row-array to scalar.
  Workaround: pass BOTH `rscale=[-x, rmax]` (prevents the internal override of
  `highlightNegative` back to True) AND `highlightNegative=False` (skips the buggy
  loop). Neither alone is sufficient: `rscale[0]<0` leaves `highlightNegative`
  at its default (`True`); `highlightNegative=False` alone is overridden to `True`
  when `rscale` is None.
- Pre-existing bug 4: `Plotter.plotArray` — `labels=None` (default) raises `TypeError`
  at `labels[0]`; always supply a labels list explicitly
- Long lines (>120 chars): 54 lines, all in docstrings
- `__main__` block: starts L4034, 685 lines of demo code

### Phase status
| Phase | Status | Notes |
|-------|--------|-------|
| 0 — Understand | ✓ complete | See File facts above |
| 1 — Baseline tests | ✓ complete | test/test_ryplot.py written; 29 assertions verified in sandbox |
| 2 — Refactor plan | ✓ approved | User approved on 2026-04-22 |
| 3 — Refactor | ✓ complete | All 5 steps done; 16/16 sandbox checks pass |
| 4 — Conda yaml | ✓ complete | environment.yml already covered all ryplot imports |
| 5 — README | ✓ complete | ryplot added to all sections; Outstanding cleared |
| 6 — Final verify | ✓ complete | 86 passed, 0 failed in conda `pyradi` env on Windows |

### Phase 3 Step tracker
| Step | Description | Status |
|------|-------------|--------|
| A | VCS keywords, `__version__`, dead `__all__` locals, f-string | ✓ done |
| B | Remove `import os` (unused at module level) | ✓ done |
| C | Wrap 54 long lines — 46 fixed, 8 URL comments exempt | ✓ done |
| D | Add docstrings: `RGBToPyCmap`, `clippedcolorbar` | ✓ done |
| E | Preserve & document `shadoAllpe` bug (Note: comment added) | ✓ done |

### Phase 3 changes summary
- Removed `#  $Id$` and `#  $HeadURL$` header lines
- `__version__ = '$Revision$'` → `__version__ = '0.1.0'`
- Removed 4 dead `__all__ = [...]` local variable assignments inside
  `FilledMarker.__init__`, `Markers.__init__`, `ProcessImage.__init__`,
  `Plotter.__init__`
- `'{0}'.format(vnum)` → `f'{vnum}'` in matplotlib version guard
- Removed `import os` (only appeared in commented-out `__main__` lines)
- 46 docstring lines > 120 chars wrapped; 8 URL-only comments exempt
- Google-style docstrings added to `RGBToPyCmap` and `clippedcolorbar`
- Pre-existing bug `turbo_colormap_data.shadoAllpe[0]` documented with
  a `Note:` comment block explaining the typo, why it is preserved, and
  why the branch is dead code in practice

### Resume instructions (next session)
1. Read REFACTOR_BRIEF.md → ryplot section to confirm status
2. Read /mnt/python-refactor/SKILL.md for skill rules
3. Run Phase 4 (update environment.yml) — scan all imports in ryplot.py
4. Run Phase 5 (update README.md) — add ryplot to refactoring status table,
   project structure, commands table, and public API summary
5. Run Phase 6 verification in conda `pyradi` env on Windows:
   ```
   pytest test/test_ryplot.py -v --tb=short
   ```
6. Update Outstanding table in this file: move ryplot.py to Completed

---

## Skill file

All learnings are captured incrementally in /mnt/python-refactor/SKILL.md.

Notable recent additions:
  - __all__ phantom audit
  - from __future__ removal
  - xml.etree.cElementTree simplification
  - import sys post-cleanup audit
  - pre-existing bug preservation protocol
  - autouse fixture pattern for sys.modules test isolation
  - ryplanck.planck wavelength-units gotcha (um not SI metres)
  - Windows/Linux mount sync fix pattern (Write tool + python script, not heredoc)
