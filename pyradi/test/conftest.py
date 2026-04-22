"""conftest.py — ensure the local pyradi source tree takes priority over any
installed pyradi package in the conda environment.

pytest collects all test files before running any tests. If the conda
environment contains an installed pyradi, it gets cached in sys.modules before
individual test files can manipulate sys.path. This conftest runs before
collection and:
  1. Inserts the local pyradi/ directory and its parent at the front of sys.path.
  2. Evicts any cached pyradi.* entries so Python re-imports from the local tree.
"""
import os
import sys

# ── local paths ──────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))          # .../pyradi/test/
_pyradi_pkg = os.path.abspath(os.path.join(_here, '..'))        # .../pyradi/
_pyradi_parent = os.path.abspath(os.path.join(_here, '..', '..'))  # .../

for _p in (_pyradi_parent, _pyradi_pkg):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── evict any installed (non-local) pyradi cached before this conftest ran ───
_to_evict = [k for k in list(sys.modules)
             if k == 'pyradi' or k.startswith('pyradi.')]
for _k in _to_evict:
    del sys.modules[_k]
