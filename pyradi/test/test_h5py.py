"""
test_h5py.py — smoke test for h5py inside the comrad conda environment.

Run with:
    python test_h5py.py

A successful run prints a PASS summary.  Any DLL / import / HDF5 error will
surface here before it bites you in a real script.
"""

import sys
import os
import tempfile
import pathlib

# ── 1. Import ────────────────────────────────────────────────────────────────
print(f"Python : {sys.version}")
try:
    import h5py
    import numpy as np
    print(f"h5py   : {h5py.__version__}")
    print(f"HDF5   : {h5py.version.hdf5_version}")
    print(f"numpy  : {np.__version__}")
except ImportError as exc:
    print(f"\nFAIL — import error: {exc}")
    sys.exit(1)

# ── 2. Create, write, read back ──────────────────────────────────────────────
tmp = pathlib.Path(tempfile.mkdtemp()) / "test_comrad.h5"

rng = np.random.default_rng(42)
data_f = rng.random((50, 100), dtype=np.float64)
data_i = rng.integers(0, 1000, size=(20,), dtype=np.int32)
label   = "hello from comrad"

try:
    # --- write ---
    with h5py.File(tmp, "w") as f:
        grp = f.create_group("sensors")
        grp.create_dataset("radiance",    data=data_f, compression="gzip")
        grp.create_dataset("counts",      data=data_i)
        grp.attrs["description"] = label
        grp.attrs["n_rows"]      = data_f.shape[0]

    # --- read back ---
    with h5py.File(tmp, "r") as f:
        rad_read   = f["sensors/radiance"][:]
        cnt_read   = f["sensors/counts"][:]
        desc_read  = f["sensors"].attrs["description"]
        nrows_read = f["sensors"].attrs["n_rows"]

    # --- verify ---
    assert np.allclose(rad_read, data_f),       "radiance array mismatch"
    assert np.array_equal(cnt_read, data_i),    "counts array mismatch"
    assert desc_read == label,                  f"attribute mismatch: {desc_read!r}"
    assert nrows_read == data_f.shape[0],       "n_rows attribute mismatch"

    print(f"\nFile   : {tmp}")
    print(f"Shape  : {rad_read.shape}  dtype={rad_read.dtype}")
    print("\n*** PASS — h5py is working correctly ***")

except Exception as exc:
    print(f"\nFAIL — {type(exc).__name__}: {exc}")
    sys.exit(1)

finally:
    if tmp.exists():
        tmp.unlink()
