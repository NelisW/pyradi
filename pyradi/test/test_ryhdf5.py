"""Regression tests for ryhdf5.py — HDF5 file I/O utilities."""

import os
import sys
import types
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import tempfile

import numpy as np
import pytest
import h5py

# ── Mock pyradi package before import ────────────────────────────────────────
# ryhdf5's plotting functions do `from . import ryplot` (relative import).
# Those are only executed when the plotting functions are called, not at import
# time, so we only need a stub on sys.modules so the relative import resolves.
#
# _ryplot_mock is always a MagicMock, created unconditionally at module level.
# The module-level sys.modules entries are set here for the import of ryhdf5;
# but another test file (test_rylookup.py) evicts all pyradi.* entries during
# its own collection so it can load the real pyradi package.  By the time the
# tests in this file actually run, the mock may be gone.  The autouse fixture
# _reinstall_ryplot_mock (below) reinstalls it before every test so that
# `from . import ryplot` inside ryhdf5 function bodies always gets our mock.
_ryplot_mock = MagicMock()
if 'pyradi' not in sys.modules:
    _pyradi_stub = types.ModuleType('pyradi')
    _pyradi_stub.ryplot = _ryplot_mock
    sys.modules['pyradi'] = _pyradi_stub
sys.modules['pyradi.ryplot'] = _ryplot_mock
sys.modules['pyradi'].ryplot = _ryplot_mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import ryhdf5 as rhdf
# `ryhdf5` is imported as a standalone module, so __package__ is '' and
# `from . import ryplot` (a relative import) would raise ImportError.
# Setting __package__ to 'pyradi' makes Python resolve the dot against
# sys.modules['pyradi.ryplot'], which is our mock above.
# __spec__ is set to None so Python 3.14+ does not raise
# DeprecationWarning: __package__ != __spec__.parent
# (__spec__.parent is a read-only property and cannot be updated to match).
rhdf.__package__ = 'pyradi'
rhdf.__spec__ = None


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_h5(path, datasets=None):
    """Create a minimal HDF5 file for testing.  Returns the file path."""
    with h5py.File(path, 'w') as f:
        if datasets:
            for name, data in datasets.items():
                f.create_dataset(name, data=data)
    return path


# ── autouse fixture — keeps the ryplot mock alive across the full suite ───────

@pytest.fixture(autouse=True)
def _reinstall_ryplot_mock():
    """Reinstall _ryplot_mock into sys.modules before every test in this file.

    test_rylookup.py evicts all sys.modules['pyradi.*'] entries at collection
    time so it can load the real pyradi package.  By the time the tests here
    run, sys.modules['pyradi.ryplot'] may point to the real module (no
    reset_mock).  Reinstalling here, at test-run time, ensures every test in
    this file always gets the MagicMock regardless of collection order.
    """
    sys.modules['pyradi.ryplot'] = _ryplot_mock
    sys.modules['pyradi'].ryplot = _ryplot_mock
    yield


# ── open_HDF ─────────────────────────────────────────────────────────────────

class TestOpenHDF:
    def test_returns_h5py_file_object(self, tmp_path):
        p = str(_make_h5(tmp_path / 'test.h5'))
        f = rhdf.open_HDF(p, mode='r')
        assert isinstance(f, h5py.File)
        f.close()

    def test_default_mode_is_read(self, tmp_path):
        p = str(_make_h5(tmp_path / 'test.h5'))
        f = rhdf.open_HDF(p)
        assert f.mode == 'r'
        f.close()

    def test_write_mode_accepted(self, tmp_path):
        p = str(tmp_path / 'new.h5')
        f = rhdf.open_HDF(p, mode='w')
        assert f.mode == 'r+'  # h5py reports 'r+' for writable files
        f.close()


# ── erase_create_HDF ─────────────────────────────────────────────────────────

class TestEraseCreateHDF:
    def test_creates_new_file(self, tmp_path):
        p = str(tmp_path / 'new.h5')
        f = rhdf.erase_create_HDF(p)
        assert isinstance(f, h5py.File)
        f.close()
        assert os.path.isfile(p)

    def test_truncates_existing_file(self, tmp_path):
        p = str(tmp_path / 'existing.h5')
        # Write some data into the file first
        _make_h5(p, datasets={'arr': np.arange(10)})
        size_before = os.path.getsize(p)

        f = rhdf.erase_create_HDF(p)
        f.close()
        # File was re-created from scratch — no datasets remain
        with h5py.File(p, 'r') as f2:
            assert list(f2.keys()) == []

    def test_returns_writable_file(self, tmp_path):
        p = str(tmp_path / 'write.h5')
        f = rhdf.erase_create_HDF(p)
        # Should be able to write a dataset without raising
        f.create_dataset('test', data=np.array([1, 2, 3]))
        f.close()


# ── get_HDF_DS ───────────────────────────────────────────────────────────────

class TestGetHDFDS:
    def test_returns_dataset_when_path_exists(self, tmp_path):
        p = str(_make_h5(tmp_path / 'test.h5', datasets={'sensor/data': np.ones((5,))}))
        with h5py.File(p, 'r') as f:
            result = rhdf.get_HDF_DS(f, 'sensor/data')
            assert result is not None
            assert isinstance(result, h5py.Dataset)

    def test_returns_none_when_path_missing(self, tmp_path):
        p = str(_make_h5(tmp_path / 'test.h5'))
        with h5py.File(p, 'r') as f:
            result = rhdf.get_HDF_DS(f, 'nonexistent/path')
            assert result is None

    def test_data_readable_from_returned_dataset(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0])
        p = str(_make_h5(tmp_path / 'test.h5', datasets={'values': arr}))
        with h5py.File(p, 'r') as f:
            ds = rhdf.get_HDF_DS(f, 'values')
            np.testing.assert_array_equal(ds[:], arr)


# ── print_HDF5_text ───────────────────────────────────────────────────────────

class TestPrintHDF5Text:
    def test_prints_given_string(self, capsys):
        rhdf.print_HDF5_text('hello world')
        captured = capsys.readouterr()
        assert 'hello world' in captured.out

    def test_prints_path_string(self, capsys):
        rhdf.print_HDF5_text('/sensors/radiance')
        captured = capsys.readouterr()
        assert '/sensors/radiance' in captured.out

    def test_returns_none(self):
        result = rhdf.print_HDF5_text('anything')
        assert result is None


# ── print_HDF5_dataset_value ─────────────────────────────────────────────────

class TestPrintHDF5DatasetValue:
    def test_prints_for_dataset(self, tmp_path, capsys):
        p = str(_make_h5(tmp_path / 'test.h5', datasets={'arr': np.array([1, 2, 3])}))
        with h5py.File(p, 'r') as f:
            ds = f['arr']
            rhdf.print_HDF5_dataset_value('arr', ds)
            captured = capsys.readouterr()
            assert 'arr' in captured.out

    def test_silent_for_group(self, tmp_path, capsys):
        p = str(tmp_path / 'test.h5')
        with h5py.File(p, 'w') as f:
            grp = f.create_group('mygroup')
            rhdf.print_HDF5_dataset_value('mygroup', grp)
            captured = capsys.readouterr()
            assert captured.out == ''

    def test_returns_none(self, tmp_path):
        p = str(_make_h5(tmp_path / 'test.h5', datasets={'d': np.array([1])}))
        with h5py.File(p, 'r') as f:
            result = rhdf.print_HDF5_dataset_value('d', f['d'])
            assert result is None


# ── get_HDF_branches ──────────────────────────────────────────────────────────

class TestGetHDFBranches:
    def test_empty_file_returns_none(self, tmp_path):
        """visit() on an empty file never calls the callback; returns None."""
        p = str(_make_h5(tmp_path / 'empty.h5'))
        with h5py.File(p, 'r') as f:
            result = rhdf.get_HDF_branches(f)
        assert result is None

    def test_nonempty_file_raises(self, tmp_path):
        """Known bug: get_HDF_branches passes itself as visitor.
        visit() calls get_HDF_branches(name_string), which then calls
        name_string.visit(...) → AttributeError because str has no .visit().
        Behaviour is preserved unchanged (cardinal rule)."""
        p = str(_make_h5(tmp_path / 'nonempty.h5', datasets={'x': np.array([1])}))
        with h5py.File(p, 'r') as f:
            with pytest.raises(AttributeError):
                rhdf.get_HDF_branches(f)


# ── plotHDF5Bitmaps ───────────────────────────────────────────────────────────

class TestPlotHDF5Bitmaps:
    """plotHDF5Bitmaps uses skimage.imsave and requires arrays with non-zero values."""

    def test_skips_scalar_datasets(self, tmp_path):
        """Datasets with shape==() must be silently skipped."""
        p = str(tmp_path / 'test.h5')
        with h5py.File(p, 'w') as f:
            f.create_dataset('scalar', data=np.float64(3.14))
            with patch('ryhdf5.imsave') as mock_imsave:
                rhdf.plotHDF5Bitmaps(f, str(tmp_path / 'out'), lstimgs=['scalar'])
                mock_imsave.assert_not_called()

    def test_skips_all_zero_array(self, tmp_path):
        """Arrays where max==0 must be silently skipped (avoid divide-by-zero)."""
        p = str(tmp_path / 'test.h5')
        with h5py.File(p, 'w') as f:
            f.create_dataset('zeros', data=np.zeros((4, 4)))
            with patch('ryhdf5.imsave') as mock_imsave:
                rhdf.plotHDF5Bitmaps(f, str(tmp_path / 'out'), lstimgs=['zeros'])
                mock_imsave.assert_not_called()

    def test_saves_image_for_nonzero_array(self, tmp_path):
        """Non-zero 2-D array must produce one imsave call."""
        p = str(tmp_path / 'test.h5')
        arr = np.ones((4, 4), dtype=np.float32)
        with h5py.File(p, 'w') as f:
            f.create_dataset('img', data=arr)
            with patch('ryhdf5.imsave') as mock_imsave:
                rhdf.plotHDF5Bitmaps(f, str(tmp_path / 'out'), pformat='png', lstimgs=['img'])
                mock_imsave.assert_called_once()
                # saved array must be uint8
                saved_arr = mock_imsave.call_args[0][1]
                assert saved_arr.dtype == np.uint8


# ── plotHDF5Images ────────────────────────────────────────────────────────────

class TestPlotHDF5Images:
    def test_skips_scalar_datasets(self, tmp_path):
        p = str(tmp_path / 'test.h5')
        with h5py.File(p, 'w') as f:
            f.create_dataset('scalar', data=np.float64(1.0))
            mock_ryplot = sys.modules['pyradi.ryplot']
            mock_ryplot.reset_mock()
            rhdf.plotHDF5Images(f, str(tmp_path / 'out'), lstimgs=['scalar'])
            mock_ryplot.savePlot.assert_not_called()

    def test_calls_savePlot_for_array(self, tmp_path):
        p = str(tmp_path / 'test.h5')
        arr = np.ones((4, 4))
        with h5py.File(p, 'w') as f:
            f.create_dataset('img', data=arr)
            mock_ryplot = sys.modules['pyradi.ryplot']
            mock_ryplot.reset_mock()
            # savePlot is used as a context manager
            mock_ryplot.savePlot.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ryplot.savePlot.return_value.__exit__ = MagicMock(return_value=False)
            rhdf.plotHDF5Images(f, str(tmp_path / 'out'), lstimgs=['img'])
            mock_ryplot.savePlot.assert_called_once()


# ── plotHDF5Histograms ────────────────────────────────────────────────────────

class TestPlotHDF5Histograms:
    def test_calls_savePlot(self, tmp_path):
        p = str(tmp_path / 'test.h5')
        arr = np.random.default_rng(0).random((20,))
        with h5py.File(p, 'w') as f:
            f.create_dataset('data', data=arr)
            mock_ryplot = sys.modules['pyradi.ryplot']
            mock_ryplot.reset_mock()
            mock_ryplot.savePlot.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_ryplot.savePlot.return_value.__exit__ = MagicMock(return_value=False)
            rhdf.plotHDF5Histograms(f, str(tmp_path / 'out'), lstimgs=['data'], bins=10)
            mock_ryplot.savePlot.assert_called_once()
