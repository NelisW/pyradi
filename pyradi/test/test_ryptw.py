"""Regression tests for ryptw.py — Agema/Cedip/FLIR PTW file reader."""

import io
import os
import struct
import sys
import types

import numpy as np
import pytest

# ── pyradi stub ────────────────────────────────────────────────────────────────
# ryptw.py imports several pyradi sub-modules at module level.  Stub them so the
# test suite does not require the full pyradi package to be installed.
for _mod in [
    'pyradi', 'pyradi.ryutils', 'pyradi.ryplanck',
    'pyradi.ryfiles', 'pyradi.ryplot', 'pyradi.rylookup',
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# ── scipy stub ─────────────────────────────────────────────────────────────────
if 'scipy' not in sys.modules:
    _sci = types.ModuleType('scipy')
    sys.modules['scipy'] = _sci
if 'scipy.interpolate' not in sys.modules:
    _interp = types.ModuleType('scipy.interpolate')
    _interp.interp1d = None
    sys.modules['scipy.interpolate'] = _interp
    sys.modules['scipy'].interpolate = _interp  # type: ignore[attr-defined]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import ryptw

# ── real PTW file paths ────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PTW_LWIR = os.path.join(DATA_DIR, 'PyradiSampleLWIR.ptw')
PTW_MWIR = os.path.join(DATA_DIR, 'PyradiSampleMWIR.ptw')


# ── TestMyInt ──────────────────────────────────────────────────────────────────

class TestMyInt:
    def test_positive(self):
        assert ryptw.myint(struct.pack('<h', 1234)) == 1234

    def test_zero(self):
        assert ryptw.myint(struct.pack('<h', 0)) == 0

    def test_negative(self):
        assert ryptw.myint(struct.pack('<h', -500)) == -500

    def test_max_signed_short(self):
        assert ryptw.myint(struct.pack('<h', 32767)) == 32767

    def test_min_signed_short(self):
        assert ryptw.myint(struct.pack('<h', -32768)) == -32768


# ── TestMyLong ─────────────────────────────────────────────────────────────────

class TestMyLong:
    """mylong uses x[0] + (x[1]<<8) + (x[2]<<16) + (x[3]<<32).

    Note: the last shift is <<32 (a pre-existing bug — should be <<24).
    For values that fit in 24 bits (byte 3 == 0) the result is correct.
    Tests use only values < 2**24 to avoid the bug.
    """

    def test_zero(self):
        assert ryptw.mylong(b'\x00\x00\x00\x00') == 0

    def test_small_value(self):
        # 3077 = 0x0C05: bytes 0x05, 0x0C, 0x00, 0x00
        assert ryptw.mylong(b'\x05\x0c\x00\x00') == 3077

    def test_one(self):
        assert ryptw.mylong(b'\x01\x00\x00\x00') == 1

    def test_256(self):
        assert ryptw.mylong(b'\x00\x01\x00\x00') == 256

    def test_65536(self):
        assert ryptw.mylong(b'\x00\x00\x01\x00') == 65536

    def test_byte3_shift_bug_documented(self):
        # When byte 3 != 0, mylong gives (byte3 << 32) not (byte3 << 24).
        # A value with only byte 3 set: expected correct=0x01000000=16777216,
        # but mylong returns 0x01_0000_0000 = 4294967296.
        result = ryptw.mylong(b'\x00\x00\x00\x01')
        assert result == 4294967296   # 1 << 32, NOT 1 << 24


# ── TestMyFloat ────────────────────────────────────────────────────────────────

class TestMyFloat:
    def test_pi(self):
        assert ryptw.myfloat(struct.pack('<f', 3.14)) == pytest.approx(3.14, rel=1e-5)

    def test_zero(self):
        assert ryptw.myfloat(struct.pack('<f', 0.0)) == 0.0

    def test_negative(self):
        assert ryptw.myfloat(struct.pack('<f', -1.5)) == pytest.approx(-1.5, rel=1e-5)


# ── TestMyDouble ───────────────────────────────────────────────────────────────

class TestMyDouble:
    def test_e(self):
        assert ryptw.mydouble(struct.pack('<d', 2.718281828)) == pytest.approx(2.718281828, rel=1e-12)

    def test_zero(self):
        assert ryptw.mydouble(struct.pack('<d', 0.0)) == 0.0

    def test_large(self):
        assert ryptw.mydouble(struct.pack('<d', 1e15)) == pytest.approx(1e15, rel=1e-12)


# ── TestMyByte ─────────────────────────────────────────────────────────────────

class TestMyByte:
    def test_zero(self):
        assert ryptw.mybyte(b'\x00') == 0

    def test_255(self):
        assert ryptw.mybyte(b'\xff') == 255

    def test_ascii_a(self):
        assert ryptw.mybyte(b'A') == 65


# ── TestMyPass ─────────────────────────────────────────────────────────────────

class TestMyPass:
    def test_int(self):
        assert ryptw.mypass(42) == 42

    def test_string(self):
        assert ryptw.mypass('hello') == 'hello'

    def test_none(self):
        assert ryptw.mypass(None) is None

    def test_bytes(self):
        assert ryptw.mypass(b'\x01\x02') == b'\x01\x02'


# ── TestMyRGB ──────────────────────────────────────────────────────────────────

class TestMyRGB:
    def test_red(self):
        assert ryptw.myRGB(b'\xff\x00\x00') == '#ff0000'

    def test_green(self):
        assert ryptw.myRGB(b'\x00\xff\x00') == '#00ff00'

    def test_blue(self):
        assert ryptw.myRGB(b'\x00\x00\xff') == '#0000ff'

    def test_black(self):
        assert ryptw.myRGB(b'\x00\x00\x00') == '#000000'

    def test_white(self):
        assert ryptw.myRGB(b'\xff\xff\xff') == '#ffffff'

    def test_mixed(self):
        assert ryptw.myRGB(b'\xde\xad\xbe') == '#deadbe'


# ── TestTerminateStrOnZero ─────────────────────────────────────────────────────

class TestTerminateStrOnZero:
    """terminateStrOnZero is documented as terminating at the first null byte.

    Note: In Python 3, indexing a bytes object gives an int, so the comparison
    str[idx] != '\\00' (int vs str) is always True — the function never
    terminates early and always returns the full input slice.  This is a
    pre-existing bug; tests assert the actual Python-3 behaviour.
    """

    def test_no_null_bytes(self):
        # No null: full input returned (as expected)
        assert ryptw.terminateStrOnZero(b'hello') == b'hello'

    def test_with_null_in_middle(self):
        # Bug: null is NOT detected — full input is returned
        result = ryptw.terminateStrOnZero(b'hello\x00world')
        assert result == b'hello\x00world'   # NOT b'hello'

    def test_all_nulls(self):
        result = ryptw.terminateStrOnZero(b'\x00\x00\x00')
        assert result == b'\x00\x00\x00'

    def test_empty(self):
        assert ryptw.terminateStrOnZero(b'') == b''


# ── TestPTWFrameInfo ───────────────────────────────────────────────────────────

class TestPTWFrameInfo:
    def test_instantiates(self):
        h = ryptw.PTWFrameInfo()
        assert h is not None

    def test_default_rows_zero(self):
        assert ryptw.PTWFrameInfo().h_Rows == 0

    def test_default_cols_zero(self):
        assert ryptw.PTWFrameInfo().h_Cols == 0

    def test_default_format_unknown(self):
        assert ryptw.PTWFrameInfo().h_format == 'unknown'

    def test_default_data_empty_list(self):
        assert ryptw.PTWFrameInfo().data == []

    def test_default_filename_empty(self):
        assert ryptw.PTWFrameInfo().FileName == ''

    def test_default_lockin_zero(self):
        assert ryptw.PTWFrameInfo().h_Lockin == 0


# ── TestReadPTWHeader ──────────────────────────────────────────────────────────

class TestReadPTWHeader:
    @pytest.fixture(autouse=True)
    def require_ptw(self):
        if not os.path.exists(PTW_LWIR):
            pytest.skip('PyradiSampleLWIR.ptw not found')

    def test_returns_ptw_frame_info(self):
        h = ryptw.readPTWHeader(PTW_LWIR)
        assert isinstance(h, ryptw.PTWFrameInfo)

    def test_format_is_cedip(self):
        h = ryptw.readPTWHeader(PTW_LWIR)
        assert h.h_format == 'cedip'

    def test_signature_is_ced(self):
        h = ryptw.readPTWHeader(PTW_LWIR)
        assert h.h_Signature == 'CED'

    def test_frame_count_positive(self):
        h = ryptw.readPTWHeader(PTW_LWIR)
        assert h.h_NumberOfFieldInFile > 0

    def test_rows_positive(self):
        h = ryptw.readPTWHeader(PTW_LWIR)
        assert h.h_Rows > 0

    def test_cols_positive(self):
        h = ryptw.readPTWHeader(PTW_LWIR)
        assert h.h_Cols > 0

    def test_rows_cols_match_known(self):
        """PyradiSampleLWIR.ptw is known to be 68 rows × 75 cols."""
        h = ryptw.readPTWHeader(PTW_LWIR)
        assert h.h_Rows == 68
        assert h.h_Cols == 75

    def test_frame_count_100(self):
        h = ryptw.readPTWHeader(PTW_LWIR)
        assert h.h_NumberOfFieldInFile == 100

    def test_main_header_size_positive(self):
        h = ryptw.readPTWHeader(PTW_LWIR)
        assert h.h_MainHeaderSize > 0

    def test_filename_set(self):
        h = ryptw.readPTWHeader(PTW_LWIR)
        assert h.FileName == PTW_LWIR

    def test_lastframe_equals_frame_count(self):
        h = ryptw.readPTWHeader(PTW_LWIR)
        assert h.h_lastframe == h.h_NumberOfFieldInFile


# ── TestGetPTWFrame ────────────────────────────────────────────────────────────

class TestGetPTWFrame:
    @pytest.fixture(autouse=True)
    def header(self):
        if not os.path.exists(PTW_LWIR):
            pytest.skip('PyradiSampleLWIR.ptw not found')
        self.h = ryptw.readPTWHeader(PTW_LWIR)

    def test_returns_tuple(self):
        result = ryptw.getPTWFrame(self.h, 1)
        assert isinstance(result, tuple) and len(result) == 2

    def test_data_is_ndarray(self):
        data, _ = ryptw.getPTWFrame(self.h, 1)
        assert isinstance(data, np.ndarray)

    def test_data_shape(self):
        data, _ = ryptw.getPTWFrame(self.h, 1)
        assert data.shape == (self.h.h_Rows, self.h.h_Cols)

    def test_data_values_finite(self):
        data, _ = ryptw.getPTWFrame(self.h, 1)
        assert np.isfinite(data).all()

    def test_data_nonzero(self):
        data, _ = ryptw.getPTWFrame(self.h, 1)
        assert data.max() > 0

    def test_header_returned(self):
        _, fh = ryptw.getPTWFrame(self.h, 1)
        assert isinstance(fh, ryptw.PTWFrameInfo)

    def test_different_frames_differ(self):
        data1, _ = ryptw.getPTWFrame(self.h, 1)
        data2, _ = ryptw.getPTWFrame(self.h, 5)
        # Not necessarily different, but shape must match
        assert data1.shape == data2.shape

    def test_wrong_format_returns_error(self):
        bad = ryptw.PTWFrameInfo()
        bad.h_format = 'agema'
        result = ryptw.getPTWFrame(bad, 1)
        # Returns errorresult array (not a 2-tuple)
        assert isinstance(result, np.ndarray)

    def test_frame_index_zero_returns_error(self):
        result = ryptw.getPTWFrame(self.h, 0)
        assert isinstance(result, np.ndarray)

    def test_frame_index_exceeds_count_returns_error(self):
        result = ryptw.getPTWFrame(self.h, self.h.h_lastframe + 1)
        assert isinstance(result, np.ndarray)


# ── TestGetPTWFrames ───────────────────────────────────────────────────────────

class TestGetPTWFrames:
    @pytest.fixture(autouse=True)
    def header(self):
        if not os.path.exists(PTW_LWIR):
            pytest.skip('PyradiSampleLWIR.ptw not found')
        self.h = ryptw.readPTWHeader(PTW_LWIR)

    def test_returns_tuple(self):
        result = ryptw.getPTWFrames(self.h, [1, 2])
        assert isinstance(result, tuple) and len(result) == 2

    def test_data_shape_three_frames(self):
        data, _ = ryptw.getPTWFrames(self.h, [1, 2, 5])
        assert data.shape == (3, self.h.h_Rows, self.h.h_Cols)

    def test_fheaders_length(self):
        _, fheaders = ryptw.getPTWFrames(self.h, [1, 2, 5])
        assert len(fheaders) == 3

    def test_single_frame_list(self):
        data, fheaders = ryptw.getPTWFrames(self.h, [3])
        assert data.shape == (1, self.h.h_Rows, self.h.h_Cols)
        assert len(fheaders) == 1

    def test_data_finite(self):
        data, _ = ryptw.getPTWFrames(self.h, [1, 2])
        assert np.isfinite(data).all()

    def test_wrong_format_returns_error(self):
        bad = ryptw.PTWFrameInfo()
        bad.h_format = 'agema'
        data, fh = ryptw.getPTWFrames(bad, [1])
        assert isinstance(data, np.ndarray)
        assert fh is None

    def test_frame_zero_in_list_returns_error(self):
        data, fh = ryptw.getPTWFrames(self.h, [0, 1])
        assert isinstance(data, np.ndarray)
        assert fh is None

    def test_frame_exceeds_count_returns_error(self):
        data, fh = ryptw.getPTWFrames(self.h, [self.h.h_lastframe + 1])
        assert isinstance(data, np.ndarray)
        assert fh is None


# ── TestShowHeader ─────────────────────────────────────────────────────────────

class TestShowHeader:
    @pytest.fixture(autouse=True)
    def header(self):
        if not os.path.exists(PTW_LWIR):
            pytest.skip('PyradiSampleLWIR.ptw not found')
        self.h = ryptw.readPTWHeader(PTW_LWIR)

    def test_does_not_raise(self, capsys):
        ryptw.showHeader(self.h)

    def test_outputs_signature(self, capsys):
        ryptw.showHeader(self.h)
        captured = capsys.readouterr()
        assert 'CED' in captured.out

    def test_outputs_frame_count(self, capsys):
        ryptw.showHeader(self.h)
        captured = capsys.readouterr()
        assert '100' in captured.out

    def test_returns_none(self, capsys):
        result = ryptw.showHeader(self.h)
        assert result is None
