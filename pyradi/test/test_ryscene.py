"""Regression tests for ryscene.py — Scene orientation locus utilities."""

import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ── stub transformations ──────────────────────────────────────────────────────
# transformations is a compiled wheel (Gohlke) that may not be present in every
# environment.  Stub it with MagicMock returns that match the real shapes so
# the tests can run without the real library installed.
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

_tr_stub = types.ModuleType('transformations')
_tr_stub.quaternion_from_euler = MagicMock(return_value=_IDENTITY_QUAT.copy())
_tr_stub.quaternion_slerp = MagicMock(return_value=_IDENTITY_QUAT.copy())
_tr_stub.euler_from_quaternion = MagicMock(return_value=(0.0, 0.0, 0.0))
sys.modules['transformations'] = _tr_stub

# ── stub pyradi sub-modules (only needed to satisfy module-level imports) ─────
for _mod in ['pyradi', 'pyradi.ryutils', 'pyradi.ryplanck']:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import ryscene as rs


# ── helpers ───────────────────────────────────────────────────────────────────

_LOCUS_CSV = """\
time,x,y,z,yawDeg,pitchDeg,rollDeg
0,0,0,0,0,0,0
1,0,0,0,30,0,0
2,0,0,0,30,15,0
3,0,0,0,30,15,20
4,0,0,0,0,0,0
"""


def _scene_with_locus(tmp_path):
    """Return a Scene with the sample locus already loaded."""
    csv = tmp_path / 'locus.csv'
    csv.write_text(_LOCUS_CSV)
    sc = rs.Scene()
    sc.readLocus(str(csv))
    return sc


# ── TestDfEuler2Quat ──────────────────────────────────────────────────────────

class TestDfEuler2Quat:
    """df_euler2quat converts a DataFrame row to a quaternion wrapped in a list."""

    def _make_row(self, yaw=0.0, pitch=0.0, roll=0.0, seq='rzyx'):
        return pd.Series({
            'yawRad': yaw,
            'pitchRad': pitch,
            'rollRad': roll,
            'sequence': seq,
        })

    def test_returns_list(self):
        """Result must be a list so pandas can store it as a single cell."""
        row = self._make_row()
        result = rs.df_euler2quat(row)
        assert isinstance(result, list)

    def test_list_contains_one_element(self):
        row = self._make_row()
        result = rs.df_euler2quat(row)
        assert len(result) == 1

    def test_element_is_array(self):
        row = self._make_row()
        result = rs.df_euler2quat(row)
        assert isinstance(result[0], np.ndarray)

    def test_quaternion_shape(self):
        row = self._make_row()
        result = rs.df_euler2quat(row)
        assert result[0].shape == (4,)

    def test_calls_quaternion_from_euler_with_angles(self):
        """Verify the Euler angles and sequence are forwarded to tr."""
        _tr_stub.quaternion_from_euler.reset_mock()
        row = self._make_row(yaw=0.1, pitch=0.2, roll=0.3, seq='rzyx')
        rs.df_euler2quat(row)
        _tr_stub.quaternion_from_euler.assert_called_once_with(0.1, 0.2, 0.3, 'rzyx')


# ── TestSceneInit ─────────────────────────────────────────────────────────────

class TestSceneInit:
    def test_dfLocus_is_none(self):
        sc = rs.Scene()
        assert sc.dfLocus is None

    def test_sequence_is_rzyx(self):
        sc = rs.Scene()
        assert sc.sequence == 'rzyx'


# ── TestReadLocus ─────────────────────────────────────────────────────────────

class TestReadLocus:
    @pytest.fixture
    def sc(self, tmp_path):
        return _scene_with_locus(tmp_path)

    def test_dfLocus_is_dataframe(self, sc):
        assert isinstance(sc.dfLocus, pd.DataFrame)

    def test_row_count(self, sc):
        assert len(sc.dfLocus) == 5

    def test_yawRad_column_present(self, sc):
        assert 'yawRad' in sc.dfLocus.columns

    def test_pitchRad_column_present(self, sc):
        assert 'pitchRad' in sc.dfLocus.columns

    def test_rollRad_column_present(self, sc):
        assert 'rollRad' in sc.dfLocus.columns

    def test_sequence_column_present(self, sc):
        assert 'sequence' in sc.dfLocus.columns

    def test_quat_column_present(self, sc):
        assert 'quat' in sc.dfLocus.columns

    def test_degree_to_radian_yaw(self, sc):
        """30 ° → π/6 rad."""
        assert sc.dfLocus.loc[1, 'yawRad'] == pytest.approx(np.pi / 6, rel=1e-9)

    def test_degree_to_radian_pitch(self, sc):
        """15 ° → π/12 rad."""
        assert sc.dfLocus.loc[2, 'pitchRad'] == pytest.approx(np.pi / 12, rel=1e-9)

    def test_degree_to_radian_roll(self, sc):
        """20 ° → π/9 rad."""
        assert sc.dfLocus.loc[3, 'rollRad'] == pytest.approx(np.pi / 9, rel=1e-9)

    def test_zero_angles_zero_radians(self, sc):
        assert sc.dfLocus.loc[0, 'yawRad'] == 0.0
        assert sc.dfLocus.loc[0, 'pitchRad'] == 0.0
        assert sc.dfLocus.loc[0, 'rollRad'] == 0.0

    def test_sequence_column_value(self, sc):
        """Every row carries the 'rzyx' sequence string."""
        assert (sc.dfLocus['sequence'] == 'rzyx').all()

    def test_quat_column_contains_lists(self, sc):
        """quat column entries must be lists (so pandas stores them as one cell)."""
        for entry in sc.dfLocus['quat']:
            assert isinstance(entry, list)

    def test_quat_entries_contain_arrays(self, sc):
        for entry in sc.dfLocus['quat']:
            assert isinstance(entry[0], np.ndarray)

    def test_quat_shape(self, sc):
        for entry in sc.dfLocus['quat']:
            assert entry[0].shape == (4,)


# ── TestInterpLocus ───────────────────────────────────────────────────────────

class TestInterpLocus:
    """interpLocus interpolates orientation between two locus time-steps.

    Note: the original code used DataFrame.ix which was removed in pandas 1.0
    and raises AttributeError.  These tests only pass after the .ix → .loc fix.
    """

    @pytest.fixture
    def sc(self, tmp_path):
        return _scene_with_locus(tmp_path)

    def test_returns_tuple_of_4(self, sc):
        result = sc.interpLocus(0.5)
        assert isinstance(result, tuple) and len(result) == 4

    def test_yaw_is_float(self, sc):
        yaw, _, _, _ = sc.interpLocus(0.5)
        assert isinstance(float(yaw), float)

    def test_pitch_is_float(self, sc):
        _, pit, _, _ = sc.interpLocus(0.5)
        assert isinstance(float(pit), float)

    def test_roll_is_float(self, sc):
        _, _, rol, _ = sc.interpLocus(0.5)
        assert isinstance(float(rol), float)

    def test_quat_shape(self, sc):
        _, _, _, q = sc.interpLocus(0.5)
        assert q.shape == (4,)

    def test_calls_slerp(self, sc):
        """quaternion_slerp must be called when interpolating."""
        _tr_stub.quaternion_slerp.reset_mock()
        sc.interpLocus(0.5)
        _tr_stub.quaternion_slerp.assert_called_once()

    def test_parametric_midpoint(self, sc):
        """t=0.5 between time=0 and time=1 must pass t=0.5 to slerp."""
        _tr_stub.quaternion_slerp.reset_mock()
        sc.interpLocus(0.5)
        call_args = _tr_stub.quaternion_slerp.call_args
        t_arg = call_args[0][2]  # third positional arg is the parameter t
        assert t_arg == pytest.approx(0.5)

    def test_parametric_quarter_point(self, sc):
        """t=0.25 between time=0 and time=1 must pass t=0.25 to slerp."""
        _tr_stub.quaternion_slerp.reset_mock()
        sc.interpLocus(0.25)
        t_arg = _tr_stub.quaternion_slerp.call_args[0][2]
        assert t_arg == pytest.approx(0.25)

    def test_euler_from_quaternion_called(self, sc):
        _tr_stub.euler_from_quaternion.reset_mock()
        sc.interpLocus(1.5)
        _tr_stub.euler_from_quaternion.assert_called_once()

    def test_different_intervals_produce_different_t(self, sc):
        """Interpolating at different fractions through different intervals
        must produce different parametric t values passed to slerp."""
        _tr_stub.quaternion_slerp.reset_mock()
        sc.interpLocus(0.25)
        t1 = _tr_stub.quaternion_slerp.call_args[0][2]

        _tr_stub.quaternion_slerp.reset_mock()
        sc.interpLocus(0.75)
        t2 = _tr_stub.quaternion_slerp.call_args[0][2]

        assert t1 != t2
