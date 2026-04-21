"""Regression tests for rymodtran.py — MODTRAN tape7 reader and utilities."""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Mock pyradi dependencies before import ───────────────────────────────────
# rymodtran imports pyradi.ryfiles at module level; mock it so the module can
# be imported in environments where the full pyradi package is not installed.
#
# IMPORTANT: use types.ModuleType (not MagicMock) for the top-level 'pyradi'
# entry in sys.modules. pytest inspects every module it loads for a
# 'pytest_plugins' attribute. MagicMock auto-creates attributes on access, so
# getattr(mock, 'pytest_plugins', []) returns a MagicMock instead of the []
# default, causing "pytest.UsageError: Plugins may be specified as a sequence
# ..." for every test class. A real ModuleType object has no 'pytest_plugins',
# so the getattr falls back to [] as intended.
_mock_ryfiles = MagicMock()
if 'pyradi' not in sys.modules:
    _pyradi_stub = types.ModuleType('pyradi')
    _pyradi_stub.ryfiles = _mock_ryfiles
    sys.modules['pyradi'] = _pyradi_stub
sys.modules.setdefault('pyradi.ryfiles', _mock_ryfiles)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import rymodtran as rym

# Paths to the bundled tape7 data files
_DATA = Path(__file__).parent.parent / 'data'


# ── fixHeaders ────────────────────────────────────────────────────────────────

class TestFixHeaders:
    def test_plus_replaced(self):
        assert rym.fixHeaders('AER+CLD') == 'AERpCLD'

    def test_minus_replaced(self):
        # double-minus maps to 'mm', single '-' at start maps to 'p' per intab/outtab
        result = rym.fixHeaders('-LOG')
        assert '+' not in result
        assert '-' not in result

    def test_square_brackets_replaced(self):
        result = rym.fixHeaders('BBODY_T[K]')
        assert '[' not in result and ']' not in result

    def test_at_replaced(self):
        result = rym.fixHeaders('SOL@OBS')
        assert '@' not in result

    def test_plain_string_unchanged(self):
        assert rym.fixHeaders('FREQ') == 'FREQ'
        assert rym.fixHeaders('TOT_TRANS') == 'TOT_TRANS'

    def test_returns_string(self):
        assert isinstance(rym.fixHeaders('AER+CLD'), str)


# ── fixHeadersList ────────────────────────────────────────────────────────────

class TestFixHeadersList:
    def test_applies_to_each_element(self):
        headers = ['FREQ_CM-1', 'AER+CLD_TRANS', 'BBODY_T[K]']
        result = rym.fixHeadersList(headers)
        assert len(result) == len(headers)
        for h in result:
            assert '+' not in h
            assert '[' not in h
            assert ']' not in h

    def test_plain_headers_unchanged(self):
        headers = ['FREQ', 'TOT_TRANS', 'PTH_THRML']
        assert rym.fixHeadersList(headers) == headers

    def test_empty_list(self):
        assert rym.fixHeadersList([]) == []


# ── loadtape7 — IEMSCT=0 (two-line header, transmission only) ─────────────────

@pytest.mark.skipif(not (_DATA / 'tape7-01').exists(), reason='tape7-01 data file not found')
class TestLoadTape7_IEMSCT0:
    _file = str(_DATA / 'tape7-01')
    _cols = ['FREQ_CM-1', 'COMBIN_TRANS', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'AER+CLD_abTRNS']

    def test_returns_array(self):
        result = rym.loadtape7(self._file, self._cols)
        assert isinstance(result, np.ndarray)

    def test_correct_column_count(self):
        result = rym.loadtape7(self._file, self._cols)
        assert result.shape[1] == len(self._cols)

    def test_frequency_column_positive(self):
        result = rym.loadtape7(self._file, self._cols)
        assert np.all(result[:, 0] > 0)

    def test_transmittance_in_unit_interval(self):
        """Transmittance columns must lie in [0, 1]."""
        result = rym.loadtape7(self._file, self._cols)
        assert np.all(result[:, 1] >= 0) and np.all(result[:, 1] <= 1.0 + 1e-6)

    def test_frequency_monotone(self):
        result = rym.loadtape7(self._file, self._cols)
        diff = np.diff(result[:, 0])
        # wavenumber vector must be either strictly increasing or decreasing
        assert np.all(diff > 0) or np.all(diff < 0)

    def test_single_column_request(self):
        result = rym.loadtape7(self._file, ['FREQ_CM-1', 'COMBIN_TRANS'])
        assert result.shape[1] == 2


@pytest.mark.skipif(not (_DATA / 'tape7-01b').exists(), reason='tape7-01b data file not found')
class TestLoadTape7_IEMSCT0b:
    def test_loads_successfully(self):
        cols = ['FREQ_CM-1', 'COMBIN_TRANS', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'AER+CLD_abTRNS']
        result = rym.loadtape7(str(_DATA / 'tape7-01b'), cols)
        assert result is not None
        assert result.shape[1] == len(cols)


# ── loadtape7 — IEMSCT=1 (thermal, some empty columns) ───────────────────────

@pytest.mark.skipif(not (_DATA / 'tape7-02').exists(), reason='tape7-02 data file not found')
class TestLoadTape7_IEMSCT1:
    _file = str(_DATA / 'tape7-02')
    _cols = ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS',
             'GRND_RFLT', 'TOTAL_RAD', 'DEPTH', 'DIR_EM', 'BBODY_TbKb']

    def test_returns_array(self):
        cols = ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'TOTAL_RAD']
        result = rym.loadtape7(self._file, cols)
        assert isinstance(result, np.ndarray)

    def test_correct_column_count(self):
        cols = ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'TOTAL_RAD']
        result = rym.loadtape7(self._file, cols)
        assert result.shape[1] == len(cols)

    def test_frequency_positive(self):
        cols = ['FREQ', 'TOT_TRANS']
        result = rym.loadtape7(self._file, cols)
        assert np.all(result[:, 0] > 0)


@pytest.mark.skipif(not (_DATA / 'tape7-02b').exists(), reason='tape7-02b data file not found')
class TestLoadTape7_IEMSCT1b:
    def test_loads_successfully(self):
        cols = ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'GRND_RFLT', 'TOTAL_RAD', 'DEPTH', 'DIR_EM', 'BBODY_TbKb']
        result = rym.loadtape7(str(_DATA / 'tape7-02b'), cols)
        assert result is not None


# ── loadtape7 — IEMSCT=2 (full radiance) ─────────────────────────────────────

@pytest.mark.skipif(not (_DATA / 'tape7-03').exists(), reason='tape7-03 data file not found')
class TestLoadTape7_IEMSCT2:
    _file = str(_DATA / 'tape7-03')

    def test_returns_array(self):
        cols = ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'TOTAL_RAD']
        result = rym.loadtape7(self._file, cols)
        assert isinstance(result, np.ndarray)

    def test_correct_shape(self):
        cols = ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'TOTAL_RAD']
        result = rym.loadtape7(self._file, cols)
        assert result.ndim == 2
        assert result.shape[1] == len(cols)


# ── loadtape7 — IEMSCT=3 (solar transmittance) ───────────────────────────────

@pytest.mark.skipif(not (_DATA / 'tape7-04').exists(), reason='tape7-04 data file not found')
class TestLoadTape7_IEMSCT3:
    _file = str(_DATA / 'tape7-04')
    _cols = ['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH']

    def test_returns_array(self):
        result = rym.loadtape7(self._file, self._cols)
        assert isinstance(result, np.ndarray)

    def test_correct_column_count(self):
        result = rym.loadtape7(self._file, self._cols)
        assert result.shape[1] == len(self._cols)

    def test_transmittance_range(self):
        result = rym.loadtape7(self._file, self._cols)
        assert np.all(result[:, 1] >= 0) and np.all(result[:, 1] <= 1.0 + 1e-6)


# ── loadtape7 — error handling ───────────────────────────────────────────────

class TestLoadTape7Errors:
    def test_too_few_lines_returns_none(self, tmp_path):
        """Files with fewer than 10 lines must return None without crashing."""
        f = tmp_path / 'short.tape7'
        f.write_text('T F 6    1    0\n' * 3)
        result = rym.loadtape7(str(f), ['FREQ'])
        assert result is None


# ── variationTape5File ────────────────────────────────────────────────────────

class TestVariationTape5File:
    def test_creates_output_file(self, tmp_path):
        """variationTape5File must create a new sub-directory with the tape5 file."""
        scenario_dir = tmp_path / 'scenario1'
        scenario_dir.mkdir()
        tape5_content = 'AAAA 1234.5678 BBBB\n'
        (scenario_dir / 'tape5').write_text(tape5_content)

        result = rym.variationTape5File(
            str(scenario_dir), 500, 'tape5',
            varFunc=None, fmtstr='8.4f', srcstr='1234.5678',
        )
        assert result == '500'
        out_file = scenario_dir / '500' / 'tape5'
        assert out_file.exists()

    def test_replaces_search_string(self, tmp_path):
        scenario_dir = tmp_path / 'scenario2'
        scenario_dir.mkdir()
        (scenario_dir / 'tape5').write_text('RANGE=   5.0000KM\n')

        rym.variationTape5File(
            str(scenario_dir), 10.0, 'tape5',
            varFunc=None, fmtstr='8.4f', srcstr='   5.0000',
        )
        out = (scenario_dir / '10.0' / 'tape5').read_text()
        assert ' 10.0000' in out  # f'{10.0:8.4f}' → ' 10.0000' (1 leading space, 8 chars wide)

    def test_applies_varfunc(self, tmp_path):
        """variationTape5File must apply varFunc before formatting."""
        scenario_dir = tmp_path / 'scenario3'
        scenario_dir.mkdir()
        (scenario_dir / 'tape5').write_text('ALT=1000\n')

        rym.variationTape5File(
            str(scenario_dir), 1000, 'tape5',
            varFunc=lambda x: x / 1000.0,
            fmtstr='6.3f', srcstr='1000',
        )
        out = (scenario_dir / '1000' / 'tape5').read_text()
        assert '1.000' in out

    def test_mismatched_list_lengths_return_none(self, tmp_path):
        """Mismatched list lengths must return None."""
        scenario_dir = tmp_path / 'scenario4'
        scenario_dir.mkdir()
        (scenario_dir / 'tape5').write_text('X\n')

        result = rym.variationTape5File(
            str(scenario_dir), [1, 2], 'tape5',
            varFunc=[None],        # wrong length
            fmtstr=['8.4f', '8.4f'], srcstr=['A', 'B'],
        )
        assert result is None

    def test_multi_variation_combined_dirname(self, tmp_path):
        """Multiple variations must produce a hyphen-joined directory name."""
        scenario_dir = tmp_path / 'scenario5'
        scenario_dir.mkdir()
        (scenario_dir / 'tape5').write_text('A=1.0000 B=2.0000\n')

        result = rym.variationTape5File(
            str(scenario_dir), [1.0, 2.0], 'tape5',
            varFunc=[None, None], fmtstr=['6.4f', '6.4f'], srcstr=['1.0000', '2.0000'],
        )
        assert result == '1.0-2.0'
        assert (scenario_dir / '1.0-2.0' / 'tape5').exists()
