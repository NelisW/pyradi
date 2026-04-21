"""Regression tests for rypflux.py — photon-flux estimates for low-light conditions."""

import os
import sys
import types

import numpy as np
import pytest

# ── Stub pyradi package before importing rypflux ─────────────────────────────
# rypflux does top-level absolute imports:
#   import pyradi.ryutils as ryutils
#   import pyradi.ryplanck as ryplanck
# We intercept both with lightweight stubs so the test suite does not require
# the full pyradi package.

if 'pyradi' not in sys.modules:
    _pyradi_stub = types.ModuleType('pyradi')
    sys.modules['pyradi'] = _pyradi_stub

# ── ryutils stub ─────────────────────────────────────────────────────────────
_ryutils_stub = types.ModuleType('pyradi.ryutils')

def _luminousEfficiency(vlamtype, wavelen, eqnapprox=False):
    """Return an all-ones efficiency curve over the supplied wavelength vector."""
    wl = np.asarray(wavelen, dtype=float)
    return np.ones_like(wl), wl

_ryutils_stub.luminousEfficiency = _luminousEfficiency
sys.modules['pyradi.ryutils'] = _ryutils_stub
sys.modules['pyradi'].ryutils = _ryutils_stub  # type: ignore[attr-defined]

# ── ryplanck stub ─────────────────────────────────────────────────────────────
_ryplanck_stub = types.ModuleType('pyradi.ryplanck')

def _planckel(wl, temps):
    """Return ones shaped (len(wl), len(temps)) — pure geometry, no physics."""
    wl_arr   = np.asarray(wl).ravel()
    temp_arr = np.asarray(temps).ravel()
    return np.ones((len(wl_arr), len(temp_arr)))

def _planck(spec, temps, stype):
    """Return ones shaped (len(spec.ravel()), len(temps)) — pure geometry."""
    spec_arr = np.asarray(spec).ravel()
    temp_arr = np.asarray(temps).ravel()
    return np.ones((len(spec_arr), len(temp_arr)))

_ryplanck_stub.planckel = _planckel
_ryplanck_stub.planck   = _planck
sys.modules['pyradi.ryplanck'] = _ryplanck_stub
sys.modules['pyradi'].ryplanck = _ryplanck_stub  # type: ignore[attr-defined]

# ── import module under test ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import rypflux as rpf
import pandas as pd


# ── PFlux.__init__ ───────────────────────────────────────────────────────────

class TestPFluxInit:
    def test_creates_instance(self):
        pf = rpf.PFlux()
        assert isinstance(pf, rpf.PFlux)

    def test_lllux_is_plain_dict(self):
        pf = rpf.PFlux()
        assert type(pf.lllux) is dict

    def test_lllux_has_ten_conditions(self):
        pf = rpf.PFlux()
        assert len(pf.lllux) == 10

    def test_lllux_expected_keys(self):
        pf = rpf.PFlux()
        expected = {
            'Sun light', 'Full sky light', 'Overcast day', 'Very dark day',
            'Twilight', 'Deep twilight', 'Full moon', 'Quarter moon',
            'Star light', 'Overcast night',
        }
        assert set(pf.lllux.keys()) == expected

    def test_lllux_each_entry_has_three_values(self):
        pf = rpf.PFlux()
        for key, val in pf.lllux.items():
            assert len(val) == 3, f'{key!r} should have 3 values (irradiance, CT, photop_frac)'

    def test_lllux_irradiance_positive(self):
        """All irradiance values must be strictly positive."""
        pf = rpf.PFlux()
        for key, (irr, ct, fp) in pf.lllux.items():
            assert irr > 0, f'{key!r}: irradiance must be positive'

    def test_lllux_colour_temp_positive(self):
        pf = rpf.PFlux()
        for key, (irr, ct, fp) in pf.lllux.items():
            assert ct > 0, f'{key!r}: colour temperature must be positive'

    def test_lllux_fracphotop_in_range(self):
        """Photopic fraction must be in [0, 1]."""
        pf = rpf.PFlux()
        for key, (irr, ct, fp) in pf.lllux.items():
            assert 0.0 <= fp <= 1.0, f'{key!r}: photopic fraction {fp} out of [0,1]'

    def test_specranges_has_five_bands(self):
        pf = rpf.PFlux()
        assert set(pf.specranges.keys()) == {'VIS', 'NIR', 'SWIR', 'MWIR', 'LWIR'}

    def test_specranges_each_band_has_three_entries(self):
        pf = rpf.PFlux()
        for band, entry in pf.specranges.items():
            assert len(entry) == 3, f'{band!r} entry should be [spec_vec, resp_vec, type_str]'

    def test_specranges_type_string_valid(self):
        pf = rpf.PFlux()
        for band, entry in pf.specranges.items():
            assert entry[2] in ('wl', 'wn'), f'{band!r}: type string must be "wl" or "wn"'

    def test_llluxcols_correct(self):
        pf = rpf.PFlux()
        assert pf.llluxCols == ['Irradiance-lm/m2', 'ColourTemp', 'FracPhotop']


# ── PFlux.lllPhotonrates ──────────────────────────────────────────────────────

class TestLllPhotonrates:
    """All tests in this class use the default (built-in) specranges."""

    @pytest.fixture(autouse=True)
    def pf(self):
        self._pf = rpf.PFlux()
        self._result = self._pf.lllPhotonrates()

    def test_returns_dataframe(self):
        assert isinstance(self._result, pd.DataFrame)

    def test_has_ten_rows(self):
        assert len(self._result) == 10

    def test_has_irradiance_column(self):
        assert 'Irradiance-lm/m2' in self._result.columns

    def test_has_colour_temp_column(self):
        assert 'ColourTemp' in self._result.columns

    def test_has_fracphotop_column(self):
        assert 'FracPhotop' in self._result.columns

    def test_has_k_column(self):
        assert 'k' in self._result.columns

    def test_has_radiance_column_for_each_band(self):
        for band in ('VIS', 'NIR', 'SWIR', 'MWIR', 'LWIR'):
            col = f'Radiance-q/(s.m2.sr)-{band}'
            assert col in self._result.columns, f'missing column {col!r}'

    def test_k_values_positive(self):
        assert (self._result['k'] > 0).all()

    def test_radiance_values_positive(self):
        for band in ('VIS', 'NIR', 'SWIR', 'MWIR', 'LWIR'):
            col = f'Radiance-q/(s.m2.sr)-{band}'
            assert (self._result[col] > 0).all(), f'{col!r}: all values should be positive'

    def test_rows_sorted_ascending_by_irradiance(self):
        irr = self._result['Irradiance-lm/m2'].values
        assert all(irr[i] <= irr[i + 1] for i in range(len(irr) - 1))

    def test_result_stored_as_attribute(self):
        assert hasattr(self._pf, 'dfPhotRates')
        assert isinstance(self._pf.dfPhotRates, pd.DataFrame)

    def test_result_attribute_matches_return_value(self):
        pd.testing.assert_frame_equal(self._result, self._pf.dfPhotRates)

    def test_columns_sorted_alphabetically(self):
        cols = list(self._result.columns)
        assert cols == sorted(cols), 'columns should be in alphabetical order'

    def test_sunlight_higher_k_than_overcast_night(self):
        """Sun light must produce a much larger scaling factor than overcast night."""
        k_sun     = self._result.loc['Sun light', 'k']
        k_dark    = self._result.loc['Overcast night', 'k']
        assert k_sun > k_dark


class TestLllPhotonratesCustomSpecranges:
    def test_single_custom_band_appears_in_output(self):
        pf = rpf.PFlux()
        numpts = 20
        custom = {
            'MYBAND': [
                np.linspace(1.5, 2.5, numpts).reshape(-1, 1),
                np.ones((numpts, 1)),
                'wl',
            ],
        }
        result = pf.lllPhotonrates(specranges=custom)
        assert 'Radiance-q/(s.m2.sr)-MYBAND' in result.columns

    def test_default_bands_absent_when_custom_supplied(self):
        """When a full custom specranges dict is provided the default bands are replaced."""
        pf = rpf.PFlux()
        numpts = 20
        custom = {
            'X': [np.linspace(4.0, 5.0, numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
        }
        result = pf.lllPhotonrates(specranges=custom)
        for band in ('VIS', 'NIR', 'SWIR', 'MWIR', 'LWIR'):
            assert f'Radiance-q/(s.m2.sr)-{band}' not in result.columns

    def test_wavenumber_band_accepted(self):
        """specranges entry with type 'wn' (wavenumber) should not raise."""
        pf = rpf.PFlux()
        numpts = 20
        custom = {
            'MWIR_wn': [
                np.linspace(2000, 3000, numpts).reshape(-1, 1),
                np.ones((numpts, 1)),
                'wn',
            ],
        }
        result = pf.lllPhotonrates(specranges=custom)
        assert 'Radiance-q/(s.m2.sr)-MWIR_wn' in result.columns

    def test_custom_specranges_stored_on_instance(self):
        """After a custom call, self.specranges should reflect the custom dict."""
        pf = rpf.PFlux()
        numpts = 20
        custom = {
            'Z': [np.linspace(0.5, 1.0, numpts).reshape(-1, 1), np.ones((numpts, 1)), 'wl'],
        }
        pf.lllPhotonrates(specranges=custom)
        assert 'Z' in pf.specranges
