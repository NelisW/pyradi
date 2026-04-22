"""Regression tests for rylookup.py — RadLookup radiometric lookup class."""

import os
import sys

import numpy as np
import pytest

# Add the pyradi package directory so bare `import rylookup` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Add the *parent* of pyradi/ so that `import pyradi.ryplot` (and the other
# absolute pyradi sub-module imports inside rylookup.py) resolve to the real
# installed package files rather than failing with ModuleNotFoundError.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# When the full test suite runs, test_ryhdf5.py (collected first) installs a
# bare types.ModuleType stub as sys.modules['pyradi'].  That stub has no
# __path__, so `import pyradi.ryfiles` inside rylookup.py fails with
# "pyradi is not a package".  Detect and evict any non-package stub so Python
# falls back to the real installed package on sys.path.
if 'pyradi' in sys.modules and not hasattr(sys.modules['pyradi'], '__path__'):
    for _k in [k for k in list(sys.modules) if k == 'pyradi' or k.startswith('pyradi.')]:
        del sys.modules[_k]

import rylookup as rl


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _nu():
    """Wavenumber vector 700–1665 cm⁻¹ in 5 cm⁻¹ steps (LWIR band)."""
    return np.linspace(700, 1665, int(1 + (1665 - 700) / 5))


def _lut_simple():
    """Minimal RadLookup with unity spectrals and no calibration data."""
    return rl.RadLookup('test', _nu(), 300.0, 800.0, 10.0)


def _cal_data():
    """Two-temperature calibration dataset (Tlo=17.1°C, Thi=34.4°C) matching __main__."""
    arrLo = np.asarray([
        [50., 100., 150., 200., 250., 300., 350., 400., 450.],
        [4571., 5132., 5906., 6887., 8034., 9338., 10834., 12386., 14042.],
    ]).T
    arrHi = np.asarray([
        [50., 100., 150., 200., 250., 300., 350., 400., 450.],
        [5477, 6050, 6817, 7789, 8922, 10262, 11694, 13299, 14921],
    ]).T
    arrLo[:, 0] += 273.16
    arrHi[:, 0] += 273.16
    Tlo, Thi = 17.1, 34.4
    return (
        {Tlo: arrLo, Thi: arrHi},
        {Tlo: 10.0, Thi: 10.0},
        {Tlo: 3625.0, Thi: 4210.0},
        Tlo, Thi,
    )


def _lut_with_cal():
    """RadLookup with unity spectrals AND calibration data."""
    nu = _nu()
    dicCaldata, dicPower, dicFloor, Tlo, Thi = _cal_data()
    return rl.RadLookup(
        'test_cal', nu, 300.0, 800.0, 10.0,
        sigMin=0.0, sigMax=2.0 ** 14, sigInc=2.0 ** 6,
        dicCaldata=dicCaldata, dicPower=dicPower, dicFloor=dicFloor,
    )


# ── Instantiation and table calculation ──────────────────────────────────────

class TestInstantiation:
    def test_spectrals_loaded(self):
        lut = _lut_simple()
        assert lut.spectralsLoaded is True

    def test_hires_tables_calculated(self):
        lut = _lut_simple()
        assert lut.hiresTablesCalculated is True

    def test_cal_tables_not_calculated_without_caldata(self):
        lut = _lut_simple()
        assert lut.calTablesCalculated is False

    def test_table_temp_rad_shape(self):
        """TableTempRad must have three columns: temperature, L_with_filter, L_no_filter."""
        lut = _lut_simple()
        n_rows = 1 + int((800.0 - 300.0) / 10.0)
        assert lut.TableTempRad.shape == (n_rows, 3)

    def test_temperature_column_range(self):
        lut = _lut_simple()
        assert lut.TableTempRad[0, 0] == pytest.approx(300.0)
        assert lut.TableTempRad[-1, 0] == pytest.approx(800.0)

    def test_radiance_columns_positive(self):
        lut = _lut_simple()
        assert np.all(lut.TableTempRad[:, 1] > 0)
        assert np.all(lut.TableTempRad[:, 2] > 0)

    def test_radiance_monotone_increasing(self):
        """Planck radiance integrated over a fixed spectral band increases with temperature."""
        lut = _lut_simple()
        rad = lut.TableTempRad[:, 1]
        assert np.all(np.diff(rad) > 0)

    def test_name_stored(self):
        lut = rl.RadLookup('my_lut', _nu(), 300.0, 500.0, 50.0)
        assert lut.specName == 'my_lut'

    def test_unity_spectrals_with_and_without_filter_equal(self):
        """With all spectral inputs None (unity), with-filter and no-filter radiance must match."""
        lut = _lut_simple()
        assert np.allclose(lut.TableTempRad[:, 1], lut.TableTempRad[:, 2])


# ── Info ─────────────────────────────────────────────────────────────────────

class TestInfo:
    def test_returns_string(self):
        assert isinstance(_lut_simple().Info(), str)

    def test_contains_name(self):
        lut = rl.RadLookup('myname', _nu(), 300.0, 500.0, 50.0)
        assert 'myname' in lut.Info()

    def test_contains_nu_range(self):
        lut = _lut_simple()
        info = lut.Info()
        assert '700' in info


# ── LookupTempRad / LookupRadTemp ─────────────────────────────────────────────

class TestLookupTempRad:
    def test_scalar_input(self):
        lut = _lut_simple()
        rad = lut.LookupTempRad(500.0)
        assert rad > 0

    def test_array_input(self):
        lut = _lut_simple()
        temps = [300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
        rad = lut.LookupTempRad(temps)
        assert len(rad) == len(temps)

    def test_monotone_increasing(self):
        lut = _lut_simple()
        temps = np.linspace(310.0, 790.0, 20)
        rad = lut.LookupTempRad(temps)
        assert np.all(np.diff(rad) > 0)

    def test_with_filter_false(self):
        lut = _lut_simple()
        rad_wf = lut.LookupTempRad(500.0, withFilter=True)
        rad_nf = lut.LookupTempRad(500.0, withFilter=False)
        # With unity spectrals both must be equal
        assert rad_wf == pytest.approx(rad_nf, rel=1e-9)

    def test_returns_none_when_tables_missing(self):
        lut = _lut_simple()
        lut.hiresTablesCalculated = False
        assert lut.LookupTempRad(500.0) is None


class TestLookupRadTemp:
    def test_scalar_input(self):
        lut = _lut_simple()
        rad = lut.LookupTempRad(500.0)
        t = lut.LookupRadTemp(rad)
        assert t == pytest.approx(500.0, abs=1.0)  # 1 K tolerance (table step = 10 K)

    def test_round_trip(self):
        """LookupRadTemp(LookupTempRad(T)) ≈ T for interior table temperatures."""
        lut = _lut_simple()
        temps_in = [300.0, 307.5, 505.0, 792.5, 800.0]
        rad = lut.LookupTempRad(temps_in)
        temps_out = lut.LookupRadTemp(rad)
        assert np.allclose(temps_in, temps_out, atol=5.0)  # within half a table step

    def test_with_filter_false_round_trip(self):
        lut = _lut_simple()
        temps_in = [400.0, 600.0]
        rad = lut.LookupTempRad(temps_in, withFilter=False)
        temps_out = lut.LookupRadTemp(rad, withFilter=False)
        assert np.allclose(temps_in, temps_out, atol=5.0)

    def test_returns_none_when_tables_missing(self):
        lut = _lut_simple()
        lut.hiresTablesCalculated = False
        assert lut.LookupRadTemp(1.0) is None


# ── Calibration-data dependent methods ───────────────────────────────────────

class TestWithCalibration:
    @pytest.fixture(scope='class')
    def lut(self):
        return _lut_with_cal()

    def test_cal_tables_calculated(self, lut):
        assert lut.calTablesCalculated is True

    def test_lookup_dl_rad_array(self, lut):
        _, _, _, Tlo, _ = _cal_data()
        dl = np.array([5000.0, 8000.0, 12000.0])
        rad = lut.LookupDLRad(dl, Tlo)
        assert rad is not None
        assert rad.shape == (3,)

    def test_lookup_dl_rad_positive(self, lut):
        _, _, _, Tlo, _ = _cal_data()
        dl = np.linspace(5000.0, 14000.0, 10)
        rad = lut.LookupDLRad(dl, Tlo)
        assert np.all(rad > 0)

    def test_lookup_dl_temp_returns_temperatures(self, lut):
        _, _, _, Tlo, _ = _cal_data()
        dl = np.array([5000.0, 9000.0, 13000.0])
        temps = lut.LookupDLTemp(dl, Tlo)
        assert temps is not None
        assert np.all(temps > 0)

    def test_lookup_dl_rad_monotone(self, lut):
        """Higher DL must correspond to higher radiance (positive linear calibration)."""
        _, _, _, Tlo, _ = _cal_data()
        dl = np.linspace(5000.0, 14000.0, 20)
        rad = lut.LookupDLRad(dl, Tlo)
        assert np.all(np.diff(rad) > 0)

    def test_lookup_dl_rad_tint_interpolation(self, lut):
        """Radiance at Tint between Tlo and Thi must lie between the two extremes."""
        _, _, _, Tlo, Thi = _cal_data()
        dl = np.array([8000.0])
        rad_lo = lut.LookupDLRad(dl, Tlo)
        rad_hi = lut.LookupDLRad(dl, Thi)
        Tmid = (Tlo + Thi) / 2.0
        rad_mid = lut.LookupDLRad(dl, Tmid)
        assert rad_lo[0] <= rad_mid[0] <= rad_hi[0] or rad_hi[0] <= rad_mid[0] <= rad_lo[0]

    def test_lookup_dl_helper_single_cal_temp(self, lut):
        """LookupDLRadHelper with paraK=0 must return the same as full lookup at Tlo."""
        _, _, _, Tlo, _ = _cal_data()
        dl = np.array([7000.0, 10000.0])
        rad_full = lut.LookupDLRad(dl, Tlo)
        rad_helper = lut.LookupDLRadHelper(dl, 0)
        assert np.allclose(rad_full, rad_helper, rtol=1e-9)


# ── Lookup returns None without calibration ───────────────────────────────────

class TestNullReturnWithoutCal:
    def test_lookup_dl_rad_none(self):
        lut = _lut_simple()
        assert lut.LookupDLRad(5000.0, 20.0) is None

    def test_lookup_dl_temp_none(self):
        lut = _lut_simple()
        assert lut.LookupDLTemp(5000.0, 20.0) is None


# ── ResetAllContainers ────────────────────────────────────────────────────────

class TestResetAllContainers:
    def test_flags_cleared(self):
        lut = _lut_simple()
        assert lut.hiresTablesCalculated is True
        lut.ResetAllContainers()
        assert lut.hiresTablesCalculated is False
        assert lut.calTablesCalculated is False

    def test_tables_set_to_none(self):
        lut = _lut_simple()
        lut.ResetAllContainers()
        assert lut.TableTempRad is None
