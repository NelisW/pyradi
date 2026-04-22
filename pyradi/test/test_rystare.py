"""Regression tests for rystare.py — CCD/CMOS staring array sensor model.

Run with:
    cd K:/WorkN/pyradi/pyradi/pyradi
    pytest test/test_rystare.py -v
"""

import os
import sys

import numpy as np
import pytest
import scipy.constants as const

# Add pyradi/ to sys.path so bare `import rystare` works.
# Add the parent of pyradi/ so `import pyradi.*` package imports work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import rystare as rs


# ---------------------------------------------------------------------------
# define_metrics
# ---------------------------------------------------------------------------
class TestDefineMetrics:
    def test_returns_seven_values(self):
        result = rs.define_metrics()
        assert len(result) == 7

    def test_m_is_one(self):
        m, *_ = rs.define_metrics()
        assert m == 1

    def test_cm_value(self):
        m, cm, mm, mum, nm, rad, mrad = rs.define_metrics()
        assert cm == pytest.approx(0.01, rel=1e-9)

    def test_mm_value(self):
        m, cm, mm, mum, nm, rad, mrad = rs.define_metrics()
        assert mm == pytest.approx(0.001, rel=1e-9)

    def test_mum_value(self):
        m, cm, mm, mum, nm, rad, mrad = rs.define_metrics()
        assert mum == pytest.approx(1e-6, rel=1e-9)

    def test_nm_value(self):
        m, cm, mm, mum, nm, rad, mrad = rs.define_metrics()
        assert nm == pytest.approx(1e-9, rel=1e-9)

    def test_rad_is_one(self):
        m, cm, mm, mum, nm, rad, mrad = rs.define_metrics()
        assert rad == 1

    def test_mrad_value(self):
        m, cm, mm, mum, nm, rad, mrad = rs.define_metrics()
        assert mrad == pytest.approx(0.001, rel=1e-9)


# ---------------------------------------------------------------------------
# limitzero
# ---------------------------------------------------------------------------
class TestLimitzero:
    def test_above_threshold_unchanged(self):
        """Values at or above thr pass through unmodified."""
        a = np.array([0.7, 1.0, 2.0])
        result = rs.limitzero(a, thr=0.6)
        np.testing.assert_allclose(result, a)

    def test_below_threshold_remapped_via_exp(self):
        """Values below thr are mapped through thr * exp((a - thr) / thr)."""
        thr = 0.6
        a = np.array([0.0])
        expected = thr * np.exp((0.0 - thr) / thr)
        result = rs.limitzero(a, thr=thr)
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_value_at_threshold_maps_to_threshold(self):
        thr = 0.6
        a = np.array([thr])
        result = rs.limitzero(a, thr=thr)
        assert result[0] == pytest.approx(thr, rel=1e-9)

    def test_output_shape_preserved_2d(self):
        a = np.ones((3, 4)) * 0.5
        result = rs.limitzero(a)
        assert result.shape == (3, 4)

    def test_all_outputs_positive(self):
        """limitzero must guarantee no negative outputs."""
        a = np.linspace(-10, 2, 200)
        result = rs.limitzero(a, thr=0.6)
        assert np.all(result > 0)

    def test_default_threshold_is_0_6(self):
        """Calling without thr should use the default threshold of 0.6."""
        a = np.array([0.3])
        with_default = rs.limitzero(a)
        explicit = rs.limitzero(a, thr=0.6)
        assert with_default[0] == pytest.approx(explicit[0], rel=1e-12)


# ---------------------------------------------------------------------------
# kTCnoiseCsn
# ---------------------------------------------------------------------------
class TestKTCnoiseCsn:
    def test_known_formula(self):
        """kTCnoiseCsn = sqrt(k * T * Csn) / e"""
        T = 300.0
        Csn = 50e-15
        expected = np.sqrt(const.k * T * Csn) / const.e
        result = rs.kTCnoiseCsn(T, Csn)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_scales_with_temperature(self):
        """Noise increases with temperature."""
        Csn = 50e-15
        r_low = rs.kTCnoiseCsn(200.0, Csn)
        r_high = rs.kTCnoiseCsn(400.0, Csn)
        assert r_high > r_low

    def test_scales_with_capacitance(self):
        """Noise increases with sense-node capacitance."""
        T = 300.0
        r_small = rs.kTCnoiseCsn(T, 10e-15)
        r_large = rs.kTCnoiseCsn(T, 100e-15)
        assert r_large > r_small

    def test_positive_result(self):
        assert rs.kTCnoiseCsn(300.0, 50e-15) > 0


# ---------------------------------------------------------------------------
# kTCnoiseGv
# ---------------------------------------------------------------------------
class TestKTCnoiseGv:
    def test_known_formula(self):
        """kTCnoiseGv = sqrt(k * T / (e * gv))"""
        T = 300.0
        gv = 5e-6
        expected = np.sqrt(const.k * T / (const.e * gv))
        result = rs.kTCnoiseGv(T, gv)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_scales_with_temperature(self):
        gv = 5e-6
        r_low = rs.kTCnoiseGv(200.0, gv)
        r_high = rs.kTCnoiseGv(400.0, gv)
        assert r_high > r_low

    def test_higher_gain_gives_lower_noise(self):
        """Larger sense-node gain reduces kTC noise."""
        T = 300.0
        r_small_gv = rs.kTCnoiseGv(T, 1e-6)
        r_large_gv = rs.kTCnoiseGv(T, 10e-6)
        assert r_small_gv > r_large_gv

    def test_positive_result(self):
        assert rs.kTCnoiseGv(300.0, 5e-6) > 0


# ---------------------------------------------------------------------------
# darkcurrentnoise
# ---------------------------------------------------------------------------
class TestDarkcurrentnoise:
    def test_known_formula(self):
        """darkcurrentnoise = sqrt(inttime * 2.55e15 * detarea * DFM * T^1.5 * exp(-Egap/(2*k*T)))"""
        inttime = 0.035
        detarea = 25e-12
        temptr = 300.0
        Egap = 1.12
        DFM = 0.5e-5
        keV = const.physical_constants['Boltzmann constant in eV/K'][0]
        ndarkcur = inttime * 2.55e15 * detarea * DFM * (temptr ** 1.5) * np.exp(-Egap / (2 * keV * temptr))
        expected = np.sqrt(ndarkcur)
        result = rs.darkcurrentnoise(inttime, detarea, temptr, Egap, DFM)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_default_DFM(self):
        """Default DFM=0.5e-5 matches explicit call."""
        args = (0.035, 25e-12, 300.0, 1.12)
        assert rs.darkcurrentnoise(*args) == pytest.approx(rs.darkcurrentnoise(*args, DFM=0.5e-5), rel=1e-12)

    def test_positive_result(self):
        assert rs.darkcurrentnoise(0.035, 25e-12, 300.0, 1.12) > 0

    def test_scales_with_integration_time(self):
        """Longer integration accumulates more dark current noise."""
        n1 = rs.darkcurrentnoise(0.01, 25e-12, 300.0, 1.12)
        n2 = rs.darkcurrentnoise(0.1, 25e-12, 300.0, 1.12)
        assert n2 > n1


# ---------------------------------------------------------------------------
# shotnoise
# ---------------------------------------------------------------------------
class TestShotnoise:
    def test_output_shape_preserved(self):
        arr = np.full((5, 5), 1000.0)
        result = rs.shotnoise(arr)
        assert result.shape == arr.shape

    def test_output_is_finite(self):
        arr = np.ones((4, 4)) * 500.0
        result = rs.shotnoise(arr)
        assert np.all(np.isfinite(result))

    def test_zero_input_gives_zero_output(self):
        """Poisson(0) is always 0."""
        arr = np.zeros((3, 3))
        result = rs.shotnoise(arr)
        np.testing.assert_array_equal(result, 0)

    def test_output_nonnegative(self):
        """Shot noise counts cannot be negative."""
        arr = np.abs(np.random.randn(10, 10)) * 100
        result = rs.shotnoise(arr)
        assert np.all(result >= 0)

    def test_large_signal_mean_close_to_input(self):
        """For a large uniform signal, Poisson mean ≈ lambda."""
        lam = 10000.0
        arr = np.full((200, 200), lam)
        result = rs.shotnoise(arr)
        assert np.mean(result) == pytest.approx(lam, rel=0.02)


# ---------------------------------------------------------------------------
# FPN_models
# ---------------------------------------------------------------------------
class TestFPNModels:
    def test_janesick_gaussian_pixel_shape(self):
        out = rs.FPN_models(10, 12, 'pixel', 'Janesick-Gaussian', 0.01)
        assert out.shape == (10, 12)

    def test_janesick_gaussian_column_shape(self):
        out = rs.FPN_models(10, 12, 'column', 'Janesick-Gaussian', 0.01)
        assert out.shape == (10, 12)

    def test_janesick_gaussian_row_shape(self):
        out = rs.FPN_models(10, 12, 'row', 'Janesick-Gaussian', 0.01)
        assert out.shape == (10, 12)

    def test_lognormal_pixel_shape(self):
        out = rs.FPN_models(8, 8, 'pixel', 'LogNormal', 0.3)
        assert out.shape == (8, 8)

    def test_lognormal_column_shape(self):
        out = rs.FPN_models(8, 8, 'column', 'LogNormal', 0.3)
        assert out.shape == (8, 8)

    def test_wald_pixel_shape(self):
        out = rs.FPN_models(8, 8, 'pixel', 'Wald', 0.5)
        assert out.shape == (8, 8)

    def test_wald_column_shape(self):
        out = rs.FPN_models(8, 8, 'column', 'Wald', 0.5)
        assert out.shape == (8, 8)

    def test_ar_elgamal_pixel_shape(self):
        out = rs.FPN_models(8, 8, 'pixel', 'AR-ElGamal', None, filter_params=[1., 0.5])
        assert out.shape == (8, 8)

    def test_ar_elgamal_column_shape(self):
        out = rs.FPN_models(8, 8, 'column', 'AR-ElGamal', None, filter_params=[1., 0.5])
        assert out.shape == (8, 8)

    def test_ar_elgamal_no_filter_returns_none(self, capsys):
        """AR-ElGamal without filter_params prints a message and returns None."""
        out = rs.FPN_models(8, 8, 'pixel', 'AR-ElGamal', None, filter_params=None)
        assert out is None

    def test_janesick_gaussian_no_spread_returns_none(self, capsys):
        """Janesick-Gaussian with spread=None prints a message and returns None."""
        out = rs.FPN_models(8, 8, 'pixel', 'Janesick-Gaussian', None)
        assert out is None

    def test_wald_no_spread_returns_none(self, capsys):
        out = rs.FPN_models(8, 8, 'pixel', 'Wald', None)
        assert out is None

    def test_lognormal_no_spread_returns_none(self, capsys):
        out = rs.FPN_models(8, 8, 'pixel', 'LogNormal', None)
        assert out is None

    def test_janesick_gaussian_spread_controls_stddev(self):
        """Larger spread → larger std dev of output (statistical, seeded)."""
        np.random.seed(42)
        out_small = rs.FPN_models(100, 100, 'pixel', 'Janesick-Gaussian', 0.01)
        np.random.seed(42)
        out_large = rs.FPN_models(100, 100, 'pixel', 'Janesick-Gaussian', 0.10)
        assert np.std(out_large) > np.std(out_small)


# ---------------------------------------------------------------------------
# nEcntLLightPhotL
# ---------------------------------------------------------------------------
class TestNEcntLLightPhotL:
    def test_known_formula(self):
        """n = pi * inttime * pfrac * detarea * L * cosTarg / (4*fno^2)
        where L = quantEff * tauOpt * tauFilt * tauAtmo * rhoTarg * photRadiance"""
        tauAtmo = 0.9
        tauFilt = 0.8
        tauOpt = 0.85
        quantEff = 0.7
        rhoTarg = 0.3
        cosTarg = 0.866
        inttime = 0.01
        pfrac = 1.0
        detarea = 25e-12
        fno = 3.2
        photRadiance = 1e14

        L = quantEff * tauOpt * tauFilt * tauAtmo * rhoTarg * photRadiance
        expected = np.pi * inttime * pfrac * detarea * L * cosTarg / (4 * fno ** 2)
        result = rs.nEcntLLightPhotL(
            tauAtmo, tauFilt, tauOpt, quantEff, rhoTarg,
            cosTarg, inttime, pfrac, detarea, fno, photRadiance
        )
        assert result == pytest.approx(expected, rel=1e-9)

    def test_scales_linearly_with_integration_time(self):
        common = dict(tauAtmo=0.9, tauFilt=0.8, tauOpt=0.85, quantEff=0.7,
                      rhoTarg=0.3, cosTarg=1.0, pfrac=1.0, detarea=25e-12,
                      fno=3.2, photRadiance=1e14)
        n1 = rs.nEcntLLightPhotL(inttime=0.01, **common)
        n2 = rs.nEcntLLightPhotL(inttime=0.02, **common)
        assert n2 == pytest.approx(2 * n1, rel=1e-9)

    def test_positive_result(self):
        assert rs.nEcntLLightPhotL(0.9, 0.8, 0.85, 0.7, 0.3, 1.0,
                                   0.01, 1.0, 25e-12, 3.2, 1e14) > 0


# ---------------------------------------------------------------------------
# nElecCntThermalScene
# ---------------------------------------------------------------------------
class TestNElecCntThermalScene:
    def _wl(self):
        # ryplanck.planck(type='ql') expects wavelength in µm, not SI metres.
        # The rystare docstring says "SI units" but the implementation passes wl
        # straight to planckql — a pre-existing units inconsistency.
        return np.linspace(3.0, 5.0, 50)  # µm, MWIR band

    def test_positive_result(self):
        wl = self._wl()
        n = rs.nElecCntThermalScene(wl, 300.0, 0.9, 0.8, 0.9, 0.85, 0.7,
                                    0.01, 1.0, 25e-12, 3.2)
        assert n > 0

    def test_scales_linearly_with_emissivity(self):
        """n ∝ emissivity (all other factors identical)."""
        wl = self._wl()
        n1 = rs.nElecCntThermalScene(wl, 300.0, 0.5, 0.8, 0.9, 0.85, 0.7,
                                     0.01, 1.0, 25e-12, 3.2)
        n2 = rs.nElecCntThermalScene(wl, 300.0, 1.0, 0.8, 0.9, 0.85, 0.7,
                                     0.01, 1.0, 25e-12, 3.2)
        assert n2 == pytest.approx(2 * n1, rel=1e-6)

    def test_higher_temperature_more_electrons(self):
        wl = self._wl()
        n_cool = rs.nElecCntThermalScene(wl, 300.0, 0.9, 0.8, 0.9, 0.85, 0.7,
                                         0.01, 1.0, 25e-12, 3.2)
        n_hot = rs.nElecCntThermalScene(wl, 500.0, 0.9, 0.8, 0.9, 0.85, 0.7,
                                        0.01, 1.0, 25e-12, 3.2)
        assert float(n_hot) > float(n_cool)  # float() handles array([x]) shape


# ---------------------------------------------------------------------------
# nEcntThermalOptics
# ---------------------------------------------------------------------------
class TestNEcntThermalOptics:
    def _wl(self):
        # Same units note as TestNElecCntThermalScene: µm required.
        return np.linspace(3.0, 5.0, 50)  # µm, MWIR band

    def test_positive_result(self):
        wl = self._wl()
        n = rs.nEcntThermalOptics(wl, 300.0, 0.9, 0.85, 0.7, 0.01, 1.0, 25e-12, 3.2)
        assert n > 0

    def test_higher_optics_temperature_more_electrons(self):
        wl = self._wl()
        n_cool = rs.nEcntThermalOptics(wl, 300.0, 0.9, 0.85, 0.7, 0.01, 1.0, 25e-12, 3.2)
        n_hot = rs.nEcntThermalOptics(wl, 500.0, 0.9, 0.85, 0.7, 0.01, 1.0, 25e-12, 3.2)
        assert float(n_hot) > float(n_cool)  # float() handles array([x]) shape


# ---------------------------------------------------------------------------
# nElecCntReflSun
# ---------------------------------------------------------------------------
class TestNElecCntReflSun:
    def _wl(self):
        # ryplanck.planck(type='ql') expects µm — see note in TestNElecCntThermalScene.
        return np.linspace(0.4, 2.5, 100)  # µm, VIS/SWIR band

    def test_positive_result(self):
        wl = self._wl()
        tauSun = np.ones_like(wl)
        n = rs.nElecCntReflSun(wl, tauSun)
        assert np.all(n > 0)

    def test_scales_linearly_with_integration_time(self):
        wl = self._wl()
        tauSun = np.ones_like(wl)
        n1 = rs.nElecCntReflSun(wl, tauSun, inttime=0.01)
        n2 = rs.nElecCntReflSun(wl, tauSun, inttime=0.02)
        assert n2 == pytest.approx(2 * n1, rel=1e-9)

    def test_default_call_succeeds(self):
        wl = self._wl()
        tauSun = np.ones_like(wl)
        result = rs.nElecCntReflSun(wl, tauSun)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# get_summary_stats — pre-existing bug assertion
# ---------------------------------------------------------------------------
class TestGetSummaryStats:
    def test_none_hdffilename_returns_empty_string(self):
        """Pre-existing bug: get_summary_stats creates a StringIO() buffer but routes
        all data through print() to stdout instead of writing to the buffer.
        output.getvalue() therefore always returns ''.  This test asserts the
        buggy behaviour so any accidental future fix is immediately caught."""
        result = rs.get_summary_stats(None)
        assert result == ''
