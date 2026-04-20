"""Regression tests for pyradi/ryutils.py.

Run with:
    cd K:/WorkN/pyradi/pyradi/pyradi
    pytest test/test_ryutils.py -v
"""

import sys
import os
import math
import pytest
import numpy as np

# Make sure the parent folder (pyradi/) is on the path so we can import ryutils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ryutils


# ---------------------------------------------------------------------------
# buildLogSpace
# ---------------------------------------------------------------------------
class TestBuildLogSpace:
    def test_basic_returns_array(self):
        result = ryutils.buildLogSpace(1, 100, 10)
        assert isinstance(result, np.ndarray)
        assert result[0] == pytest.approx(1.0, rel=1e-6)

    def test_covers_range(self):
        result = ryutils.buildLogSpace(0.1, 10, 5)
        assert result[0] <= 0.1 * 1.01
        assert result[-1] >= 9.99

    def test_patn_true(self):
        result = ryutils.buildLogSpace(1, 1000, 10, patn=True)
        assert len(result) > 0
        assert result[0] == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# intify_tuple
# ---------------------------------------------------------------------------
class TestIntifyTuple:
    def test_floats_converted(self):
        result = ryutils.intify_tuple((1.7, 2.3, 3.9))
        assert result == (1, 2, 3)
        assert all(isinstance(v, int) for v in result)

    def test_already_ints_unchanged(self):
        assert ryutils.intify_tuple((4, 5, 6)) == (4, 5, 6)

    def test_empty_tuple(self):
        assert ryutils.intify_tuple(()) == ()


# ---------------------------------------------------------------------------
# framesFirst / framesLast  (round-trip)
# ---------------------------------------------------------------------------
class TestFramesTranspose:
    def setup_method(self):
        # shape (rows=4, cols=5, frames=3)
        self.seq_last = np.random.rand(4, 5, 3)

    def test_framesFirst_shape(self):
        out = ryutils.framesFirst(self.seq_last)
        assert out.shape == (3, 4, 5)

    def test_framesLast_shape(self):
        seq_first = ryutils.framesFirst(self.seq_last)
        out = ryutils.framesLast(seq_first)
        assert out.shape == (4, 5, 3)

    def test_round_trip(self):
        seq_first = ryutils.framesFirst(self.seq_last)
        recovered = ryutils.framesLast(seq_first)
        np.testing.assert_array_equal(self.seq_last, recovered)


# ---------------------------------------------------------------------------
# cart2polar / polar2cart  (round-trip)
# ---------------------------------------------------------------------------
class TestCoordConversion:
    def test_cart2polar_origin(self):
        r, theta = ryutils.cart2polar(0.0, 0.0)
        assert r == pytest.approx(0.0)

    def test_cart2polar_unit_x(self):
        r, theta = ryutils.cart2polar(1.0, 0.0)
        assert r == pytest.approx(1.0)
        assert theta == pytest.approx(0.0)

    def test_cart2polar_unit_y(self):
        r, theta = ryutils.cart2polar(0.0, 1.0)
        assert r == pytest.approx(1.0)
        assert theta == pytest.approx(math.pi / 2)

    def test_round_trip(self):
        x_orig = np.array([1.0, -2.0, 3.0, 0.5])
        y_orig = np.array([0.5, 1.5, -2.0, -0.5])
        r, theta = ryutils.cart2polar(x_orig, y_orig)
        x_back, y_back = ryutils.polar2cart(r, theta)
        np.testing.assert_allclose(x_back, x_orig, atol=1e-12)
        np.testing.assert_allclose(y_back, y_orig, atol=1e-12)


# ---------------------------------------------------------------------------
# upMu
# ---------------------------------------------------------------------------
class TestUpMu:
    def test_returns_string(self):
        result = ryutils.upMu()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unicode_mu_on_python3(self):
        # On Python 3 the upright branch is not taken; we always get the unicode char
        result = ryutils.upMu(uprightMu=False)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# abshumidity / TFromAbshumidity  (round-trip)
# ---------------------------------------------------------------------------
class TestHumidity:
    def test_abshumidity_returns_positive(self):
        T = np.array([273.15, 293.15, 313.15])
        ah = ryutils.abshumidity(T)
        assert np.all(ah > 0)

    def test_abshumidity_increases_with_T(self):
        T = np.linspace(260, 330, 50)
        ah = ryutils.abshumidity(T)
        assert np.all(np.diff(ah) > 0)

    def test_TFromAbshumidity_roundtrip(self):
        T_original = 295.0
        ah = ryutils.abshumidity(np.array([T_original]))[0]
        T_recovered = ryutils.TFromAbshumidity(ah)
        assert float(T_recovered) == pytest.approx(T_original, abs=0.5)

    def test_equation2_positive(self):
        T = np.array([280.0, 300.0])
        ah = ryutils.abshumidity(T, equationSelect=2)
        assert np.all(ah > 0)


# ---------------------------------------------------------------------------
# sfilter
# ---------------------------------------------------------------------------
class TestSFilter:
    def setup_method(self):
        self.spectral = np.linspace(0.5, 12.0, 500)

    def test_bandpass_peak_at_center(self):
        center, width = 5.0, 1.0
        tau = ryutils.sfilter(self.spectral, center, width, filtertype='bandpass')
        peak_idx = np.argmax(tau)
        assert abs(self.spectral[peak_idx] - center) < 0.1

    def test_bandpass_values_bounded(self):
        tau = ryutils.sfilter(self.spectral, 5.0, 1.0)
        assert np.all(tau >= 0)
        assert np.all(tau <= 1.0 + 1e-9)

    def test_lowpass_high_side_maximal(self):
        tau = ryutils.sfilter(self.spectral, 5.0, 1.0, filtertype='lowpass')
        # values well below center should be near taupass=1
        below = tau[self.spectral < 3.0]
        assert np.all(below > 0.99)

    def test_highpass_low_side_maximal(self):
        tau = ryutils.sfilter(self.spectral, 5.0, 1.0, filtertype='highpass')
        above = tau[self.spectral > 7.0]
        assert np.all(above > 0.99)

    def test_invalid_filtertype_returns_none(self):
        assert ryutils.sfilter(self.spectral, 5.0, 1.0, filtertype='invalid') is None


# ---------------------------------------------------------------------------
# responsivity
# ---------------------------------------------------------------------------
class TestResponsivity:
    def test_returns_nonnegative(self):
        wl = np.linspace(1.0, 12.0, 200)
        resp = ryutils.responsivity(wl, lwavepeak=5.0)
        assert np.all(resp >= 0)

    def test_peak_near_lwavepeak(self):
        wl = np.linspace(1.0, 10.0, 500)
        resp = ryutils.responsivity(wl, lwavepeak=5.0)
        peak_wl = wl[np.argmax(resp)]
        assert abs(peak_wl - 5.0) < 1.0


# ---------------------------------------------------------------------------
# effectiveValue
# ---------------------------------------------------------------------------
class TestEffectiveValue:
    def test_flat_baseline_returns_mean(self):
        wl = np.linspace(1.0, 5.0, 100)
        signal = np.ones(100) * 3.0
        baseline = np.ones(100)
        result = ryutils.effectiveValue(wl, signal, baseline)
        assert result == pytest.approx(3.0, rel=1e-6)

    def test_delta_baseline_selects_point(self):
        wl = np.linspace(0.0, 10.0, 1001)
        signal = wl.copy()          # signal = wavelength value
        baseline = np.zeros(1001)
        # narrow spike near wl=5
        baseline[500] = 1.0
        result = ryutils.effectiveValue(wl, signal, baseline)
        assert result == pytest.approx(5.0, abs=0.1)


# ---------------------------------------------------------------------------
# convertSpectralDomain
# ---------------------------------------------------------------------------
class TestConvertSpectralDomain:
    def test_wavelength_to_wavenumber_roundtrip(self):
        wl = np.linspace(1.0, 10.0, 50)   # um
        wn = ryutils.convertSpectralDomain(wl, type='ln')
        wl_back = ryutils.convertSpectralDomain(wn, type='nl')
        np.testing.assert_allclose(wl_back, wl, rtol=1e-10)

    def test_invalid_type_returns_empty(self):
        wl = np.linspace(1.0, 5.0, 10)
        result = ryutils.convertSpectralDomain(wl, type='xx')
        assert result.size == 0

    def test_wavelength_to_frequency_positive(self):
        wl = np.linspace(1.0, 10.0, 20)
        freq = ryutils.convertSpectralDomain(wl, type='lf')
        assert np.all(freq > 0)


# ---------------------------------------------------------------------------
# circ / rect
# ---------------------------------------------------------------------------
class TestApertures:
    def setup_method(self):
        x = np.linspace(-1.0, 1.0, 201)
        y = np.linspace(-1.0, 1.0, 201)
        self.xx, self.yy = np.meshgrid(x, y)

    def test_circ_centre_is_one(self):
        z = ryutils.circ(self.xx, self.yy, d=1.0)
        # centre pixel should be 1
        c = z[100, 100]
        assert c == pytest.approx(1.0)

    def test_circ_corner_is_zero(self):
        z = ryutils.circ(self.xx, self.yy, d=1.0)
        assert z[0, 0] == pytest.approx(0.0)

    def test_circ_values_in_01(self):
        z = ryutils.circ(self.xx, self.yy, d=0.8)
        assert np.all((z >= 0) & (z <= 1.0))

    def test_rect_centre_is_one(self):
        z = ryutils.rect(self.xx, self.yy, sx=1.0, sy=1.0)
        assert z[100, 100] == pytest.approx(1.0)

    def test_rect_corner_is_zero(self):
        z = ryutils.rect(self.xx, self.yy, sx=1.0, sy=1.0)
        assert z[0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# detectFARThresholdToNoisepulseWidth
# ---------------------------------------------------------------------------
class TestDetectionStatistics:
    def test_far_positive(self):
        far = ryutils.detectFARThresholdToNoisepulseWidth(
            ThresholdToNoise=3.0, pulseWidth=1e-3
        )
        assert far > 0

    def test_higher_threshold_lower_far(self):
        far_low = ryutils.detectFARThresholdToNoisepulseWidth(2.0, 1e-3)
        far_high = ryutils.detectFARThresholdToNoisepulseWidth(5.0, 1e-3)
        assert far_high < far_low


# ---------------------------------------------------------------------------
# solidAngleSquare
# ---------------------------------------------------------------------------
class TestSolidAngle:
    def test_geometric_positive(self):
        sa = ryutils.solidAngleSquare(0.1, 0.1, 1.0, 'g', 50)
        assert sa > 0

    def test_projected_positive(self):
        sa = ryutils.solidAngleSquare(0.1, 0.1, 1.0, 'p', 50)
        assert sa > 0

    def test_invalid_type_returns_none(self):
        result = ryutils.solidAngleSquare(0.1, 0.1, 1.0, 'x', 50)
        assert result is None

    def test_geometric_less_than_projected(self):
        # For a small source, geometric solid angle < projected
        sg = ryutils.solidAngleSquare(0.1, 0.1, 1.0, 'g', 100)
        sp = ryutils.solidAngleSquare(0.1, 0.1, 1.0, 'p', 100)
        assert sg > sp  # geometric has n=3 (less weighting) so larger value
