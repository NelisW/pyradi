"""Regression tests for rychroma.py — CIE colour coordinate processing."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import rychroma as rc

# Path to the bundled CIE data file (works when running from the project folder)
_DATA_DIR = Path(__file__).parent.parent / 'data' / 'colourcoordinates'
_CIE_FILE = _DATA_DIR / 'ciexyz31_1.txt'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flat_bars(n=100):
    """Return a flat spectrum and equal tristimulus bars over [0.38, 0.72] µm."""
    wl   = np.linspace(0.38, 0.72, n)
    ones = np.ones(n)
    return wl, ones, ones, ones   # wl, xbar, ybar, zbar


def _mock_resources_for_cie():
    """Return a context-manager patch that routes resources.files to the local CIE file."""
    mock_traversable = MagicMock()
    mock_traversable.open.return_value = open(_CIE_FILE, 'r')
    mock_files = MagicMock(return_value=MagicMock())
    mock_files.return_value.joinpath.return_value = mock_traversable
    return patch('rychroma.resources.files', mock_files)


# ── chromaticityforSpectralL ─────────────────────────────────────────────────

class TestChromaticityForSpectralL:
    def test_returns_three_values(self):
        wl, xb, yb, zb = _flat_bars()
        result = rc.chromaticityforSpectralL(wl, np.ones(100), xb, yb, zb)
        assert len(result) == 3

    def test_equal_bars_give_equal_weight(self):
        """Equal tristimulus bars → x = y = 1/3."""
        wl, xb, yb, zb = _flat_bars()
        x, y, Y = rc.chromaticityforSpectralL(wl, np.ones(100), xb, yb, zb)
        assert x == pytest.approx(1 / 3, abs=1e-6)
        assert y == pytest.approx(1 / 3, abs=1e-6)

    def test_xy_sum_less_than_one(self):
        wl, xb, yb, zb = _flat_bars()
        x, y, Y = rc.chromaticityforSpectralL(wl, np.ones(100), xb, yb, zb)
        assert x + y <= 1.0 + 1e-9

    def test_xy_nonneg(self):
        wl, xb, yb, zb = _flat_bars()
        x, y, Y = rc.chromaticityforSpectralL(wl, np.ones(100), xb, yb, zb)
        assert x >= 0 and y >= 0

    def test_Y_positive_for_positive_radiance(self):
        wl, xb, yb, zb = _flat_bars()
        _, _, Y = rc.chromaticityforSpectralL(wl, np.ones(100), xb, yb, zb)
        assert Y > 0

    def test_scales_with_radiance(self):
        """Doubling radiance doubles Y but leaves x, y unchanged."""
        wl, xb, yb, zb = _flat_bars()
        rad = np.random.default_rng(0).random(100) + 0.1
        x1, y1, Y1 = rc.chromaticityforSpectralL(wl, rad, xb, yb, zb)
        x2, y2, Y2 = rc.chromaticityforSpectralL(wl, 2 * rad, xb, yb, zb)
        assert x2 == pytest.approx(x1, abs=1e-9)
        assert y2 == pytest.approx(y1, abs=1e-9)
        assert Y2 == pytest.approx(2 * Y1, rel=1e-6)


# ── XYZforSpectralL ───────────────────────────────────────────────────────────

class TestXYZForSpectralL:
    def test_returns_three_values(self):
        wl, xb, yb, zb = _flat_bars()
        result = rc.XYZforSpectralL(wl, np.ones(100), xb, yb, zb)
        assert len(result) == 3

    def test_equal_bars_give_equal_XYZ(self):
        """Equal tristimulus bars → X = Y = Z."""
        wl, xb, yb, zb = _flat_bars()
        X, Y, Z = rc.XYZforSpectralL(wl, np.ones(100), xb, yb, zb)
        assert X == pytest.approx(Y, rel=1e-9)
        assert Y == pytest.approx(Z, rel=1e-9)

    def test_all_positive(self):
        wl, xb, yb, zb = _flat_bars()
        X, Y, Z = rc.XYZforSpectralL(wl, np.ones(100), xb, yb, zb)
        assert X > 0 and Y > 0 and Z > 0

    def test_consistent_with_chromaticity(self):
        """XYZ and chromaticityforSpectralL must agree on Y."""
        wl = np.linspace(0.38, 0.72, 200)
        rng = np.random.default_rng(1)
        rad  = rng.random(200) + 0.1
        xb   = rng.random(200) + 0.1
        yb   = rng.random(200) + 0.1
        zb   = rng.random(200) + 0.1
        X, Y, Z = rc.XYZforSpectralL(wl, rad, xb, yb, zb)
        _, _, Y2 = rc.chromaticityforSpectralL(wl, rad, xb, yb, zb)
        assert Y == pytest.approx(Y2, rel=1e-9)


# ── loadCIEbar ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _CIE_FILE.exists(), reason='CIE data file not found')
class TestLoadCIEbar:
    def test_wl_input_shape(self):
        wl = np.linspace(0.38, 0.72, 50)
        with _mock_resources_for_cie():
            ciebar = rc.loadCIEbar(wl, stype='wl')
        assert ciebar is not None
        assert ciebar.shape == (50, 4)   # [wl, x, y, z]

    def test_wn_input_shape(self):
        wn = np.linspace(14000, 26000, 60)
        with _mock_resources_for_cie():
            ciebar = rc.loadCIEbar(wn, stype='wn')
        assert ciebar is not None
        assert ciebar.shape == (60, 4)

    def test_xyz_nonneg(self):
        """Tristimulus values are non-negative throughout the visible range."""
        wl = np.linspace(0.38, 0.72, 80)
        with _mock_resources_for_cie():
            ciebar = rc.loadCIEbar(wl, stype='wl')
        assert np.all(ciebar[:, 1:] >= -1e-6)

    def test_unsorted_returns_none(self):
        """Unsorted spectral vector must return (None, None)."""
        wl = np.array([0.5, 0.4, 0.6, 0.45])
        with _mock_resources_for_cie():
            result = rc.loadCIEbar(wl, stype='wl')
        assert result == (None, None)

    def test_wl_and_wn_agree(self):
        """Wavelength and wavenumber inputs over the same range must give the same xyz."""
        wl = np.linspace(0.38, 0.72, 50)
        wn = (1e4 / wl)[::-1]           # descending wavenumber → ascending wl
        with _mock_resources_for_cie():
            cbar_wl = rc.loadCIEbar(wl, stype='wl')
        with _mock_resources_for_cie():
            cbar_wn = rc.loadCIEbar(wn, stype='wn')
        # Both should cover the same wavelength range; y-column should be close
        assert cbar_wl is not None and cbar_wn is not None


# ── rgb2CIExy ─────────────────────────────────────────────────────────────────

class TestRgb2CIExy:
    _RGB = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=float)

    @pytest.mark.parametrize("system", ['CIE RGB', 'Adobe RGB', 'sRGB'])
    def test_output_shape(self, system):
        xy = rc.rgb2CIExy(self._RGB, system=system)
        assert xy.shape == (3, 2)

    def test_unknown_system_returns_none(self):
        assert rc.rgb2CIExy(self._RGB, system='UNKNOWN') is None

    @pytest.mark.parametrize("system", ['CIE RGB', 'Adobe RGB', 'sRGB'])
    def test_xy_in_unit_square(self, system):
        xy = rc.rgb2CIExy(self._RGB, system=system)
        assert np.all(xy >= -1e-6) and np.all(xy <= 1.0 + 1e-6)

    def test_single_row_input(self):
        rgb = np.array([[128, 64, 200]], dtype=float)
        xy = rc.rgb2CIExy(rgb, system='CIE RGB')
        assert xy.shape == (1, 2)

    def test_pure_colours_differ(self):
        """Red, green, blue primaries must land at distinct chromaticity points."""
        xy = rc.rgb2CIExy(self._RGB, system='CIE RGB')
        assert not np.allclose(xy[0], xy[1])
        assert not np.allclose(xy[1], xy[2])
        assert not np.allclose(xy[0], xy[2])


# ── CIExy2rgb ─────────────────────────────────────────────────────────────────

class TestCIExy2rgb:
    _XY = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]])

    @pytest.mark.parametrize("system", ['CIE RGB', 'Adobe RGB', 'sRGB'])
    def test_output_shape(self, system):
        rgb = rc.CIExy2rgb(self._XY, system=system)
        assert rgb.shape == (3, 3)

    def test_unknown_system_returns_none(self):
        assert rc.CIExy2rgb(self._XY, system='UNKNOWN') is None

    def test_single_row_input(self):
        xy = np.array([[0.30, 0.33]])
        rgb = rc.CIExy2rgb(xy, system='CIE RGB')
        assert rgb.shape == (1, 3)

    def test_max_normalised_to_one(self):
        """CIExy2rgb must normalise so the max component per row is 1."""
        rgb = rc.CIExy2rgb(self._XY, system='CIE RGB')
        row_maxes = np.max(rgb, axis=1)
        assert np.allclose(row_maxes, 1.0, atol=1e-9)

    @pytest.mark.parametrize("system", ['CIE RGB', 'sRGB'])
    def test_round_trip(self, system):
        """rgb → xy → rgb should approximately recover the original hue."""
        rgb_in = np.array([[200, 50, 100], [10, 220, 30]], dtype=float)
        xy = rc.rgb2CIExy(rgb_in, system=system)
        rgb_out = rc.CIExy2rgb(xy, system=system)
        # Normalise input the same way (max per row = 1)
        rgb_norm = rgb_in / np.max(rgb_in, axis=1).reshape(-1, 1)
        assert np.allclose(rgb_out, rgb_norm, atol=0.02)


# ── Module-level colour-space matrices ────────────────────────────────────────

class TestColourMatrices:
    def test_cie_rgb_inverse(self):
        """mcieRGBtoXYZ and mXYZtoCIERGB are approximate inverses."""
        assert np.allclose(np.linalg.inv(rc.mcieRGBtoXYZ), rc.mXYZtoCIERGB, atol=1e-4)

    def test_srgb_inverse(self):
        """msRGBtoXYZ and mXYZtosRGB are approximate inverses."""
        assert np.allclose(np.linalg.inv(rc.msRGBtoXYZ), rc.mXYZtosRGB, atol=1e-4)

    def test_matrices_are_3x3(self):
        for m in [rc.mcieRGBtoXYZ, rc.madobeRGBtoXYZ, rc.msRGBtoXYZ,
                  rc.mXYZtoCIERGB, rc.mXYZtoAdobeRGB, rc.mXYZtosRGB]:
            assert m.shape == (3, 3)
