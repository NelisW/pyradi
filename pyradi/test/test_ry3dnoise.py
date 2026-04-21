"""Regression tests for ry3dnoise.py — 3D noise parameter calculation."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import ry3dnoise as ry3d

# ── Fixtures ──────────────────────────────────────────────────────────────────

FRAMES, ROWS, COLS = 5, 4, 6


@pytest.fixture
def zeros():
    return np.zeros((FRAMES, ROWS, COLS))


@pytest.fixture
def ones():
    return np.ones((FRAMES, ROWS, COLS))


@pytest.fixture
def rng_seq():
    rng = np.random.default_rng(42)
    return rng.standard_normal((FRAMES, ROWS, COLS))


@pytest.fixture
def temporal_ramp():
    """Each frame is a constant equal to its frame index (0, 1, 2, …)."""
    seq = np.zeros((FRAMES, ROWS, COLS))
    for i in range(FRAMES):
        seq[i] = float(i)
    return seq


@pytest.fixture
def known():
    """Small deterministic array for exact-value operator tests."""
    return np.arange(24, dtype=float).reshape(2, 3, 4)


# ── Operator output shapes ────────────────────────────────────────────────────

class TestOperatorShapes:
    def test_oprDV_shape(self, ones):
        assert ry3d.oprDV(ones).shape == (FRAMES, 1, COLS)

    def test_oprDH_shape(self, ones):
        assert ry3d.oprDH(ones).shape == (FRAMES, ROWS, 1)

    def test_oprDT_shape(self, ones):
        assert ry3d.oprDT(ones).shape == (1, ROWS, COLS)

    def test_oprSDT_shape(self, rng_seq):
        assert ry3d.oprSDT(rng_seq).shape == (FRAMES, ROWS, COLS)

    def test_oprSDH_shape(self, rng_seq):
        assert ry3d.oprSDH(rng_seq).shape == (FRAMES, ROWS, COLS)

    def test_oprSDV_shape(self, rng_seq):
        assert ry3d.oprSDV(rng_seq).shape == (FRAMES, ROWS, COLS)


# ── Operator values ───────────────────────────────────────────────────────────

class TestOperatorValues:
    def test_oprDV_zeros(self, zeros):
        assert np.allclose(ry3d.oprDV(zeros), 0)

    def test_oprDH_zeros(self, zeros):
        assert np.allclose(ry3d.oprDH(zeros), 0)

    def test_oprDT_zeros(self, zeros):
        assert np.allclose(ry3d.oprDT(zeros), 0)

    def test_oprDV_ones(self, ones):
        assert np.allclose(ry3d.oprDV(ones), 1)

    def test_oprDH_ones(self, ones):
        assert np.allclose(ry3d.oprDH(ones), 1)

    def test_oprDT_ones(self, ones):
        assert np.allclose(ry3d.oprDT(ones), 1)

    def test_oprSDT_zero_on_constant(self, ones):
        """Constant array: temporal deviation is everywhere zero."""
        assert np.allclose(ry3d.oprSDT(ones), 0)

    def test_oprSDH_zero_on_constant(self, ones):
        assert np.allclose(ry3d.oprSDH(ones), 0)

    def test_oprSDV_zero_on_constant(self, ones):
        assert np.allclose(ry3d.oprSDV(ones), 0)

    def test_oprDV_known(self, known):
        """DV averages over rows (axis=1); verify against numpy directly."""
        result = ry3d.oprDV(known)
        assert result.shape == (2, 1, 4)
        assert np.allclose(result[0, 0, :], np.mean(known[0], axis=0))
        assert np.allclose(result[1, 0, :], np.mean(known[1], axis=0))

    def test_oprDH_known(self, known):
        """DH averages over cols (axis=2)."""
        result = ry3d.oprDH(known)
        assert result.shape == (2, 3, 1)
        assert np.allclose(result[0, :, 0], np.mean(known[0], axis=1))
        assert np.allclose(result[1, :, 0], np.mean(known[1], axis=1))

    def test_oprDT_known(self, known):
        """DT averages over frames (axis=0)."""
        result = ry3d.oprDT(known)
        assert result.shape == (1, 3, 4)
        assert np.allclose(result[0], np.mean(known, axis=0))

    def test_oprSDT_removes_mean(self, rng_seq):
        """After SDT the temporal mean of each pixel is zero."""
        result = ry3d.oprSDT(rng_seq)
        assert np.allclose(np.mean(result, axis=0), 0, atol=1e-12)

    def test_oprSDH_removes_col_mean(self, rng_seq):
        """After SDH the column mean for each (frame, row) is zero."""
        result = ry3d.oprSDH(rng_seq)
        assert np.allclose(np.mean(result, axis=2), 0, atol=1e-12)

    def test_oprSDV_removes_row_mean(self, rng_seq):
        """After SDV the row mean for each (frame, col) is zero."""
        result = ry3d.oprSDV(rng_seq)
        assert np.allclose(np.mean(result, axis=1), 0, atol=1e-12)


# ── getS ──────────────────────────────────────────────────────────────────────

class TestGetS:
    def test_zeros(self, zeros):
        assert ry3d.getS(zeros) == pytest.approx(0.0)

    def test_ones(self, ones):
        assert ry3d.getS(ones) == pytest.approx(1.0)

    def test_constant(self):
        seq = np.full((3, 4, 5), 7.5)
        assert ry3d.getS(seq) == pytest.approx(7.5)

    def test_known(self, rng_seq):
        assert ry3d.getS(rng_seq) == pytest.approx(float(np.mean(rng_seq)))


# ── Noise component return types and shapes ───────────────────────────────────

class TestNoiseComponentReturnTypes:
    """Every getNXX function must return (float, np.ndarray)."""

    @pytest.mark.parametrize("fn", [
        ry3d.getNT, ry3d.getNVH, ry3d.getNTV,
        ry3d.getNTH, ry3d.getNV, ry3d.getNH, ry3d.getNTVH,
    ])
    def test_returns_float_and_array(self, fn, rng_seq):
        std, img = fn(rng_seq)
        assert isinstance(float(std), float)
        assert isinstance(img, np.ndarray)

    def test_getNT_image_shape(self, rng_seq):
        _, img = ry3d.getNT(rng_seq)
        assert img.shape == (FRAMES, 1, 1)

    def test_getNVH_image_shape(self, rng_seq):
        _, img = ry3d.getNVH(rng_seq)
        assert img.shape == (1, ROWS, COLS)

    def test_getNTV_image_shape(self, rng_seq):
        _, img = ry3d.getNTV(rng_seq)
        assert img.shape == (FRAMES, ROWS, 1)

    def test_getNTH_image_shape(self, rng_seq):
        _, img = ry3d.getNTH(rng_seq)
        assert img.shape == (FRAMES, 1, COLS)

    def test_getNV_image_shape(self, rng_seq):
        _, img = ry3d.getNV(rng_seq)
        assert img.shape == (1, ROWS, 1)

    def test_getNH_image_shape(self, rng_seq):
        _, img = ry3d.getNH(rng_seq)
        assert img.shape == (1, 1, COLS)

    def test_getNTVH_image_shape(self, rng_seq):
        _, img = ry3d.getNTVH(rng_seq)
        assert img.shape == (FRAMES, ROWS, COLS)


# ── Noise components — zero on constant input ─────────────────────────────────

class TestNoiseZeroOnConstant:
    """A constant array has no variation: every noise component must be zero."""

    @pytest.mark.parametrize("fn", [
        ry3d.getNT, ry3d.getNVH, ry3d.getNTV,
        ry3d.getNTH, ry3d.getNV, ry3d.getNH, ry3d.getNTVH,
    ])
    def test_zero(self, fn, ones):
        std, _ = fn(ones)
        assert std == pytest.approx(0.0, abs=1e-12)


# ── Noise components — physical behaviour ─────────────────────────────────────

class TestNoisePhysics:
    def test_getNT_detects_temporal_variation(self, temporal_ramp):
        """Array where each frame differs from others → NT must be nonzero."""
        std, _ = ry3d.getNT(temporal_ramp)
        assert std > 0

    def test_getNT_zero_when_no_temporal_variation(self, ones):
        std, _ = ry3d.getNT(ones)
        assert std == pytest.approx(0.0, abs=1e-12)

    def test_getNVH_detects_spatial_nonuniformity(self):
        """Each pixel has a unique fixed value → NVH must be nonzero."""
        rng = np.random.default_rng(7)
        pattern = rng.standard_normal((ROWS, COLS))
        # Repeat same pattern across all frames (no temporal change)
        seq = np.stack([pattern] * FRAMES, axis=0)
        std, _ = ry3d.getNVH(seq)
        assert std > 0

    def test_all_noise_nonneg(self, rng_seq):
        for fn in [ry3d.getNT, ry3d.getNVH, ry3d.getNTV,
                   ry3d.getNTH, ry3d.getNV, ry3d.getNH, ry3d.getNTVH]:
            std, _ = fn(rng_seq)
            assert std >= 0


# ── getTotal ──────────────────────────────────────────────────────────────────

class TestGetTotal:
    def test_zero_on_constant(self, ones):
        assert ry3d.getTotal(ones) == pytest.approx(0.0, abs=1e-12)

    def test_nonneg(self, rng_seq):
        assert ry3d.getTotal(rng_seq) >= 0

    def test_pythagorean_sum(self, rng_seq):
        """getTotal must equal sqrt(sum of squares of all seven components)."""
        nt   = ry3d.getNT(rng_seq)[0]
        nvh  = ry3d.getNVH(rng_seq)[0]
        ntv  = ry3d.getNTV(rng_seq)[0]
        nth  = ry3d.getNTH(rng_seq)[0]
        nv   = ry3d.getNV(rng_seq)[0]
        nh   = ry3d.getNH(rng_seq)[0]
        ntvh = ry3d.getNTVH(rng_seq)[0]
        expected = np.sqrt(nt**2 + nvh**2 + ntv**2 + nth**2 + nv**2 + nh**2 + ntvh**2)
        assert ry3d.getTotal(rng_seq) == pytest.approx(expected)
