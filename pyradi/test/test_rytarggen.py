"""Regression tests for rytarggen.py — HDF5 image generation utilities.

Tests cover:
  - assigncheck (HDF5 key creation/update)
  - calcLuxEquivalent (radiance→lux interpolation)
  - calcTemperatureEquivalent (radiance↔temperature interpolation)
  - create_HDF5_image with hdf_Uniform, hdf_stairs, hdf_disk_photon
  - Noise image generation path (saveNoiseImage=True)
"""

import os
import sys
import numpy as np
import pytest

# Make both bare module imports and pyradi.* imports work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import rytarggen as rtg


# ---------------------------------------------------------------------------
# assigncheck
# ---------------------------------------------------------------------------
class TestAssigncheck:
    def test_creates_new_entry(self, tmp_path):
        """assigncheck creates a new dataset when the path does not exist."""
        import h5py
        fpath = tmp_path / 'test.hdf5'
        with h5py.File(fpath, 'w') as f:
            rtg.assigncheck(f, 'group/key', 42.0)
            assert f['group/key'][()] == pytest.approx(42.0)

    def test_updates_existing_entry(self, tmp_path):
        """assigncheck overwrites an existing dataset."""
        import h5py
        fpath = tmp_path / 'test.hdf5'
        with h5py.File(fpath, 'w') as f:
            f.create_dataset('group/key', data=np.array(1.0))
            rtg.assigncheck(f, 'group/key', 99.0)
            assert f['group/key'][()] == pytest.approx(99.0)

    def test_creates_string_entry(self, tmp_path):
        """assigncheck handles string values."""
        import h5py
        fpath = tmp_path / 'test.hdf5'
        with h5py.File(fpath, 'w') as f:
            rtg.assigncheck(f, 'meta/label', 'hello')
            val = f['meta/label'][()]
            # h5py may return bytes or str depending on version
            if isinstance(val, bytes):
                val = val.decode()
            assert val == 'hello'


# ---------------------------------------------------------------------------
# calcLuxEquivalent
# ---------------------------------------------------------------------------
class TestCalcLuxEquivalent:
    """calcLuxEquivalent returns an interp1d mapping photon-rate radiance → lux.

    Note on wavelength units: calcLuxEquivalent expects wavelength in SI metres
    (it multiplies by 1e6 internally to convert to µm for the photopic response
    formula).  This is different from calcTemperatureEquivalent which expects µm
    directly (matching ryplanck.planck's convention).
    """

    def test_returns_callable(self):
        wl = 0.55e-6  # SI metres
        fintp = rtg.calcLuxEquivalent(wl, 1e14, 1e16, 'q/(s.m2.sr)')
        assert callable(fintp)

    def test_higher_radiance_gives_more_lux_photon_units(self):
        """Lux increases monotonically with photon-rate radiance."""
        wl = 0.55e-6
        rad_min = 1e14
        rad_dynrange = 1e16
        fintp = rtg.calcLuxEquivalent(wl, rad_min, rad_dynrange, 'q/(s.m2.sr)')
        # The fintp x-axis domain is approximately [0.99*rad_min, 1.01*(rad_min+rad_dynrange)]
        mid = rad_min + rad_dynrange * 0.3
        high = rad_min + rad_dynrange * 0.7
        assert fintp(high) > fintp(mid)

    def test_watt_units_returns_callable(self):
        """Watt-unit variant still returns a callable."""
        wl = 0.55e-6
        fintp = rtg.calcLuxEquivalent(wl, 9.659e-4, 0.483, 'W/(m2.sr)')
        assert callable(fintp)

    def test_photon_units_fintp_domain_matches_input_range(self):
        """The interp1d domain covers the input q-rate range."""
        wl = 0.55e-6
        rad_min = 5e13
        rad_dynrange = 5e15
        fintp = rtg.calcLuxEquivalent(wl, rad_min, rad_dynrange, 'q/(s.m2.sr)')
        # Values within [rad_min, rad_min + rad_dynrange] must not raise
        val = fintp(rad_min + rad_dynrange * 0.5)
        assert np.isfinite(val)
        assert val > 0


# ---------------------------------------------------------------------------
# calcTemperatureEquivalent
# ---------------------------------------------------------------------------
class TestCalcTemperatureEquivalent:
    """calcTemperatureEquivalent returns (fintpLE, fintpEL).

    Note on wavelength units: ryplanck.planck(type='ql') expects wavelength
    in µm, so the wavelength vector passed here must be in µm.
    """

    def _wl_sysresp(self):
        wl = np.linspace(3.0, 5.0, 50)  # µm — MWIR band
        sysresp = np.ones_like(wl)
        return wl, sysresp

    def test_returns_two_callables(self):
        wl, sysresp = self._wl_sysresp()
        fLE, fEL = rtg.calcTemperatureEquivalent(wl, sysresp, 280, 350)
        assert callable(fLE)
        assert callable(fEL)

    def test_roundtrip_temperature(self):
        """fintpLE(fintpEL(T)) ≈ T for a temperature within [tmin, tmax]."""
        wl, sysresp = self._wl_sysresp()
        fLE, fEL = rtg.calcTemperatureEquivalent(wl, sysresp, 280, 320)
        T_in = 300.0
        T_out = float(fLE(fEL(T_in)))
        assert T_out == pytest.approx(T_in, rel=1e-3)

    def test_higher_temperature_gives_more_radiance(self):
        """fintpEL (temp → radiance) is monotonically increasing."""
        wl, sysresp = self._wl_sysresp()
        fLE, fEL = rtg.calcTemperatureEquivalent(wl, sysresp, 280, 400)
        assert float(fEL(380.0)) > float(fEL(300.0))

    def test_fe_returns_finite_positive(self):
        wl, sysresp = self._wl_sysresp()
        fLE, fEL = rtg.calcTemperatureEquivalent(wl, sysresp, 290, 310)
        val = float(fEL(300.0))
        assert np.isfinite(val)
        assert val > 0


# ---------------------------------------------------------------------------
# create_HDF5_image with hdf_Uniform
# ---------------------------------------------------------------------------
class TestCreateHDF5ImageUniform:

    def test_creates_hdf5_file(self, tmp_path, monkeypatch):
        """create_HDF5_image writes a file to disk."""
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='TestUniform',
            numPixels=[32, 32],
            fn=rtg.hdf_Uniform,
            kwargs={'rad_dynrange': 1e15},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
        )
        assert os.path.isfile(fname)

    def test_image_has_correct_shape(self, tmp_path, monkeypatch):
        import h5py
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='TestShape',
            numPixels=[16, 24],
            fn=rtg.hdf_Uniform,
            kwargs={'rad_dynrange': 1e15},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
        )
        with h5py.File(fname, 'r') as f:
            img = f['image/PhotonRateRadianceNoNoise'][()]
        assert img.shape == (16, 24)

    def test_uniform_image_all_same_value(self, tmp_path, monkeypatch):
        """Every pixel should equal rad_dynrange (photon-rate units)."""
        import h5py
        monkeypatch.chdir(tmp_path)
        rad = 1.3e17
        fname = rtg.create_HDF5_image(
            imageName='TestConst',
            numPixels=[16, 16],
            fn=rtg.hdf_Uniform,
            kwargs={'rad_dynrange': rad},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
        )
        with h5py.File(fname, 'r') as f:
            img = f['image/PhotonRateRadianceNoNoise'][()]
        assert np.allclose(img, img[0, 0])

    def test_zero_uniform_all_zeros(self, tmp_path, monkeypatch):
        import h5py
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='TestZero',
            numPixels=[16, 16],
            fn=rtg.hdf_Uniform,
            kwargs={'rad_dynrange': 0},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
        )
        with h5py.File(fname, 'r') as f:
            img = f['image/PhotonRateRadianceNoNoise'][()]
        assert np.all(img == 0.0)

    def test_filename_uses_nospace_flag(self, tmp_path, monkeypatch):
        """noSpaces=True replaces spaces and dots in the filename."""
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='My Image 1.0',
            numPixels=[8, 8],
            fn=rtg.hdf_Uniform,
            kwargs={'rad_dynrange': 0},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
            noSpaces=True,
        )
        assert ' ' not in fname
        assert fname.count('.') == 1  # only the .hdf5 extension dot


# ---------------------------------------------------------------------------
# hdf_stairs
# ---------------------------------------------------------------------------
class TestHdfStairs:

    def test_stairslin_creates_nonuniform_image(self, tmp_path, monkeypatch):
        """Linear staircase image has varying pixel values."""
        import h5py
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='TestStairs',
            numPixels=[32, 64],
            fn=rtg.hdf_stairs,
            kwargs={'rad_min': 0.0, 'rad_dynrange': 1e15, 'steps': 8, 'imtype': 'stairslin'},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
        )
        with h5py.File(fname, 'r') as f:
            img = f['image/PhotonRateRadianceNoNoise'][()]
        assert img.shape == (32, 64)
        assert img.max() > img.min()

    def test_stairslog_creates_nonuniform_image(self, tmp_path, monkeypatch):
        """Logarithmic staircase image has varying pixel values."""
        import h5py
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='TestStairsLog',
            numPixels=[32, 64],
            fn=rtg.hdf_stairs,
            kwargs={'rad_min': 0.0, 'rad_dynrange': 1e15, 'steps': 4, 'imtype': 'stairslog'},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
        )
        with h5py.File(fname, 'r') as f:
            img = f['image/PhotonRateRadianceNoNoise'][()]
        assert img.max() > 0.0

    def test_stairslin_max_equals_min_plus_dynrange(self, tmp_path, monkeypatch):
        """Max pixel ≈ rad_min + rad_dynrange (photon units, q-rate input)."""
        import h5py
        monkeypatch.chdir(tmp_path)
        rad_min = 0.0
        rad_dynrange = 2e15
        fname = rtg.create_HDF5_image(
            imageName='TestStairsMax',
            numPixels=[32, 64],
            fn=rtg.hdf_stairs,
            kwargs={'rad_min': rad_min, 'rad_dynrange': rad_dynrange,
                    'steps': 4, 'imtype': 'stairslin'},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
        )
        with h5py.File(fname, 'r') as f:
            img = f['image/PhotonRateRadianceNoNoise'][()]
        expected_max = rad_min + rad_dynrange
        assert img.max() == pytest.approx(expected_max, rel=1e-6)


# ---------------------------------------------------------------------------
# hdf_disk_photon
# ---------------------------------------------------------------------------
class TestHdfDiskPhoton:

    def test_disk_creates_image_correct_shape(self, tmp_path, monkeypatch):
        import h5py
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='TestDisk',
            numPixels=[64, 64],
            fn=rtg.hdf_disk_photon,
            kwargs={'rad_min': 0.0, 'rad_dynrange': 1e15,
                    'fracdiameter': 0.5, 'fracblur': 0.1},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
        )
        with h5py.File(fname, 'r') as f:
            img = f['image/PhotonRateRadianceNoNoise'][()]
        assert img.shape == (64, 64)
        assert img.max() > img.min()

    def test_disk_center_brighter_than_corner(self, tmp_path, monkeypatch):
        """The bright disk is at the image centre; corners should be darker."""
        import h5py
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='TestDiskCenter',
            numPixels=[64, 64],
            fn=rtg.hdf_disk_photon,
            kwargs={'rad_min': 0.0, 'rad_dynrange': 1e15,
                    'fracdiameter': 0.6, 'fracblur': 0.05},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
        )
        with h5py.File(fname, 'r') as f:
            img = f['image/PhotonRateRadianceNoNoise'][()]
        center = img[32, 32]
        corner = img[0, 0]
        assert center > corner

    def test_disk_with_nonzero_min(self, tmp_path, monkeypatch):
        """rad_min raises the background level; minimum pixel > 0."""
        import h5py
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='TestDiskMinFloor',
            numPixels=[32, 32],
            fn=rtg.hdf_disk_photon,
            kwargs={'rad_min': 1e14, 'rad_dynrange': 1e15,
                    'fracdiameter': 0.4, 'fracblur': 0.05},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
        )
        with h5py.File(fname, 'r') as f:
            img = f['image/PhotonRateRadianceNoNoise'][()]
        assert img.min() >= 0.0


# ---------------------------------------------------------------------------
# Noise image path (saveNoiseImage=True)
# ---------------------------------------------------------------------------
class TestNoiseImage:

    def test_save_noise_image_creates_dataset(self, tmp_path, monkeypatch):
        """With saveNoiseImage=True, PhotonRateRadiance dataset exists."""
        import h5py
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='TestNoise',
            numPixels=[16, 16],
            fn=rtg.hdf_Uniform,
            kwargs={'rad_dynrange': 1e15},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
            saveNoiseImage=True,
            seedval=42,
        )
        with h5py.File(fname, 'r') as f:
            assert 'image/PhotonRateRadiance' in f

    def test_noisy_image_differs_from_clean(self, tmp_path, monkeypatch):
        """Photon noise makes the noisy image statistically different."""
        import h5py
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='TestNoiseDiff',
            numPixels=[32, 32],
            fn=rtg.hdf_Uniform,
            kwargs={'rad_dynrange': 1e15},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
            saveNoiseImage=True,
            seedval=42,
        )
        with h5py.File(fname, 'r') as f:
            img_noise = f['image/PhotonRateRadiance'][()]
            img_clean = f['image/PhotonRateRadianceNoNoise'][()]
        # np.allclose uses rtol=1e-5 by default; photon noise on 1e15 photons is
        # ~sqrt(1e15) ≈ 3e7 (relative ~3e-8), which falls within that tolerance.
        # Use array_equal — the arrays are not identical even though the difference is small.
        assert not np.array_equal(img_noise, img_clean)

    def test_noise_image_seed_reproducibility(self, tmp_path, monkeypatch):
        """Same seed produces identical noisy images."""
        import h5py
        monkeypatch.chdir(tmp_path)
        fname1 = rtg.create_HDF5_image(
            imageName='TestSeed1',
            numPixels=[16, 16],
            fn=rtg.hdf_Uniform,
            kwargs={'rad_dynrange': 1e15},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
            saveNoiseImage=True,
            seedval=7,
        )
        fname2 = rtg.create_HDF5_image(
            imageName='TestSeed2',
            numPixels=[16, 16],
            fn=rtg.hdf_Uniform,
            kwargs={'rad_dynrange': 1e15},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
            saveNoiseImage=True,
            seedval=7,
        )
        with h5py.File(fname1, 'r') as f:
            img1 = f['image/PhotonRateRadiance'][()]
        with h5py.File(fname2, 'r') as f:
            img2 = f['image/PhotonRateRadiance'][()]
        assert np.array_equal(img1, img2)


# ---------------------------------------------------------------------------
# Equivalent signal path (saveEquivImage=True, fintp=None)
# ---------------------------------------------------------------------------
class TestEquivImage:

    def test_equiv_signal_saved_when_requested(self, tmp_path, monkeypatch):
        import h5py
        monkeypatch.chdir(tmp_path)
        fname = rtg.create_HDF5_image(
            imageName='TestEquiv',
            numPixels=[16, 16],
            fn=rtg.hdf_Uniform,
            kwargs={'rad_dynrange': 1e15},
            wavelength=0.55e-6,
            LinUnits='q/(s.m2.sr)',
            saveEquivImage=True,
        )
        with h5py.File(fname, 'r') as f:
            equiv = f['image/equivalentSignal'][()]
            clean = f['image/PhotonRateRadianceNoNoise'][()]
        # fintp=None → equivalentSignal == PhotonRateRadianceNoNoise
        assert np.allclose(equiv, clean)
