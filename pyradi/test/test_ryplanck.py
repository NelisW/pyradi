"""Regression tests for ryplanck.py — Planck law spectral exitance functions.

Physics ground-truth values used throughout:
  Stefan-Boltzmann constant  σ_e ≈ 5.67037e-8  W/(m²·K⁴)
  Wien displacement (radiant, wavelength)  λ_max·T ≈ 2897.8  µm·K
  Wien displacement (photon,  wavelength)  λ_max·T ≈ 3669.7  µm·K
  Wien displacement (radiant, wavenumber)  ν_max/T ≈ 1.9610  cm⁻¹/K
"""

import os
import sys
import types

import numpy as np
import pytest
import scipy.constants as const

# ── Stub pyradi.ryutils before importing ryplanck ─────────────────────────────
# ryplanck uses ryutils.convertSpectralDomain only inside planckInt.
# We provide a correct minimal implementation to keep planckInt testable.

if 'pyradi' not in sys.modules:
    _pyradi_stub = types.ModuleType('pyradi')
    sys.modules['pyradi'] = _pyradi_stub

_ryutils_stub = types.ModuleType('pyradi.ryutils')


def _convertSpectralDomain(value, type):  # noqa: A002
    """Minimal stub — only the conversions used by planckInt."""
    v = np.asarray(value, dtype=float)
    if type == 'ln':    # wavelength µm → wavenumber cm⁻¹
        return 1e4 / v
    if type == 'fn':    # frequency Hz → wavenumber cm⁻¹
        return v / (100.0 * const.c)
    return v            # 'en' — wavenumber passes through unchanged


_ryutils_stub.convertSpectralDomain = _convertSpectralDomain
sys.modules['pyradi.ryutils'] = _ryutils_stub
sys.modules['pyradi'].ryutils = _ryutils_stub  # type: ignore[attr-defined]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import ryplanck as rp


# ── PlanckConstants ───────────────────────────────────────────────────────────

class TestPlanckConstants:
    def test_module_singleton_exists(self):
        assert isinstance(rp.pconst, rp.PlanckConstants)

    def test_c2l_value(self):
        """c2l = h·c/k × 1e6 ≈ 14388 µm·K."""
        expected = const.h * const.c / const.k * 1e6
        assert rp.pconst.c2l == pytest.approx(expected, rel=1e-6)

    def test_sigmae_matches_scipy(self):
        assert rp.pconst.sigmae == pytest.approx(const.sigma, rel=1e-8)

    def test_a5_root(self):
        """a5 is the root of 5(1-exp(-x))-x; known value ≈ 4.9651."""
        x = rp.pconst.a5
        assert 5 * (1 - np.exp(-x)) - x == pytest.approx(0.0, abs=1e-10)

    def test_wel_wiens_constant(self):
        """wel = λ_max · T; should be ≈ 2897.8 µm·K."""
        assert rp.pconst.wel == pytest.approx(2897.8, rel=1e-3)


# ── fixDimensions — shape handling via planckel ───────────────────────────────

class TestFixDimensions:
    """Test the fixDimensions decorator via planckel as a proxy."""

    def test_scalar_spec_scalar_temp_returns_scalar(self):
        result = rp.planckel(5.0, 300.0)
        assert np.ndim(result) == 0   # numpy scalar

    def test_array_spec_scalar_temp_returns_1d(self):
        wl = np.linspace(1.0, 10.0, 50)
        result = rp.planckel(wl, 300.0)
        assert result.shape == (50,)

    def test_array_spec_list_temp_single_returns_1d(self):
        wl = np.linspace(1.0, 10.0, 50)
        result = rp.planckel(wl, [300.0])
        assert result.shape == (50,)

    def test_array_spec_array_temp_returns_2d(self):
        wl = np.linspace(1.0, 10.0, 20)
        temps = np.array([300.0, 500.0, 1000.0])
        result = rp.planckel(wl, temps)
        assert result.shape == (20, 3)

    def test_scalar_spec_array_temp_returns_shape_1_M(self):
        temps = np.array([200.0, 300.0, 400.0, 500.0])
        result = rp.planckel(5.0, temps)
        assert result.shape == (1, 4)

    def test_2d_temperature_returns_none(self, capsys):
        bad_temp = np.array([[300.0, 400.0], [500.0, 600.0]])
        result = rp.planckel(5.0, bad_temp)
        assert result is None

    def test_row_vector_spectral_returns_none(self, capsys):
        bad_spec = np.array([[1.0, 2.0, 3.0]])   # shape (1, 3) — row vector
        result = rp.planckel(bad_spec, 300.0)
        assert result is None

    def test_zero_temperature_no_nan_or_inf(self):
        result = rp.planckel(5.0, 0.0)
        assert np.isfinite(result)
        assert float(result) == pytest.approx(0.0, abs=1e-100)


# ── planckel — radiant exitance, wavelength domain ───────────────────────────

class TestPlanckel:
    T = 1000.0   # K — used throughout this class

    def test_values_positive(self):
        wl = np.linspace(0.5, 20.0, 100)
        assert np.all(rp.planckel(wl, self.T) > 0)

    def test_wien_peak_location(self):
        """Peak of planckel should occur near λ_max = 2897.9/T µm."""
        wl = np.linspace(0.5, 20.0, 2000)
        M = rp.planckel(wl, self.T)
        peak_wl = wl[np.argmax(M)]
        expected = 2897.9 / self.T
        assert peak_wl == pytest.approx(expected, rel=0.02)

    def test_stefan_boltzmann_integration(self):
        """∫planckel dλ over wide range ≈ σT⁴ within 2%."""
        wl = np.linspace(0.05, 200.0, 8000)
        M = rp.planckel(wl, self.T)
        integral = np.trapezoid(M, wl)
        sbl = rp.stefanboltzman(self.T, 'e')
        assert integral == pytest.approx(float(sbl), rel=0.02)

    def test_higher_temperature_higher_exitance(self):
        """Planckel(λ_peak, T_high) > Planckel(λ_peak, T_low)."""
        peak_wl = 2897.9 / self.T
        assert float(rp.planckel(peak_wl, 1200.0)) > float(rp.planckel(peak_wl, 800.0))

    def test_list_temperature_input(self):
        wl = np.linspace(1.0, 5.0, 10)
        result = rp.planckel(wl, [300, 400, 500])
        assert result.shape == (10, 3)
        assert np.all(result > 0)


# ── planckql — photon-rate exitance, wavelength domain ───────────────────────

class TestPlanckql:
    def test_values_positive(self):
        wl = np.linspace(0.5, 20.0, 100)
        assert np.all(rp.planckql(wl, 1000.0) > 0)

    def test_wien_peak_location(self):
        """Photon-rate peak at λ_max ≈ 3669.7/T µm."""
        T = 1000.0
        wl = np.linspace(0.5, 20.0, 2000)
        M = rp.planckql(wl, T)
        peak_wl = wl[np.argmax(M)]
        assert peak_wl == pytest.approx(3669.7 / T, rel=0.02)

    def test_stefan_boltzmann_photon_integration(self):
        """∫planckql dλ ≈ σ_q·T³ within 2%."""
        T = 1000.0
        wl = np.linspace(0.05, 200.0, 8000)
        M = rp.planckql(wl, T)
        integral = np.trapezoid(M, wl)
        sbl = rp.stefanboltzman(T, 'q')
        assert integral == pytest.approx(float(sbl), rel=0.02)


# ── plancken — radiant exitance, wavenumber domain ───────────────────────────

class TestPlancken:
    def test_values_positive(self):
        n = np.linspace(100.0, 5000.0, 100)
        assert np.all(rp.plancken(n, 1000.0) > 0)

    def test_wien_peak_location(self):
        """Radiant wavenumber peak ≈ 1.9610·T cm⁻¹."""
        T = 1000.0
        n = np.linspace(100.0, 5000.0, 2000)
        M = rp.plancken(n, T)
        peak_n = n[np.argmax(M)]
        assert peak_n == pytest.approx(1.9610 * T, rel=0.02)

    def test_shape_matches_spectral_input(self):
        n = np.linspace(500.0, 3000.0, 75)
        assert rp.plancken(n, 500.0).shape == (75,)


# ── planckef, planckqf, planckqn — brief positivity / shape ──────────────────

class TestPlanckef:
    def test_positive(self):
        f = np.linspace(1e12, 1e15, 50)
        assert np.all(rp.planckef(f, 1000.0) > 0)

    def test_shape(self):
        f = np.linspace(1e12, 1e15, 30)
        assert rp.planckef(f, 1000.0).shape == (30,)


class TestPlanckqf:
    def test_positive(self):
        f = np.linspace(1e12, 1e15, 50)
        assert np.all(rp.planckqf(f, 1000.0) > 0)

    def test_shape(self):
        f = np.linspace(1e12, 1e15, 30)
        assert rp.planckqf(f, 1000.0).shape == (30,)


class TestPlanckqn:
    def test_positive(self):
        n = np.linspace(100.0, 5000.0, 50)
        assert np.all(rp.planckqn(n, 1000.0) > 0)

    def test_shape(self):
        n = np.linspace(500.0, 3000.0, 40)
        assert rp.planckqn(n, 1000.0).shape == (40,)


# ── stefanboltzman ────────────────────────────────────────────────────────────

class TestStefanboltzman:
    def test_radiant_formula(self):
        """σ_e·T⁴ using scipy.constants.sigma."""
        T = 500.0
        result = rp.stefanboltzman(T, 'e')
        assert float(result) == pytest.approx(const.sigma * T**4, rel=1e-6)

    def test_photon_formula(self):
        """σ_q·T³; σ_q ≈ 1.5204e15 q/(s·m²·K³)."""
        T = 1000.0
        result = rp.stefanboltzman(T, 'q')
        expected = rp.pconst.sigmaq * T**3
        assert float(result) == pytest.approx(expected, rel=1e-6)

    def test_invalid_type_returns_minus_one(self):
        result = rp.stefanboltzman(300.0, 'x')
        assert float(result) == pytest.approx(-1.0)

    def test_array_temperature(self):
        temps = np.array([200.0, 300.0, 400.0])
        result = rp.stefanboltzman(temps, 'e')
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_2d_temp_returns_minus_one(self, capsys):
        bad = np.array([[300.0, 400.0], [500.0, 600.0]])
        result = rp.stefanboltzman(bad, 'e')
        assert int(result) == -1


# ── planck dispatcher ─────────────────────────────────────────────────────────

class TestPlanck:
    def test_el_matches_planckel(self):
        wl = np.linspace(1.0, 10.0, 30)
        T = 800.0
        assert np.allclose(rp.planck(wl, T, 'el'), rp.planckel(wl, T))

    def test_qn_matches_planckqn(self):
        n = np.linspace(500.0, 4000.0, 20)
        T = 600.0
        assert np.allclose(rp.planck(n, T, 'qn'), rp.planckqn(n, T))

    def test_invalid_type_returns_none(self):
        assert rp.planck(5.0, 300.0, 'zz') is None

    def test_all_valid_types_return_positive(self):
        wl = np.linspace(2.0, 5.0, 20)
        n  = np.linspace(1000.0, 3000.0, 20)
        f  = np.linspace(1e13, 1e14, 20)
        for spec, t in [('el', wl), ('ql', wl), ('en', n), ('qn', n), ('ef', f), ('qf', f)]:
            result = rp.planck(t, 1000.0, spec)
            assert np.all(result > 0), f'type={spec!r} returned non-positive values'


# ── dplanck and derivative functions ─────────────────────────────────────────

class TestDplnckel:
    def test_positive(self):
        """dM/dT is always positive for Planck radiation."""
        wl = np.linspace(1.0, 10.0, 50)
        assert np.all(rp.dplnckel(wl, 1000.0) > 0)

    def test_finite_difference_consistency(self):
        """dplnckel should match a numerical finite difference of planckel."""
        wl = 5.0    # µm
        T  = 800.0  # K
        dT = 0.01   # K — small perturbation
        fd = (float(rp.planckel(wl, T + dT)) - float(rp.planckel(wl, T))) / dT
        analytic = float(rp.dplnckel(wl, T))
        assert analytic == pytest.approx(fd, rel=1e-3)

    def test_shape(self):
        wl = np.linspace(2.0, 8.0, 40)
        assert rp.dplnckel(wl, 500.0).shape == (40,)


class TestDplanck:
    def test_el_matches_dplnckel(self):
        wl = np.linspace(1.0, 10.0, 25)
        T = 700.0
        assert np.allclose(rp.dplanck(wl, T, 'el'), rp.dplnckel(wl, T))

    def test_all_valid_types_return_positive(self):
        wl = np.linspace(2.0, 5.0, 20)
        n  = np.linspace(1000.0, 3000.0, 20)
        f  = np.linspace(1e13, 1e14, 20)
        for spec, t in [('el', wl), ('ql', wl), ('en', n), ('qn', n), ('ef', f), ('qf', f)]:
            result = rp.dplanck(t, 1000.0, spec)
            assert np.all(result > 0), f'type={spec!r} returned non-positive values'

    def test_invalid_type_returns_negative_ones(self):
        wl = np.linspace(1.0, 5.0, 10)
        result = rp.dplanck(wl, 300.0, 'zz')
        assert np.all(result == -1.0)


# ── planck_integral (internal helper) ────────────────────────────────────────

class TestPlanckInternal:
    """planck_integral computes ∫ν_lo^∞ M dν in wavenumber space."""

    def test_radiant_full_range_approx_stefan_boltzmann(self):
        """planck_integral(ν≈0, T) - planck_integral(ν≈∞, T) ≈ σ·T⁴."""
        T = 1000.0
        lo = rp.planck_integral(1.0,     T, radiant=True)   # ν=1 cm⁻¹ ≈ 0
        hi = rp.planck_integral(200000.0, T, radiant=True)  # ν=200000 cm⁻¹ ≈ ∞
        integral = lo - hi
        assert integral == pytest.approx(const.sigma * T**4, rel=0.01)

    def test_photon_full_range_approx_stefan_boltzmann_q(self):
        T = 1000.0
        lo = rp.planck_integral(1.0,     T, radiant=False)
        hi = rp.planck_integral(200000.0, T, radiant=False)
        integral = lo - hi
        assert integral == pytest.approx(float(rp.stefanboltzman(T, 'q')), rel=0.01)


# ── planckInt ─────────────────────────────────────────────────────────────────

class TestPlanckInt:
    def test_en_wide_band_radiant_approx_stefan_boltzmann(self):
        """Wide wavenumber band 'en' ≈ σT⁴ within 1%."""
        T = 1000.0
        result = rp.planckInt(1.0, 200000.0, T, type='en')
        assert result == pytest.approx(const.sigma * T**4, rel=0.01)

    def test_qn_wide_band_photon_approx_stefan_boltzmann_q(self):
        T = 1000.0
        result = rp.planckInt(1.0, 200000.0, T, type='qn')
        assert result == pytest.approx(float(rp.stefanboltzman(T, 'q')), rel=0.01)

    def test_el_wide_band_uses_mock_convertSpectralDomain(self):
        """'el' input is wavelength — ryutils converts it to wavenumber internally."""
        T = 1000.0
        # wide wavelength range: 0.05 µm to 1000 µm
        result = rp.planckInt(0.05, 1000.0, T, type='el')
        assert result == pytest.approx(const.sigma * T**4, rel=0.02)

    def test_ql_wide_band(self):
        T = 1000.0
        result = rp.planckInt(0.05, 1000.0, T, type='ql')
        assert result == pytest.approx(float(rp.stefanboltzman(T, 'q')), rel=0.02)

    def test_zero_temperature_returns_none(self):
        assert rp.planckInt(1000.0, 5000.0, 0.0, type='en') is None

    def test_invalid_type_returns_none(self):
        assert rp.planckInt(1000.0, 5000.0, 300.0, type='zz') is None

    def test_reversed_limits_same_as_forward(self):
        """planckInt(lo, hi, T) == planckInt(hi, lo, T) — limits are auto-sorted."""
        T = 800.0
        fwd = rp.planckInt(500.0, 5000.0, T, type='en')
        rev = rp.planckInt(5000.0, 500.0, T, type='en')
        assert fwd == pytest.approx(rev, rel=1e-10)
