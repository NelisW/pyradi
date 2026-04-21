"""Regression tests for rydetector.py — PV infrared detector simulation."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import rydetector as ryd

# Physical constants (scipy.constants values, used for independent verification)
_e     = 1.602176634e-19    # electron charge [C]
_k     = 1.380649e-23       # Boltzmann [J/K]
_h     = 6.62607015e-34     # Planck [J·s]
_c     = 299792458.0        # speed of light [m/s]
_m_e   = 9.1093837015e-31   # electron mass [kg]
_sigma = 5.670374419e-8     # Stefan-Boltzmann [W/m²/K⁴]


# ── eVtoJoule / JouleTeEv ────────────────────────────────────────────────────

class TestEnergyConversion:
    def test_eVtoJoule_one_eV(self):
        """1 eV must equal the electron charge in Joules."""
        assert ryd.eVtoJoule(1.0) == pytest.approx(_e, rel=1e-6)

    def test_JouleTeEv_roundtrip(self):
        """Round-trip conversion must recover the original value."""
        for val in [0.1, 1.0, 2.5, 10.0]:
            assert ryd.JouleTeEv(ryd.eVtoJoule(val)) == pytest.approx(val, rel=1e-9)

    def test_eVtoJoule_zero(self):
        assert ryd.eVtoJoule(0.0) == 0.0

    def test_JouleTeEv_zero(self):
        assert ryd.JouleTeEv(0.0) == 0.0

    def test_eVtoJoule_array(self):
        arr = np.array([0.5, 1.0, 1.5])
        result = ryd.eVtoJoule(arr)
        assert result.shape == (3,)
        assert np.allclose(result, arr * _e)


# ── FermiDirac ────────────────────────────────────────────────────────────────

class TestFermiDirac:
    def test_at_fermi_level_is_half(self):
        """F-D probability at E = Ef is exactly 0.5 for any T > 0."""
        Ef = ryd.eVtoJoule(0.1)
        for T in [77, 200, 300]:
            assert ryd.FermiDirac(Ef, Ef, T) == pytest.approx(0.5, abs=1e-10)

    def test_well_below_fermi_level_approaches_one(self):
        """At E << Ef the occupancy approaches 1."""
        Ef = ryd.eVtoJoule(1.0)
        E  = ryd.eVtoJoule(0.0)
        val = ryd.FermiDirac(Ef, E, 300)
        assert val > 0.99

    def test_well_above_fermi_level_approaches_zero(self):
        """At E >> Ef the occupancy approaches 0."""
        Ef = ryd.eVtoJoule(0.0)
        E  = ryd.eVtoJoule(2.0)
        val = ryd.FermiDirac(Ef, E, 300)
        assert val < 0.01

    def test_output_in_unit_interval(self):
        Ef = ryd.eVtoJoule(0.3)
        E  = ryd.eVtoJoule(np.linspace(0.0, 0.6, 50))
        vals = ryd.FermiDirac(Ef, E, 300)
        assert np.all(vals >= 0) and np.all(vals <= 1)

    def test_array_input(self):
        Ef = ryd.eVtoJoule(0.1)
        E  = ryd.eVtoJoule(np.array([0.05, 0.1, 0.15]))
        result = ryd.FermiDirac(Ef, E, 300)
        assert result.shape == (3,)


# ── EgVarshni ─────────────────────────────────────────────────────────────────

class TestEgVarshni:
    # InSb-like parameters
    _E0 = 0.24
    _A  = 6e-4
    _B  = 500.0

    def test_returns_positive(self):
        Eg = ryd.EgVarshni(self._E0, self._A, self._B, 80.0)
        assert Eg > 0

    def test_decreases_with_temperature(self):
        """Varshni bandgap must decrease as temperature rises."""
        Eg_low  = ryd.EgVarshni(self._E0, self._A, self._B, 80.0)
        Eg_high = ryd.EgVarshni(self._E0, self._A, self._B, 300.0)
        assert Eg_low > Eg_high

    def test_near_zero_temperature_approaches_E0(self):
        """At T → 0 the Varshni result approaches E0."""
        Eg = ryd.EgVarshni(self._E0, self._A, self._B, 1e-6)
        assert Eg == pytest.approx(self._E0, rel=1e-4)

    def test_known_value(self):
        """Spot-check against manual calculation."""
        T = 80.0
        expected = self._E0 - (self._A * T**2 / (T + self._B))
        assert ryd.EgVarshni(self._E0, self._A, self._B, T) == pytest.approx(expected)


# ── Absorption ────────────────────────────────────────────────────────────────

class TestAbsorption:
    _wl     = np.linspace(1e-6, 6e-6, 200)    # wavelength [m]
    _Eg     = 0.24                              # bandgap [eV]
    _T      = 80.0
    _a0     = 1.9e4 * 100
    _a0p    = 800 * 100

    def test_output_shape(self):
        result = ryd.Absorption(self._wl, self._Eg, self._T, self._a0, self._a0p)
        assert result.shape == self._wl.shape

    def test_nonneg(self):
        result = ryd.Absorption(self._wl, self._Eg, self._T, self._a0, self._a0p)
        assert np.all(result >= 0)

    def test_above_bandgap_uses_eq35(self):
        """Above the bandgap the absorption should be larger (eq 3.5 dominant)."""
        wl_above = np.array([1e-6])   # very short wavelength → high photon energy
        wl_below = np.array([10e-6])  # very long wavelength → below bandgap
        a_above = ryd.Absorption(wl_above, self._Eg, self._T, self._a0, self._a0p)
        a_below = ryd.Absorption(wl_below, self._Eg, self._T, self._a0, self._a0p)
        assert a_above[0] > a_below[0]


# ── AbsorptionFile ────────────────────────────────────────────────────────────

class TestAbsorptionFile:
    def test_reads_and_interpolates(self, tmp_path):
        """AbsorptionFile must read a two-column file and return matching arrays."""
        # Write a minimal absorption data file (wavelength [um], absorption [m-1])
        data = np.column_stack([
            np.linspace(1.0, 5.0, 10),
            np.linspace(1e4, 5e4, 10),
        ])
        fpath = tmp_path / 'test_abs.txt'
        np.savetxt(str(fpath), data)

        wl = np.linspace(1.5, 4.5, 20)
        wl_out, absorb_out = ryd.AbsorptionFile(wl, str(fpath))
        assert wl_out.shape == absorb_out.shape
        assert len(wl_out) > 0

    def test_clips_to_file_range(self, tmp_path):
        """Wavelengths outside the file range must be silently excluded."""
        data = np.column_stack([np.linspace(2.0, 4.0, 5), np.ones(5) * 1e4])
        fpath = tmp_path / 'test_abs2.txt'
        np.savetxt(str(fpath), data)

        wl = np.linspace(1.0, 6.0, 50)   # extends well beyond file range
        wl_out, absorb_out = ryd.AbsorptionFile(wl, str(fpath))
        assert np.all(wl_out >= 2.0) and np.all(wl_out <= 4.0)


# ── QuantumEfficiency ─────────────────────────────────────────────────────────

class TestQuantumEfficiency:
    _wl    = np.linspace(1e-6, 6e-6, 200)
    _Eg    = 0.24
    _T     = 80.0
    _a0    = 1.9e4 * 100
    _a0p   = 800 * 100

    def test_output_shape(self):
        absorb = ryd.Absorption(self._wl, self._Eg, self._T, self._a0, self._a0p)
        qe = ryd.QuantumEfficiency(absorb, d1=0, d2=5e-6, theta1=0.0,
                                   nFront=1.0, nMaterial=3.42)
        assert qe.shape == self._wl.shape

    def test_in_zero_one_range(self):
        absorb = ryd.Absorption(self._wl, self._Eg, self._T, self._a0, self._a0p)
        qe = ryd.QuantumEfficiency(absorb, d1=0, d2=5e-6, theta1=0.0,
                                   nFront=1.0, nMaterial=3.42)
        assert np.all(qe >= 0) and np.all(qe <= 1.0 + 1e-9)

    def test_zero_thickness_gives_zero_qe(self):
        """d1 == d2 → no depletion layer → QE must be zero."""
        absorb = ryd.Absorption(self._wl, self._Eg, self._T, self._a0, self._a0p)
        qe = ryd.QuantumEfficiency(absorb, d1=5e-6, d2=5e-6, theta1=0.0,
                                   nFront=1.0, nMaterial=3.42)
        assert np.allclose(qe, 0, atol=1e-12)


# ── Responsivity ──────────────────────────────────────────────────────────────

class TestResponsivity:
    def test_output_shape(self):
        wl = np.linspace(1e-6, 5e-6, 50)
        qe = np.ones(50) * 0.8
        r = ryd.Responsivity(wl, qe)
        assert r.shape == (50,)

    def test_nonneg(self):
        wl = np.linspace(1e-6, 5e-6, 50)
        qe = np.linspace(0, 1, 50)
        assert np.all(ryd.Responsivity(wl, qe) >= 0)

    def test_proportional_to_wavelength(self):
        """At constant QE, responsivity ∝ wavelength."""
        qe = np.ones(2)
        wl = np.array([2e-6, 4e-6])
        r = ryd.Responsivity(wl, qe)
        assert r[1] == pytest.approx(2 * r[0], rel=1e-9)

    def test_zero_qe_gives_zero(self):
        wl = np.linspace(1e-6, 5e-6, 20)
        assert np.allclose(ryd.Responsivity(wl, np.zeros(20)), 0)


# ── DStar ─────────────────────────────────────────────────────────────────────

class TestDStar:
    def test_positive(self):
        resp = np.ones(50) * 1.0
        assert np.all(ryd.DStar(1e-8, 100.0, 1e-12, resp) > 0)

    def test_scales_with_area(self):
        """D* ∝ sqrt(area)."""
        resp = np.array([1.0])
        d1 = ryd.DStar(1e-8, 1.0, 1e-12, resp)
        d2 = ryd.DStar(4e-8, 1.0, 1e-12, resp)
        assert d2 == pytest.approx(2 * d1, rel=1e-9)


# ── NEP ───────────────────────────────────────────────────────────────────────

class TestNEP:
    def test_returns_two_arrays(self):
        resp = np.linspace(0.5, 2.0, 20)
        nep, det = ryd.NEP(1e-12, resp)
        assert nep.shape == resp.shape
        assert det.shape == resp.shape

    def test_detectivity_proportional_to_responsivity(self):
        """detectivity = responsivity / noise."""
        noise = 1e-12
        resp  = np.array([1.0, 2.0, 4.0])
        _, det = ryd.NEP(noise, resp)
        assert np.allclose(det, resp / noise)

    def test_nep_positive(self):
        resp = np.linspace(0.5, 2.0, 10)
        nep, _ = ryd.NEP(1e-12, resp)
        assert np.all(nep > 0)

    def test_zero_responsivity_handled(self):
        """Zero responsivity must not raise — NEP uses sys.float_info.max guard."""
        resp = np.array([0.0, 1.0])
        nep, det = ryd.NEP(1e-12, resp)
        assert np.all(np.isfinite(nep))


# ── Isaturation ───────────────────────────────────────────────────────────────

class TestIsaturation:
    # InSb-like parameters
    _kwargs = dict(
        mobE=100.0, tauE=1e-8, mobH=1.0, tauH=1e-8,
        me=0.014 * _m_e, mh=0.43 * _m_e,
        na=1e16, nd=1e16, Eg=0.24, tDetec=80.0, areaDet=(200e-6)**2,
    )

    def test_returns_tuple(self):
        result = ryd.Isaturation(**self._kwargs)
        assert len(result) == 2

    def test_I0_positive(self):
        I0, ni = ryd.Isaturation(**self._kwargs)
        assert I0 > 0

    def test_ni_positive(self):
        _, ni = ryd.Isaturation(**self._kwargs)
        assert ni > 0

    def test_zero_na_handled(self):
        """na=0 must not raise (electron term set to zero)."""
        kwargs = dict(self._kwargs, na=0)
        I0, ni = ryd.Isaturation(**kwargs)
        assert np.isfinite(I0)

    def test_zero_nd_handled(self):
        """nd=0 must not raise (hole term set to zero)."""
        kwargs = dict(self._kwargs, nd=0)
        I0, ni = ryd.Isaturation(**kwargs)
        assert np.isfinite(I0)


# ── IXV ───────────────────────────────────────────────────────────────────────

class TestIXV:
    def test_at_zero_bias_no_photo(self):
        """At V=0 with iPhoto=0, diode current is 0."""
        result = ryd.IXV(0.0, 1.0, 80.0, 0.0, 1e-10)
        assert result == pytest.approx(0.0, abs=1e-20)

    def test_at_zero_bias_with_photo(self):
        """At V=0, IXV = -iPhoto."""
        iPhoto = 1e-6
        result = ryd.IXV(0.0, 1.0, 80.0, iPhoto, 1e-10)
        assert result == pytest.approx(-iPhoto, rel=1e-6)

    def test_forward_bias_increases_current(self):
        """Forward bias must increase current above zero-bias value."""
        I0     = 1e-10
        iPhoto = 0.0
        I_zero = ryd.IXV(0.0,   1.0, 80.0, iPhoto, I0)
        I_fwd  = ryd.IXV(0.05,  1.0, 80.0, iPhoto, I0)
        assert I_fwd > I_zero

    def test_array_bias(self):
        V = np.linspace(-0.1, 0.1, 50)
        result = ryd.IXV(V, 1.0, 80.0, 0.0, 1e-10)
        assert result.shape == (50,)


# ── Noise ─────────────────────────────────────────────────────────────────────

class TestNoise:
    def test_returns_four_values(self):
        result = ryd.Noise(80.0, 1.7, 1e-10, 1e-7)
        assert len(result) == 4

    def test_all_positive(self):
        noise, R0, johnson, shot = ryd.Noise(80.0, 1.7, 1e-10, 1e-7)
        assert noise > 0 and R0 > 0 and johnson > 0 and shot > 0

    def test_total_noise_is_quadrature_sum(self):
        """Total noise = sqrt(johnson^2 + shot^2)."""
        noise, _, johnson, shot = ryd.Noise(80.0, 1.7, 1e-10, 1e-7)
        assert noise == pytest.approx(np.sqrt(johnson**2 + shot**2), rel=1e-9)

    def test_r0_increases_with_temperature(self):
        """Dynamic resistance R0 = IVbeta·kT/(Isat·e) scales linearly with temperature."""
        # Johnson PSD = 4kT/R0 = 4·Isat·e/IVbeta — temperature cancels, so it's
        # constant for fixed Isat. What does grow with T is R0 itself.
        _, R0_low,  _, _ = ryd.Noise( 80.0, 1.0, 1e-10, 0.0)
        _, R0_high, _, _ = ryd.Noise(300.0, 1.0, 1e-10, 0.0)
        assert R0_high == pytest.approx(R0_low * (300.0 / 80.0), rel=1e-6)


# ── DstarSpectralFlatPhotonLim ────────────────────────────────────────────────

class TestDstarSpectralFlatPhotonLim:
    def test_positive(self):
        assert ryd.DstarSpectralFlatPhotonLim(77.0, 300.0, 1.0) > 0

    def test_cooler_detector_gives_higher_dstar(self):
        """A colder detector has lower thermal noise → higher D*."""
        d_cold = ryd.DstarSpectralFlatPhotonLim( 77.0, 300.0, 1.0)
        d_warm = ryd.DstarSpectralFlatPhotonLim(300.0, 300.0, 1.0)
        assert d_cold > d_warm

    def test_scales_with_sqrt_epsilon(self):
        """D* ∝ sqrt(epsilon)."""
        d1 = ryd.DstarSpectralFlatPhotonLim(77.0, 300.0, 1.0)
        d4 = ryd.DstarSpectralFlatPhotonLim(77.0, 300.0, 4.0)
        assert d4 == pytest.approx(2 * d1, rel=1e-9)
