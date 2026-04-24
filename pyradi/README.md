# pyradi

General-purpose radiometry utility functions for the [PyRadi](https://github.com/NelisW/pyradi) toolkit.

## Refactoring status

Last updated: 2026-04-22  (all 18 files complete; 745 tests green)

### Completed

| File | Date |
|------|------|
| `ryutils.py` | 2026-04-19 |
| `06-Diverse-pyradi-utilities.ipynb` | 2026-04-19 |
| `ryfiles.py` | 2026-04-19 |
| `ry3dnoise.py` | 2026-04-20 |
| `rychroma.py` | 2026-04-20 |
| `rydetector.py` | 2026-04-20 |
| `rylookup.py` | 2026-04-20 |
| `rymodtran.py` | 2026-04-20 |
| `ryhdf5.py` | 2026-04-21 |
| `rypflux.py` | 2026-04-21 |
| `ryplanck.py` | 2026-04-21 |
| `ryprob.py` | 2026-04-21 |
| `ryplotspherical.py` | 2026-04-21 |
| `ryptw.py` | 2026-04-21 |
| `ryscene.py` | 2026-04-21 |
| `rystare.py` | 2026-04-21 |
| `rytarggen.py` | 2026-04-21 |
| `ryplot.py` | 2026-04-22 |

### Outstanding

_(none — all files complete)_

---

## Quick start

```bash
conda env create -f environment.yml
conda activate pyradi
python ryutils.py        # runs the built-in demo / self-test
python ry3dnoise.py      # runs the 3D noise demo (requires data files)
```

Run the unit tests:

```bash
pytest test/test_ryutils.py test/test_ryfiles.py test/test_ry3dnoise.py test/test_rychroma.py test/test_rydetector.py test/test_rylookup.py test/test_rymodtran.py test/test_ryhdf5.py test/test_rypflux.py test/test_ryplanck.py test/test_ryprob.py test/test_ryplotspherical.py test/test_ryptw.py test/test_ryscene.py test/test_rystare.py test/test_rytarggen.py test/test_ryplot.py -v
```

Or run the full suite in the conda environment in one command:

```bash
conda run -n pyradi --no-capture-output pytest test/ -v --tb=short
```

---

## Tech stack

| Package | Purpose |
|---------|---------|
| numpy | Array maths throughout |
| scipy | Interpolation, signal processing, constants |
| matplotlib | Plotting helpers and aperture patches |
| pillow | Image I/O in `drawCheckerboard` and `extractGraph` |
| ipython | `update_progress` progress bar, `VersionInformation` display |
| imageio | Raw frame I/O in `rawFrameToImageFile` |
| pandas | DataFrame merge utilities in `mergeDFS` |
| scikit-learn | Kernel density estimation in `ryprob` |
| h5py | HDF5 file read/write in `ryhdf5` |
| scikit-image | Medial-axis skeleton in `extractGraph`; bitmap image save in `ryhdf5.plotHDF5Bitmaps` |
| transformations | 3-D homogeneous transforms in `ryscene` (pip wheel) |

---

## Project structure

```
pyradi/
├── ry3dnoise.py           # 3D noise parameter analysis (D'Agostino & Webb)
├── ryfiles.py             # File I/O, text/LaTeX/raw-frame utilities
├── ryutils.py             # General radiometry utility functions
├── rychroma.py            # Chromaticity calculations
├── rydetector.py          # Detector modelling
├── rylookup.py            # Look-up table utilities
├── rymodtran.py           # MODTRAN interface
├── rypflux.py             # Photon flux / low-light radiance estimates
├── ryplanck.py            # Planck radiation functions
├── ryplot.py              # Plotting helpers
├── ryplotspherical.py     # Spherical plot utilities
├── ryprob.py              # Probability and statistics
├── ryptw.py               # PTW file format reader
├── ryscene.py             # Scene modelling
├── ryhdf5.py              # HDF5 file I/O utilities
├── rystare.py             # CCD/CMOS staring array sensor model
├── rytarggen.py           # HDF5 test image generation utilities
├── environment.yml        # Conda environment spec
├── README.md              # This file
└── test/
    ├── test_ry3dnoise.py   # pytest tests for ry3dnoise
    ├── test_rychroma.py    # pytest tests for rychroma
    ├── test_rydetector.py  # pytest tests for rydetector
    ├── test_ryfiles.py     # pytest tests for ryfiles
    ├── test_ryhdf5.py      # pytest tests for ryhdf5
    ├── test_rylookup.py    # pytest tests for rylookup
    ├── test_rypflux.py     # pytest tests for rypflux
    ├── test_ryplanck.py    # pytest tests for ryplanck
    ├── test_ryplotspherical.py  # pytest tests for ryplotspherical
    ├── test_ryprob.py      # pytest tests for ryprob
    ├── test_ryptw.py       # pytest tests for ryptw
    ├── test_ryscene.py     # pytest tests for ryscene
    ├── test_rystare.py     # pytest tests for rystare
    ├── test_rytarggen.py   # pytest tests for rytarggen
    ├── test_rymodtran.py   # pytest tests for rymodtran
    ├── test_ryutils.py     # pytest tests for ryutils
    └── test_ryplot.py      # pytest tests for ryplot
```

---

## Available scripts / commands

| Command | Description |
|---------|-------------|
| `python ry3dnoise.py` | Run the 3D noise demo (requires sensor data files) |
| `python rydetector.py` | Run the built-in detector modelling demo |
| `python ryfiles.py` | Run the built-in file I/O demo |
| `python ryhdf5.py` | Run the built-in HDF5 I/O demo |
| `python rylookup.py` | Run the built-in lookup-table demo |
| `python rymodtran.py` | Run the built-in MODTRAN reader demo |
| `python rypflux.py` | Run the built-in photon flux demo |
| `python ryplanck.py` | Run the built-in Planck radiation demo |
| `python ryplotspherical.py` | Run the built-in spherical trajectory demo (requires scipy) |
| `python ryprob.py` | Run the built-in probability demo |
| `python ryptw.py` | Run the built-in PTW file reader demo (requires data/*.ptw and calibration XML) |
| `python ryscene.py` | Run the built-in scene locus / SLERP demo (requires data/scene/sensorlocus.csv and transformations) |
| `python rystare.py` | Run the built-in CCD/CMOS sensor simulation demo (requires data/ image HDF5 files) |
| `python rytarggen.py` | Run the built-in HDF5 image generation demo (requires data/ raw image files) |
| `python rychroma.py` | Run the built-in chromaticity demo |
| `python ryutils.py` | Run the built-in utility demo and self-test examples |
| `python ryplot.py` | Run the built-in plotting demo (saves PNG files to cwd) |
| `pytest test/ -v` | Run the full regression test suite |
| `pytest test/test_ryutils.py -v` | Tests for ryutils only |
| `pytest test/test_ryfiles.py -v` | Tests for ryfiles only |
| `pytest test/test_ry3dnoise.py -v` | Tests for ry3dnoise only |
| `pytest test/test_rychroma.py -v` | Tests for rychroma only |
| `pytest test/test_rydetector.py -v` | Tests for rydetector only |
| `pytest test/test_rylookup.py -v` | Tests for rylookup only |
| `pytest test/test_rymodtran.py -v` | Tests for rymodtran only |
| `pytest test/test_ryhdf5.py -v` | Tests for ryhdf5 only |
| `pytest test/test_rypflux.py -v` | Tests for rypflux only |
| `pytest test/test_ryplanck.py -v` | Tests for ryplanck only |
| `pytest test/test_ryprob.py -v` | Tests for ryprob only |
| `pytest test/test_ryplotspherical.py -v` | Tests for ryplotspherical only |
| `pytest test/test_ryptw.py -v` | Tests for ryptw only |
| `pytest test/test_ryscene.py -v` | Tests for ryscene only |
| `pytest test/test_rystare.py -v` | Tests for rystare only |
| `pytest test/test_rytarggen.py -v` | Tests for rytarggen only |
| `pytest test/test_ryplot.py -v` | Tests for ryplot only |

---

## Public API summary

### ry3dnoise — 3D noise analysis

Based on the D'Agostino & Webb methodology (SPIE vol. 1488, 1991).
All functions operate on `np.ndarray` of shape `(frames, rows, cols)`.

#### Averaging operators

| Function | Description |
|----------|-------------|
| `oprDT(imgSeq)` | Average over frames → shape `(1, rows, cols)` |
| `oprDV(imgSeq)` | Average over rows → shape `(frames, 1, cols)` |
| `oprDH(imgSeq)` | Average over columns → shape `(frames, rows, 1)` |
| `oprSDT(imgSeq)` | Subtract temporal mean from each frame |
| `oprSDV(imgSeq)` | Subtract row mean from each pixel |
| `oprSDH(imgSeq)` | Subtract column mean from each pixel |

#### Noise components

Each returns `(scalar_std: float, component_image: np.ndarray)`.

| Function | Description |
|----------|-------------|
| `getS(imgSeq)` | Grand mean over all pixels and frames |
| `getNT(imgSeq)` | Temporal frame noise — frame-to-frame mean variation |
| `getNVH(imgSeq)` | Fixed spatial (pixel non-uniformity) noise |
| `getNTV(imgSeq)` | Row temporal noise — row-average variation per frame |
| `getNTH(imgSeq)` | Column temporal noise — column-average variation per frame |
| `getNV(imgSeq)` | Fixed row noise — row-average variation fixed in time |
| `getNH(imgSeq)` | Fixed column noise — column-average variation fixed in time |
| `getNTVH(imgSeq)` | Temporal pixel noise — random per-pixel, per-frame noise |
| `getTotal(imgSeq)` | Total system noise — RSS of all seven components |

---

### rychroma — CIE colour coordinate processing

Computes CIE 1931 chromaticity from arbitrary spectra and converts between colour spaces.
Colour-space conversion matrices (`mcieRGBtoXYZ`, `mXYZtoCIERGB`, `madobeRGBtoXYZ`,
`mXYZtoAdobeRGB`, `msRGBtoXYZ`, `mXYZtosRGB`) are available as module-level constants.

| Function | Description |
|----------|-------------|
| `chromaticityforSpectralL(spectral, radiance, xbar, ybar, zbar)` | CIE xy chromaticity and Y from a spectrum and tristimulus curves |
| `XYZforSpectralL(spectral, radiance, xbar, ybar, zbar)` | CIE XYZ tristimulus values from a spectrum |
| `loadCIEbar(specvec, stype)` | Load CIE 1931 x̄ ȳ z̄ functions interpolated to a wavelength or wavenumber vector |
| `rgb2CIExy(rgb, system)` | Convert RGB colours to CIE xy — systems: `'CIE RGB'`, `'Adobe RGB'`, `'sRGB'` |
| `CIExy2rgb(xy, system)` | Convert CIE xy coordinates to RGB — same systems as above |

---

### rydetector — PV infrared photodetector simulation

Models photovoltaic infrared detectors using classical semiconductor equations
(Dereniak & Boreman, Rogalski). Uses InSb parameters by default; any III-V
compound can be substituted by changing material parameters. Requires
`scipy.constants` for physical constants.

| Function | Description |
|----------|-------------|
| `eVtoJoule(EeV)` | Convert energy from eV to Joules |
| `JouleTeEv(EJ)` | Convert energy from Joules to eV |
| `FermiDirac(Ef, EJ, T)` | Fermi-Dirac occupancy probability |
| `EgVarshni(E0, VarshniA, VarshniB, tempDet)` | Bandgap at detector temperature via the Varshni equation [eV] |
| `Absorption(wavelength, Eg, tempDet, a0, a0p)` | Spectral absorption coefficient [m⁻¹] (Dereniak Eq 3.5/3.6) |
| `AbsorptionFile(wavelength, filename)` | Read absorption coefficient from file and interpolate onto wavelength vector |
| `QuantumEfficiency(absorption, d1, d2, theta1, nFront, nMaterial)` | Spectral quantum efficiency for a depletion-layer detector |
| `Responsivity(wavelength, quantumEffic)` | Spectral responsivity [A/W] |
| `DStar(areaDet, deltaFreq, iNoise, responsivity)` | Spectral D* [cm√Hz/W] |
| `NEP(iNoise, responsivity)` | Noise equivalent power [W] and detectivity |
| `Isaturation(mobE, tauE, mobH, tauH, me, mh, na, nd, Eg, tDetec, areaDet)` | Reverse saturation current I₀ [A] and intrinsic carrier concentration nᵢ [m⁻³] |
| `IXV(V, IVbeta, tDetec, iPhoto, I0)` | Diode I–V curve including photocurrent [A] |
| `Noise(tempDet, IVbeta, Isat, iPhoto, vBias)` | Noise PSD [A/√Hz]: total, R₀, Johnson only, shot only |
| `DstarSpectralFlatPhotonLim(Tdetec, Tenvironment, epsilon)` | Photon-noise-limited D* for a spectrally flat detector [cm√Hz/W] |

---

### rylookup — radiometric lookup tables

Bidirectional lookup between source temperature, spectral radiance, and sensor signal
(digital level). A single class `RadLookup` builds high-resolution lookup tables from
spectral curves and optional camera calibration data; supports two-point internal-temperature
interpolation for calibrated sensors.

**Constructor**

```python
RadLookup(specName, nu, tmprLow, tmprHi, tmprInc,
          sensorResp=None, opticsTau=None, filterTau=None, atmoTau=None,
          sourceEmis=None, sigMin=None, sigMax=None, sigInc=None,
          dicCaldata=None, dicPower=None, dicFloor=None)
```

Spectral parameters may be a filename string, a `np.ndarray` of shape `(N,1)`, or
`None` (unity). Calibration parameters (`sigMin`/`sigMax`/`sigInc`/`dicCaldata`) are
optional; omitting them gives radiance-only lookups.

**Lookup methods**

| Method | Description |
|--------|-------------|
| `LookupTempRad(temperature, withFilter=True)` | Temperature [K] → radiance [W/(sr·m²)] |
| `LookupRadTemp(radiance, withFilter=True)` | Radiance → temperature [K] |
| `LookupDLRad(DL, Tint)` | Sensor signal + internal temperature → radiance (calibration required) |
| `LookupDLRadHelper(DL, paraK)` | Low-level DL→radiance with pre-computed Tint parametric |
| `LookupDLTemp(DL, Tint)` | Sensor signal + internal temperature → temperature K (calibration required) |
| `Info()` | Return human-readable summary string of loaded data and coefficients |

**Plot methods** (`saveFig=False` by default; all accept `savePath`, `saveExt`, `saveFig`)

| Method | Description |
|--------|-------------|
| `PlotSpectrals(...)` | All five spectral curves on one graph |
| `PlotCalSpecRadiance(...)` | Spectral radiance at each calibration temperature |
| `PlotCalDLRadiance(...)` | Digital level vs radiance with calibration fit lines |
| `PlotTempRadiance(withFilter, ...)` | Temperature vs radiance (with or without filter) |
| `PlotCalTempRadiance(...)` | Calibration temperature vs radiance per Tint |
| `PlotCalTintRad(...)` | Internal instrument temperature vs optics radiance |
| `PlotCalDLTemp(...)` | Digital level vs temperature per Tint |
| `PlotCalTempDL(...)` | Temperature vs digital level per Tint |

---

### rymodtran — MODTRAN tape7 reader and utilities

Reads MODTRAN tape7 output files for all four IEMSCT modes (transmission, thermal,
full radiance, solar) and provides helpers for automated MODTRAN parametric studies.

| Function | Description |
|----------|-------------|
| `fixHeaders(instr)` | Replace special characters (`+`, `-`, `[`, `]`, `@`) in a column header string so it is a valid Python identifier |
| `fixHeadersList(headers)` | Apply `fixHeaders` to every element of a list |
| `loadtape7(filename, colspec)` | Read a tape7 file and return a 2-D numpy array with the requested columns (supports IEMSCT 0–3) |
| `variationTape5File(scenario, variation, tape5, ...)` | Copy a tape5 template into a new sub-directory with one or more numeric values substituted; returns the variation directory name |
| `runModtranAndCopy(root, research, pathToModtranBin, execname, ...)` | Find tape5 input files recursively, run MODTRAN, and copy tape6/7/8 outputs back to the input directory |
| `runModtranModrootIn(root, filerootname, pathToModtranBin, execname, ...)` | Same as above but drives MODTRAN via a `modroot.in` file (MODTRAN 6 convention) |

**Key classes (internal, used by `loadtape7`)**

| Class | Description |
|-------|-------------|
| `extracttape6` | Parses a MODTRAN tape6 file to extract atmospheric profile data (temperature, pressure, humidity, aerosols) as a list of dicts |

---

### ryhdf5 — HDF5 file I/O utilities

Thin wrappers around [h5py](https://docs.h5py.org) for opening, inspecting, and
visualising HDF5 files. Plotting functions depend on `ryplot` (lazy import) and are
only needed when visualisation is requested.

| Function | Description |
|----------|-------------|
| `open_HDF(filename, mode='r')` | Open an existing HDF5 file and return the `h5py.File` object; default mode is read-only |
| `erase_create_HDF(filename)` | Delete `filename` if it exists then create and return a new writable `h5py.File` |
| `get_HDF_DS(hdf5File, path)` | Return the dataset at `path` inside an open HDF5 file, or `None` if the path is absent |
| `print_HDF5_text(vartext)` | Print a string — used as an h5py visitor callback to list hierarchy names |
| `print_HDF5_dataset_value(var, obj)` | Print the name of `var` when `obj` is a `Dataset` (silent for Groups) — h5py visitor callback |
| `get_HDF_branches(hdf5File)` | Visit every node in the file and print its path (note: contains a known bug — raises `AttributeError` on non-empty files because it passes itself as the visitor callable) |
| `plotHDF5Bitmaps(hfd5f, prefix, pformat, lstimgs, debug)` | Save each listed dataset as a scaled uint8 bitmap (PNG/JPEG); skips scalars and all-zero arrays |
| `plotHDF5Images(hfd5f, prefix, colormap, cbarshow, lstimgs, logscale, debug)` | Plot each listed dataset as a false-colour image using `ryplot.savePlot`; supports log scale |
| `plotHDF5Histograms(hfd5f, prefix, format, lstimgs, bins)` | Plot a histogram of each listed dataset using `ryplot.savePlot` |

---

### rypflux — photon flux estimates for low-light conditions

Provides order-of-magnitude estimates of photon flux radiance across natural-light
illumination levels from sunlight down to overcast night.  The module contains a single
class `PFlux`; the companion `ryplanck` and `ryutils` modules supply the underlying
spectral and luminous-efficiency calculations.

#### PFlux

| Method / attribute | Description |
|--------------------|-------------|
| `PFlux()` | Constructor — populates `lllux` (ten built-in illumination conditions), `llluxCols`, and `specranges` (five default spectral bands: VIS, NIR, SWIR, MWIR, LWIR) |
| `lllPhotonrates(specranges=None)` | Compute spectrally integrated photon radiance for every illumination condition; returns a `pd.DataFrame` indexed by condition name with columns for `k` (luminance-to-photon-rate scale factor) and `Radiance-q/(s.m2.sr)-<band>` for each band |
| `lllux` | `dict` mapping condition name → `[irradiance_lm_m2, colour_temp_K, photopic_fraction]` |
| `specranges` | `dict` mapping band name → `[spectral_vector, response_vector, type_str]`; `type_str` is `'wl'` (µm) or `'wn'` (cm⁻¹) |
| `dfPhotRates` | `pd.DataFrame` cached after the last `lllPhotonrates()` call |

---

### ryplanck — Planck blackbody radiation functions

Spectral radiance and photon-rate functions for blackbody sources over wavelength
(µm), wavenumber (cm⁻¹), and frequency (Hz) domains.  All spectral functions are
decorated with `fixDimensions` which normalises scalar and vector inputs and returns
output of the expected shape: scalar → scalar, (N,) × scalar → (N,), scalar × (M,) →
(1, M), (N,) × (M,) → (N, M).

#### Physical constants — `PlanckConstants`

Module-level instance `pconst = PlanckConstants()` exposes all constants as
attributes.  Call `pconst.printConstants()` to list them.

| Attribute | Value | Description |
|-----------|-------|-------------|
| `c1` | 3.741 774 9 × 10⁻¹⁶ W·m² | First radiation constant (energy domain) |
| `c1q` | 1.191 042 9 × 10⁻¹² q·m²/s | First radiation constant (photon domain) |
| `c2` | 1.438 776 9 × 10⁻² m·K | Second radiation constant |
| `smalls` | 5.670 37 × 10⁻⁸ W/(m²·K⁴) | Stefan–Boltzmann constant |
| `smallsq` | 1.520 5 × 10¹⁵ q/(s·m²·K³) | Stefan–Boltzmann constant (photon) |
| `c3e` | 2 897.8 µm·K | Wien displacement (energy, wavelength) |
| `c3q` | 3 669.7 µm·K | Wien displacement (photon, wavelength) |
| `c3en` | 1.9610 cm⁻¹/K | Wien displacement (energy, wavenumber) |
| `an` | series coefficients | Bernoulli-series coefficients for `planck_integral` |

#### Module-level spectral radiance functions

All accept `spectral` (scalar or 1-D array) and `temperature` (scalar or 1-D array).
Return shape follows `fixDimensions` rules.

| Function | Input domain | Spectral variable | Output units |
|----------|-------------|-------------------|--------------|
| `planckel(spectral, temperature)` | wavelength [µm] | λ | W/(m²·sr·µm) |
| `planckef(spectral, temperature)` | frequency [Hz] | ν | W/(m²·sr·Hz) |
| `plancken(spectral, temperature)` | wavenumber [cm⁻¹] | ñ | W/(m²·sr·cm⁻¹) |
| `planckql(spectral, temperature)` | wavelength [µm] | λ | q/(s·m²·sr·µm) |
| `planckqf(spectral, temperature)` | frequency [Hz] | ν | q/(s·m²·sr·Hz) |
| `planckqn(spectral, temperature)` | wavenumber [cm⁻¹] | ñ | q/(s·m²·sr·cm⁻¹) |
| `dplnckel(spectral, temperature)` | wavelength [µm] | λ | ∂L/∂T [W/(K·m²·sr·µm)] |
| `dplnckef(spectral, temperature)` | frequency [Hz] | ν | ∂L/∂T [W/(K·m²·sr·Hz)] |
| `dplncken(spectral, temperature)` | wavenumber [cm⁻¹] | ñ | ∂L/∂T [W/(K·m²·sr·cm⁻¹)] |
| `dplnckql(spectral, temperature)` | wavelength [µm] | λ | ∂Lq/∂T [q/(K·s·m²·sr·µm)] |
| `dplnckqf(spectral, temperature)` | frequency [Hz] | ν | ∂Lq/∂T [q/(K·s·m²·sr·Hz)] |
| `dplnckqn(spectral, temperature)` | wavenumber [cm⁻¹] | ñ | ∂Lq/∂T [q/(K·s·m²·sr·cm⁻¹)] |

#### Integrated and convenience functions

| Function | Description |
|----------|-------------|
| `planck(spectral, temperature, type)` | Dispatch to the correct spectral function by `type` string: `'el'`, `'ef'`, `'en'`, `'ql'`, `'qf'`, `'qn'` |
| `dplanck(spectral, temperature, type)` | Dispatch to the correct derivative function by `type` string |
| `planckInt(temperature, type)` | Integrate Planck radiance from 0 to ∞ (Stefan–Boltzmann); `type` values: `'el'`, `'ql'`, `'ef'`, `'qf'`, `'en'`, `'qn'` |
| `planck_integral(lowval, temperature, n)` | Bernoulli-series integral of Planck function from `lowval` (wavenumber, cm⁻¹) to ∞; `n` is the number of series terms (default 2) |

---

### ryprob — probability utility functions

Provides random-variate generators for several continuous distributions, a bounded
Kernel Density Estimation wrapper around scikit-learn, and a uniform-on-sphere point
generator.  Most functions are consumed by the high-level staring-array model `rystare`.

> **Note — pre-existing dispatcher behaviour**: `distributions_generator` always
> forwards `distribParams=None` to each sub-function, so every distribution uses its
> own built-in defaults regardless of what the caller passes.  This is preserved
> behaviour.

#### Distribution generators (direct call)

Each function takes a pre-allocated `out` array (whose shape defines the sample size)
and returns the filled array of random variates.

| Function | Default params | Output range |
|----------|---------------|--------------|
| `distribution_exp(distribParams, out, funcName)` | `[1.]` (lambda=1) | y ≥ 0 |
| `distribution_lognormal(distribParams, out, funcName)` | `[0., 1.]` (mu=0, sigma=1) | y > 0 |
| `distribution_inversegauss(distribParams, out, funcName)` | `[0., 1.]` (mu=0 fails > 0 → zeros) | y > 0 |
| `distribution_logistic(distribParams, out, funcName)` | `[0., 1.]` (a=0, k=1) | −∞ < y < ∞ |
| `distribution_wald(distribParams, out, funcName)` | `[0.]` (chi=0 fails > 0 → zeros) | y > 0 |

#### Dispatcher

| Function | Description |
|----------|-------------|
| `distributions_generator(distribName, distribParams, sampleSize)` | Generate random variates by distribution name; `sampleSize=None` → scalar, `int N` → N×N array, `tuple/list` → shaped array. Accepted names: `'exp'`/`'exponential'`, `'lognorm'`/`'lognormal'`/`'cobbdouglas'`/`'antilognormal'`, `'ig'`/`'inversegauss'`/`'invgauss'`, `'logistic'`, `'wald'` |

#### KDE

| Function | Description |
|----------|-------------|
| `KDEbounded(x_d, x, bandwidth, lowbnd, uppbnd, kernel)` | Kernel Density Estimation with optional boundary reflection to correct edge bias; uses scikit-learn `KernelDensity`. Returns `(x_d, pdf, bandwidth, kernel)`. Pass `bandwidth=np.nan` to auto-select via cross-validation (slow). |

#### Validation helpers

| Function | Description |
|----------|-------------|
| `validateParam(funcName, distribName, runDistribName, distribParamsName, paramName, param, conditionStr)` | Validate a scalar parameter against a list of condition strings (e.g. `['> 0', '<= 1']`); returns `bool` |
| `checkParamsNum(funcName, distribName, runDistribName, distribParams, correctNum)` | Check that `len(distribParams)` is in `correctNum`; returns `bool` |

#### Geometry

| Function | Description |
|----------|-------------|
| `unifomOnSphere(number, dimension=3)` | Generate `number` unit-norm vectors uniformly distributed on an n-dimensional sphere; returns `np.array` of shape `(number, dimension)` |

---

### ryplotspherical — spherical trajectory utilities for OSSIM

Reads and writes [Object-oriented Simple Simulator (OSSIM)](https://ossim.trac.sourceforge.net/)
trajectory files based on the OFF (Object File Format) mesh format.  Generates 6-DOF
(x, y, z, roll, pitch, yaw) trajectory tables either from an OFF mesh file or from an
explicit elevation/azimuth grid.

> **Note**: Mayavi-based visualisation functions (`plotSpherical`, `plotOSSIMSpherical`,
> `sphericalPlotElevAzim`, etc.) are present as commented-out stubs; they require an
> optional Mayavi installation and are not part of the public API.

| Function | Description |
|----------|-------------|
| `readOffFile(filename)` | Read an OFF file; return `(vertices, triangles)` as numpy arrays, or `(None, None)` if the header is invalid |
| `getRotateFromOffFile(filename, xPos, yPos, zPos)` | Build a Rotate-type 6-DOF trajectory (sensor fixed at `(xPos, yPos, zPos)`, target at each mesh vertex); return `(x, y, z, roll, pitch, yaw, vertices, triangles)` |
| `getOrbitFromOffFile(filename, xTarget, yTarget, zTarget, distance)` | Build an Orbit-type 6-DOF trajectory (sensor placed at `distance` from target along each mesh vertex direction); same 8-tuple return |
| `writeOSSIMTrajOFFFile(filename, trajtype, numSamples, xTarget, yTarget, zTarget, ...)` | Write OSSIM `.lut`, Vertices `.dat`, and Triangles `.dat` files from an OFF mesh; `trajtype` is `'Rotate'` or `'Orbit'` |
| `getRotateFromElevAzim(azimuth, elevation, xPos, yPos, zPos)` | Build a Rotate-type trajectory from azimuth/elevation grid arrays; return `(x, y, z, roll, pitch, yaw, azel)` where `azel` is `(N, 2)` |
| `getOrbitFromElevAzim(azimuth, elevation, xTarget, yTarget, zTarget, distance)` | Build an Orbit-type trajectory from azimuth/elevation grid; same 7-tuple return |
| `writeOSSIMTrajElevAzim(numSamples, filename, trajtype, ...)` | Write OSSIM `.lut` and AzelAngles `.dat` files from an elevation/azimuth grid; `trajtype` is `'Rotate'` or `'Orbit'` |

---

### ryptw — Agema/Cedip/FLIR PTW thermal-image file reader

Reads binary PTW files produced by FLIR Jade and related cameras (Agema, Cedip formats).
Provides low-level byte decoders, a header parser, and frame-extraction functions.
Also includes `JadeCalibrationData`, which loads an XML calibration file and builds a
`RadLookup` table for DL ↔ temperature / radiance conversion.

> **Note**: The copyright to the proprietary PTW file format remains the property of
> FLIR Inc. This reader was released with FLIR's permission.

#### Low-level byte decoders

| Function | Description |
|----------|-------------|
| `myint(x)` | Decode 2-byte little-endian signed int from bytes slice |
| `mylong(x)` | Decode 4-byte little-endian unsigned int (note: byte-3 shift is `<<32` not `<<24` — safe for values < 2²⁴) |
| `myfloat(x)` | Decode 4-byte IEEE 754 single-precision float |
| `mydouble(x)` | Decode 8-byte IEEE 754 double-precision float |
| `mybyte(x)` | Decode single unsigned byte (0–255) from a length-1 bytes slice |
| `mypass(x)` | Identity — return *x* unchanged |
| `myRGB(x)` | Decode 3 bytes (R, G, B) to an `'#rrggbb'` hex string |
| `terminateStrOnZero(str)` | Intended to truncate bytes at the first null; returns the full input in Python 3 due to a pre-existing int/str comparison bug |

#### Header and frame I/O

| Function / Class | Description |
|------------------|-------------|
| `PTWFrameInfo` | Data container for PTW file-level header and per-frame fields; populated by `readPTWHeader` and `GetPTWFrameFromFile` |
| `readPTWHeader(ptwfilename)` | Read the main PTW header and return a populated `PTWFrameInfo` object |
| `GetPTWFrameFromFile(header)` | Load the frame at `header.h_framepointer` into `header.data`; update frame-time and FPA-temperature fields; return the updated header |
| `getPTWFrame(header, frameindex)` | Retrieve one frame by 1-based index; return `(data, header)` or `np.asarray([0])` on error |
| `getPTWFrames(header, loadFrames)` | Retrieve multiple frames by index list; return `(data, fheaders)` of shape `(N, rows, cols)` or `(np.asarray([0]), None)` on error |
| `showHeader(Header)` | Print all header fields to stdout |

#### Calibration

| Class / Method | Description |
|----------------|-------------|
| `JadeCalibrationData(filename, datafileroot)` | Load XML calibration file; build `RadLookup` table (`self.LU`) covering all instrument temperatures in the file |
| `JadeCalibrationData.Info()` | Return a human-readable summary string of the loaded calibration parameters |

---

### ryscene — scene orientation locus modelling

Models a sensor's 6-DOF orientation locus in world space using quaternion SLERP
interpolation.  Depends on the ``transformations`` wheel by Christoph Gohlke.

**Coordinate system:** Right-Hand, North (x), East (y), Down (z).
Euler sequence: yaw (z) → pitch (y) → roll (x), encoded as ``'rzyx'``.

| Name | Type | Description |
|------|------|-------------|
| `df_euler2quat(row)` | function | Convert a DataFrame row's Euler angles to a ``[w,x,y,z]`` quaternion wrapped in a list; use with `DataFrame.apply` |
| `Scene` | class | Reads a time-stamped CSV locus, converts Euler angles to quaternions, and SLERP-interpolates orientation at arbitrary times |
| `Scene.readLocus(filename)` | method | Load locus CSV; add ``yawRad``, ``pitchRad``, ``rollRad``, ``sequence``, ``quat`` columns |
| `Scene.interpLocus(ltime)` | method | Return ``(yaw, pitch, roll, q)`` interpolated at ``ltime`` seconds |

**Evaluation helpers** (not public API — for use from ``__main__``):

| Function | Description |
|----------|-------------|
| `unitxrotate(sc)` | Print rotation quaternion and rotated unit-x vector for each locus pose |
| `applyInterpLocus(sc, t)` | Return a labelled Series ``{yaw, pit, rol, quat}`` for use with ``DataFrame.apply`` |
| `locusPlot(sc, ryplot)` | Plot interpolated Euler angles over the full time span; saves ``ryscene-quatslerp.png`` |

---

### ryfiles — file I/O and text utilities

| Function | Description |
|----------|-------------|
| `saveHeaderArrayTextFile(filename, dataArray, ...)` | Write a numpy array to a text file with optional header |
| `loadColumnTextFile(filename, ...)` | Load a column-format text file into a numpy array |
| `loadHeaderTextFile(filename, ...)` | Load a text file that has a header block |
| `cleanFilename(inputFilename)` | Strip non-safe characters from a filename |
| `listFiles(root, patterns, ...)` | Recursively list files matching glob or regex patterns |
| `readRawFrames(filename, rows, cols, vartype, ...)` | Read frames from a raw binary image file |
| `writeRawFrames(filename, frames)` | Write frames to a raw binary image file |
| `rawFrameToImageFile(filename, image, ...)` | Save a raw frame as a PNG/TIFF image |
| `arrayToLaTex(filename, arr, ...)` | Write a numpy array as a LaTeX tabular environment |
| `epsLaTexFigure(filename, epsfile, ...)` | Generate a LaTeX figure block for an EPS file |
| `execOnFiles(wildcard, function, ...)` | Apply a function to every file matching a wildcard |
| `read2DLookupTable(filename)` | Read a 2-D lookup table from a text file |
| `downloadFileUrl(url, filename)` | Download a file from a URL |
| `unzipGZipfile(gzipfile, ...)` | Decompress a gzip archive |
| `untarTarfile(tarfilename, ...)` | Extract a tar archive |
| `downloadUntar(url, ...)` | Download and extract a tar.gz archive |
| `mergeDFS(df1, df2, ...)` | Merge two time-indexed DataFrames by interpolation |
| `latex_escape(text)` | Escape special LaTeX characters in a string |

---

### ryutils — general radiometry utilities

#### Spectral domain

| Function | Description |
|----------|-------------|
| `convertSpectralDomain(domain, type)` | Convert between wavelength [µm], wavenumber [cm⁻¹] and frequency [Hz] |
| `convertSpectralDensity(domain, quantity, type)` | Convert spectral density between domains |
| `effectiveValue(domain, signal, baseline)` | Weighted spectral integral (normalised) |
| `sfilter(spectral, center, width, ...)` | Maximally-flat bandpass / lowpass / highpass filter |
| `responsivity(wavelength, lwavepeak, ...)` | Photon-detector spectral responsivity shape |
| `convolve(inspectral, ...)` | Spectral convolution with configurable window |
| `savitzkyGolay1D(y, window_size, order)` | Savitzky–Golay smoothing filter |

#### Coordinate transforms

| Function | Description |
|----------|-------------|
| `cart2polar(x, y)` | Cartesian → polar |
| `polar2cart(r, theta)` | Polar → Cartesian |
| `framesFirst(seq)` | Reorder 3-D image sequence: frames to axis 0 |
| `framesLast(seq)` | Reorder 3-D image sequence: frames to axis 2 |
| `index_coords(data, origin)` | Generate (x, y) coordinate arrays for an array |

#### Detection statistics

| Function | Description |
|----------|-------------|
| `detectFARThresholdToNoisepulseWidth(T2N, pw)` | FAR from threshold-to-noise and pulse width |
| `detectThresholdToNoiseTpFAR(pw, FAR)` | Threshold-to-noise from pulse width and FAR |
| `detectSignalToNoiseThresholdToNoisePd(T2N, Pd)` | SNR from threshold-to-noise and Pd |
| `detectThresholdToNoiseSignalToNoisepD(SNR, Pd)` | Threshold-to-noise from SNR and Pd |
| `detectProbabilityThresholdToNoiseSignalToNoise(T2N, SNR)` | Pd from threshold-to-noise and SNR |
| `rangeEquation(I, E, rangeTab, tauTab)` | Solve point-source range equation |

#### Image / geometry

| Function | Description |
|----------|-------------|
| `circ(x, y, d)` | Circular aperture array |
| `rect(x, y, sx, sy)` | Rectangular aperture array |
| `poissonarray(inp, ...)` | Poisson (shot) noise for image arrays |
| `drawCheckerboard(rows, cols, ...)` | Generate a checkerboard test pattern |
| `extractGraph(filename, xmin, xmax, ...)` | Digitise a graph from a bitmap image |
| `makemotionsequence(imgfile, mtnfile, ...)` | Build video frames from a still + motion file |
| `draw_siemens_star(outfile, n, dpi)` | Render a Siemens star MTF test pattern |
| `calcMTFwavefrontError(sample, wfdisplmnt, ...)` | Compute MTF from wavefront error map |

#### Atmosphere / photometry

| Function | Description |
|----------|-------------|
| `abshumidity(T)` | Absolute humidity [g/m³] from temperature [K] |
| `TFromAbshumidity(AH)` | Temperature [K] from absolute humidity |
| `luminousEfficiency(vlamtype, wavelen)` | CIE photopic / scotopic luminous efficiency V(λ) |

---

### rystare — CCD/CMOS staring array sensor model

High-level simulation of the complete photosensor signal chain: photon irradiance →
electrons → voltage → digital numbers. Based on Konnik & Willers (SPIE Proc 10036).
All major noise sources (shot noise, dark current, PRNU, kTC reset noise, source-follower
noise, ADC non-linearity) are modelled. Parameters are passed via an HDF5 file handle.

#### Sensor pipeline (HDF5-based)

| Function | Description |
|----------|-------------|
| `photosensor(strh5, initialise)` | Top-level pipeline: drives the full photon-to-DN conversion |
| `set_photosensor_constants(strh5, initialise)` | Compute band-gap energy and store fundamental constants |
| `check_create_datasets(strh5, initialise)` | Allocate all image-sized arrays in the HDF5 store |
| `source_follower(strh5, initialise)` | Apply source-follower gain and optional V/V non-linearity |
| `fixed_pattern_offset(strh5)` | Add CMOS dark fixed-pattern offset (column FPN) |
| `cds(strh5)` | Apply correlated double sampling gain |
| `adc(strh5)` | Analogue-to-digital conversion with optional ILE non-linearity |
| `charge_to_voltage(strh5)` | Sense-node charge-to-voltage conversion with optional V/e non-linearity |
| `sense_node_reset_noise(strh5, seed)` | Compute kTC sense-node reset noise matrix |
| `dark_current_and_dark_noises(strh5)` | Add dark current, dark shot noise, and dark FPN |
| `source_follower_noise(strh5, initialise)` | Compute source-follower noise spectrum (white + 1/f + RTS) |
| `multiply_detector_area(strh5)` | Convert electron-rate irradiance to electron rate per pixel |
| `multiply_integration_time(strh5)` | Convert electron rate to accumulated electron count |
| `convert_to_electrons(strh5)` | Multiply photon rate by quantum efficiency to get electron rate |
| `shotnoise(sensor_signal_in)` | Apply Poisson shot noise to a signal array |
| `responsivity_FPN_light(strh5)` | Apply photo-response non-uniformity (PRNU) |
| `responsivity_FPN_dark(strh5)` | Apply dark-current fixed-pattern noise |
| `FPN_models(rows, cols, noisetype, dist, spread, filter_params)` | Generate FPN noise matrix (Janesick-Gaussian, LogNormal, Wald, AR-ElGamal) |

#### Standalone radiometry helpers

| Function | Description |
|----------|-------------|
| `nEcntLLightDF(tauAtmo, ..., dfPhotRates)` | Electron count from a low-light photon-radiance DataFrame |
| `nEcntLLightPhotL(tauAtmo, ..., photRadiance)` | Electron count from a scalar in-band photon radiance |
| `nElecCntThermalScene(wl, tmptr, emis, ...)` | Electron count from a thermal scene source (wl in um) |
| `nEcntThermalOptics(wl, tmptrOpt, ...)` | Electron count from self-emission of warm optics (wl in um) |
| `nElecCntReflSun(wl, tauSun, ...)` | Electron count from reflected sunlight (wl in um) |
| `darkcurrentnoise(inttime, detarea, temptr, Egap, DFM)` | Dark-current noise as electron count (rms) |
| `kTCnoiseCsn(temptr, sensecapacity)` | kTC reset noise from sense-node capacitance |
| `kTCnoiseGv(temptr, gv)` | kTC reset noise from sense-node gain |
| `define_metrics()` | Return SI scaling factors (m, cm, mm, um, nm, rad, mrad) |
| `limitzero(a, thr)` | Asymmetric clipping to prevent negative values in FPN arrays |
| `run_example(doTest, outfilename, ...)` | Run a Simple or Advanced sensor simulation example |
| `get_summary_stats(hdffilename)` | Print summary statistics from a completed simulation HDF5 file |

#### Utilities

| Function | Description |
|----------|-------------|
| `buildLogSpace(Vmin, Vmax, nDec)` | Log-spaced vector |
| `solidAngleSquare(width, breadth, height, stype)` | Solid angle of a rectangular flat surface |
| `upMu()` | LaTeX micro symbol |

---

### rytarggen -- HDF5 test image generation utilities

Creates synthetic HDF5 image files for use as inputs to the pyradi sensor model
(rystare). Each image type is generated by calling create_HDF5_image with
a corresponding generating function. The module also provides wavelength-to-lux
and wavelength-to-temperature interpolation helpers.

#### HDF5 helper

| Function | Description |
|----------|-------------|
| `assigncheck(hdf5, path, value)` | Write value to path in an open HDF5 file; creates a new dataset if the path does not exist, overwrites in-place if it does |

#### Image generating functions (pass as fn to create_HDF5_image)

| Function | Description |
|----------|-------------|
| `hdf_Uniform(imghd5, rad_dynrange)` | Fill the image with a constant photon-rate value equal to rad_dynrange |
| `hdf_disk_photon(imghd5, rad_min, rad_dynrange, fracdiameter, fracblur)` | Illuminated circle with blurred boundary; fracdiameter and fracblur are fractions of the minimum image dimension |
| `hdf_stairs(imghd5, rad_min, rad_dynrange, steps, imtype)` | Staircase image; imtype is stairslin (linear steps) or stairslog (logarithmic steps) |
| `hdf_Raw(imghd5, filename, inputSize, outputSize, ...)` | Load a np.float64 raw binary image, optionally subsample, blur, and scale to rad_min range |

#### Top-level orchestrator

| Function | Description |
|----------|-------------|
| `create_HDF5_image(imageName, numPixels, fn, kwargs, wavelength, ...)` | Create an HDF5 image file, populate mandatory metadata, call the generating function, optionally add Poisson noise and equivalent-signal datasets; return the filename |

#### Analysis and visualisation

| Function | Description |
|----------|-------------|
| `analyse_HDF5_image(imghd5, plotfile, ...)` | Print statistics and plot images from an open HDF5 file handle |
| `analyse_HDF5_imageFile(hdffilename, ...)` | Open an HDF5 file by name and call analyse_HDF5_image |

#### Radiometry helpers

| Function | Description |
|----------|-------------|
| `calcLuxEquivalent(wavelength, rad_min, rad_dynrange, units)` | Build an interp1d mapping photon-rate radiance to lux for a monochromatic source. wavelength in SI metres. |
| `calcTemperatureEquivalent(wavelength, sysresp, tmin, tmax)` | Build paired interp1d objects mapping equivalent photon-rate radiance to temperature (lower and upper dynamic-range ends); wavelength in micrometres (um) |

---

### ryplot — matplotlib plotting wrapper

Provides compact, high-level wrappers around matplotlib for cartesian, polar,
3-D, contour, image, and specialised "pie" (angular/radial) plots.  A single
`Plotter` instance manages a multi-subplot figure; all plot methods return the
underlying matplotlib `Axes` object for further customisation.

#### Helper classes

| Class | Description |
|-------|-------------|
| `FilledMarker(...)` | Encapsulates filled-marker style properties (colour, shape, size, fillstyle) |
| `Markers(...)` | Collects a list of `FilledMarker` locations and draws them with `Markers.plot(ax)` |
| `ProcessImage()` | Image pre-processing helpers: histogram equalisation and polar reprojection |

#### ProcessImage methods

| Method | Description |
|--------|-------------|
| `compressEqualizeImage(image, selectCompressSet, numCbarlevels, cbarformat)` | Compress (linear/log/sqrt) then histogram-equalise an image; return equalised array and colourbar tick labels |
| `reprojectImageIntoPolar(data, origin, framesFirst, cval)` | Reproject a 3-D array from Cartesian to polar coordinates |

#### Plotter — construction and utilities

| Method | Description |
|--------|-------------|
| `Plotter(fignumber, subpltnrow, subpltncol, figuretitle, figsize, ...)` | Constructor: create a new multi-subplot figure |
| `buildPlotCol(plotCol, n)` | Set / extend the sequence of default plot colours |
| `nextPlotCol()` | Return the next colour in the cycle |
| `resetPlotCol()` | Reset the colour cycle to the beginning |
| `saveFig(filename, dpi, ...)` | Save the figure to disk (PNG, PDF, EPS, SVG …) |
| `getPlot()` | Return the underlying `matplotlib.figure.Figure` handle |
| `getSubPlot(subplotNum)` | Return the `Axes` handle for a subplot, or `None` if not yet created |
| `labelSubplot(spax, ptitle, xlabel, ylabel, zlabel, ...)` | Set subplot title and axis labels |
| `getXLim(subplotNum)` | Return x-axis limits as `np.ndarray([lo, hi])` |
| `getYLim(subplotNum)` | Return y-axis limits as `np.ndarray([lo, hi])` |
| `verticalLineCoords(subplotNum, x)` | Return `(x_arr, y_arr)` for a vertical line at `x` |
| `horizontalLineCoords(subplotNum, y)` | Return `(x_arr, y_arr)` for a horizontal line at `y` |
| `emptyPlot(plotnum, projection)` | Return an empty `Axes` for manual matplotlib use |
| `clippedcolorbar(CS, **kwargs)` | Add a colorbar clipped to the contour level range (static method) |

#### Plotter — plot methods

| Method | Description |
|--------|-------------|
| `plot(plotnum, x, y, ...)` | Linear cartesian plot; y may be multi-column |
| `logLog(plotnum, x, y, ...)` | Log–log cartesian plot |
| `semilogX(plotnum, x, y, ...)` | Semi-log plot (log x axis) |
| `semilogY(plotnum, x, y, ...)` | Semi-log plot (log y axis) |
| `stackplot(plotnum, x, y, ...)` | Stacked area chart |
| `polar(plotnum, theta, r, ...)` | Polar plot (theta in radians) |
| `showImage(plotnum, img, ...)` | Display a 2-D image with optional colourbar |
| `meshContour(plotnum, xvals, yvals, zvals, ...)` | Filled contour plot of a 2-D scalar field |
| `mesh3D(plotnum, xvals, yvals, zvals, ...)` | 3-D surface / mesh plot |
| `plot3d(plotnum, x, y, z, ...)` | 3-D line plot |
| `polar3d(plotnum, theta, radial, zvals, ...)` | 3-D polar surface plot |
| `polarMesh(plotnum, theta, radial, zvals, ...)` | Polar mesh / colourmap plot |
| `plotArray(plotnum, inarray, slicedim, ...)` | Auto-subplot grid from a 2-D array |
| `pie(plotnum, theta, radius, ...)` | Specialised angular/radial "pie-polar" plot |

#### Module-level functions

| Function | Description |
|----------|-------------|
| `savePlot(fignumber, ..., saveName)` | Context manager: create a `Plotter`, yield it, then auto-save on exit |
| `cubehelixcmap(start, rot, gamma, hue, reverse, nlev)` | Return a `LinearSegmentedColormap` using Dave Green's cubehelix algorithm |
| `RGBToPyCmap(rgbdata)` | Convert an `(N, 3)` RGB array to a matplotlib colourmap data dict |

---

## F-string conversion

All `ry*.py` files were converted from `.format()` and `%`-style string formatting to f-strings in April 2026 (~650 conversions across 17 files). All converted files pass `py_compile` syntax validation and the full test suite (784 tests).

### Unconverted strings

The following strings were intentionally left unconverted.

#### Variable-template `.format()` calls

These cannot be mechanically converted because the format template is a runtime variable, not a string literal.

| File | Line | Reason |
|------|------|--------|
| `ryfiles.py` | 438 | `cmdline.format(filename)` — `cmdline` is built from user-supplied arguments at runtime |
| `ryplot.py` | 399 | `fstr.format(...)` — `fstr` is constructed dynamically as `'{0:' + cbarformat + '}'` from a caller-supplied format string |

#### Commented-out code

These lines are inside `#`-prefixed comments and have no runtime effect. They were left as-is to preserve the original diagnostic intent.

| File | Line | Content |
|------|------|---------|
| `rychroma.py` | 540 | `#print('{0:15s} {1:15s} ({2:.4f},{3:.4f})'.format(...))` |
| `rychroma.py` | 553 | `#print('{0} um ({1},{2})'.format(...))` |
| `rylookup.py` | 419 | `# print('straight line fit DL = {} L + {}'.format(...))` |
| `ryplot.py` | 382 | `#        print('Image irradiance range minimum={0} maximum={1}'.format(...))` |
| `ryplot.py` | 390 | `#        H.saveFig('cumhist{0}.png'.format(...))` |
| `ryplotspherical.py` | 701 | `#         ptitle = '{0} {1}'.format(plottitle, specBand[i])` |
| `ryplotspherical.py` | 704 | `#     ptitle = '{0} {1}'.format(plottitle, 'log(abs(MWIR/LWIR))')` |
| `ryplotspherical.py` | 707 | `#     ptitle = '{0} {1}'.format(plottitle, 'log(abs(MWIR/SWIR1))')` |
| `rystare.py` | 2514 | `# fo.write('{:26}, ...'.format(...))` |
| `rystare.py` | 2515 | `# fo.write('{:26}, ...'.format(...))` |
| `rystare.py` | 2587 | `# print('Input rystare/EinUnit : {}'.format(...))` |
| `ryutils.py` | 1847 | `# print('{} {} integrate image {}'.format(...))` |
| `ryutils.py` | 1858 | `# print('{} {}'.format(...))` |
| `ryutils.py` | 2319 | `# print(eval('self.{}'.format(var)).shape)` |
