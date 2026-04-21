# pyradi

General-purpose radiometry utility functions for the [PyRadi](https://github.com/NelisW/pyradi) toolkit.

## Refactoring status

Last updated: 2026-04-21  (ryplanck.py added)

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

### Outstanding

ryplotspherical.py, ryprob.py, ryptw.py, ryscene.py, rystare.py, rytarggen.py, ryplot.py

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
pytest test/test_ryutils.py test/test_ryfiles.py test/test_ry3dnoise.py test/test_rychroma.py test/test_rydetector.py test/test_rylookup.py test/test_rymodtran.py test/test_ryhdf5.py test/test_rypflux.py test/test_ryplanck.py -v
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
├── rystare.py             # Staring array modelling (deferred)
├── rytarggen.py           # Target generation (deferred)
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
    ├── test_rymodtran.py   # pytest tests for rymodtran
    └── test_ryutils.py     # pytest tests for ryutils
```

---

## Available scripts / commands

| Command | Description |
|---------|-------------|
| `python ryutils.py` | Run the built-in demo and self-test examples |
| `python ry3dnoise.py` | Run the 3D noise demo (requires sensor data files) |
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

#### Utilities

| Function | Description |
|----------|-------------|
| `buildLogSpace(Vmin, Vmax, nDec)` | Log-spaced vector |
| `solidAngleSquare(width, breadth, height, stype)` | Solid angle of a rectangular flat surface |
| `upMu()` | LaTeX micro symbol |              