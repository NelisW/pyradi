# ryutils

General-purpose radiometry utility functions for the [PyRadi](https://github.com/NelisW/pyradi) toolkit.

## Refactoring in process 

status on 2026-04-19 

Files completed:  
ryutils.py ✓ refactored 2026-04-19, 06-Diverse-pyradi-utilities.ipynb ✓ refactored 2026-04-19 
ryfiles.py ✓ refactored 2026-04-19  


Files outstanding: 
ry3dnoise.py
rychroma.py
rydetector.py
rylookup.py
rymodtran.py
rypflux.py
ryplanck.py
ryplot.py
ryplotspherical.py
ryprob.py
ryptw.py
ryscene.py

Files not to refactored (HDF dependencies, to be refactored later):
ryhdf5.py
rystare.py
rytarggen.py

## Quick start

```bash
conda env create -f environment.yml
conda activate pyradi
python ryutils.py        # runs the built-in demo / self-test
```

Run the unit tests:

```bash
pytest test/test_ryutils.py test/test_ryfiles.py -v
```

## Tech stack

| Package | Purpose |
|---------|---------|
| numpy | Array maths throughout |
| scipy | Interpolation, signal processing, constants |
| matplotlib | Plotting helpers and aperture patches |
| scikit-image | Medial-axis skeleton in `extractGraph` |
| pillow | Image I/O in `drawCheckerboard` and `extractGraph` |
| ipython | `update_progress` progress bar, `VersionInformation` display |

## Project structure

```
pyradi/
├── ryutils.py             # This module — utility functions
├── environment.yml        # Conda environment spec
├── README.md              # This file
└── test/
    └── test_ryutils.py    # pytest regression tests
```

## Available scripts / commands

| Command | Description |
|---------|-------------|
| `python ryutils.py` | Run the built-in demo and self-test examples |
| `pytest test/test_ryutils.py -v` | Run the regression test suite |

## Public API summary

### Spectral domain

| Function | Description |
|----------|-------------|
| `convertSpectralDomain(domain, type)` | Convert between wavelength [µm], wavenumber [cm⁻¹] and frequency [Hz] |
| `convertSpectralDensity(domain, quantity, type)` | Convert spectral density between domains |
| `effectiveValue(domain, signal, baseline)` | Weighted spectral integral (normalised) |
| `sfilter(spectral, center, width, ...)` | Maximally-flat bandpass / lowpass / highpass filter |
| `responsivity(wavelength, lwavepeak, ...)` | Photon-detector spectral responsivity shape |
| `convolve(inspectral, ...)` | Spectral convolution with configurable window |
| `savitzkyGolay1D(y, window_size, order)` | Savitzky–Golay smoothing filter |

### Coordinate transforms

| Function | Description |
|----------|-------------|
| `cart2polar(x, y)` | Cartesian → polar |
| `polar2cart(r, theta)` | Polar → Cartesian |
| `framesFirst(seq)` | Reorder 3-D image sequence: frames to axis 0 |
| `framesLast(seq)` | Reorder 3-D image sequence: frames to axis 2 |
| `index_coords(data, origin)` | Generate (x, y) coordinate arrays for an array |

### Detection statistics

| Function | Description |
|----------|-------------|
| `detectFARThresholdToNoisepulseWidth(T2N, pw)` | FAR from threshold-to-noise and pulse width |
| `detectThresholdToNoiseTpFAR(pw, FAR)` | Threshold-to-noise from pulse width and FAR |
| `detectSignalToNoiseThresholdToNoisePd(T2N, Pd)` | SNR from threshold-to-noise and Pd |
| `detectThresholdToNoiseSignalToNoisepD(SNR, Pd)` | Threshold-to-noise from SNR and Pd |
| `detectProbabilityThresholdToNoiseSignalToNoise(T2N, SNR)` | Pd from threshold-to-noise and SNR |
| `rangeEquation(I, E, rangeTab, tauTab)` | Solve point-source range equation |

### Image / geometry

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

### Atmosphere / photometry

| Function | Description |
|----------|-------------|
| `abshumidity(T)` | Absolute humidity [g/m³] from temperature [K] |
| `TFromAbshumidity(AH)` | Temperature [K] from absolute humidity |
| `luminousEfficiency(vlamtype, wavelen)` | CIE photopic / scotopic luminous efficiency V(λ) |

### Utilities

| Function | Description |
|----------|-------------|
| `buildLogSpace(Vmin, Vmax, nDec)` | Log-spaced vector |
| `solidAngleSquare(width, breadth, height, stype)` | Solid angle of a rectangular flat surface |
| `upMu()` | LaTeX micro symbol