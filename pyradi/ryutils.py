# -*- coding: utf-8 -*-


################################################################
# The contents of this file are subject to the BSD 3Clause (New) License
# you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://directory.fsf.org/wiki/License:BSD_3Clause

# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations
# under the License.

# The Original Code is part of the PyRadi toolkit.

# The Initial Developer of the Original Code is CJ Willers,
# Portions created by CJ Willers are Copyright (C) 2006-2012
# All Rights Reserved.

# Contributor(s): ______________________________________.
################################################################
"""
This module provides various utility functions for radiometry calculations.
Functions are provided for a maximally flat spectral filter, a simple photon
detector spectral response, effective value calculation, conversion of spectral
domain variables between [um], [cm^-1] and [Hz], conversion of spectral
density quantities between [um], [cm^-1] and [Hz] and spectral convolution.

See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students
and readers of the book Electro-Optical System Analysis and Design: A Radiometry
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""

__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['sfilter', 'responsivity', 'effectiveValue', 'convertSpectralDomain',
         'convertSpectralDensity', 'convolve', 'savitzkyGolay1D','abshumidity', 
         'rangeEquation','_rangeEquationCalc','detectThresholdToNoiseTpFAR', 
         'detectSignalToNoiseThresholdToNoisePd',
         'detectThresholdToNoiseSignalToNoisepD',
         'detectProbabilityThresholdToNoiseSignalToNoise',
         'detectFARThresholdToNoisepulseWidth', 'upMu',
         'cart2polar', 'polar2cart','index_coords','framesFirst','framesLast',
         'rect', 'circ','poissonarray','draw_siemens_star','drawCheckerboard',
         'makemotionsequence','extractGraph','luminousEfficiency','Spectral',
         'Atmo','Sensor','Target','calcMTFwavefrontError',
         'polar2cartesian','warpPolarImageToCartesianImage',
         'intify_tuple','differcommonfiles','blurryextract'
         ]

import sys
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import os
import pkg_resources
from numbers import Number
if sys.version_info[0] > 2:
    from io import StringIO
else:
    from StringIO import StringIO



##############################################################################
##
def intify_tuple(tup):
    """Make tuple entries int type
    """
    tup_int = ()
    for tup_ent in tup:
        tup_int = tup_int + (int(tup_ent),)
    return tup_int


##############################################################################
##
def framesFirst(imageSequence):
    """Image sequence with frames along axis=2 (last index), reordered such that
    frames are along axis=0 (first index).

    Image sequences are stored in three-dimensional arrays, in rows, columns and frames.
    Not all libraries share the same sequencing, some store frames along axis=0 and
    others store frames along axis=2.  This function reorders an image sequence with
    frames along axis=2  to an image sequence with frames along axis=0. The function
    uses np.transpose(imageSequence, (2,0,1))

    Args:
        | imageSequence (3-D np.array): image sequence in three-dimensional array, frames along axis=2


    Returns:
        |  ((3-D np.array): reordered three-dimensional array (view or copy)


    Raises:
        | No exception is raised.
    """
    return np.transpose(imageSequence, (2,0,1))

##############################################################################
##
def framesLast(imageSequence):
    """Image sequence with frames along axis=0 (first index), reordered such that
    frames are along axis=2 (last index).

    Image sequences are stored in three-dimensional arrays, in rows, columns and frames.
    Not all libraries share the same sequencing, some store frames along axis=0 and
    others store frames along axis=2.  This function reorders an image sequence with
    frames along axis=0  to an image sequence with frames along axis=2.  The function
    uses np.transpose(imageSequence, (1,2,0))

    Args:
        | imageSequence (3-D np.array): image sequence in three-dimensional array, frames along axis=0


    Returns:
        |  ((3-D np.array): reordered three-dimensional array (view or copy)


    Raises:
        | No exception is raised.
    """
    return np.transpose(imageSequence, (1,2,0))



##############################################################################
##
def index_coords(data, origin=None, framesFirst=True):
    """Creates (x,y) zero-based coordinate arrrays for a numpy array indices, relative to some origin.

    This function calculates two meshgrid arrays containing the coordinates of the
    input array.  The origin of the new coordinate system  defaults to the
    center of the image, unless the user supplies a new origin.

    The data format can be data.shape = (rows, cols, frames) or
    data.shape = (frames, rows, cols), the format of which is indicated by the
    framesFirst parameter.

    Args:
        | data (np.array): array for which coordinates must be calculated.
        | origin ( (x-orig, y-orig) ): data-coordinates of where origin should be
        | framesFirst (bool): True if data.shape is (frames, rows, cols), False if
            data.shape is (rows, cols, frames)

    Returns:
        | x (float np.array): x coordinates in array format.
        | y (float np.array): y coordinates in array format.

    Raises:
        | No exception is raised.

    original code by Joe Kington
    https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
    """
    if framesFirst:
        ny, nx = data.shape[1:3]
    else:
        ny, nx = data.shape[:2]

    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

##############################################################################
##
def cart2polar(x, y):
    """Converts from cartesian to polar coordinates, given (x,y) to (r,theta).

    Args:
        | x (float np.array): x values in array format.
        | y (float np.array): y values in array format.

    Returns:
        | r (float np.array): radial component for given (x,y).
        | theta (float np.array): angular component for given (x,y).

    Raises:
        | No exception is raised.

    original code by Joe Kington
    https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

##############################################################################
##
def polar2cart(r, theta):
    """Converts from polar to cartesian coordinates, given (r,theta) to (x,y).

    Args:
        | r (float np.array): radial values in array format.
        | theta (float np.array): angular values in array format.

    Returns:
        | x (float np.array): x component for given (r, theta).
        | y (float np.array): y component for given (r, theta).

    Raises:
        | No exception is raised.

    original code by Joe Kington
    https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


##############################################################################
##
def upMu(uprightMu=True, textcomp=False):
    """Returns a LaTeX micron symbol, either an upright version or the normal symbol.

    The upright symbol requires that the siunitx LaTeX package be installed on the
    computer running the code.  This function also changes the Matplotlib rcParams
    file.

    Args:
        | uprightMu (bool): signals upright (True) or regular (False) symbol (optional).
        | textcomp (bool): if True use the textcomp package, else use siunitx package (optional).

    Returns:
        | range (string): LaTeX code for the micro symbol.

    Raises:
        | No exception is raised.
    """
    if sys.version_info[0] < 3:

        if uprightMu:
            from matplotlib import rc, font_manager
            import matplotlib as mpl
            rc('text', usetex=True)
            # set up the use of external latex, fonts and packages
            if not textcomp :
                mpl.rcParams['text.latex.preamble'] = [
                # r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
                '\\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
                '\\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
                '\\usepackage{helvet}',  # set the normal font here
                '\\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
                '\\sansmath']  # <- tricky! -- gotta actually tell tex to use!
                upmu = '\si{\micro}'
            else:
                mpl.rcParams['text.latex.preamble'] = [
                '\\usepackage{textcomp}',   # i need upright \micro symbols, but you need...
                '\\usepackage{helvet}',  # set the normal font here
                '\\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
                '\\sansmath'  # <- tricky! -- gotta actually tell tex to use!
                ]
                upmu = '\\textmu{}'
        else:
            upmu = '$\\mu$'
    else:
        upmu = '\u00B5'
        
    return upmu


##############################################################################
##
def detectFARThresholdToNoisepulseWidth(ThresholdToNoise, pulseWidth):
    """ Solve for the FAR, given the threshold to noise ratio and pulse width, for matched filter.

    References:

    "Electro-optics handbook," Tech. Rep. EOH-11, RCA, 1974. RCA Technical Series Publication.

    R. D. Hippenstiel, Detection Theory: Applications and Digital Signal Pro-cessing, CRC Press, 2002

    Args:
        | ThresholdToNoise (float): the threshold to noise ratio.
        | pulseWidth (float): the signal pulse width in [s].

    Returns:
        | FAR (float): the false alarm rate in [alarms/s]

    Raises:
        | No exception is raised.
    """

    FAR = np.exp(- (ThresholdToNoise ** 2) / 2.) / (2. * pulseWidth * np.sqrt(3))

    return FAR


##############################################################################
##
def detectThresholdToNoiseTpFAR(pulseWidth, FAR):
    """ Solve for threshold to noise ratio, given pulse width and FAR, for matched filter.

    Using the theory of matched filter design, calculate the
    threshold to noise ratio, to achieve a required false alarm rate.

    References:

    "Electro-optics handbook," Tech. Rep. EOH-11, RCA, 1974. RCA Technical Series Publication.

    R. D. Hippenstiel, Detection Theory: Applications and Digital Signal Pro-cessing, CRC Press, 2002

    Args:
        | pulseWidth (float): the signal pulse width in [s].
        | FAR (float): the false alarm rate in [alarms/s]

    Returns:
        | range (float): threshold to noise ratio

    Raises:
        | No exception is raised.
    """

    ThresholdToNoise = np.sqrt(-2 * np.log (2 * pulseWidth * np.sqrt(3) * FAR ))

    return ThresholdToNoise



##############################################################################
##
def detectSignalToNoiseThresholdToNoisePd(ThresholdToNoise, pD):
    """ Solve for the signal to noise ratio, given the threshold to noise ratio and
    probability of detection.

    Using the theory of matched filter design, calculate the
    signal to noise ratio, to achieve a required probability of detection.

    References:

    "Electro-optics handbook," Tech. Rep. EOH-11, RCA, 1974. RCA Technical Series Publication.

    R. D. Hippenstiel, Detection Theory: Applications and Digital Signal Pro-cessing, CRC Press, 2002

    Args:
        | ThresholdToNoise (float): the threshold to noise ratio [-]
        | pD (float): the probability of detection [-]

    Returns:
        | range (float): signal to noise ratio

    Raises:
        | No exception is raised.
    """

    import scipy.special

    SignalToNoise = np.sqrt(2) * scipy.special.erfinv(2 * pD -1) + ThresholdToNoise

    return SignalToNoise

##############################################################################
##
def detectThresholdToNoiseSignalToNoisepD(SignalToNoise, pD):
    """ Solve for the threshold to noise ratio, given the signal to noise ratio and
    probability of detection.

    References:

    "Electro-optics handbook," Tech. Rep. EOH-11, RCA, 1974. RCA Technical Series Publication.

    R. D. Hippenstiel, Detection Theory: Applications and Digital Signal Pro-cessing, CRC Press, 2002

    Args:
        | SignalToNoise (float): the signal to noise ratio [-]
        | pD (float): the probability of detection [-]

    Returns:
        | range (float): signal to noise ratio

    Raises:
        | No exception is raised.
    """

    import scipy.special

    ThresholdToNoise = SignalToNoise - np.sqrt(2) * scipy.special.erfinv(2 * pD -1)

    return ThresholdToNoise


##############################################################################
##
def detectProbabilityThresholdToNoiseSignalToNoise(ThresholdToNoise, SignalToNoise):
    """ Solve for the probability of detection, given the signal to noise ratio and
     threshold to noise ratio

    References:

    "Electro-optics handbook," Tech. Rep. EOH-11, RCA, 1974. RCA Technical Series Publication.

    R. D. Hippenstiel, Detection Theory: Applications and Digital Signal Pro-cessing, CRC Press, 2002

    Args:
        | ThresholdToNoise (float): the threshold to noise ratio [-]
        | SignalToNoise (float): the signal to noise ratio [-]

    Returns:
        | range (float): probability of detection

    Raises:
        | No exception is raised.
    """

    import scipy.special

    pD   = 0.5 * (scipy.special.erf((SignalToNoise - ThresholdToNoise) / np.sqrt(2)) + 1)

    return pD


##############################################################################
##
def rangeEquation(Intensity, Irradiance, rangeTab, tauTab, rangeGuess = 1, n = 2):
    """ Solve the range equation for arbitrary transmittance vs range.

    This function solve for the range :math:`R` in the range equation

    .. math::

     E = \\frac{I\\tau_a(R)}{R^n}

    where :math:`E` is the threshold irradiance in [W/m2],
    and :math:`I` is the intensity in [W/sr]. This range equation holds for
    the case where the target is smaller than the field of view.

    The range :math:`R` must be in [m], and :math:`\\tau_a(R)`
    is calculated from a lookup table of atmospheric transmittance vs. range.
    The transmittance lookup table  can be calculated from the simple Bouguer law,
    or it can have any arbitrary shape, provided it decreases with increasing range.
    The user supplies the lookup table in the form of an array of range values and
    an associated array of transmittance values.  The range values need not be on
    constant linear range increment.

    The parameter :math:`n`

    * :math:`n=2` (default value) the general case of a radiating source
      smaller than the field of view.

    * :math:`n=4` the special case of a laser range finder illuminating a target
      smaller than the field of view, viewed against the sky. In this case there
      is an :math:`R^2` attenuation from the laser to the source and another
      :math:`R^2` attenuation from the source to the receiver, hence
      :math:`R^4` overall.

    If the range solution is doubtful (e.g. not a trustworthy solution) the
    returned value is made negative.

    Args:
        | Intensity (float or np.array[N,] or [N,1]):  in  [W/sr].
        | Irradiance (float or np.array[N,] or [N,1]):  in  [W/m2].
        | rangeTab (np.array[N,] or [N,1]):  range vector for tauTab lookup in [m]
        | tauTab (np.array[N,] or [N,1]):   transmittance vector for lookup in [m]
        | rangeGuess (float): starting value range estimate in [m] (optional)
        | n (float): range power (2 or 4) (optional)

    Returns:
        | range (float or np.array[N,] or [N,1]): Solution to the range equation in [m].
          Value is negative if calculated range exceeds the top value in range table,
          or if calculated range is too near the lower resolution limit.

    Raises:
        | No exception is raised.
    """

    from scipy.interpolate import  interp1d
    from scipy.optimize import fsolve

    tauTable = interp1d(rangeTab, tauTab, kind = 'linear')

    Range = fsolve(_rangeEquationCalc, rangeGuess,
        args = (Intensity,Irradiance,tauTable,n,np.max(rangeTab),))

    #near the bottom (minimum) range of the table
    if(Range < rangeTab[2] ):
        Range = - Range

        # beyond the top of the range table
    if(Range >  rangeTab[-1] ):
        Range = - Range

    return Range


##############################################################################
##
def _rangeEquationCalc(r,i,e,tauTable,n,rMax):
    if r > rMax:
        return 0
    return i * tauTable(r) / (r ** n) - e



##############################################################################
##
def abshumidity(T, equationSelect = 1):
    """ Atmopsheric absolute humidity [g/m3] for temperature in [K] between 248 K and 342 K.

    This function provides two similar equations, but with different constants.


    Args:
        | temperature (np.array[N,] or [N,1]):  in  [K].
        | equationSelect (int): select the equation to be used.


    Returns:
        | absolute humidity (np.array[N,] or [N,1]):  abs humidity in [g/m3]

    Raises:
        | No exception is raised.
    """

    #there are two options, the fist one seems more accurate (relative to test set)
    if equationSelect == 1:
        #http://www.vaisala.com/Vaisala%20Documents/Application%20notes/Humidity_Conversion_Formulas_B210973EN-D.pdf
        return ( 1325.2520998 * 10 **(7.5892*(T - 273.15)/(T -32.44)))/T

    else:
        #http://www.see.ed.ac.uk/~shs/Climate%20change/Data%20sources/Humidity%20with%20altidude.pdf
        return (1324.37872 * 2.718281828459046 **(17.67*(T - 273.16)/(T - 29.66)))/T



##############################################################################
##
def sfilter(spectral,center, width, exponent=6, taupass=1.0,  \
            taustop=0.0, filtertype = 'bandpass' ):
    """ Calculate a symmetrical filter response of shape exp(-x^n)


    Given a number of parameters, calculates maximally flat,
    symmetrical transmittance.  The function parameters controls
    the width, pass-band and stop-band transmittance and sharpness
    of cutoff. This function is not meant to replace the use of
    properly measured filter responses, but rather serves as a
    starting point if no other information is available.
    This function does not calculate ripple in the pass-band
    or cut-off band.

    Filter types supported include band pass, high (long) pass and
    low (short) pass filters. High pass filters have maximal
    transmittance for all spectral values higher than the central
    value. Low pass filters have maximal transmittance for all
    spectral values lower than the central value.

    Args:
        | spectral (np.array[N,] or [N,1]): spectral vector in  [um] or [cm-1].
        | center (float): central value for filter passband
        | width (float): proportional to width of filter passband
        | exponent (float): even integer, define the sharpness of cutoff.
        |                     If exponent=2        then gaussian
        |                     If exponent=infinity then square
        | taupass (float): the transmittance in the pass band (assumed constant)
        | taustop (float): peak transmittance in the stop band (assumed constant)
        | filtertype (string): filter type, one of 'bandpass', 'lowpass' or 'highpass'

    Returns:
        | transmittance (np.array[N,] or [N,1]):  transmittances at "spectral" intervals.

    Raises:
        | No exception is raised.
        | If an invalid filter type is specified, return None.
        | If negative spectral is specified, return None.
    """

    maxexp = np.log(sys.float_info.max)/np.log(np.max(2*np.abs(spectral-center)/width))
    # minexp = np.log(sys.float_info.min)/np.log(np.min(2*(spectral-center)/width))
    exponent = maxexp if exponent > maxexp else exponent
    # exponent = minexp if exponent < minexp else exponent
    tau = taustop+(taupass-taustop)*np.exp(-(2*np.abs(spectral-center)/width)**exponent)
    maxtau=np.max(tau)
    if filtertype == 'bandpass':
        pass
    elif filtertype == 'lowpass':
        tau = tau * np.greater(spectral,center) + \
                maxtau * np.ones(spectral.shape) * np.less(spectral,center)
    elif filtertype == 'highpass':
        tau = tau * np.less(spectral,center) + \
                maxtau * np.ones(spectral.shape) * np.greater(spectral,center)
    else:
        return None

    return tau



##############################################################################
##
def responsivity(wavelength,lwavepeak, cuton=1, cutoff=20, scaling=1.0):
    """ Calculate a photon detector wavelength spectral responsivity

    Given a number of parameters, calculates a shape that is somewhat similar to a photon
    detector spectral response, on wavelength scale. The function parameters controls the
    cutoff wavelength and shape of the response. This function is not meant to replace the use
    of properly measured  spectral responses, but rather serves as a starting point if no other
    information is available.

    Args:
        | wavelength (np.array[N,] or [N,1]):  vector in  [um].
        | lwavepeak (float): approximate wavelength  at peak response
        | cutoff (float): cutoff strength  beyond peak, 5 < cutoff < 50
        | cuton (float): cuton sharpness below peak, 0.5 < cuton < 5
        | scaling (float): scaling factor

    Returns:
        | responsivity (np.array[N,] or [N,1]):  responsivity at wavelength intervals.

    Raises:
        | No exception is raised.
    """

    responsivity=scaling *( ( wavelength / lwavepeak) **cuton - ( wavelength / lwavepeak) **cutoff)
    responsivity= responsivity * (responsivity > 0)

    return responsivity


################################################################
##
def effectiveValue(spectraldomain,  spectralToProcess,  spectralBaseline):
    """Normalise a spectral quantity to a scalar, using a weighted mapping by another spectral quantity.

    Effectivevalue =  integral(spectralToProcess * spectralBaseline) / integral( spectralBaseline)

    The data in  spectralToProcess and  spectralBaseline must both be sampled at the same
    domain values     as specified in spectraldomain.

    The integral is calculated with numpy/scipy trapz trapezoidal integration function.

    Args:
        | inspectraldomain (np.array[N,] or [N,1]):  spectral domain in wavelength, frequency or wavenumber.
        | spectralToProcess (np.array[N,] or [N,1]):  spectral quantity to be normalised
        | spectralBaseline (np.array[N,] or [N,1]):  spectral serving as baseline for normalisation

    Returns:
        | (float):  effective value
        | Returns None if there is a problem

    Raises:
        | No exception is raised.
    """

    num=np.trapz(spectralToProcess.reshape(-1, 1)*spectralBaseline.reshape(-1, 1),spectraldomain, axis=0)[0]
    den=np.trapz(spectralBaseline.reshape(-1, 1),spectraldomain, axis=0)[0]
    return num/den


################################################################
##
def convertSpectralDomain(inspectraldomain,  type=''):
    """Convert spectral domains, i.e. between wavelength [um], wavenummber [cm^-1] and frequency [Hz]

    In string variable type, the 'from' domain and 'to' domains are indicated each with a single letter:
    'f' for temporal frequency, 'l' for wavelength and 'n' for wavenumber
    The 'from' domain is the first letter and the 'to' domain the second letter.

    Note that the 'to' domain vector is a direct conversion of the 'from' domain
    to the 'to' domain (not interpolated or otherwise sampled.

    Args:
        | inspectraldomain (np.array[N,] or [N,1]):  spectral domain in wavelength, frequency or wavenumber.
        |    wavelength vector in  [um]
        |    frequency vector in  [Hz]
        |    wavenumber vector in   [cm^-1]
        | type (string):  specify from and to domains:
        |    'lf' convert from wavelength to per frequency
        |    'ln' convert from wavelength to per wavenumber
        |    'fl' convert from frequency to per wavelength
        |    'fn' convert from frequency to per wavenumber
        |    'nl' convert from wavenumber to per wavelength
        |    'nf' convert from wavenumber to per frequency

    Returns:
        | [N,1]: outspectraldomain
        | Returns zero length array if type is illegal, i.e. not one of the expected values

    Raises:
        | No exception is raised.
    """

    #use dictionary to switch between options, lambda fn to calculate, default zero
    outspectraldomain = {
              'lf': lambda inspectraldomain:  constants.c / (inspectraldomain * 1.0e-6),
              'ln': lambda inspectraldomain:  (1.0e4/inspectraldomain),
              'fl': lambda inspectraldomain:  constants.c  / (inspectraldomain * 1.0e-6),
              'fn': lambda inspectraldomain:  (inspectraldomain / 100) / constants.c ,
              'nl': lambda inspectraldomain:  (1.0e4/inspectraldomain),
              'nf': lambda inspectraldomain:  (inspectraldomain * 100) * constants.c,
              }.get(type, lambda inspectraldomain: np.zeros(shape=(0, 0)) )(inspectraldomain)

    return outspectraldomain



################################################################
##
def convertSpectralDensity(inspectraldomain,  inspectralquantity, type=''):
    """Convert spectral density quantities, i.e. between W/(m^2.um), W/(m^2.cm^-1) and W/(m^2.Hz).

    In string variable type, the 'from' domain and 'to' domains are indicated each with a
    single letter:
    'f' for temporal frequency, 'w' for wavelength and ''n' for wavenumber
    The 'from' domain is the first letter and the 'to' domain the second letter.

    The return values from this function are always positive, i.e. not mathematically correct,
    but positive in the sense of radiance density.

    The spectral density quantity input is given as a two vectors: the domain value vector
    and the density quantity vector. The output of the function is also two vectors, i.e.
    the 'to' domain value vector and the 'to' spectral density. Note that the 'to' domain
    vector is a direct conversion of the 'from' domain to the 'to' domain (not interpolated
    or otherwise sampled).

    Args:
        | inspectraldomain (np.array[N,] or [N,1]):  spectral domain in wavelength,
            frequency or wavenumber.
        | inspectralquantity (np.array[N,] or [N,1]):  spectral density in same domain
           as domain vector above.
        |    wavelength vector in  [um]
        |    frequency vector in  [Hz]
        |    wavenumber vector in   [cm^-1]
        | type (string):  specify from and to domains:
        |    'lf' convert from per wavelength interval density to per frequency interval density
        |    'ln' convert from per wavelength interval density to per wavenumber interval density
        |    'fl' convert from per frequency interval density to per wavelength interval density
        |    'fn' convert from per frequency interval density to per wavenumber interval density
        |    'nl' convert from per wavenumber interval density to per wavelength interval density
        |    'nf' convert from per wavenumber interval density to per frequency interval density

    Returns:
        | ([N,1],[N,1]): outspectraldomain and outspectralquantity
        | Returns zero length arrays is type is illegal, i.e. not one of the expected values

    Raises:
        | No exception is raised.
    """

    inspectraldomain = inspectraldomain.reshape(-1,)
    inspectralquantity = inspectralquantity.reshape(inspectraldomain.shape[0], -1)
    outspectralquantity = np.zeros(inspectralquantity.shape)

    # the meshgrid idea does not work well here, because we can have very long
    # spectral arrays and these become too large for meshgrid -> size **2
    # we have to loop this one
    spec = inspectraldomain
    for col in range(inspectralquantity.shape[1]):

        quant = inspectralquantity[:,col]

        #use dictionary to switch between options, lambda fn to calculate, default zero
        outspectraldomain = {
                  'lf': lambda spec:  constants.c / (spec * 1.0e-6),
                  'fn': lambda spec:  (spec / 100) / constants.c ,
                  'nl': lambda spec:  (1.0e4/spec),
                  'ln': lambda spec:  (1.0e4/spec),
                  'nf': lambda spec:  (spec * 100) * constants.c,
                  'fl': lambda spec:  constants.c  / (spec * 1.0e-6),
                  }.get(type, lambda spec: np.zeros(shape=(0, 0)) )(spec)

        outspectralquantity[:, col] = {
                  'lf': lambda quant: quant / (constants.c *1.0e-6 / ((spec * 1.0e-6)**2)),
                  'fn': lambda quant: quant * (100 *constants.c),
                  'nl': lambda quant: quant / (1.0e4 / spec**2) ,
                  'ln': lambda quant: quant / (1.0e4 / spec**2) ,
                  'nf': lambda quant: quant / (100 * constants.c),
                  'fl': lambda quant: quant / (constants.c *1.0e-6 / ((spec * 1.0e-6)**2)),
                  }.get(type, lambda quant: np.zeros(shape=(0, 0)) )(quant)

    return (outspectraldomain,outspectralquantity)




##############################################################################
##
def savitzkyGolay1D(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    Source: http://wiki.scipy.org/Cookbook/SavitzkyGolay

    The Savitzky Golay filter is a particular type of low-pass filter,
    well adapted for data smoothing. For further information see:
    http://www.wire.tu-bs.de/OLDWEB/mameyer/cmr/savgol.pdf


    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.


    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples:
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()

    References:
        [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
            Data by Simplified Least Squares Procedures. Analytical
            Chemistry, 1964, 36 (8), pp 1627-1639.
        [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
            W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
            Cambridge University Press ISBN-13: 9780521880688


    Args:
        | y : array_like, shape (N,) the values of the time history of the signal.
        | window_size : int the length of the window. Must be an odd integer number.
        | order : int the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        | deriv: int the order of the derivative to compute (default = 0 means only smoothing)


    Returns:
        | ys : ndarray, shape (N) the smoothed signal (or it's n-th derivative).

     Raises:
        | Exception raised for window size errors.
   """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = list(range(order+1))
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


##############################################################################
##
def convolve(inspectral, samplingresolution,  inwinwidth,  outwinwidth,  windowtype=np.bartlett):
    """ Convolve (non-circular) a spectral variable with a window function,
    given the input resolution and input and output window widths.

    This function is normally used on wavenumber-domain spectral data.  The spectral
    data is assumed sampled at samplingresolution wavenumber intervals.
    The inwinwidth and outwinwidth window function widths are full width half-max (FWHM)
    for the window functions for the inspectral and returned spectral variables, respectively.
    The Bartlett function is used as default, but the user can use a different function.
    The Bartlett function is a triangular function reaching zero at the ends. Window function
    width is correct for Bartlett and only approximate for other window functions.

    Spectral convolution is best done in frequency domain ([cm-1] units) because
    the filter or emission line shapes have better symmetry in frequency domain than
    in wavelength domain.

    The input spectral vector must be in spectral density units of cm-1.

    Args:
        | inspectral (np.array[N,] or [N,1]):  spectral variable input  vector (e.g., radiance or transmittance).
        | samplingresolution (float): wavenumber interval between inspectral samples
        | inwinwidth (float): FWHM window width used to obtain the input spectral vector (e.g., spectroradiometer window width)
        | outwinwidth (float): FWHM window width of the output spectral vector after convolution
        | windowtype (function): name of a  numpy/scipy function for the window function

    Returns:
        | outspectral (np.array[N,]):  input vector, filtered to new window width.
        | windowfn (np.array[N,]):  The window function used.

    Raises:
        | No exception is raised.
    """

    winbins = round(2*(outwinwidth/(inwinwidth*samplingresolution)), 0)
    winbins = winbins if winbins%2==1 else winbins+1
    windowfn=windowtype(winbins)
    #np.convolve is unfriendly towards unicode strings

    if sys.version_info[0] > 2:
        cmode='same'
    else:
        cmode='same'.encode('utf-8')

    outspectral = np.convolve(windowfn/(samplingresolution*windowfn.sum()),
                        inspectral.reshape(-1, ),mode=cmode)
    return outspectral,  windowfn

######################################################################################
def circ(x, y, d=1):
    """ Generation of a circular aperture.

    Args:
        | x (np.array[N,M]): x-grid, metres
        | y (np.array[N,M]): y-grid, metres
        | d (float): diameter in metres.
        | comment (string): the symbol used to comment out lines, default value is None.
        | delimiter (string): delimiter used to separate columns, default is whitespace.

    Returns:
        | z (np.array[N,M]): z-grid, 1's inside radius, meters/pixels.

    Raises:
        | No exception is raised.

    Author: Prof. Jason Schmidt, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """
    z = None
    r = np.sqrt(x ** 2 + y ** 2)
    z = np.zeros(r.shape)
    z[r < d / 2.] = 1.0
    z[r == d / 2.] = 0.5

    return z


######################################################################################
def rect(x, y, sx=1, sy=1):
    """ Generation of a rectangular aperture.

    Args:
        | x (np.array[N,M]): x-grid, metres
        | y (np.array[N,M]): x-grid, metres
        | sx (float): full size along x.
        | sy (float): full size along y.

    Returns:
        | Nothing.

    Raises:
        | No exception is raised.

    Author:  CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    z = None
    if x is not None and y is not None:
        z = np.zeros(x.shape)
        z[np.logical_and(np.abs(x) < sx/2.,np.abs(y) < sy/2.)] = 1.
        z[np.logical_and(np.abs(x) == sx/2., np.abs(y) == sy/2.)] = 0.5

    return z


######################################################################################################
def poissonarray(inp, seedval=None, tpoint=1000):
    r"""This routine calculates a Poisson random variable for an array of input values
    with potentially very high event counts.

    At high mean values the Poisson distribution calculation overflows. For
    mean values exceeding 1000, the Poisson distribution may be approximated by a
    Gaussian distribution.

    The function accepts a two-dimensional array and calculate a separate random
    value for each element in the array, using the element value as the mean value.
    A typical use case is when calculating shot noise for image data.

    From http://en.wikipedia.org/wiki/Poisson_distribution#Related_distributions
    For sufficiently large values of :math:`\lambda`, (say :math:`\lambda>1000`),
    the normal distribution with mean :math:`\lambda` and
    variance :math:`\lambda` (standard deviation :math:`\sqrt{\lambda}`)
    is an excellent approximation to the Poisson distribution.
    If :math:`\lambda` is greater than about 10, then the normal distribution
    is a good approximation if an appropriate continuity correction is performed, i.e.,
    :math:`P(X \le x)`, where (lower-case) x is a non-negative integer, is replaced by
    :math:`P(X\le\,x+0.5)`.

    :math:`F_\mathrm{Poisson}(x;\lambda)\approx\,F_\mathrm{normal}(x;\mu=\lambda,\sigma^2=\lambda)`

    This function returns values of zero when the input is zero.

    Args:
        | inp (np.array[N,M]): array with mean value
        | seedval (int): seed for random number generator, None means use system time.
        | tpoint (int): Threshold when to switch over between Poisson and Normal distributions

    Returns:
        | outp (np.array[N,M]): Poisson random variable for given mean value

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    #If seed is omitted or None, current system time is used
    np.random.seed(seedval)

    #this is a bit of a mess:
    # - for values smaller than tpoint calculate using standard Poisson distribution
    # - for values larger than tpoint but nonzero use normal approximation, add small sdelta to avoid variance==0
    # - for values larger than tpoint but zero keep at zero, sdelta added has no effect, just avoids zero divide
    sdelta = 1e-10
    outp = np.zeros(inp.shape)
    outp =  (inp<=tpoint) * np.random.poisson(inp * (inp<=tpoint) )\
                        + ((inp>tpoint) & (inp!=0)) * np.random.normal(loc=inp, scale=np.sqrt(inp+sdelta))

    outp = np.where(inp==0, 0., outp)
    return outp


######################################################################################################
def draw_siemens_star(outfile, n, dpi):
    r"""Siemens star chart generator

    by Libor Wagner, http://cmp.felk.cvut.cz/~wagnelib/utils/star.html

    Args:
        | outfile (str): output image filename (monochrome only)
        | n (int): number of spokes in the output image.
        | dpi (int): dpi in output image, determines output image size.

    Returns:
        | Nothing, creates a monochrome siemens star image

    Raises:
        | No exception is raised.

    Author: Libor Wagner, adapted by CJ Willers
    """
    from scipy import misc

    # Create figure and add patterns
    fig, ax = plt.subplots()
    ax.add_collection(gen_siemens_star((0,0), 1., n))
    plt.axis('equal')
    plt.axis([-1.03, 1.03, -1.03, 1.03])
    plt.axis('off')
    fig.savefig(outfile, figsize=(900,900), papertype='a0', bbox_inches='tight', dpi=dpi)
    #read image back in order to crop to spokes only
    imgIn = np.abs(255 - misc.imread(outfile)[:,:,0])
    nz0 = np.nonzero(np.sum(imgIn,axis=0))
    nz1 = np.nonzero(np.sum(imgIn,axis=1))
    imgOut = imgIn[(nz1[0][0]-1) : (nz1[0][-1]+2),  (nz0[0][0]-1) : (nz0[0][-1]+2)]
    imgOut = np.abs(255 - imgOut)
    misc.imsave(outfile, imgOut)

######################################################################################################
def gen_siemens_star(origin, radius, n):
    centres = np.linspace(0, 360, n+1)[:-1]
    step = (((360.0)/n)/4.0)
    patches = []
    for c in centres:
        patches.append(Wedge(origin, radius, c-step, c+step))
    return PatchCollection(patches, facecolors='k', edgecolors='none')


######################################################################################################
def drawCheckerboard(rows, cols, numPixInBlock, imageMode, colour1, colour2, imageReturnType='image',datatype=np.uint8):
    """Draw checkerboard with 8-bit pixels 
    
   From http://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
   
   Args:
        | rows (int) : number or rows in checkerboard
        | cols (int) : number of columns in checkerboard 
        | numPixInBlock (int) : number of pixels to be used in one block of the checkerboard
        | imageMode (string) : PIL image mode [e.g. L (8-bit pixels, black and white), RGB (3x8-bit pixels, true color)]
        | colour1 (int or RGB tuple) : colour 1 specified according to the imageMode
        | colour2 (int or RGB tuple) : colour 2 specified according to the imageMode
        | imageReturnType: 'image' for PIL image, 'nparray' for numpy array
        | datatype (numpy data type) : numpy data type for the returned np.array
   
    Returns:
        | img          : checkerboard numpy array or PIL image (see imageReturnType) 
    
    Raises:
        | No exception is raised.
        
    Example Usage:
    
        rows = 5
        cols = 7
        pixInBlock = 4

        color1 = 0       
        color2 = 255      
        img = drawCheckerboard(rows,cols,pixInBlock,'L',color1,color2,'nparray')
        pilImg = Img.fromarray(img, 'L')
        pilImg.save('{0}.png'.format('checkerboardL'))


        color1 = (0,0,0)          
        color2 = (255,255,255)      
        pilImage = drawCheckerboard(rows,cols,pixInBlock,'RGB',color1,color2,'image')
        pilImage.save('{0}.png'.format('checkerboardRGB'))
        
    """
    width = numPixInBlock * cols
    height = numPixInBlock * rows
    coords = np.ogrid[0:height, 0:width]
    idx = (coords[0] // numPixInBlock + coords[1] // numPixInBlock) % 2
    vals = np.array([colour1, colour2], dtype=datatype)
    img = vals[idx]
    
    if (imageReturnType == 'nparray'):
        return img
    else:
        from PIL import Image as Img
        pilImage = Img.fromarray(img, imageMode)
        return pilImage


######################################################################################################
def extractGraph(filename, xmin, xmax, ymin, ymax, outfile=None,doPlot=False,\
        xaxisLog=False, yaxisLog=False, step=None, value=None):
    """Scan an image containing graph lines and produce (x,y,value) data.
    
    This function processes an image, calculate the location of pixels on a 
    graph line, and then scale the (r,c) or (x,y) values of pixels with non-zero 
    values. The 

    Get a bitmap of the graph (scan or screen capture).
    Take care to make the graph x and y axes horizontal/vertical.
    The current version of the software does not work with rotated images.
    Bitmap edit the graph. Clean the graph to the maximum extent possible,
    by removing all the clutter, such that only the line to be scanned is visible.
    Crop only the central block that contains the graph box, by deleting
    the x and y axes notation and other clutter. The size of the cropped image 
    must cover the range in x and y values you want to cover in the scan. The 
    graph image/box must be cut out such that the x and y axes min and max
    correspond exactly with the edges of the bitmap.
    You must end up with nothing in the image except the line you want 
    to digitize.

    The current version only handles single lines on the graph, but it does
    handle vertical and horizontal lines.
    
    The function can also write out a value associated with the (x,y) coordinates 
    of the graph, as the third column. Normally these would have all the same 
    value if the line represents an iso value.

    The x,y axes can be lin/lin, lin/log, log/lin or log/log, set the flags.

    Args:
        | filename: name of the image file
        | xmin: the value corresponding to the left side (column=0)
        | xmax: the value corresponding to the right side (column=max)
        | ymin: the value corresponding to the bottom side (row=bottom)
        | ymax: the value corresponding to the top side (row=top)
        | outfile: write the sampled points to this output file
        | doPlot: plot the digitised graph for visual validation
        | xaxisLog: x-axis is in log10 scale (min max are log values)
        | yaxisLog: y-axis is in log10 scale (min max are log values)
        | step: if not None only ouput every step values
        | value: if not None, write this value as the value column

    Returns:
        | outA: a numpy array with columns (xval, yval, value)
        | side effect: a file may be written
        | side effect: a graph may be displayed
        
    Raises:
        | No exception is raised.

    Author: neliswillers@gmail.com
    """

    from scipy import ndimage
    from skimage.morphology import medial_axis
    if doPlot:
        import pylab
        import matplotlib.pyplot as pyplot
     
    #read image file, as grey scale
    img = ndimage.imread(filename, True)

    # find threshold 50% up the way
    halflevel = img.min() + (img.max()-img.min()) /2
    # form binary image by thresholding
    img = img < halflevel
    #find the skeleton one pixel wide
    imgskel = medial_axis(img)
    
    #if doPlot:
        # pylab.imshow(imgskel)
        # pylab.gray()
        # pylab.show()

    # set up indices arrays to get x and y indices
    ind = np.indices(img.shape)

    #skeletonise the graph to one pixel only
    #then get the y pixel value, using indices
    yval = ind[0,...] * imgskel.astype(float)
    
    #if doPlot:
        # pylab.imshow(yval>0)
        # pylab.gray()
        # pylab.show()
        
    # invert y-axis origin from left top to left bottom
    yval = yval.shape[0] - np.max(yval, axis=0)
    #get indices for only the pixels where we have data
    wantedIdx = np.where(np.sum(imgskel, axis = 0) > 0)

    # convert to original graph coordinates
    cvec = np.arange(0.0,img.shape[1])

    xval = xmin + (cvec[wantedIdx] / img.shape[1]) * (xmax - xmin)
    xval = xval.reshape(-1,1)
    yval = ymin + (yval[wantedIdx] / img.shape[0]) * (ymax - ymin)
    yval = yval.reshape(-1,1)

    if xaxisLog:
        xval =  10** xval

    if yaxisLog:
        yval =  10 ** yval

    #build the result array
    outA = np.hstack((xval,yval))
    if value is not None:
        outA = np.hstack((outA,value*np.ones(yval.shape)))

    # process step intervals
    if step is not None:
        # collect the first value, every step'th value, and last value
        outA = np.vstack((outA[0,:],outA[1:-2:step,:],outA[-1,:]))

    #write output file
    if outfile is not None > 0 :
        np.savetxt(outfile,outA)

    if doPlot:
        fig = pyplot.figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(xval,yval)
        if xaxisLog:
            ax.set_xscale('log')
        if yaxisLog:
            ax.set_yscale('log')
        pylab.show()

    return outA


######################################################################################################
def makemotionsequence(imgfilename, mtnfilename,postfix,intTime,frmTim,outrows,outcols,
                imgRatio,pixsize,numsamples,fnPlotInput=None):
    r"""Builds a video from a still image and a displacement motion file.

    The objective with this function is to create a video sequence from a still
    image, as if the camera moved minutely during the sensor integration time.

    A static image is moved according to the (x,y) displacement motion in an
    input file.  The input file must be at least ten times plus a bit
    larger than the required output file.  The image input file is sampled with
    appropriate displacement for each point in the displacement file and pixel vlaues
    are accumulated in the output image. All of this temporal displacement and
    accumulation takes place in the context of a frame integration time and frame
    frequency.

    The key requirements for accuracy in this method is an input image with much
    higher resolution than the output image, plus a temporal displacement file with
    much higher temporal sampling than the sensor integration time.

    The function creates a sequence of images that can be used to create a video.
    Images are numbered in sequence, using the same base name as the input image.
    The sequence is generated in the current working directory.

    The function currently processes only monochrome images (M,N) arrays.

    The motion data file must be a compressed numpy npz or text file,
    with three columns:
    First column must be time, then movement along rows, then movement along columns.
    The units and scale of the motion columns must be the same units and scale as
    the pixel size in the output image.

    imgRatio x imgRatio number of pixels in the input (hires) image are summed
    together and stored in one output image pixel.  In other words if imgRatio is ten,
    each pixel in the output image will be the sum of 100 pixels in the imput image.
    During one integration time period the hires input image will be sampled at slightly
    different offsets (according to the motion file) and accumulated in an intermediate
    internal hires file.  This intermediate internal file is collapsed as described
    above.

    The function creates a series-numbered sequence if images that can be used to
    construct a video.  One easy means to create the video is to use VirtualDub,
    available at www.virtualdub.org/index.  In VirtualDub open the first image file
    in the numbered sequence, VirtualDub will then recognise the complete sequence
    as a video. Once loaded in VirtualDub, save the video as avi.

    Args:
        | imgfilename (str): static image filename (monochrome only)
        | mtnfilename (str): motion data filename.
        | postfix (str): add this string to the end of the output filename.
        | intTime (float): sensor integration time.
        | frmTim (float): sensor frame time.
        | outrows (int): number of rows in the output image.
        | outcols (int): number of columns in the output image.
        | imgRatio (float): hires image pixel count block size of one output image pixel
        | pixsize (float): pixel size in same units as motion file.
        | numsamples (int): number of motion input samples to be processed (-1 for all).
        | fnPlotInput (str): output plot filename (None for no plot).

    Returns:
        | True if successful, message otherwise, creates numbered images in current working directory

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """

    from scipy import ndimage
    from scipy import misc
    import os

    #read in the image and motion files.
    if not os.path.exists(imgfilename):
        return '{} not found'.format(imgfilename)
    imgIn = misc.imread(imgfilename)
    centrow = imgIn.shape[0]/2
    centcol = imgIn.shape[1]/2
    motionScale = pixsize / imgRatio

    if not os.path.exists(mtnfilename):
        return '{} not found'.format(mtnfilename)
    if '.npz' in mtnfilename:
        rcmotion = np.load(mtnfilename)['arr_0']
    elif '.txt' in mtnfilename:
        rcmotion = np.loadtxt(mtnfilename)
    else:
        return '{} not in appropriate format'.format(mtnfilename)

    mtnfilenamecore = os.path.split(mtnfilename)[1]
    mtnfilenamecore = mtnfilenamecore[:mtnfilenamecore.find('.')]

    #reset time to start at zero
    times = rcmotion[:,0] - rcmotion[0,0]
    drows = rcmotion[:,1]
    dcols = rcmotion[:,2]

    if fnPlotInput is not None:
        I = ryplot.Plotter(1,3,1,'', figsize=(6,9))
        I.showImage(1, imgIn)
        I.plot(2,times,rcmotion[:,1:3],'Input motion','Time s','Displacement',label=['row','col'])
        I.plot(3,times,rcmotion[:,1:3]/pixsize,'Input motion','Time s','Displacement pixels',label=['row','col'])
        I.saveFig(fnPlotInput)

    fullframe = 0
    subframes = 0
    outimage = np.zeros((outrows*imgRatio,outcols*imgRatio))

    if times.shape[0] < numsamples:
        numsamples = times.shape[0]

    for isample,time in enumerate(times):

        if isample <= numsamples:

            fracframe =  np.floor(time / frmTim)
            if fracframe >= fullframe + 1:
                #output and reset the present image
                outfilename = os.path.split(imgfilename)[1].replace('.png',
                           '-{}-{}-{:05d}.png'.format(mtnfilenamecore,postfix,fullframe))

                outimage = outimage/subframes
                saveimage = np.array([[np.sum(vchunk) for vchunk in np.split(hchunk, outrows, 1)]
                             for hchunk in np.split(outimage, outcols)])/imgRatio**2
                misc.imsave(outfilename, saveimage)
                outimage = np.zeros((outrows*imgRatio,outcols*imgRatio))
                fullframe += 1
                subframes = 0

            if time - fullframe * frmTim < intTime:
                #integrate the frames during integration time
                # print('{} {} integrate image {}'.format(time,fracframe, fullframe))
                roffs = drows[isample] / motionScale
                coffs = dcols[isample] / motionScale
                outimage += imgIn[
                            centrow+roffs-outrows*imgRatio/2:centrow+roffs+outrows*imgRatio/2,
                            centcol+coffs-outcols*imgRatio/2:centcol+coffs+outcols*imgRatio/2
                            ]

                subframes += 1
            else:
                # this sample is not integrated in the output image
                # print('{} {}'.format(time,fracframe))
                pass

    return True

######################################################################################################
def luminousEfficiency(vlamtype='photopic', wavelen=None, eqnapprox=False):
    r"""Returns the photopic luminous efficiency function on wavelength intervals

    Type must be one of:

    photopic: CIE Photopic V(lambda) modified by Judd (1951) and Vos (1978) [also known as CIE VM(lambda)] 
    scotopic:  CIE (1951) Scotopic V'(lambda)
    CIE2008v2:  2 degree CIE "physiologically-relevant" luminous efficiency Stockman & Sharpe
    CIE2008v10:  10 degree CIE "physiologically-relevant" luminous efficiency Stockman & Sharpe

    For the equation approximations (only photoic and scotopic), if wavelength is not 
    given a vector is created 0.3-0.8 um.

    For the table data, if wavelength is not given a vector is read from the table.


    CIE Photopic V(l) modified by Judd (1951) and Vos (1978) [also known as CIE VM(l)] 
    from http://www.cvrl.org/index.htm

    Args:
        | vlamtype (str): type of curve required
        | wavelen (np.array[]): wavelength in um
        | eqnapprox (bool): if False read tables, if True use equation

    Returns:
        | luminousEfficiency (np.array[]): luminous efficiency 
        | wavelen (np.array[]): wavelength in um

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """


    if eqnapprox:
        if wavelen is None:
            wavelen = np.linspace(0.3, 0.8, 100)
        if 'photopic' in vlamtype:
            vlam = 1.019 * np.exp(-285.51 * (wavelen - 0.5591) ** 2 ).reshape(-1,)
        elif  'scotopic' in vlamtype:
            vlam = 0.99234 * np.exp(-321.1 * (wavelen - 0.502) ** 2 ).reshape(-1,)
        else:
            return None, None

    else:

        if 'photopic' in vlamtype:
            vlamname = 'vljve.csv'
        elif  'scotopic' in vlamtype:
            vlamname = 'scvle.csv'
        elif  'CIE2008v2' in vlamtype:
            vlamname = 'linCIE2008v2e_1.csv'
        elif  'CIE2008v10' in vlamtype:
            vlamname = 'linCIE2008v10e_1.csv'
        else:
            return None, None

        #load data file from the pyradi directories, not local dir
        resource_package = 'pyradi'  #__name__  ## Could be any module/package name.
        resource_path = os.path.join('data', 'photometry',vlamname)
        dat = pkg_resources.resource_string(resource_package, resource_path)
        if sys.version_info[0] > 2:
            dat = np.loadtxt(StringIO(dat.decode('utf-8')),delimiter=",")
        else:
            dat = np.genfromtxt(StringIO(dat),delimiter=",")

        if wavelen is not None:
            vlam = np.interp(wavelen*1000., dat[:,0],dat[:,1],left=dat[0,1],right=dat[-1,1])
        else:
            wavelen = dat[:,0]/1000.
            vlam = dat[:,1]

    return vlam, wavelen


##############################################################################################
##############################################################################################
##############################################################################################
# to calculate the MTF degradation from the pupil function
def calcMTFwavefrontError(sample, wfdisplmnt, xg, yg, specdef,
                     samplingStride = 1,clear='Clear'):
    """Given a mirror figure error, calculate MTF degradation from ideal
    
    An aperture has an MTF determined by its shape.  A clear aperture has
    zero phase delay and the MTF is determined only by the aperture shape.
    Any phase delay/error in the wavefront in the aperture will result in a 
    lower MTF than the clear aperture diffraction MTF.
    
    This function calculates the MTF degradation attributable to a wavefront
    error, relative to the ideal aperture MTF.
    
    The optical transfer function is the Fourier transform of the point spread 
    function, and the point spread function is the square absolute of the inverse 
    Fourier transformed pupil function. The optical transfer function can also 
    be calculated directly from the pupil function. From the convolution theorem 
    it can be seen that the optical transfer function is  the autocorrelation of 
    the pupil function <https://en.wikipedia.org/wiki/Optical_transfer_function>.  

    The pupil function comprises a masking shape (the binary shape of the pupil) 
    and a transmittance and spatial phase delay inside the mask. A perfect aperture 
    has unity transmittance and zero phase delay in the mask. Some pupils have 
    irregular pupil functions/shapes and hence the diffraction MTF has to be 
    calculated numerically using images (masks) of the pupil function.   

    From the OSA Handbook of Optics, Vol II, p 32.4:  
    For an incoherent optical system, the OTF is proportional to the two-dimensional 
    autocorrelation of the exit pupil. This calculation can account for any phase 
    factors across the pupil, such as those arising from aberrations or defocus. 
    A change of variables is required for the identification of an autocorrelation 
    (a function of position in the pupil) as a transfer function (a function of
    image-plane spatial frequency). The change of variables is


    xi = {x}/{lambda d_i}
 
    where $x$ is the autocorrelation shift distance in the pupil, $\lambda$ is 
    the wavelength, and $d_i$ is the distance from the exit pupil to the image. 
    A system with an exit pupil of full width $D$ has an image-space cutoff 
    frequency (at infinite conjugates)  of

    xi_{cutoff} ={D}/{lambda f}

    In this analysis we assume that 
    1. the sensor is operating at infinite conjugates. 
    2. the mask falls in the entrance pupil shape.

    The MTF is calculated as follows:

    1. Read in the pupil function mask and create an image of the mask.
    2. Calculate the two-dimensional autocorrelation function of the binary 
       image (using the SciPy two-dimensional correlation function `signal.correlate2d`).
    3. Scale the magnitude and $(x,y)$ dimensions according to the dimensions of 
       the physical pupil.
       
    The the array containing the wavefront displacement in the pupil must have np.nan 
    values outside the pupil. The np.nan values are ignored and not included in the 
    calculation. Obscurations can be modelled by placing np.nan in the obscuration.
    
    The specdef dictionary has a string key to identify (name) the band, with a
    single float contents which is the wavelength associated with this band.

    Args:
        | sample (string): an identifier string to be used in the plots
        | wfdisplmnt (nd.array[M,N]): wavefront displacement in m
        | xg (nd.array[M,N]): x values from meshgrid, for wfdisplmnt
        | yg (nd.array[M,N]): y values from meshgrid, for wfdisplmnt
        | specdef (dict): dictionary defining spectral wavelengths
        | samplingStride (number): sampling stride to limit size and processing
        | clear (string): defines the dict key for clear aperture reference

    Returns:
        | dictionaries below have entries for all keys in specdef.
        | wfdev (dict): subsampled wavefront error in m
        | phase (dict): subsampled wavefront error in rad
        | pupfn (dict): subsampled complex pupil function
        | MTF2D (dict): 2D MTF in (x,y) format
        | MTFpol (dict):  2D MTF in (r,theta) format
        | specdef (): specdef dictionary as passed plus clear entry
        | MTFmean (dict): mean MTF across all rotation angles
        | rho (nd.array[M,]): spatial frequency scale in cy/mrad
        | fcrit (float): cutoff or critical spatial frequency cy/mrad
        | clear (string): key used to signify the clear aperture case.

    Raises:
        | No exception is raised.
    
    """

    from scipy import signal
    import pyradi.ryplot as ryplot

    error = {}
    wfdev = {}
    phase = {}
    pupfn = {}
    pupfnz = {}
    MTF2D = {}
    MTFpol = {}
    MTFmean = {}
    freqfsm = {}
    rho = {}
    fcrit = {}

    pim = ryplot.ProcessImage()
 
    # make the clear case zero error
    wfdev[clear] = np.where(np.isnan(wfdisplmnt),np.nan,0)
    specdef[clear] = 1e300
    
    # three cases, clear is done for near infinite wavelength (=zero phase)
    for specband in specdef:
        
        # the physical deviation/error from the ideal mirror figure
        # force nan outside of valid mirror surface
        if clear not in specband:
            wfdev[specband] = np.where(np.isnan(wfdisplmnt),np.nan,wfdisplmnt)

        # resample with stride to reduce processing load
        wfdev[specband] = wfdev[specband][::samplingStride,0:wfdev[specband].shape[0]:samplingStride] 
        
        # one wavelength error is 2pi rad phase shift
        # use physical displacement and wavelength to normalise to # of wavelengths 
        phase[specband] = np.where(np.isnan(wfdev[specband]), np.nan, 2*np.pi*(wfdev[specband]/specdef[specband]))

        # phase into complex pupil function
        pupfn[specband] = np.exp(-1j * phase[specband])
        # correlation fn does not work if nan in data set, force nan to zero
        pupfnz[specband] = np.where(np.isnan(pupfn[specband]),0,pupfn[specband])
        # correlation to get optical transfer function
        corr = signal.correlate2d(pupfnz[specband], np.conj(pupfnz[specband]), boundary='fill', mode='full')
        # normalise and get abs value to get MTF
        MTF2D[specband] = np.abs(corr / np.max(corr))
        
        polar_c, _, _ = pim.reprojectImageIntoPolar( 
                            MTF2D[specband].reshape(MTF2D[specband].shape[0],MTF2D[specband].shape[1],1), 
                            None, False,cval=0.)
        MTFpol[specband] = polar_c[:,:,0]
       
        MTFmean[specband] = MTFpol[specband].mean(axis=1)

        #calculate the aperture diameter, geometric mean along x and y
        pdia = np.sqrt(np.abs(np.nanmax(xg)-np.nanmin(xg)) * np.abs(np.nanmax(yg)-np.nanmin(yg)))
        freqfsm[specband] = np.sqrt(2.) * pdia  / (specdef[specband] * 1000.)
        rho[specband] = np.linspace(0,freqfsm[specband],MTFpol[specband].shape[0]) 
        fcrit[specband] = freqfsm[specband]/np.sqrt(2.)
        
    return  wfdev, phase, pupfn, MTF2D, MTFpol, specdef, MTFmean, rho, fcrit, clear



##############################################################################################
# to map an cartesian (r,theta) image to cartesian (x,y)
def polar2cartesian(xycoords, inputshape, origin):
    """Converting polar (r,theta) array indices to Cartesian (x,y) array indices. 

    This function is called from scipy.ndimage.geometric_transform which calls this function 
    as follows:
    polar2cartesian: A callable object that accepts a tuple of length equal to the output 
    array rank, and returns the corresponding input coordinates as a tuple of length equal 
    to the input array rank.
    
    theta goes from 0 to 2pi.  the x,y coords maps from -r to +r.

    For an example application, see the function warpPolarImageToCartesianImage below
    
    http://stackoverflow.com/questions/2164570/reprojecting-polar-to-cartesian-grid 
    note that we changed the code from the original  

    Args:
        | xycoords (tuple): x,y values for which r,theta coords must be found
        | inputshape (tuple):  shape of the polar input array
        | origin (tuple):  x and y indices of where the origin should be in the output array

    Returns:
         | r,theta_index (tuple) : indices into the the r,theta array corresponding to xycoords

    Raises:
        | No exception is raised.
    
    """

    xindex, yindex = xycoords
    x0, y0 = origin
    x = xindex - x0
    y = yindex - y0

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta_index = np.round((theta + np.pi) * (inputshape[1]-1.) / (2 * np.pi))

    return (r,theta_index)

##############################################################################################
def warpPolarImageToCartesianImage(mesh):
    """Convert an image in (r,theta) format to (x,y) format
    
    The 0th and 1st axes correspond to r and theta, respectively.
    theta goes from 0 to 2pi, and r's units are just its indices.
    output image is twice the size of r length in both x and y
    
    http://stackoverflow.com/questions/2164570/reprojecting-polar-to-cartesian-grid  
    
    Example code:
    size = 100
    dset = np.random.random((size,size))
    mesh_cart = warpPolarImageToCartesianImage(dset)
    p = ryplot.Plotter(1,1,2);
    p.showImage(1, dset);
    p.showImage(2, mesh_cart);

    Args:
         | mesh (np.array) : array in r,theta coordinates

    Returns:
         | mesh_cart (np.array) : array in x,y coordinates

    Raises:
        | No exception is raised.
    """
    import scipy as sp
    from scipy import ndimage

    output_shape=(mesh.shape[0] * 2, mesh.shape[0] * 2)
    mesh_cart = sp.ndimage.geometric_transform(mesh, polar2cartesian,
                order=0,output_shape=output_shape, 
            extra_keywords={'inputshape':mesh.shape,'origin':(mesh.shape[0], mesh.shape[0])})
    return mesh_cart

##############################################################################################
##############################################################################################
##############################################################################################
class Spectral(object):
    """Generic spectral can be used for any spectral vector
    """
    ############################################################
    ##
    def __init__(self, ID, value, wl=None, wn=None, desc=None):
        """Defines a spectral variable of property vs wavelength or wavenumber

        One of wavelength or wavenunber must be supplied, the other is calculated.
        No assumption is made of the sampling interval on eiher wn or wl.

        The constructor defines the 

            Args:
                | ID (str): identification string
                | wl (np.array (N,) or (N,1)): vector of wavelength values
                | wn (np.array (N,) or (N,1)): vector of wavenumber values
                | value (np.array (N,) or (N,1)): vector of property values
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        self.ID = ID
        self.desc = desc

        self.wn = wn
        self.wl = wl
        if wn is not None:
            self.wn =  wn.reshape(-1,1)
            self.wl = 1e4 /  self.wn
        elif wl is not None:
            self.wl =  wl.reshape(-1,1)
            self.wn = 1e4 /  self.wl
        else:
            pass

        if isinstance(value, Number):
            if wn is not None:
                self.value = value * np.ones(self.wn.shape)
            else:
                self.value = value 
        elif isinstance(value, np.ndarray):
            self.value = value.reshape(-1,1)


    ############################################################
    ##
    def __str__(self):
        """Returns string representation of the object

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.wn, np.ndarray):
            numpts = self.wn.shape[0]
            stride = int(numpts / 3)

        strn = '{}\n'.format(self.ID)
        strn += ' {}-desc: {}\n'.format(self.ID,self.desc)
        # for all numpy arrays, provide subset of values
        keys = sorted(list(self.__dict__.keys()))
        for var in keys:
            # then see if it is an array
            if isinstance(eval('self.{}'.format(var)), np.ndarray):
                # print(eval('self.{}'.format(var)).shape)
                svar = (np.vstack((eval('self.{}'.format(var))[0::stride], eval('self.{}'.format(var))[-1] ))).T
                strn += ' {}-{} (subsampled.T): {}\n'.format(self.ID,var, svar)
            elif isinstance(eval('self.{}'.format(var)), Number):
                svar = eval('self.{}'.format(var))
                # print(type(svar),svar)
                strn += ' {}-{} (subsampled.T): {:.4e}\n'.format(self.ID,var, svar)

        return strn

    ############################################################
    ##
    def vecalign(self, other):
        """returns two spectral values properly interpolated and aligned to same base

        it is not intended that the function will be called directly by the user

            Args:
                | other (Spectral): the other Spectral to be used in addition

            Returns:
                | wl, wn, s, o

            Raises:
                | No exception is raised.
        """

        if self.wn is not None and other.wn is not None:
            # create new spectral in wn wider than either self or other.
            wnmin = min(np.min(self.wn),np.min(other.wn))
            wnmax = max(np.max(self.wn),np.max(other.wn))
            wninc = min(np.min(np.abs(np.diff(self.wn,axis=0))),np.min(np.abs(np.diff(other.wn,axis=0))))
            wn = np.linspace(wnmin, wnmax, (wnmax-wnmin)/wninc)
            wl = 1e4 / self.wn
            if np.mean(np.diff(self.wn,axis=0)) > 0:
                s = np.interp(wn,self.wn[:,0], self.value[:,0])
                o = np.interp(wn,other.wn[:,0], other.value[:,0])
            else:
                s = np.interp(wn,np.flipud(self.wn[:,0]), np.flipud(self.value[:,0]))
                o = np.interp(wn,np.flipud(other.wn[:,0]), np.flipud(other.value[:,0]))
        elif self.wn is     None and other.wn is not None:
            o = other.value
            s = self.value
            wl = other.wl    
            wn = other.wn    

        elif self.wn is not None and other.wn is     None:
            o = other.value
            s = self.value
            wl = self.wl    
            wn = self.wn    

        else:
            o = other.value
            s = self.value
            wl = None    
            wn = None    

        return wl, wn, s, o

    ############################################################
    ##
    def __mul__(self, other):
        """Returns a spectral product

        it is not intended that the function will be called directly by the user

            Args:
                | other (Spectral): the other Spectral to be used in multiplication

            Returns:
                | str

            Raises:
                | No exception is raised.
        """

        if isinstance(other, Spectral):
            wl, wn, s, o = self.vecalign(other)
            rtnVal = Spectral(ID='{}*{}'.format(self.ID,other.ID), value=s * o, wl=wl, wn=wn,
                    desc='{}*{}'.format(self.desc,other.desc))
        else:
            rtnVal = Spectral(ID='{}*{}'.format(self.ID,other), value=self.value * other, wl=self.wl, 
                wn=self.wn,desc='{}*{}'.format(self.desc,other))

        return rtnVal


    ############################################################
    ##
    def __add__(self, other):
        """Returns a spectral product

        it is not intended that the function will be called directly by the user

            Args:
                | other (Spectral): the other Spectral to be used in addition

            Returns:
                | str

            Raises:
                | No exception is raised.
        """

        if isinstance(other, Spectral):
            wl, wn, s, o = self.vecalign(other)
            rtnVal = Spectral(ID='{}+{}'.format(self.ID,other.ID), value=s + o, wl=wl, wn=wn,
                    desc='{}+{}'.format(self.desc,other.desc))
        else:
            rtnVal = Spectral(ID='{}+{}'.format(self.ID,other), value=self.value + other, wl=self.wl, 
                wn=self.wn,desc='{}+{}'.format(self.desc,other))

        return rtnVal


    ############################################################
    ##
    def __sub__(self, other):
        """Returns a spectral product

        it is not intended that the function will be called directly by the user

            Args:
                | other (Spectral): the other Spectral to be used in subtraction

            Returns:
                | str

            Raises:
                | No exception is raised.
        """

        if isinstance(other, Spectral):
            wl, wn, s, o = self.vecalign(other)

            rtnVal = Spectral(ID='{}-{}'.format(self.ID,other.ID), value=s - o, wl=wl, wn=wn,
                    desc='{}-{}'.format(self.desc,other.desc))
        else:
            rtnVal = Spectral(ID='{}-{}'.format(self.ID,other), value=self.value - other, wl=self.wl, 
                wn=self.wn,desc='{}-{}'.format(self.desc,other))

        return rtnVal




    ############################################################
    ##
    def __pow__(self, power):
        """Returns a spectral to some power

        it is not intended that the function will be called directly by the user

            Args:
                | power (number): spectral raised to power

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        return Spectral(ID='{}**{}'.format(self.ID,power), value=self.value ** power, 
            wl=self.wl, wn=self.wn,desc='{}**{}'.format(self.desc,power))

    ############################################################
    ##
    def plot(self, filename=None, heading=None, ytitle=''):
        """Do a simple plot of spectral variable(s)

            Args:
                | filename (str): filename for png graphic
                | heading (str): graph heading
                | ytitle (str): graph y-axis title

            Returns:
                | Nothing, writes png file to disk

            Raises:
                | No exception is raised.
        """

        if filename == None:
            filename = '{}-{}'.format(self.ID,self.desc)

        if heading == None:
            heading = self.desc

        if 'value' in self.__dict__:
            import pyradi.ryplot as ryplot
            p = ryplot.Plotter(1,2,1,figsize=(8,5))

            if isinstance(self.value, np.ndarray):
                xvall = self.wl
                xvaln = self.wn
                yval = self.value
            else:
                xvall = np.array([1,10])
                xvaln = 1e4 / xvall
                yval = np.array([self.value,self.value])

            p.plot(1,xvall,yval,heading,'Wavelength $\mu$m', ytitle)
            p.plot(2,xvaln,yval,heading,'Wavenumber cm$^{-1}$', ytitle)

            p.saveFig(ryfiles.cleanFilename(filename))



##############################################################################################
##############################################################################################
##############################################################################################
class Atmo():
    """Atmospheric spectral such as transittance or attenuation coefficient
    """
    ############################################################
    ##
    def __init__(self, ID, distance=None, wl=None, wn=None,  tran=None, atco=None, prad=None, desc=None):
        """Defines a spectral variable of property vs wavelength or wavenumber

        One of wavelength or wavenunber must be supplied, the other is calculated.
        No assumption is made of the sampling interval on either wn or wl.

        If distance is not None, tran=transmittance, prad=path radiance, at distance,
        then the atco and normalised path radiance is calculated.

        If distance is None, atco (attenuation coefficients in m^{-1}), and
        normalised prad (Lpath/(1-tauPath)) are used as supplied

        All spectral variables must be on the same spectral vector wn or wl


            Args:
                | ID (str): identification string
                | distance (scalar): distance in m if transmittance, or None if att coeff
                | wl (np.array (N,) or (N,1)): vector of wavelength values 
                | wn (np.array (N,) or (N,1)): vector of wavenumber values
                | tran (np.array (N,) or (N,1)): transmittance
                | atco (np.array (N,) or (N,1)): attenuation coeff in m-1
                | prad (np.array (N,) or (N,1)): path radiance over distance in W/(sr.m2.cm-1)
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        self.ID = ID
        self.desc = desc

        if distance is not None:
            if atco is None:
                self.atco = Spectral(self.ID+'-atco', value=-np.log(tran)/distance,wl=wl,wn=wn,desc=self.desc)
            else:
                self.atco = Spectral(self.ID+'-atco', value=atco,wl=wl,wn=wn,desc=self.desc)
               
            if prad is not None:
                self.prad = Spectral(self.ID+'-prad',value=prad.reshape(-1,1)/(1-np.exp(-self.atco.value*distance)),
                    wl=wl,wn=wn,desc=self.desc)
        else:
            self.atco = Spectral(self.ID+'-atco', value=atco,wl=wl,wn=wn,desc=self.desc)
            self.prad = Spectral(self.ID+'-prad', value=prad,wl=wl,wn=wn,desc=self.desc)
           
    ############################################################
    ##
    def __str__(self):
        """Returns string representation of the object

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """

        strn = 'ID: {}\n'.format(self.ID)
        strn += 'desc: {}\n'.format(self.desc)
        strn += '{}\n'.format(self.atco)
        strn += '{}\n'.format(self.prad)

        return strn

    ############################################################
    ##
    def tauR(self, distance):
        """Calculates the transmittance at distance 

        Distance is in m

            Args:
                | distance (scalar or np.array (M,)): distance in m if transmittance, or None if att coeff

            Returns:
                | transmittance (np.array (N,M) ): transmittance along N at distance along M

            Raises:
                | No exception is raised.
        """
        distance = np.array(distance).reshape(1,-1)
        return np.exp(-distance * self.atco.value)

    ############################################################
    ##
    def pathR(self, distance):
        """Calculates the path radiance at distance

        Distance is in m

            Args:
                | distance (scalar or np.array (M,)): distance in m if transmittance, or None if att coeff

            Returns:
                | transmittance (np.array (N,M) ): transmittance along N at distance along M

            Raises:
                | No exception is raised.
        """
        distance = np.array(distance).reshape(1,-1)
        tran = np.exp(-distance * self.atco.value)
        return self.prad.value * (1 - tran)


##############################################################################################
##############################################################################################
##############################################################################################
class Sensor():
    """Sensor characteristics
    """
    ############################################################
    ##
    def __init__(self, ID, fno, detarea, inttime, tauOpt=1, quantEff=1, 
                pfrac=1, desc=''):
        """Sensor characteristics

            Args:
                | ID (str): identification string
                | fno (scalar): optics fnumber
                | detarea (scalar): detector area
                | inttime (scalar): detector integration time
                | tauOpt (scalar or Spectral): sensor optics transmittance 
                | quantEff (scalar or Spectral): detector quantum efficiency 
                | pfrac (scalar):  fraction of optics clear aperture
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        self.ID = ID
        self.fno = fno
        self.detarea = detarea
        self.inttime = inttime
        self.pfrac = pfrac
        self.desc = desc

        if isinstance(quantEff, Number):
            self.quantEffVal = Spectral(ID='{}-quantEff'.format(self.ID), value=quantEff, desc='{}-quantEff'.format(self.desc))
        else:
            self.quantEffVal = quantEff

        if isinstance(tauOpt, Number):
            self.tauOptVal = Spectral(ID='{}-tauOpt'.format(self.ID), value=tauOpt, desc='{}-tauOpt'.format(self.desc))
        else:
            self.tauOptVal = tauOpt

           
    ############################################################
    ##
    def __str__(self):
        """Returns string representation of the object

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        strn =  'Sensor ID: {}\n'.format(self.ID)
        strn += 'desc: {}\n'.format(self.desc)
        strn += 'fno: {:.3f}\n'.format(self.fno)
        strn += 'detarea: {:.3e}\n'.format(self.detarea)
        strn += 'inttime: {:.6f}\n'.format(self.inttime)
        strn += 'pfrac: {:.3f}\n'.format(self.pfrac)
        strn += '{}\n'.format(self.tauOptVal)
        strn += '{}\n'.format(self.quantEffVal)

        return strn


    ############################################################
    ##
    def tauOpt(self):
        """Returns scaler or np.array for optics transmittance

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.tauOptVal, Spectral):
             rtnVal = self.tauOptVal.value
           
        if isinstance(self.tauOptVal, Number):
            rtnVal = self.tauOptVal 

        return rtnVal

    ############################################################
    ##
    def QE(self):
        """Returns scaler or np.array for detector quantEff

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.quantEffVal, Spectral):
             rtnVal = self.quantEffVal.value
           
        if isinstance(self.quantEffVal, Number):
            rtnVal = self.quantEffVal 

        return rtnVal

##############################################################################################
##############################################################################################
##############################################################################################
class Target(Spectral):
    """Target / Source characteristics
    """
    ############################################################
    ##
    def __init__(self, ID, tmprt, emis, refl=1, cosTarg=1, taumed=1, scale=1, desc=''):
        """Source characteristics

            This function supports two alternative models:

            1.  If it is a thermally radiating source,
                use temperature and emissivity only leave all other at unity.

            2.  If it is reflected light (i.e., sunlight), set temperature and
                emissivity to the illuminating source (6000 K, 1, if sun) and use
                refl, cosTarg, taumed and scale for reflected light on the ground.

            emis must always be a spectral, refl and taumed may be scalars or spectrals.
            The emis wavelength/number vector serves as the base spectral vector.

            see http://nbviewer.jupyter.org/github/NelisW/ComputationalRadiometry/blob/master/09b-StaringArrayDetectors.ipynb#Electron-count--for-various-sources

            Args:
                | ID (str): identification string
                | emis (Spectral):  radiance source surface emissivity 
                | tmprt (scalar): surface temperature
                | refl (scalar or Spectral): surface reflectance if reflecting
                | cosTarg (scalar): cosine between surface normal and illumintor direction
                | taumed (scalar or Spectral): transmittance between the surface and illumintor 
                | scale (scalar): surface radiance scale factor, sun: 2.17e-5, otherwise 1.
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        self.ID = ID
        if isinstance(tmprt, Number):
            self.tmprt = np.array([tmprt]).reshape(1,-1)
        else:
            self.tmprt = tmprt.reshape(1,-1)
        self.cosTarg = cosTarg
        self.scale = scale
        self.desc = desc

        if isinstance(emis, Spectral):
            self.emisVal = emis
        else:
            print('Target {} emis must be of type Spectral'.format(self.ID))
            self.emisVal = None

        if isinstance(refl, Number):
            self.reflVal = Spectral(ID='{}-refl'.format(self.ID), value=refl, desc='{}-refl'.format(self.desc))
        else:
            self.reflVal = refl

        if isinstance(taumed, Number):
            self.taumedVal = Spectral(ID='{}-taumed'.format(self.ID), value=taumed, desc='{}-taumed'.format(self.desc))
        else:
            self.taumedVal = taumed

          
    ############################################################
    ##
    def __str__(self):
        """Returns string representation of the object

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        strn =  'Sensor ID: {}\n'.format(self.ID)
        strn += 'desc: {}\n'.format(self.desc)
        strn += 'tmprt: {}\n'.format(self.tmprt)
        strn += 'cosTarg: {}\n'.format(self.cosTarg)
        strn += 'scale: {}\n'.format(self.scale)
        strn += '{}\n'.format(self.emisVal)
        strn += '{}\n'.format(self.reflVal)
        strn += '{}\n'.format(self.taumedVal)

        return strn

    ############################################################
    ##
    def emis(self):
        """Returns scaler or np.array for emissivity

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.emisVal, Spectral):
             rtnVal = self.emisVal.value
           
        if isinstance(self.emisVal, Number):
            rtnVal = self.emisVal 

        return rtnVal


    ############################################################
    ##
    def refl(self):
        """Returns scaler or np.array for reflectance

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.reflVal, Spectral):
             rtnVal = self.reflVal.value
           
        if isinstance(self.reflVal, Number):
            rtnVal = self.reflVal 

        return rtnVal


    ############################################################
    ##
    def taumed(self):
        """Returns scaler or np.array for atmospheric transmittance to illuminating source

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.taumedVal, Spectral):
             rtnVal = self.taumedVal.value
           
        if isinstance(self.taumedVal, Number):
            rtnVal = self.taumedVal 

        return rtnVal


    ############################################################
    ##
    def radiance(self, units='el'):
        """Returns radiance spectral for target

        The type of spectral is one of the following:
           type='el  [W/(m$^2$.$\mu$m)]
           type='ql'  [q/(s.m$^2$.$\mu$m)]
           type='en'  [W/(m$^2$.cm$^{-1}$)]
           type='qn'  [q/(s.m$^2$.cm$^{-1}$)]

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """

        specprod = self.emisVal * self.reflVal * self.taumedVal
        vscale =  specprod.value * self.scale * self.cosTarg

        if units[1] == 'l':
            radiance =  vscale * ryplanck.planck(specprod.wl,self.tmprt, type=units).reshape(-1,self.tmprt.shape[1])
            rtnVal = Spectral(ID='{}-{}-radiance'.format(self.ID,self.tmprt), value=radiance, wl=specprod.wl, desc='{}'.format(self.desc))
        elif units[1] == 'n':
            radiance =  vscale * ryplanck.planck(specprod.wn,self.tmprt, type=units).reshape(-1,self.tmprt.shape[1])
            rtnVal = Spectral(ID='{}-{}-radiance'.format(self.ID,self.tmprt), value=radiance, wn=specprod.wn, desc='{}'.format(self.desc))
        else:
            radiance = None

        return rtnVal


############################################################
##
def differcommonfiles(dir1,dir2, patterns='*'):
    """Find if common files in two dirs are different
    Directories are not recursed, only common files are binary compared

    Args:
        | filename (dir1): first directory name
        | filename (dir2): first directory name

    Returns:
        | file (list): list of common files
        | different (list) : list of flags of different flags

    Raises:
        | No exception is raised.
    """

    import pyradi.ryfiles as ryfiles

    bufsize = 2**8

    dlist = []
    flist = []
    file1s = ryfiles.listFiles(dir1, patterns, recurse=0, return_folders=0, useRegex=False)
    file2s = ryfiles.listFiles(dir2, patterns, recurse=0, return_folders=0, useRegex=False)
    print(file1s)
    print(file2s)
    for file1 in file1s:
        for file2 in file2s:
            if os.path.basename(file1) == os.path.basename(file2):
                print(os.path.basename(file1), os.path.basename(file1))
                flist.append(os.path.basename(file1))
                fp1 = open(file1, 'rb')
                fp2 = open(file2, 'rb')
                different = False
                while True:
                    # print('*')
                    b1 = fp1.read(bufsize)
                    b2 = fp2.read(bufsize)
                    if b1 != b2:
                        different = True
                        break
                    if not b1 or not b2:
                        break
                dlist.append(different)

    return(flist,dlist)



############################################################
##
def blurryextract(img, inputOrigin, outputSize, blocksize, sigma=0., filtermode='reflect' ):
    """Slice region from 2d array, blur it with gaussian kernel, and coalesce to lower resolution

    The image is blurred with a gaussian kernel of size sigma, using filtermode.
    Then the slice is calculated using the inputOrigin, required output size and 
    the block size.  The sliced image is then lowered in resolution by averaging
    blocks of values in the input image into the output image.

    If the input image size/bounds are to be exceeded in the calculation 
    the function returns None.

    | https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html
    | http://ajcr.net/Basic-guide-to-einsum/
    | https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
    | https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
    | https://stackoverflow.com/questions/26064594/how-does-numpy-einsum-work
    |
    | http://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    | http://stackoverflow.com/questions/21220942/sums-of-subarrays
    | https://en.wikipedia.org/wiki/Einstein_notation
    | https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.einsum.html
    | https://stackoverflow.com/questions/28652976/passing-array-range-as-argument-to-a-function
    | https://stackoverflow.com/questions/13706258/passing-python-slice-syntax-around-to-functions
    |
    | https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.filters.gaussian_filter.html


    Args:
        | img (nd.array(M,N)): 2D input array
        | inputOrigin ([lower0,lower1]): start coordinates in the input image
        | outputSize ([size0,size1]): size of the output array
        | blocksize ([block0,block1]): input image block size to be 
        |   averaged into single output image
        | sigma (float): gaussian kernel rms size in pixel counts
        | sigma (float): gaussian kernel filter mode

    Returns:
        | imgo (nd.array of floats): smaller image or None if out of bounds

    Raises:
        | No exception is raised.
    """
    import scipy.ndimage.filters as filters

    # filter the input image
    img = img.astype('float')
    if sigma > 0:
        img = filters.gaussian_filter(img,sigma=sigma, mode=filtermode)

    # slice the region from the input image
    inputUpper = [inputOrigin[0]+outputSize[0]*blocksize[0],
                  inputOrigin[1]+outputSize[1]*blocksize[1]]
    if inputUpper[0]>img.shape[0] or inputUpper[1]>img.shape[1]:
        print('blurryextract: input image too small for required output image')
        return None

    imgn = img[inputOrigin[0]:inputUpper[0],inputOrigin[1]:inputUpper[1]].copy()

    # prepare to lower resolution: make appropriate size
    h, w = imgn.shape
    h = (h // blocksize[0])*blocksize[0]
    w = (w // blocksize[1])*blocksize[1]
    # create correct size for einsum to work
    imgn = imgn[:h,:w]
    # use einstein sums to average aggregate blocks into single pixels
    imgo = np.einsum('ijkl->ik', 
          imgn.reshape(h // blocksize[0], blocksize[0], -1, blocksize[1]))\
                  / (blocksize[0]*blocksize[1])

    return imgo


################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':
    import math
    import sys
    from scipy.interpolate import interp1d
    import pyradi.ryplanck as ryplanck
    import pyradi.ryplot as ryplot
    import pyradi.ryfiles as ryfiles
    import pyradi.rymodtran as rymodtran
    import os
    import collections

    rit = intify_tuple

    figtype = ".png"  # eps, jpg, png
    # figtype = ".eps"  # eps, jpg, png

    # this path assumes that pyradi-data is cloned on the same
    # level as pradi itself.
    pathtoPyradiData = '../../pyradi-data'

    if sys.version_info[0] > 2:
        spectrals = {}
        atmos = {}
        sensors = {}
        targets = {}
    else:
        spectrals = collections.OrderedDict()   
        atmos = collections.OrderedDict()   
        sensors = collections.OrderedDict()   
        targets = collections.OrderedDict()   

    doAll = True

    if doAll:

        filename = 'rawFile.bin'
        # create a temporary image for subsampling
        img = drawCheckerboard(rows=2**11, cols=2*11, numPixInBlock=2**4, imageMode='L', 
                colour1=0, colour2=1, imageReturnType='nparray',datatype=np.float)
        # do the filtering and averaging
        imgo = blurryextract(img, inputOrigin=[10,10], outputSize=[64,64], sigma=3, blocksize=[2,4])
        # confirm image
        p = ryplot.Plotter(1,3,1,filename)
        p.showImage(1,imgo)
        p.saveFig('{}.png'.format(filename[:-4]))
        # fil = open(filename,'wb')
        # fil.write(imgo)

    if doAll:
        # test warpPolarImageToCartesianImage
        size = 100
        np.random.seed(1)
        dset = np.random.random((size,size))
        mesh_cart = warpPolarImageToCartesianImage(dset)
        p = ryplot.Plotter(1,1,2);
        p.showImage(1, dset);
        p.showImage(2, mesh_cart);
        p.saveFig('warpPolarImageToCartesianImage.png')


    if doAll:

        # test loading of spectrals
        print('\n---------------------Spectrals:')
        spectral = np.loadtxt('data/MWIRsensor.txt')
        spectrals['ID0'] = Spectral('ID1',value=.3,desc="const value")
        spectrals['ID1'] = Spectral('ID1',value=spectral[:,1],wl=spectral[:,0],desc="spec value")

        spectrals['IDp00'] = spectrals['ID0'] * spectrals['ID0']
        spectrals['IDp01'] = spectrals['ID0'] * spectrals['ID1']
        spectrals['IDp10'] = spectrals['ID1'] * spectrals['ID0']
        spectrals['IDp11'] = spectrals['ID1'] * spectrals['ID1']

        spectrals['ID0pow'] = spectrals['ID0'] ** 3
        spectrals['ID1pow'] = spectrals['ID1'] ** 3

        spectrals['ID0mul'] = spectrals['ID0'] * 1.67
        spectrals['ID1mul'] = spectrals['ID1'] * 1.67



        for key in sorted(list(spectrals.keys())):
            print(spectrals[key])
        for key in sorted(list(spectrals.keys())):
            filename ='{}-{}'.format(key,spectrals[key].desc)
            spectrals[key].plot(filename=filename,heading=spectrals[key].desc,ytitle='Value')


    if doAll:

        # test loading of atmospheric spectrals
        print('\n---------------------Atmos:')
        tape72 = rymodtran.loadtape7("data/tape7-02", ['FREQ', 'TOT_TRANS', 'PTH_THRML'] )
        distance = 2000.
        atmos['A1'] = Atmo(ID='tape7-02-1', tran=tape72[:,1], prad=1e4*tape72[:,2], distance=distance, wl=None, 
            wn=tape72[:,0], desc='tape7-02-1 raw data input')
        tape72 = rymodtran.loadtape7("data/tape7-02", ['FREQ', 'TOT_TRANS', 'PTH_THRML'] )
        atmos['A2'] = Atmo(ID='tape7-02-2', atco=-np.log(tape72[:,1])/distance, prad=1e4*tape72[:,2] / (1 - tape72[:,1]),
            wl=1e4/tape72[:,0], wn=None, desc='tape7-02-2 normalised input')
        atmos['A3'] = Atmo(ID='tape7-02-2', atco=None, prad=1e4*tape72[:,2] / (1 - tape72[:,1]),
            wl=1e4/tape72[:,0], wn=None, desc='tape7-02-2 normalised input')
        atmos['A4'] = Atmo(ID='tape7-02-2', atco=-np.log(tape72[:,1])/distance, prad=None,
            wl=1e4/tape72[:,0], wn=None, desc='tape7-02-2 normalised input')
        for key in sorted(list(atmos.keys())):
            print(atmos[key])
        for key in sorted(list(atmos.keys())):
            atmos[key].prad.plot(filename='{}-{}-{}'.format(key,atmos[key].desc,'prad'),
                heading=atmos[key].desc,ytitle='Norm Lpath W/(sr.m2.cm-1)')
            atmos[key].atco.plot(filename='{}-{}-{}'.format(key,atmos[key].desc,'atco'),
                heading=atmos[key].desc,ytitle='Attenuation m$^{-1}$')

        stride = int(atmos['A1'].atco.wn.shape[0] / 3)
        for distance in [1000,2000]:
            print('distance={} m, tran={}'.format(distance,atmos['A1'].tauR(distance)[0::stride].T))
            ID = 'Atau{:.0f}'.format(distance)
            spectrals[ID] = Spectral(ID,value=atmos['A1'].tauR(distance),wl=atmos['A1'].atco.wl,desc="MWIR transmittance "+ID)
            spectrals[ID].plot(filename='{}-{}-tau'.format(key,spectrals[ID].desc),
                heading=spectrals[ID].desc,ytitle='Transmittance')
            ID = 'Aprad{:.0f}'.format(distance)
            spectrals[ID] = Spectral(ID,value=atmos['A1'].pathR(distance),wl=atmos['A1'].atco.wl,desc="MWIR path radiance "+ID)
            spectrals[ID].plot(filename='{}-{}-prad'.format(key,spectrals[ID].desc),
                heading=spectrals[ID].desc,ytitle='Lpath W/(sr.m2.cm-1)')

        distances = np.array([1000,2000])
        print('distances={} m, tran={}'.format(distances,atmos['A1'].tauR(distances)[0::stride].T))
        print('distances={} m, atco={}'.format(distances,(-np.log(atmos['A1'].tauR(distances))/distances)[0::stride].T))

    if doAll:
        # test loading of sensors
        print('\n---------------------Sensors:')
        ID = 'S0'
        sensors[ID] = Sensor(ID=ID, fno=3.2, detarea=(10e-6)**2, inttime=0.01, 
            tauOpt=.5, quantEff=.6, pfrac=0.4, desc='Sensor one')
        ID = 'S1'
        sensors[ID] = Sensor(ID=ID, fno=3.2, detarea=(10e-6)**2, inttime=0.01, 
            tauOpt=Spectral(ID=ID,value=.5, wl=np.linspace(4.3, 5, 20), desc=''),
            quantEff=.6, pfrac=0.4, desc='Sensor one')
        spectral = np.loadtxt('data/MWIRsensor.txt')
        ID = 'S2'
        spectrals[ID] = Spectral(ID,value=spectral[:,1],wl=spectral[:,0],desc="MWIR transmittance")
        sensors[ID] = Sensor(ID=ID, fno=4, detarea=(15e-6)**2, inttime=0.02, 
            tauOpt=spectrals[ID], quantEff=spectrals[ID],  desc='Sensor two')
        for key in sorted(list(sensors.keys())):
            print(sensors[key])
            # print(sensors[key].tauOpt())
            # print(sensors[key].QE())


    if doAll:
        # test loading of targets/sources
        print('\n---------------------Sources:')
        ID = 'Rad-MWIR'
        desc = 'selfemit'
        wl = np.linspace(3,6,100)
        wn = 1e4 / wl
        emisx = Spectral('{}-emis'.format(ID),value=np.ones(wl.shape),wl=wl,desc='{}-emis'.format(desc))
        targets[ID] = Target(ID=ID, emis=emisx, tmprt=300, desc='Self emitted')

        ID = 'refl-MWIR'
        desc = 'reflsun'
        emisr = Spectral('{}-emis'.format(ID),value=np.ones(wn.shape),wn=wn,desc='{}-emis'.format(desc))
        targets[ID] = Target(ID=ID, emis=emisr, tmprt=6000, refl=.4,cosTarg=1.,
            scale=2.17e-5,taumed=0.5,desc='Reflected sunlight')
        for key in sorted(list(targets.keys())):
            print(targets[key])
            targets[key].radiance('el').plot(ytitle='Radiance')

        ssens = targets['Rad-MWIR'].radiance('el') + targets['refl-MWIR'].radiance('el')
        ssens.plot(ytitle='Radiance')

    if doAll:
        p = ryplot.Plotter(1,1,1)
        vlam,wl = luminousEfficiency(vlamtype='photopic', wavelen=None, eqnapprox=True)
        p.plot(1,wl,vlam,label=['{} equation'.format('CIE Photopic VM($\lambda$)')])
        vlam,wl = luminousEfficiency(vlamtype='scotopic', wavelen=None, eqnapprox=True)
        p.plot(1,wl,vlam,label=['{} equation'.format("CIE (1951) Scotopic V'($\lambda$)")])
        vlam,wl = luminousEfficiency(vlamtype='scotopic', wavelen=np.linspace(0.3, 0.8, 200), eqnapprox=True)
        p.plot(1,wl,vlam,label=['{} equation'.format("CIE (1951) Scotopic V'($\lambda$)")])

        vlam,wl = luminousEfficiency(vlamtype='photopic', wavelen=None, eqnapprox=False)
        p.plot(1,wl,vlam,label=['{} table'.format('CIE Photopic VM($\lambda$)')])
        vlam,wl = luminousEfficiency(vlamtype='scotopic', wavelen=None, eqnapprox=False)
        p.plot(1,wl,vlam,label=['{} table'.format("CIE (1951) Scotopic V'($\lambda$)")])
        vlam,wl = luminousEfficiency(vlamtype='CIE2008v2', wavelen=None, eqnapprox=False)
        p.plot(1,wl,vlam,label=['{} table'.format('CIE 2008 V($\lambda$) 2 deg')])
        vlam,wl = luminousEfficiency(vlamtype='CIE2008v10', wavelen=None, eqnapprox=False)
        p.plot(1,wl,vlam,'Luminous Efficiency','Wavelength $\mu$m','Efficiency',
            label=['{} table'.format('CIE 2008 V($\lambda$) 10 deg')])
        p.saveFig('vlam.png')


    if  doAll:
        imgfilename = os.path.join(pathtoPyradiData,'images/Tan-Chau-bw.png')
        mtnfilename = 'data/imgmotion/rowcol-displacement.npz'
        intTime = 0.005
        frmTim = 0.02
        makemotionsequence(imgfilename=imgfilename,mtnfilename=mtnfilename,postfix='',
                intTime=0.01,frmTim=0.02,outrows=512,outcols=512,imgRatio=10,
                pixsize=560e-6, numsamples=-1,fnPlotInput='makemotionsequence')


    if  doAll:
        #demonstrate the pulse detection algorithms
        pulsewidth = 100e-9
        FAR = 15
        probDetection = 0.999
        ThresholdToNoise = detectThresholdToNoiseTpFAR(pulsewidth,FAR)
        SignalToNoise = detectSignalToNoiseThresholdToNoisePd(ThresholdToNoise, probDetection)
        print('For a laser pulse with width={0:.3e}, a FAR={1:.3e} and Pd={2:.3e},'.format(pulsewidth,FAR,probDetection))
        print('the Threshold to Noise ratio must be {0:.3e}'.format(ThresholdToNoise))
        print('and the Signal to Noise ratio must be {0:.3e}'.format(SignalToNoise))
        print(' ')


    if  doAll:
        #demonstrate the range equation solver
        #create a range table and its associated transmittance table
        print('The first case should give an incorrect warning:')
        rangeTab = np.linspace(0, 20000, 10000)
        tauTab = np.exp(- 0.00015 * rangeTab)
        Intensity=200
        Irradiancetab=[1e-100, 1e-7, 10e-6, 10e-1]
        for Irradiance in Irradiancetab:
            r = rangeEquation(Intensity = Intensity, Irradiance = Irradiance, rangeTab = rangeTab,
                  tauTab = tauTab, rangeGuess = 1, n = 2)

            #test the solution by calculating the irradiance at this range.
            tauTable = interp1d(rangeTab, tauTab, kind = 'linear')

            if np.abs(r[0]) < rangeTab[2]:
                rr = rangeTab[2]
                strError = "Check range resolution in lookup table"
            elif np.abs(r[0]) > rangeTab[-1]:
                rr = rangeTab[-1]
                strError = "Check maximum range in lookup table"
            else:
                rr = r[0]
                strError = ""

            irrad = Intensity * tauTable(rr) / rr ** 2

            # print(type(Irradiance))
            # print(type(r[0]))
            # print(type(irrad))
            # print(type((irrad - Irradiance) / Irradiance))
            # print(type(strError))
            print('\nE={0:.3e}: Range equation solver: at range {1:.3e} the irradiance is {2:.3e}, error is {3:.3e} - {4}'.format(
                Irradiance,r[0],irrad, (irrad - Irradiance) / Irradiance, strError))
        print(' ')

    #########################################################################################
    if  doAll:
        # demo the spectral density conversions
        wavelenRef = np.asarray([0.1,  1,  10 ,  100]) # in units of um
        wavenumRef = np.asarray([1.0e5,  1.0e4,  1.0e3,  1.0e2]) # in units of cm-1
        frequenRef = np.asarray([  2.99792458e+15,   2.99792458e+14,   2.99792458e+13, 2.99792458e+12])
        print('Input spectral vectors:')
        print(wavelenRef)
        print(wavenumRef)
        print(frequenRef)

        #first we test the conversion between the domains
        # if the spectral domain conversions are correct, all following six statements should print unity vectors
        print('all following six statements should print unity vectors:')
        print(convertSpectralDomain(frequenRef, 'fl')/wavelenRef)
        print(convertSpectralDomain(wavenumRef, 'nl')/wavelenRef)
        print(convertSpectralDomain(frequenRef, 'fn')/wavenumRef)
        print(convertSpectralDomain(wavelenRef, 'ln')/wavenumRef)
        print(convertSpectralDomain(wavelenRef, 'lf')/frequenRef)
        print(convertSpectralDomain(wavenumRef, 'nf')/frequenRef)
        print('test illegal input type should have shape (0,0)')
        print(rit(convertSpectralDomain(wavenumRef, 'ng').shape))
        print(rit(convertSpectralDomain(wavenumRef, '').shape))
        print(rit(convertSpectralDomain(wavenumRef).shape))

        # now test conversion of spectral density quantities
        #create planck spectral densities at the wavelength interval
        exitancewRef = ryplanck.planck(wavelenRef, 1000,'el')
        exitancefRef = ryplanck.planck(frequenRef, 1000,'ef')
        exitancenRef = ryplanck.planck(wavenumRef, 1000,'en')
        exitance = exitancewRef.copy()
        #convert to frequency density
        print('all following eight statements should print (close to) unity vectors:')
        (freq, exitance) = convertSpectralDensity(wavelenRef, exitance, 'lf')
        print('exitance converted: wf against calculation')
        print(exitancefRef/exitance)
       #convert to wavenumber density
        (waven, exitance) = convertSpectralDensity(freq, exitance, 'fn')
        print('exitance converted: wf->fn against calculation')
        print(exitancenRef/exitance)
        #convert to wavelength density
        (wavel, exitance) = convertSpectralDensity(waven, exitance, 'nl')
        #now repeat in opposite sense
        print('exitance converted: wf->fn->nw against original')
        print(exitancewRef/exitance)
        print('spectral variable converted: wf->fn->nw against original')
        print(wavelenRef/wavel)
        #convert to wavenumber density
        exitance = exitancewRef.copy()
        (waven, exitance) = convertSpectralDensity(wavelenRef, exitance, 'ln')
        print('exitance converted: wf against calculation')
        print(exitancenRef/exitance)
       #convert to frequency density
        (freq, exitance) = convertSpectralDensity(waven, exitance, 'nf')
        print('exitance converted: wf->fn against calculation')
        print(exitancefRef/exitance)
        #convert to wavelength density
        (wavel, exitance) = convertSpectralDensity(freq, exitance, 'fl')
        # if the spectral density conversions were correct, the following two should be unity vectors
        print('exitance converted: wn->nf->fw against original')
        print(exitancewRef/exitance)
        print('spectral variable converted: wn->nf->fw against original')
        print(wavelenRef/wavel)

        ## plot some conversions
        wl = np.linspace(1, 14, 101)#.reshape(-1,1) # wavelength
        wn = convertSpectralDomain(wl, type='ln') # wavenumber
        radiancewl = ryplanck.planck(wl,1000, 'el') / np.pi
        _,radiancewn1 = convertSpectralDensity(wl,radiancewl, 'ln')
        radiancewn2 = ryplanck.planck(wn,1000, 'en') / np.pi
        p = ryplot.Plotter(2,1,3,figsize=(18,6))
        p.plot(1,wl, radiancewl,'Planck radiance','Wavelength $\mu$m', 'Radiance W/(m$^2$.sr.$\mu$m)')
        p.plot(2,wn, radiancewn1,'Planck radiance','Wavenumber cm$^{-1}$', 'Radiance W/(m$^2$.sr.cm$^{-1}$)')
        p.plot(3,wn, radiancewn2,'Planck radiance','Wavenumber cm$^{-1}$', 'Radiance W/(m$^2$.sr.cm$^{-1}$)')
        p.saveFig('densconvert.png')

    if  doAll:
        ##++++++++++++++++++++ demo the convolution ++++++++++++++++++++++++++++
        #do a few tests first to check basic functionality. Place single lines and then convolve.
        ## ----------------------- basic functionality------------------------------------------
        samplingresolution=0.5
        wavenum=np.linspace(0, 100, 100/samplingresolution)
        inspectral=np.zeros(wavenum.shape)
        inspectral[int(10/samplingresolution)] = 1
        inspectral[int(11/samplingresolution)] = 1
        inspectral[int(45/samplingresolution)] = 1
        inspectral[int(55/samplingresolution)] = 1
        inspectral[int(70/samplingresolution)] = 1
        inspectral[int(75/samplingresolution)] = 1
        inwinwidth=1
        outwinwidth=5
        outspectral,  windowfn = convolve(inspectral, samplingresolution,  inwinwidth,  outwinwidth)
        convplot = ryplot.Plotter(1, 1, 1)
        convplot.plot(1, wavenum, inspectral, "Convolution Test", r'Wavenumber cm$^{-1}$',\
                    r'Signal', ['r'],label=['Input'],legendAlpha=0.5)
        convplot.plot(1, wavenum, outspectral, "Convolution Test", r'Wavenumber cm$^{-1}$',\
                    r'Signal', ['g'],label=['Output'],legendAlpha=0.5)
        convplot.saveFig('convplot01'+figtype)

        ## ----------------------- spectral convolution practical example ----------
         # loading bunsen spectral radiance file: 4cm-1  spectral resolution, approx 2 cm-1 sampling
        specRad = ryfiles.loadColumnTextFile('data/bunsenspec.txt',  \
                        loadCol=[0,1], comment='%', delimiter=' ')
        # modtran5 transmittance 5m path, 1 cm-1 spectral resolution, sampled 1cm-1
        tauAtmo = ryfiles.loadColumnTextFile('data/atmotrans5m.txt',  \
                        loadCol=[0,1], comment='%', delimiter=' ')
        wavenum =  tauAtmo[:, 0]
        tauA = tauAtmo[:, 1]
        # convolve transmittance from 1cm-1 to 4 cm-1
        tauAtmo4,  windowfn = convolve(tauA, 1,  1,  4)
        #interpolate bunsen spectrum to atmo sampling
        #first construct the interpolating function, using bunsen
        bunInterp1 = interp1d(specRad[:,0], specRad[:,1])
        #then call the function on atmo intervals
        bunsen = bunInterp1(wavenum)

        atmoplot = tauA.copy()
        atmoplot =  np.vstack((atmoplot, tauAtmo4))
        convplot02 = ryplot.Plotter(1, 1, 1,figsize=(20,5))
        convplot02.plot(1, wavenum, atmoplot.T, "Atmospheric Transmittance", r'Wavenumber cm$^{-1}$',\
                    r'Transmittance', ['r', 'g'],label=['1 cm-1', '4 cm-1' ],legendAlpha=0.5)
        convplot02.saveFig('convplot02'+figtype)

        bunsenPlt = ryplot.Plotter(1,3, 2, figsize=(20,7))
        bunsenPlt.plot(1, wavenum, bunsen, "Bunsen Flame Measurement 4 cm-1", r'',\
                    r'Signal', ['r'], pltaxis =[2000, 4000, 0,1.5])
        bunsenPlt.plot(2, wavenum, bunsen, "Bunsen Flame Measurement 4 cm-1", r'',\
                    r'Signal', ['r'], pltaxis =[2000, 4000, 0,1.5])
        bunsenPlt.plot(3, wavenum, tauA, "Atmospheric Transmittance 1 cm-1", r'',\
                    r'Transmittance', ['r'])
        bunsenPlt.plot(4, wavenum, tauAtmo4, "Atmospheric Transmittance 4 cm-1", r'',\
                    r'Transmittance', ['r'])
        bunsenPlt.plot(5, wavenum, bunsen/tauA, "Atmospheric-corrected Bunsen Flame Measurement 1 cm-1", r'Wavenumber cm$^{-1}$',\
                    r'Signal', ['r'], pltaxis =[2000, 4000, 0,1.5])
        bunsenPlt.plot(6, wavenum, bunsen/tauAtmo4, "Atmospheric-corrected Bunsen Flame Measurement 4 cm-1", r'Wavenumber cm$^{-1}$',\
                    r'Signal', ['r'], pltaxis =[2000, 4000, 0,1.5])

        bunsenPlt.saveFig('bunsenPlt01'+figtype)

        #bunsenPlt.plot


    if  doAll:

        ##++++++++++++++++++++ demo the filter ++++++++++++++++++++++++++++
        ## ----------------------- wavelength------------------------------------------
        #create the wavelength scale to be used in all spectral calculations,
        # wavelength is reshaped to a 2-D  (N,1) column vector
        wavelength=np.linspace(0.1, 1.3, 350).reshape(-1, 1)

        ##------------------------filter -------------------------------------
        #test the different filter types
        width = 0.4
        center = 0.9
        filterExp=[2, 2,  4, 6,  8, 12, 1000]
        filterPass=[0.4, 0.5,  0.6, 0.7,  0.8, 0.9, 0.99]
        filterSupp = np.asarray(filterPass) * 0.1
        filterType=['bandpass', 'lowpass', 'highpass', 'bandpass', 'lowpass', 'highpass', 'bandpass']
        filterTxt = [r's={0}, $\tau_p$={2}, {1}'.format(s,y,z) for s,y,z in zip(filterExp, filterType, filterPass) ]
        filters = sfilter(wavelength,center, width, filterExp[0], filterPass[0],  filterSupp[0], filterType[0])
        for i, exponent in enumerate(filterExp[1:]):
            tau=sfilter(wavelength,center, width, filterExp[i],filterPass[i],  filterSupp[i], filterType[i])
            filters =  np.hstack((filters,tau))

        ##------------------------- plot sample filters ------------------------------
        smpleplt = ryplot.Plotter(1, 1, 1, figsize=(10, 4))
        smpleplt.plot(1, wavelength, filters,
            r"Optical filter for $\lambda_c$={0}, $\Delta\lambda$={1}".format(center,width),
            r'Wavelength [$\mu$m]', r'Transmittance', \
                    ['r', 'g', 'y','k', 'b', 'm'],label=filterTxt)

        smpleplt.saveFig('sfilterVar2'+figtype)

        #all passband, different shapes
        width = 0.5
        center = 0.7
        filterExp=[2,  4, 6,  8, 12, 1000]

        filterTxt = ['s={0}'.format(s) for s in filterExp ]
        filters = sfilter(wavelength,center, width, filterExp[0], 0.8,  0.1)
        for exponent in filterExp[1:]:
            filters =  np.hstack((filters, sfilter(wavelength,center, width, exponent, 0.8,  0.1)))

        ##------------------------- plot sample filters ------------------------------
        smpleplt = ryplot.Plotter(1, 1, 1, figsize=(10, 4))
        smpleplt.plot(1, wavelength, filters,
            r"Optical filter for $\lambda_c$=0.7, $\Delta\lambda$=0.5,$\tau_{s}$=0.1, $\tau_{p}$=0.8",
            r'Wavelength [$\mu$m]', r'Transmittance', \
                    ['r', 'g', 'y','k', 'b', 'm'],label=filterTxt)
        smpleplt.saveFig('sfilterVar'+figtype)


    if  doAll:
        ##++++++++++++++++++++ demo the detector ++++++++++++++++++++++++++++
        ## ----------------------- detector------------------------------------------
        lwavepeak = 1.2
        params = [(0.5, 5), (1, 10), (1, 20), (1, 30), (1, 1000), (2, 20)]
        parameterTxt = ['a={0}, n={1}'.format(s[0], s[1]) for s in params ]
        responsivities = responsivity(wavelength,lwavepeak, params[0][0], params[0][1], 1.0)
        for param in params[1:]:
            responsivities =  np.hstack((responsivities, responsivity(wavelength,lwavepeak, param[0], param[1], 1.0)))

        ##------------------------- plot sample detector ------------------------------
        smpleplt = ryplot.Plotter(1, 1, 1, figsize=(10, 4))
        smpleplt.plot(1, wavelength, responsivities, "Detector Responsivity for $\lambda_c$=1.2 $\mu$m, k=1", r'Wavelength [$\mu$m]',\
                   r'Responsivity', \
                   ['r', 'g', 'y','k', 'b', 'm'],label=parameterTxt)
        smpleplt.saveFig('responsivityVar'+figtype)

        ##--------------------filtered responsivity ------------------------------
        # here we simply multiply the responsivity and spectral filter spectral curves.
        # this is a silly example, but demonstrates the spectral integral.
        filtreps = responsivities * filters
        parameterTxt = [str(s)+' & '+str(f) for (s, f) in zip(params, filterExp) ]
        ##------------------------- plot filtered detector ------------------------------
        smpleplt = ryplot.Plotter(1, 1, 1)
        smpleplt.plot(1, wavelength, filtreps, "Filtered Detector Responsivity", r'Wavelength $\mu$m',\
                   r'Responsivity', \
                   ['r', 'g', 'y','k', 'b', 'm'],label=parameterTxt,legendAlpha=0.5)
        smpleplt.saveFig('filtrespVar'+figtype)

        ##++++++++++++++++++++ demo the demo effectivevalue  ++++++++++++++++++++++++++++

        #test and demo effective value function
        temperature = 5900
        spectralBaseline = ryplanck.planckel(wavelength,temperature)
        # do for each detector in the above example
        for i in range(responsivities.shape[1]):
            effRespo = effectiveValue(wavelength,  responsivities[:, i],  spectralBaseline)
            print('Effective responsivity {0:.3e} of detector with parameters {1} '
                 'and source temperature {2:.3e} K'.\
                  format(effRespo, params[i], temperature))

        print(' ')
        ##++++++++++++++++++++ demo the absolute humidity function ++++++++++++++++++++++++++++
        #http://rolfb.ch/tools/thtable.php?tmin=-25&tmax=50&tstep=5&hmin=10&hmax=100&hstep=10&acc=2&calculate=calculate
        # check absolute humidity function. temperature in C and humudity in g/m3
        data=np.asarray([
        [   50  ,   82.78  ]   ,
        [   45  ,   65.25    ]   ,
        [   40  ,   50.98    ]   ,
        [   35  ,   39.47    ]   ,
        [   30  ,   30.26    ]   ,
        [   25  ,   22.97  ]   ,
        [   20  ,   17.24    ]   ,
        [   15  ,   12.8    ]   ,
        [   10  ,   9.38 ]   ,
        [   5   ,   6.79 ]   ,
        [   0   ,   4.85 ]   ,
        [   -5  ,   3.41 ]   ,
        [   -10 ,   2.36 ]   ,
        [   -15 ,   1.61 ]   ,
        [   -20 ,   1.08 ]   ,
        [   -25 ,   0.71 ] ])
        temperature = data[:,0]+273.15
        absh = abshumidity(temperature).reshape(-1,1)
        data = np.hstack((data,absh))
        data = np.hstack((data, 100 * np.reshape((data[:,1]-data[:,2])/data[:,2],(-1,1))))
        print('        deg C          Testvalue           Fn value       \% Error')
        print(data)

        p=ryplot.Plotter(1)
        temperature = np.linspace(-20,70,90)
        abshum = abshumidity(temperature + 273.15).reshape(-1,1)
        p.plot(1,temperature, abshum,'Absolute humidity vs temperature','Temperature [K]','Absolute humidity g/m$^3$]')
        p.saveFig('abshum.eps')

        print('US-Std 15C 46 RH has {:.3e} g/m3'.format(0.46*abshumidity(15 + 273.15)))
        #highest ever recorded absolute humidity was at dew point of 34C
        print('Highest recorded absolute humidity was {0:.3e}, dew point {1:.3e} deg C'.\
            format(abshumidity(np.asarray([34 + 273.15]))[0],34))

        ###################################################
        print('{} renders in LaTeX as an upright symbol'.format(upMu()))
        print('{} renders in LaTeX as an upright symbol'.format(upMu(True)))
        print('{} renders in LaTeX as an italic/slanted symbol'.format(upMu(False)))

    #----------  test circ ---------------------
    if  doAll:
        a = 3.
        b = 3.
        d = 2.
        samples = 10
        varx = np.linspace(-a/2, a/2, samples*a)
        vary = np.linspace(-b/2, b/2, samples*b)
        x, y = np.meshgrid(varx, vary)
        z = circ(x, y, d)

        #calculate area two ways to confirm
        print('Circ area is {:.4e}, should be {:.4e}'.format(np.sum(z)/(samples * a * samples * b), np.pi*(d/2)**2/(a * b)))

        if samples < 20:
            with ryplot.savePlot(1,1,1,figsize=(8,8), saveName=['tool_circ.png']) as p:
                p.mesh3D(1, x, y, z, ptitle='tool-circ',
                 xlabel='x', ylabel='y', zlabel='z',
                 maxNX=3, maxNY=3, maxNZ=4, alpha=0.5);


    #----------  test rect ---------------------
    if  doAll:
        a = 3.
        b = 3.
        sx = 2.
        sy = 2.
        samples = 10
        varx = np.linspace(-a/2, a/2, samples*a)
        vary = np.linspace(-b/2, b/2, samples*b)
        x, y = np.meshgrid(varx, vary)
        z = rect(x, y, sx, sy)

        #calculate area two ways to confirm
        print('Rect area is {:.3e}, should be {:.3e}'.format(np.sum(z)/(samples * a * samples * b), sx*sy/(a * b)))

        if samples < 20:
            with ryplot.savePlot(1,1,1,figsize=(8,8), saveName=['tool_rect.png']) as p:
                p.mesh3D(1, x, y, z, ptitle='tool-rect',
                 xlabel='x', ylabel='y', zlabel='z',
                 maxNX=3, maxNY=3, maxNZ=4, alpha=0.5);

     #----------  poissonarray ---------------------
    if doAll:
        asize = 100000 # you need values of more than 10000000 to get really good stats
        tpoint = 1000
        print('Poisson for a mean value of zero is not defined and should give a zero')
        for lam in [0,10, tpoint-5, tpoint-1, tpoint, tpoint+1, tpoint+5, 20000]:
            inp = lam * np.ones((asize,1))
            out =  poissonarray(inp, seedval=0,tpoint=tpoint)

            # print(lam)
            errmean = np.nan if lam==0 else (lam-np.mean(out))/lam
            errvar = np.nan if lam==0 else (lam-np.var(out))/lam
            print('lam={:.3e} mean={:.3e} var={:.3e} err-mean={:.3e} err-var={:.3e}'.format(lam,
               np.mean(out),np.var(out), errmean, errvar))

    # ----------------test checkerboard texture------------------------
    if doAll:
        from PIL import Image as Img
        rows = 5
        cols = 7
        pixInBlock = 4

        color1 = 0       
        color2 = 255      
        img = drawCheckerboard(rows,cols,pixInBlock,'L',color1,color2,'nparray')
        pilImg = Img.fromarray(img, 'L')
        pilImg.save('{0}.png'.format('checkerboardL'))


        color1 = (0,0,0)          
        color2 = (255,255,255)      
        pilImage = drawCheckerboard(rows,cols,pixInBlock,'RGB',color1,color2,'image')
        pilImage.save('{0}.png'.format('checkerboardRGB'))

    ############################################################
    # test the mtf from wavefront code
    if doAll:

        sample = u'Mirror-001 at large stride'
        # load the data imgVal is the deviation from the ideal shape, in the direction of the axis
        b = np.load('data/mirrorerror.npz')    
        figerror = b['figerror']
        xg = b['xg']
        yg = b['yg']
        specdef = {'Visible':0.5e-6,'Infrared':4e-6}
        #this calc takes a while for small stride values, made rough here to test concept
        # wavefront displacement is twice the mirror figure error
        wfdev, phase, pupfn, MTF2D, MTFpol, specdef, MTFmean, rho, fcrit, clear = \
            calcMTFwavefrontError(sample, 2 * figerror, xg, yg, specdef,samplingStride=10);

        #avoid divide by zero error
        MTF2D[clear] = np.where(MTF2D[clear]==0,1e-100,MTF2D[clear])
        MTFmean[clear] = np.where(MTFmean[clear]==0,1e-100,MTFmean[clear])

        #  plot the wavefront error
        I1 = ryplot.Plotter(1,2,len(specdef.keys()),'sample {} '.format(sample), figsize=(14,10));
        keys = sorted(list(wfdev.keys()))
        for i,specband in enumerate(keys):
            I1.showImage(i+1, wfdev[specband], ptitle='{} wavefront displacement in m'.format(specband), 
                         cmap=ryplot.cubehelixcmap(),titlefsize=10, cbarshow=True);
            I1.showImage(i+4, phase[specband], ptitle='{} wavefront displacement in rad'.format(specband), 
                         cmap=ryplot.cubehelixcmap(),titlefsize=10, cbarshow=True);
        I1.saveFig('mirrorerror-WFerror.png')

        # plot the 2D MTF
        I1 = ryplot.Plotter(1,1,len(specdef.keys()),'Degraded MTF: sample {} '.format(sample),
                            figsize=(14,5));
        keys = sorted(list(MTF2D.keys()))
        for i,specband in enumerate(keys):
            I1.showImage(i+1, MTF2D[specband], ptitle='{} MTF'.format(specband), cmap=ryplot.cubehelixcmap(), 
                         titlefsize=10, cbarshow=True);
        I1.saveFig('mirrorerror-2D-MTF.png')

        #  plot the 2D MTF degradation
        I1 = ryplot.Plotter(2,1,len(specdef.keys()),'Degraded MTF/MTFdiffraction: sample {} '.format(sample),
                            figsize=(14,5));
        keys = sorted(list(MTF2D.keys()))
        for i,specband in enumerate(keys):
            I1.showImage(i+1, MTF2D[specband]/MTF2D[clear], ptitle='{} MTF'.format(specband), 
                         cmap=ryplot.cubehelixcmap(),titlefsize=10, cbarshow=True);
        I1.saveFig('mirrorerror-2D-MTF-ratio.png')

        # plot the rotational MTF
        I1 = ryplot.Plotter(3,1,len(specdef.keys())-1,figsize=(8*(len(specdef.keys())-1),5));
        j = 0
        keys = sorted(list(MTFpol.keys()))
        for specband in keys:
            if clear not in specband:
                for i in range(0,MTFpol[specband].shape[1]):
                    I1.plot(j+1,rho[specband],MTFpol[specband][:,i],'Sample {} {}'.format(sample,specband),
                           'Frequency cy/mrad','MTF',pltaxis=[0,fcrit[specband],0,1])
                j += 1
        I1.saveFig('mirrorerror-rotational-MTF.png')

        # plot the mean MTF
        I1 = ryplot.Plotter(4,len(specdef.keys())-1,2,figsize=(12,4*(len(specdef.keys())-1)),
                figuretitle='Mean MTF relative to diffraction limit: sample {}'.format(sample));
        j = 0
        keys = sorted(list(MTFmean.keys()))
        for specband in keys:
            if clear not in specband:
                I1.plot(j*2+1,rho[specband],MTFmean[specband],'',
                       'Frequency cy/mrad','MTF',pltaxis=[0,fcrit[specband],0,1],label=[specband])
                I1.plot(j*2+2,rho[specband],MTFmean[specband]/MTFmean[clear],'','Frequency cy/mrad',
                        'MTF/MTFdiffraction',pltaxis=[0,fcrit[specband],0,1],label=[specband])
                I1.plot(j*2+1,rho[specband],MTFmean[clear],'',
                       'Frequency cy/mrad','MTF',pltaxis=[0,fcrit[specband],0,1],label=[clear])
                j += 1
        I1.saveFig('mirrorerror-mean-MTF.png')



    print('\n\nmodule ryutils done!')
