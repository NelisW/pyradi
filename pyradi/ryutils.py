#  $Id$
#  $HeadURL$

################################################################
# The contents of this file are subject to the Mozilla Public License
# Version 1.1 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/

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

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['sfilter', 'responsivity', 'effectiveValue', 'convertSpectralDomain',
         'convertSpectralDensity', 'convolve', 'abshumidity', 'rangeEquation',
         'detectThresholdToNoise','detectSignalToNoise'
         ]

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)


import numpy as np
from scipy import constants


def upMu(uprightMu=True):
    #all this to get an upright micron
    if uprightMu:
      from matplotlib import rc, font_manager
      import matplotlib as mpl
      # set up the use of external latex, fonts and packages
      rc('text', usetex=True)
      mpl.rcParams['text.latex.preamble'] = [
         r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
         r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
         r'\usepackage{helvet}',  # set the normal font here
         r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
         r'\sansmath']  # <- tricky! -- gotta actually tell tex to use!
      upmu = '\si{\micro}'
    else:
      upmu = '$\mu$'
    return upmu


##############################################################################
##
def detectThresholdToNoise(pulseWidth, FAR):
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
def detectSignalToNoise(ThresholdToNoise, pD):
    """ Solve for signal to noise ratio, given the threshold to noise ratio and
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
    or it can have any abritrary shape, provided it decreases with increasing range.
    The user supplies the lookup table in the form of an array of range values and
    an associated array of transmittance values.  The range values need not be on
    constant linear range increment.

    The parameter :math:`n`

    * :math:`n=2` (default value) the general case of a radiating source
      smaller than the field of view.

    * :math:`n=4` the special case of a laser rangefinder illuminating a target
      smaller than the field of view, viewed against the sky. In this case there
      is an :math:`R^2` attenuation from the laser to the source and another
      :math:`R^2` attenuation from the source to the receiver, hence
      :math:`R^4` overall.

    If the range solution is doubtful (e.g. not a trustworthy solution) the
    returned value is made negative.

    Args:
        | Intensity (float or np.array[N,] or [N,1]):  in  [W/sr].
        | Irradiance (float or np.array[N,] or [N,1]):  in  [W/m2].
        | rangeTab (np.array[N,] or [N,1]):  range vector for lookup
        | tauTab (np.array[N,] or [N,1]):   transmittance vector for lookup
        | rangeGuess (float): starting value range estimate
        | n (float): range power (2 or 4)

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
    """ Absolute humidity [g/m3] for temperature in [K] between 248 K and 342 K.

    This function provides two similar equations, but with different constants.


    Args:
        | temperature (np.array[N,] or [N,1]):  in  [K].

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
    """

    tau = taustop+(taupass-taustop)*np.exp(-(2*(spectral-center)/width)**exponent)
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
def convolve(inspectral, samplingresolution,  inwinwidth,  outwinwidth,  windowtype=np.bartlett):
    """ Convolve (non-circular) a spectral variable with a window function,
    given the input resolution and input and output window widths.

    This function is normally used on wavenumber-domain spectral data.  The spectral
    data is assumed sampled at samplingresolution wavenumber intervals.
    The inwinwidth and outwinwidth window function widths are full width half-max (FWHM)
    for the window functions for the inspectral and returned spectral variables, respectively.
    The Bartlett function is used as default, but the user can use a different function.
    The Bartlett function is a triangular function reaching zero at the ends. Window functio
    width is correct for Bartlett and only approximate for other window functions.

    Args:
        | inspectral (np.array[N,] or [N,1]):  vector in  [cm-1].
        | samplingresolution (float): wavenumber interval between inspectral samples
        | inwinwidth (float): FWHM window width of the input spectral vector
        | outwinwidth (float): FWHM window width of the output spectral vector
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
    outspectral = np.convolve(windowfn/(samplingresolution*windowfn.sum()),
                        inspectral.reshape(-1, ),mode='same'.encode('utf-8'))
    return outspectral,  windowfn

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

    figtype = ".png"  # eps, jpg, png
    figtype = ".eps"  # eps, jpg, png


    #demonstrate the pulse detection algorithms
    pulsewidth = 100e-9
    FAR = 15
    probDetection = 0.999
    ThresholdToNoise = detectThresholdToNoise(pulsewidth,FAR)
    SignalToNoise = detectSignalToNoise(ThresholdToNoise, probDetection)
    print('For a laser pulse with width={0}, a FAR={1} and Pd={2},'.format(pulsewidth,FAR,probDetection))
    print('the Threshold to Noise ratio must be {0}'.format(ThresholdToNoise))
    print('and the Signal to Noise ratio must be {0}'.format(SignalToNoise))
    print(' ')



    #demonstrate the range equation solver
    #create a range table and its associated transmittance table
    rangeTab = np.linspace(0, 10000, 1000)
    tauTab = np.exp(- 0.00015 * rangeTab)
    Intensity=200
    Irradiancetab=[10e-100, 10e-6, 10e-1]
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

        print('Range equation solver: at range {0} the irradiance is {1}, error is {2}. {3}'.format(
            r[0],irrad, (irrad - Irradiance) / Irradiance, strError))
    print(' ')

    #########################################################################################
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
    print(convertSpectralDomain(wavenumRef, 'ng').shape)
    print(convertSpectralDomain(wavenumRef, '').shape)
    print(convertSpectralDomain(wavenumRef).shape)

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


    ##++++++++++++++++++++ demo the convolution ++++++++++++++++++++++++++++
    #do a few tests first to check basic functionality. Place single lines and then convolve.
    ## ----------------------- basic functionality------------------------------------------
    samplingresolution=0.5
    wavenum=np.linspace(0, 100, 100/samplingresolution)
    inspectral=np.zeros(wavenum.shape)
    inspectral[10/samplingresolution] = 1
    inspectral[11/samplingresolution] = 1
    inspectral[45/samplingresolution] = 1
    inspectral[55/samplingresolution] = 1
    inspectral[70/samplingresolution] = 1
    inspectral[75/samplingresolution] = 1
    inwinwidth=1
    outwinwidth=5
    outspectral,  windowfn = convolve(inspectral, samplingresolution,  inwinwidth,  outwinwidth)
    convplot = ryplot.Plotter(1, 1, 1)
    convplot.plot(1, wavenum, inspectral, "Convolution Test", r'Wavenumber cm$^{-1}$',\
                r'Signal', ['r-'],label=['Input'],legendAlpha=0.5)
    convplot.plot(1, wavenum, outspectral, "Convolution Test", r'Wavenumber cm$^{-1}$',\
                r'Signal', ['g-'],label=['Output'],legendAlpha=0.5)
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
                r'Transmittance', ['r-', 'g-'],label=['1 cm-1', '4 cm-1' ],legendAlpha=0.5)
    convplot02.saveFig('convplot02'+figtype)

    bunsenPlt = ryplot.Plotter(1,3, 2, figsize=(20,7))
    bunsenPlt.plot(1, wavenum, bunsen, "Bunsen Flame Measurement 4 cm-1", r'',\
                r'Signal', ['r-'], pltaxis =[2000, 4000, 0,1.5])
    bunsenPlt.plot(2, wavenum, bunsen, "Bunsen Flame Measurement 4 cm-1", r'',\
                r'Signal', ['r-'], pltaxis =[2000, 4000, 0,1.5])
    bunsenPlt.plot(3, wavenum, tauA, "Atmospheric Transmittance 1 cm-1", r'',\
                r'Transmittance', ['r-'])
    bunsenPlt.plot(4, wavenum, tauAtmo4, "Atmospheric Transmittance 4 cm-1", r'',\
                r'Transmittance', ['r-'])
    bunsenPlt.plot(5, wavenum, bunsen/tauA, "Atmospheric-corrected Bunsen Flame Measurement 1 cm-1", r'Wavenumber cm$^{-1}$',\
                r'Signal', ['r-'], pltaxis =[2000, 4000, 0,1.5])
    bunsenPlt.plot(6, wavenum, bunsen/tauAtmo4, "Atmospheric-corrected Bunsen Flame Measurement 4 cm-1", r'Wavenumber cm$^{-1}$',\
                r'Signal', ['r-'], pltaxis =[2000, 4000, 0,1.5])

    bunsenPlt.saveFig('bunsenPlt01'+figtype)


    #bunsenPlt.plot

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
                ['r-', 'g-', 'y-','g--', 'b-', 'm-'],label=filterTxt)

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
                ['r-', 'g-', 'y-','g--', 'b-', 'm-'],label=filterTxt)
    smpleplt.saveFig('sfilterVar'+figtype)



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
               ['r-', 'g-', 'y-','g--', 'b-', 'm-'],label=parameterTxt)
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
               ['r-', 'g-', 'y-','g--', 'b-', 'm-'],label=parameterTxt,legendAlpha=0.5)
    smpleplt.saveFig('filtrespVar'+figtype)

    ##++++++++++++++++++++ demo the demo effectivevalue  ++++++++++++++++++++++++++++

    #test and demo effective value function
    temperature = 5900
    spectralBaseline = ryplanck.planckel(wavelength,temperature)
    # do for each detector in the above example
    for i in range(responsivities.shape[1]):
        effRespo = effectiveValue(wavelength,  responsivities[:, i],  spectralBaseline)
        print('Effective responsivity {0} of detector with parameters {1} '
             'and source temperature {2} K'.\
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


    #highest ever recorded absolute humidity was at dew point of 34C
    print('Highest recorded absolute humidity was {0}, dew point {1} deg C'.\
        format(abshumidity(np.asarray([34 + 273.15]))[0],34))

    print('module ryutils done!')
