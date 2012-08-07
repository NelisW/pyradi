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
Functions are provided for a maximally flat spectral filter, a simple photon detector spectral response, effective value calculation, conversion of spectral domain variables between [um], [cm^-1] and [Hz], conversion of spectral density quantities between [um], [cm^-1] and [Hz] and spectral convolution.

See the __main__ function for examples of use.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__= 'CJ Willers'
__all__= ['sfilter', 'responsivity']

import numpy
from scipy import constants

##############################################################################
##
def sfilter(spectral,center, width, exponent=6, taupass=1.0,  taustop=0.0 ):
    """ Calculate a symmetrical filter response of shape exp(-x^n)
    
    Given a number of parameters, calculates maximally flat, symmetrical transmittance.
    The function parameters controls the width, pass-band and stop-band transmittance and
    sharpness of cutoff. This function is not meant to replace the use of properly measured
    filter responses, but rather serves as a starting point if no other information is available.
    This function does not calculate ripple in the pass-band or cut-off band.
    
    Args:
        | spectral (np.array[N,] or [N,1]): spectral vector in  [um] or [cm-1].
        | center (float): central value for filter passband
        | width (float): proportional to width of filter passband
        | exponent (float): even integer, define the sharpness of cutoff. 
        |                     If exponent=2        then gaussian
        |                     If exponent=infinity then square 
        | taupass (float): the transmittance in the pass band (assumed constant)
        | taustop (float): peak transmittance in the stop band (assumed constant)
        
    Returns:
        | transmittance (np.array[N,] or [N,1]):  transmittances at "spectral" intervals.
        
    Raises:
        | No exception is raised.
    """
    
    tau = taustop+(taupass-taustop)*numpy.exp(-(2*(spectral-center)/width)**exponent)
   
    return tau


##############################################################################
##
def responsivity(wavelength,lwavepeak, cuton=1, cutoff=20, scaling=1.0):
    """ Calculate a photon detector wavelength spectral responsisivity
    
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

    num=numpy.trapz(spectralToProcess.reshape(-1, 1)*spectralBaseline.reshape(-1, 1),spectraldomain, axis=0)[0]
    den=numpy.trapz(spectralBaseline.reshape(-1, 1),spectraldomain, axis=0)[0]
    return num/den   
    

################################################################
##
def convertSpectralDomain(inspectraldomain,  type=''):
    """Convert spectral domains, i.e. between wavelength [um], wavenummber [cm^-1] and frequency [Hz]
    
    In string variable type, the 'from' domain and 'to' domains are indicated each with a single letter:
    'f' for temporal frequency, 'w' for wavelength and ''n' for wavenumber
    The 'from' domain is the first letter and the 'to' domain the second letter.
    
    Note that the 'to' domain vector is a direct conversion of the 'from' domain 
    to the 'to' domain (not interpolated or otherwise sampled.
    
    Args:
        | inspectraldomain (np.array[N,] or [N,1]):  spectral domain in wavelength, frequency or wavenumber.
        |    wavelength vector in  [um]
        |    frequency vector in  [Hz]
        |    wavenumber vector in   [cm^-1]
        | type (string):  specify from and to domains:
        |    'wf' convert from wavelength to per frequency 
        |    'wn' convert from wavelength to per wavenumber 
        |    'fw' convert from frequency to per wavelength 
        |    'fn' convert from frequency to per wavenumber
        |    'nw' convert from wavenumber to per wavelength 
        |    'nf' convert from wavenumber to per frequency
        
    Returns:
        | [N,1]: outspectraldomain 
        | Returns zero length array if type is illegal, i.e. not one of the expected values
            
    Raises:
        | No exception is raised.
    """

    #use dictionary to switch between options, lambda fn to calculate, default zero
    outspectraldomain = {
              'wf': lambda inspectraldomain:  constants.c / (inspectraldomain * 1.0e-6), 
              'wn': lambda inspectraldomain:  (1.0e4/inspectraldomain), 
              'fw': lambda inspectraldomain:  constants.c  / (inspectraldomain * 1.0e-6), 
              'fn': lambda inspectraldomain:  (inspectraldomain / 100) / constants.c , 
              'nw': lambda inspectraldomain:  (1.0e4/inspectraldomain), 
              'nf': lambda inspectraldomain:  (inspectraldomain * 100) * constants.c, 
              }.get(type, lambda inspectraldomain: numpy.zeros(shape=(0, 0)) )(inspectraldomain)
              
    return outspectraldomain


################################################################
##
def convertSpectralDensity(inspectraldomain,  inspectralquantity, type=''):
    """Convert spectral density quantities, i.e. between W/(m^2.um), W/(m^2.cm^-1) and W/(m^2.Hz). Return always positive.
    
    In string variable type, the 'from' domain and 'to' domains are indicated each with a single letter:
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
        | inspectraldomain (np.array[N,] or [N,1]):  spectral domain in wavelength, frequency or wavenumber.
        | inspectralquantity (np.array[N,] or [N,1]):  spectral density in same domain as domain vector above.
        |    wavelength vector in  [um]
        |    frequency vector in  [Hz]
        |    wavenumber vector in   [cm^-1]
        | type (string):  specify from and to domains:
        |    'wf' convert from per wavelength interval density to per frequency interval density
        |    'wn' convert from per wavelength interval density to per wavenumber interval density
        |    'fw' convert from per frequency interval density to per wavelength interval density
        |    'fn' convert from per frequency interval density to per wavenumber interval density
        |    'nw' convert from per wavenumber interval density to per wavelength interval density
        |    'nf' convert from per wavenumber interval density to per frequency interval density
        
    Returns:
        | ([N,1],[N,1]): outspectraldomain and outspectralquantity
        | Returns zero length arrays is type is illegal, i.e. not one of the expected values
            
    Raises:
        | No exception is raised.
    """

    #use dictionary to switch between options, lambda fn to calculate, default zero
    outspectraldomain = {
              'wf': lambda inspectraldomain:  constants.c / (inspectraldomain * 1.0e-6), 
              'wn': lambda inspectraldomain:  (1.0e4/inspectraldomain), 
              'fw': lambda inspectraldomain:  constants.c  / (inspectraldomain * 1.0e-6), 
              'fn': lambda inspectraldomain:  (inspectraldomain / 100) / constants.c , 
              'nw': lambda inspectraldomain:  (1.0e4/inspectraldomain), 
              'nf': lambda inspectraldomain:  (inspectraldomain * 100) * constants.c, 
              }.get(type, lambda inspectraldomain: numpy.zeros(shape=(0, 0)) )(inspectraldomain)
              
    outspectralquantity = {
              'wf': lambda inspectralquantity: inspectralquantity / (constants.c * 1.0e-6 / ((inspectraldomain * 1.0e-6)**2)), 
              'wn': lambda inspectralquantity: inspectralquantity / (1.0e4 / inspectraldomain**2) , 
              'fw': lambda inspectralquantity: inspectralquantity / (constants.c * 1.0e-6  / ((inspectraldomain * 1.0e-6)**2)), 
              'fn': lambda inspectralquantity: inspectralquantity / (100 * constants.c), 
              'nw': lambda inspectralquantity: inspectralquantity / (1.0e4 / inspectraldomain**2) , 
              'nf': lambda inspectralquantity: inspectralquantity / (100* constants.c), 
              }.get(type, lambda inspectralquantity: numpy.zeros(shape=(0, 0)) )(inspectralquantity)
    
    return (outspectraldomain,outspectralquantity )


##############################################################################
##
def convolve(inspectral, samplingresolution,  inwinwidth,  outwinwidth,  windowtype=numpy.bartlett):
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
    #numpy.convolve is unfriendly towards unicode strings
    outspectral = numpy.convolve(windowfn/(samplingresolution*windowfn.sum()),
                        inspectral.reshape(-1, ),mode='same'.encode('utf-8'))
    return outspectral,  windowfn

################################################################
##

if __name__ == '__init__':
    pass
    
if __name__ == '__main__':

    import pyradi.ryplanck as ryplanck
    import pyradi.ryplot as ryplot
    import pyradi.ryfiles as ryfiles

    figtype = ".png"  # eps, jpg, png
    #figtype = ".eps"  # eps, jpg, png
        
    import math
    import sys
    from scipy.interpolate import interp1d


    # demo the spectral density conversions
    wavelenRef = numpy.asarray([0.1,  1,  10 ,  100]) # in units of um
    wavenumRef = numpy.asarray([1.0e5,  1.0e4,  1.0e3,  1.0e2]) # in units of cm-1
    frequenRef = numpy.asarray([  2.99792458e+15,   2.99792458e+14,   2.99792458e+13, 2.99792458e+12])
    print(wavelenRef)
    print(wavenumRef)
    print(frequenRef)
    
    #first we test the conversion between the domains
    # if the spectral domain conversions are correct, all following six statements should print unity vectors
    print(convertSpectralDomain(frequenRef, 'fw')/wavelenRef)
    print(convertSpectralDomain(wavenumRef, 'nw')/wavelenRef)
    print(convertSpectralDomain(frequenRef, 'fn')/wavenumRef)
    print(convertSpectralDomain(wavelenRef, 'wn')/wavenumRef)
    print(convertSpectralDomain(wavelenRef, 'wf')/frequenRef)
    print(convertSpectralDomain(wavenumRef, 'nf')/frequenRef)
    #test illegal input type should havw shape (0,0)
    print(convertSpectralDomain(wavenumRef, 'ng').shape)
    print(convertSpectralDomain(wavenumRef, '').shape)
    print(convertSpectralDomain(wavenumRef).shape)

    # now test conversion of spectral density quantities
    #create planck spectral densities at the wavelength interval
    emittanceRef = ryplanck.planck(wavelenRef, 1000,'el')
    emittance = emittanceRef.copy()
    #convert to frequency density
    print('emittance converted: wf->fn->nw->wn->nf->fw')
    (freq, emittance) = convertSpectralDensity(wavelenRef, emittance, 'wf')
   #convert to wavenumber density
    (waven, emittance) = convertSpectralDensity(freq, emittance, 'fn')
    #convert to wavelength density
    (wavel, emittance) = convertSpectralDensity(waven, emittance, 'nw')
    #now repeat in opposite sense
    #convert to frequency density
    (waven, emittance) = convertSpectralDensity(wavelenRef, emittance, 'wn')
   #convert to wavenumber density
    (freq, emittance) = convertSpectralDensity(waven, emittance, 'nf')
    #convert to wavelength density
    (wavel, emittance) = convertSpectralDensity(freq, emittance, 'fw')
    # if the spectral density conversions were correct, the following two should be unity vectors
    print(emittanceRef/emittance)
    print(wavelenRef/wavel)

    
    
    ##++++++++++++++++++++ demo the convolution ++++++++++++++++++++++++++++
    #do a few tests first to check basic functionality. Place single lines and then convolve.
    ## ----------------------- basic functionality------------------------------------------
    samplingresolution=0.5
    wavenum=numpy.linspace(0, 100, 100/samplingresolution)
    inspectral=numpy.zeros(wavenum.shape)
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
    convplot.plot(1, "Convolution Test", r'Wavenumber cm$^{-1}$',\
                r'Signal', wavenum, inspectral, ['r-'],['Input'],0.5)
    convplot.plot(1, "Convolution Test", r'Wavenumber cm$^{-1}$',\
                r'Signal', wavenum, outspectral, ['g-'],['Output'],0.5)
    convplot.saveFig('convplot01'+figtype)
    
    ## ----------------------- practical example ------------------------------------------
    # loading bunsen spectral radiance file: 4cm-1  spectral resolution, approx 2 cm-1 sampling
    specRad = ryfiles.loadColumnTextFile('data/bunsenspec.txt',  \
                    loadCol=[0,1], comment='%', delimiter=' ')
    # modtran5 transmittance 5m path, 1 cm-1 spectral resolution, sampled 1cm-1 
    tauAtmo = ryfiles.loadColumnTextFile('data/atmotrans5m.txt',  \
                    loadCol=[0,1], comment='%', delimiter=' ')
    wavenum =  tauAtmo[:, 0]
    # convolve transmittance from 1cm-1 to 4 cm-1
    tauAtmo4,  windowfn = convolve(tauAtmo[:, 1], 1,  1,  4)
    #interpolate bunsen spectrum to atmo sampling
    #first construct the interpolating function, using bunsen
    bunInterp1 = interp1d(specRad[:,0], specRad[:,1])
    #then call the function on atmo intervals
    bunsen = bunInterp1(wavenum)
   
    atmoplot = tauAtmo[:, 1].copy()
    atmoplot =  numpy.vstack((atmoplot, tauAtmo4)) 
    convplot02 = ryplot.Plotter(1, 1, 1,figsize=(20,5))
    convplot02.plot(1, "Atmospheric Transmittance", r'Wavenumber cm$^{-1}$',\
                r'Transmittance', wavenum, atmoplot.T, ['r-', 'g-'],['1 cm-1', '4 cm-1' ],0.5)
    convplot02.saveFig('convplot02'+figtype)

    bunsenPlt = ryplot.Plotter(1,3, 2, figsize=(20,7))
    bunsenPlt.plot(1, "Bunsen Flame Measurement 4 cm-1", r'',\
                r'Signal', wavenum, bunsen, ['r-'],[], legendAlpha=0.5)
    bunsenPlt.plot(2, "Bunsen Flame Measurement 4 cm-1", r'',\
                r'Signal', wavenum, bunsen, ['r-'],[],legendAlpha=0.5)
    bunsenPlt.plot(3, "Atmospheric Transmittance 1 cm-1", r'',\
                r'Transmittance', wavenum, tauAtmo[:, 1], ['r-'],[],legendAlpha=0.5)
    bunsenPlt.plot(4, "Atmospheric Transmittance 4 cm-1", r'',\
                r'Transmittance', wavenum, tauAtmo4, ['r-'],[],legendAlpha=0.5)
    bunsenPlt.plot(5, "Corrected Bunsen Flame Measurement 1 cm-1", r'Wavenumber cm$^{-1}$',\
                r'Signal', wavenum, bunsen/tauAtmo[:, 1], ['r-'],[],legendAlpha=0.5, pltaxis =[2000, 4000, 0,1.4])
    bunsenPlt.plot(6, "Corrected Bunsen Flame Measurement 4 cm-1", r'Wavenumber cm$^{-1}$',\
                r'Signal', wavenum, bunsen/tauAtmo4, ['r-'],[],legendAlpha=0.5)
                
    bunsenPlt.saveFig('bunsenPlt01'+figtype)


    #bunsenPlt.plot

    ##++++++++++++++++++++ demo the filter ++++++++++++++++++++++++++++
    ## ----------------------- wavelength------------------------------------------
    #create the wavelength scale to be used in all spectral calculations, 
    # wavelength is reshaped to a 2-D  (N,1) column vector
    wavelength=numpy.linspace(0.1, 1.3, 350).reshape(-1, 1)
   
    ##------------------------filter -------------------------------------
    width = 0.5
    center = 0.7
    filterExp=[2,  4, 6,  8, 12, 1000]
    filterTxt = [str(s) for s in filterExp ]
    filters = sfilter(wavelength,center, width, filterExp[0], 0.8,  0.1)
    for exponent in filterExp[1:]:
        filters =  numpy.hstack((filters, sfilter(wavelength,center, width, exponent, 0.8,  0.1))) 

    ##------------------------- plot sample filters ------------------------------
    smpleplt = ryplot.Plotter(1, 1, 1)
    smpleplt.plot(1, "Filter Transmittance: exponent", r'Wavelength $\mu$m',\
                r'Transmittance', wavelength, filters, \
                ['r-', 'g-', 'y-','g--', 'b-', 'm-'],filterTxt,0.5)
    smpleplt.saveFig('sfilterVar'+figtype)


    ##++++++++++++++++++++ demo the detector ++++++++++++++++++++++++++++
    ## ----------------------- detector------------------------------------------
    lwavepeak = 1.2
    params = [(0.5, 5), (1, 10), (1, 20), (1, 30), (1, 1000), (2, 20)]
    parameterTxt = [str(s) for s in params ]
    responsivities = responsivity(wavelength,lwavepeak, params[0][0], params[0][1], 1.0)
    for param in params[1:]:
        responsivities =  numpy.hstack((responsivities, responsivity(wavelength,lwavepeak, param[0], param[1], 1.0))) 


    ##------------------------- plot sample detector ------------------------------
    smpleplt = ryplot.Plotter(1, 1, 1)
    smpleplt.plot(1, "Detector Responsivity", r'Wavelength $\mu$m',\
               r'Responsivity', wavelength, responsivities, \
               ['r-', 'g-', 'y-','g--', 'b-', 'm-'],parameterTxt,0.5)
    smpleplt.saveFig('responsivityVar'+figtype)

    ##--------------------filtered responsivity ------------------------------
    # here we simply multiply the responsivity and spectral filter spectral curves.
    # this is a silly example, but demonstrates the spectral integral.
    filtreps = responsivities * filters
    parameterTxt = [str(s)+' & '+str(f) for (s, f) in zip(params, filterExp) ]
    ##------------------------- plot filtered detector ------------------------------
    smpleplt = ryplot.Plotter(1, 1, 1)
    smpleplt.plot(1, "Filtered Detector Responsivity", r'Wavelength $\mu$m',\
               r'Responsivity', wavelength, filtreps, \
               ['r-', 'g-', 'y-','g--', 'b-', 'm-'],parameterTxt,0.5)
    smpleplt.saveFig('filtrespVar'+figtype)

    
    #test and demo effective value function
    temperature = 5900
    spectralBaseline = ryplanck.planckel(wavelength,temperature)
    # do for each detector in the above example
    for i in range(responsivities.shape[1]):
        effRespo = effectiveValue(wavelength,  responsivities[:, i],  spectralBaseline)
        print('Effective responsivity {0} of detector with parameters {1} '
             'and source temperature {2} K'.\
              format(effRespo, params[i], temperature))
    
    print('module ryutils done!')
