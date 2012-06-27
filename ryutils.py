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
This module provides various utilityfunctions for radiometry calculations.

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

if __name__ == '__init__':
    pass
    
if __name__ == '__main__':
        
    import math
    import sys

    import pyradi.planck as radiometry
    import pyradi.ryplot as ryplot
    import pyradi.ryfiles as ryfiles

    figtype = ".png"  # eps, jpg, png
    #figtype = ".eps"  # eps, jpg, png

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
    smpleplt = ryplot.plotter(1, 1, 1)
    smpleplt.Plot(1, "Filter Transmittance: exponent", r'Wavelength $\mu$m',\
                r'Transmittance', wavelength, filters, \
                ['r-', 'g-', 'y-','g--', 'b-', 'm-'],filterTxt,0.5)
    smpleplt.SaveFig('sfilterVar'+figtype)
 

    ## ----------------------- detector------------------------------------------
    lwavepeak = 1.2
    params = [(0.5, 5), (1, 10), (1, 20), (1, 30), (1, 1000), (2, 20)]
    parameterTxt = [str(s) for s in params ]
    responsivities = responsivity(wavelength,lwavepeak, params[0][0], params[0][1], 1.0)
    for param in params[1:]:
        responsivities =  numpy.hstack((responsivities, responsivity(wavelength,lwavepeak, param[0], param[1], 1.0)))


    ##------------------------- plot sample detector ------------------------------
    smpleplt = ryplot.plotter(1, 1, 1)
    smpleplt.Plot(1, "Detector Responsivity", r'Wavelength $\mu$m',\
                r'Responsivity', wavelength, responsivities, \
                ['r-', 'g-', 'y-','g--', 'b-', 'm-'],parameterTxt,0.5)
    smpleplt.SaveFig('responsivityVar'+figtype)
 
    ##--------------------filtered responsivity ------------------------------
    # here we simply multiply the responsivity and spectral filter spectral curves.
    # this is a silly example, but demonstrates the spectral integral.
    filtreps = responsivities * filters
    parameterTxt = [str(s)+' & '+str(f) for (s, f) in zip(params, filterExp) ]
   ##------------------------- plot filtered detector ------------------------------
    smpleplt = ryplot.plotter(1, 1, 1)
    smpleplt.Plot(1, "Filtered Detector Responsivity", r'Wavelength $\mu$m',\
                r'Responsivity', wavelength, filtreps, \
                ['r-', 'g-', 'y-','g--', 'b-', 'm-'],parameterTxt,0.5)
    smpleplt.SaveFig('filtrespVar'+figtype)
 
    
    
    
    
    
