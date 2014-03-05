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
This module provides rudimentary colour coordinate processing.
Calculate the CIE 1931 rgb chromaticity coordinates for an arbitrary spectrum.

See the __main__ function for examples of use.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['chromaticityforSpectralL']

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)


import numpy

##############################################################################
##
def chromaticityforSpectralL(spectral,radiance,xbar,ybar,zbar):
    """ Calculate the CIE chromaticity coordinates for an arbitrary spectrum.

    Given a spectral radiance vector and CIE tristimulus curves,
    calculate the CIE chromaticity coordinates. It is assumed that the
    radiance spectral density is given in the same units as the spectral
    vector (i.e. [1/um] or [1/cm-1], corresponding to [um] or [cm-1] respectively.
    It is furthermore accepted that the tristimulus curves are also sampled at
    the same spectral intervals as the radiance. See
    http://en.wikipedia.org/wiki/CIE_1931_color_space
    for more information on CIE tristimulus spectral curves.

    Args:
        | spectral (np.array[N,] or [N,1]): spectral vector in  [um] or [cm-1].
        | radiance (np.array[N,] or [N,1]): the spectral radiance (any units), (sampled at spectral).
        | xbar (np.array[N,] or [N,1]): CIE x tristimulus spectral curve (sampled at spectral values).
        | ybar (np.array[N,] or [N,1]): CIE y tristimulus spectral curve (sampled at spectral values).
        | zbar (np.array[N,] or [N,1]): CIE z tristimulus spectral curve (sampled at spectral values).

    Returns:
        | [x,y,Y]: color coordinates x, y, and Y.

    Raises:
        | No exception is raised.
    """

    X=numpy.trapz(radiance.reshape(-1, 1)*xbar.reshape(-1, 1),spectral, axis=0)
    Y=numpy.trapz(radiance.reshape(-1, 1)*ybar.reshape(-1, 1),spectral, axis=0)
    Z=numpy.trapz(radiance.reshape(-1, 1)*zbar.reshape(-1, 1),spectral, axis=0)

    x=X/(X+Y+Z)
    y=Y/(X+Y+Z)

    return [x[0], y[0], Y[0]]



################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    import math
    import sys

    import pyradi.ryplanck as ryplanck
    import pyradi.ryplot as ryplot
    import pyradi.ryfiles as ryfiles

    #figtype = ".png"  # eps, jpg, png
    figtype = ".eps"  # eps, jpg, png

    ## ----------------------- wavelength------------------------------------------
    #create the wavelength scale to be used in all spectral calculations,
    # wavelength is reshaped to a 2-D  (N,1) column vector
    wavelength=numpy.linspace(0.38, 0.72, 350).reshape(-1, 1)

    ## ----------------------- colour tristimulus ---------------------------------
    # read csv file with wavelength in nm, x, y, z cie tristimulus values (x,y,z).
    # return values are 2-D (N,3) array scaled and interpolated.
    bar = ryfiles.loadColumnTextFile('data/colourcoordinates/ciexyz31_1.txt', abscissaOut=wavelength,
                    loadCol=[1,2,3],  comment='%', delimiter=',', abscissaScale=1e-3)


    ## ------------------------ sources ------------------------------------------
    #build a 2-D array with the source radiance values, where each column
    #represents a different source. Wavelength extends along rows.
    #Spectral interval for all source spectra is the same, which is 'wavelength'
    #Blackbody radiance spectra are calculated at the required wavelength intervals
    #Data read from files are interpolated to the required wavelength intervals
    #Use numpy.hstack to stack columns horizontally.

    sources = ryfiles.loadColumnTextFile('data/colourcoordinates/fluorescent.txt',
        abscissaOut=wavelength,comment='%', normalize=1).reshape(-1,1)
    sources = numpy.hstack((sources, ryplanck.planckel(wavelength,5900).reshape(-1,1)))
    sources = numpy.hstack((sources, ryplanck.planckel(wavelength,2850).reshape(-1,1)))
    sources = numpy.hstack((sources, ryfiles.loadColumnTextFile(
                            'data/colourcoordinates/LowPressureSodiumLamp.txt',
                            abscissaOut=wavelength, comment='%', normalize=1).reshape(-1,1)))
    #label sources in order of appearance
    sourcesTxt=['Fluorescent', 'Planck 5900 K', 'Planck 2850 K', 'Sodium']

    #normalize the source data (along axis-0, which is along columns)
    #this is not really necessary for CIE xy calc, which normalizes itself.
    #It is however useful for plotting the curves.
    sources /= numpy.max(sources,axis=0)

    ##------------------------- sample data ----------------------------------------
    # read space separated file containing wavelength in um, then sample data.
    # select the samples to be read in and then load all in one call!
    # first line in file contains labels for columns.
    samplesSelect = [1,2,3,8,10,11]
    samples = ryfiles.loadColumnTextFile('data/colourcoordinates/samples.txt',
         abscissaOut=wavelength, loadCol=samplesSelect,  comment='%')
    samplesTxt=ryfiles.loadHeaderTextFile('data/colourcoordinates/samples.txt',
                loadCol=samplesSelect, comment='%')

    ##------------------------- plot sample spectra ------------------------------
    smpleplt = ryplot.Plotter(1, 1, 1)
    smpleplt.plot(1, wavelength, samples, "Sample reflectance", r'Wavelength $\mu$m',
                r'Reflectance', ['r-', 'g-', 'y-','g--', 'b-', 'm-'],samplesTxt,0.5)
    smpleplt.saveFig('SampleReflectance'+figtype)

    ##------------------------- plot source spectra ------------------------------
    srceplt = ryplot.Plotter(2, 1, 1)
    srceplt.plot(1, wavelength, sources, "Normalized source radiance",
                r'Wavelength $\mu$m', r'Radiance',
                ['k:', 'k-.', 'k--', 'k-'],sourcesTxt,0.5 )
    srceplt.saveFig('SourceRadiance'+figtype)

    ##------------------------- plot cie tristimulus spectra ---------------------
    cietriplt = ryplot.Plotter(3, 1, 1)
    cietriplt.plot(1, wavelength, bar,"CIE tristimulus values",r'Wavelength $\mu$m',
            r'Response', 'k--', ['$\\bar{x}$','$\\bar{y}$','$\\bar{z}$'],0.5)
    cietriplt.saveFig('tristimulus'+figtype)


    ##------------------------- calculate cie xy for samples and sources ---------
    xs = numpy.zeros((samples.shape[1],sources.shape[1]))
    ys = numpy.zeros((samples.shape[1],sources.shape[1]))
    for iSmpl in range(samples.shape[1]):
        for iSrc in range(sources.shape[1]):
            [ xs[iSmpl,iSrc], ys[iSmpl,iSrc], Y]=\
                chromaticityforSpectralL(wavelength,
                (samples[:,iSmpl]*sources[:,iSrc]).reshape(-1, 1),
                bar[:,0], bar[:,1], bar[:,2])
            #print('{0:15s} {1:15s} ({2:.4f},{3:.4f})'.format(samplesTxt[iSmpl],
            #    sourcesTxt[iSrc], xs[iSmpl,iSrc], ys[iSmpl,iSrc]))

    ##---------------------- calculate cie xy for monochromatic  -----------------
    xm=numpy.zeros(wavelength.shape)
    ym=numpy.zeros(wavelength.shape)
    #create a series of data points with unity at specific wavelength
    for iWavel in range(wavelength.shape[0]):
        monospectrum=numpy.zeros(wavelength.shape)
        monospectrum[iWavel] = 1
        #calc xy for single mono wavelength point
        [xm[iWavel],ym[iWavel],Y]=chromaticityforSpectralL(wavelength,
                monospectrum, bar[:,0], bar[:,1], bar[:,2])
        #print('{0} um ({1},{2})'.format(wavelength[iWavel],xm[iWavel],ym[iWavel]))

    ##---------------------- plot chromaticity diagram  ---------------------------
    ciexyplt = ryplot.Plotter(4, 1, 1)
    #plot monochromatic horseshoe
    ciexyplt.plot(1, xm, ym,"CIE chromaticity diagram", r'x', r'y', ['k-'])
    #plot chromaticity loci for samples
    styleSample=['r--', 'g-.', 'y-', 'g-', 'b-', 'k-']
    for iSmpl in range(samples.shape[1]):
        ciexyplt.plot(1,xs[iSmpl],ys[iSmpl],"CIE chromaticity diagram", r'x', r'y',
                [styleSample[iSmpl]] ,[samplesTxt[iSmpl]],0.5 )
    #plot source markers
    styleSource=['bo', 'yo', 'ro', 'go']
    for iSmpl in range(samples.shape[1]):
        for iSrc in range(sources.shape[1]):
            if iSmpl==0:
                legend=[sourcesTxt[iSrc]]
            else:
                legend=''
            ciexyplt.plot(1,xs[iSmpl,iSrc],ys[iSmpl,iSrc],"CIE chromaticity diagram", r'x',r'y',\
                    [styleSource[iSrc]],legend,0.5 )

    ciexyplt.saveFig('chromaticity'+figtype)

    print('module chroma done!')
