#  $Id$
#  $HeadURL$

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
This module provides rudimentary colour coordinate processing.
Calculate the CIE 1931 rgb chromaticity coordinates for an arbitrary spectrum.

See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students 
and readers of the book Electro-Optical System Analysis and Design: A Radiometry 
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""





__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['chromaticityforSpectralL','XYZforSpectralL','loadCIEbar','CIErgbCIExy','CIExyCIErgb']

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)

import os
import pkg_resources
from StringIO import StringIO
from numbers import Number

import numpy as np
from scipy.interpolate import  interp1d

# convert from CIE RGB to XYZ coordinates
mRGBtoXYZ = 5.6507 * np.asarray([
    [0.49,0.31,0.20],
    [0.17697,0.81240,0.01063],
    [0.00,0.01,0.99]])

# convert from CIE xy to CIE RGB coordinates
mXYZtoRGB = np.asarray([
    [0.41847, -0.15866, -0.082835],
    [-0.091169, 0.25243, 0.015708],
    [0.00092090, -0.0025498, 0.17860]])

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

    X=np.trapz(radiance.reshape(-1, 1)*xbar.reshape(-1, 1),spectral, axis=0)
    Y=np.trapz(radiance.reshape(-1, 1)*ybar.reshape(-1, 1),spectral, axis=0)
    Z=np.trapz(radiance.reshape(-1, 1)*zbar.reshape(-1, 1),spectral, axis=0)

    x=X/(X+Y+Z)
    y=Y/(X+Y+Z)

    return [x[0], y[0], Y[0]]



##############################################################################
##
def XYZforSpectralL(spectral,radiance,xbar,ybar,zbar):
    """ Calculate the CIE chromaticity coordinates for an arbitrary spectrum.

    Given a spectral radiance vector and CIE tristimulus curves,
    calculate the XYZ chromaticity coordinates. It is assumed that the
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
        | [X,Y,Z]: color coordinates X,Y,Z.

    Raises:
        | No exception is raised.
    """

    X=np.trapz(radiance.reshape(-1, 1)*xbar.reshape(-1, 1),spectral, axis=0)
    Y=np.trapz(radiance.reshape(-1, 1)*ybar.reshape(-1, 1),spectral, axis=0)
    Z=np.trapz(radiance.reshape(-1, 1)*zbar.reshape(-1, 1),spectral, axis=0)


    return [X[0], Y[0], Z[0]]



##############################################################################
##
def loadCIEbar(specvec, stype):
    """ 

    Args:
        | specvec (np.array[N,] or [N,1]): spectral vector in  [um] or [cm-1].
        | stype (str]): type spectral vector wl=wavelength, wn=wavenumber.

    Returns:
        | CIE tristimilus (np.array[:,4]: cols=[specvec,x,y,z])

    Raises:
        | No exception is raised.

    """
    specvec = np.asarray(specvec)
    # lookup assumes wavelength
    if stype == 'wn':
        wl = 1e4 / specvec.reshape(-1,1)
        ciebar = 1e4 / wl
    else:
        wl = specvec.reshape(-1,1)
        ciebar = wl
    
    #input vectors must be sorted either ascending or descending
    # this can also work: np.all(a[1:] >= a[:-1]) 
    wldescending = False

    if not wl.shape == ():
        wldiff = np.diff(wl,axis=0)
        if not (np.all(wldiff<=0) or np.all(wldiff>=0)):
            print('Function loadCIEbar(): spectral vector not sorted')
            return None, None
        
        # if the wl data are in descending order, output must flipud
        # because the calling function sorts  ascending
        if np.all(wldiff<=0):
            wl = np.flipud(wl)
            ciebar = np.flipud(ciebar)
            wldescending = True

    #load data file from the pyradi directories, not local dir
    resource_package = 'pyradi'  #__name__  ## Could be any module/package name.
    resource_path = os.path.join('data', 'colourcoordinates','ciexyz31_1.txt')
    dat = pkg_resources.resource_string(resource_package, resource_path)
    cie = np.genfromtxt(StringIO(dat),delimiter=",",skip_header=1)

    #interpolate cie spectral data to samples wavelength range
    #first construct the interpolating function then call the function on data
    for i in [1, 2, 3]:
        interpfun = interp1d(cie[:,0]/1.0e3, cie[:,i], bounds_error=False, fill_value=0.0)
        ciebar = np.hstack((ciebar, (interpfun(wl)).reshape(-1,1)))

    if wldescending:
        ciebar = np.flipud(ciebar)
        
    return ciebar

##############################################################################
##
def CIErgbCIExy(rgb):
    """ Convert from CIE RGB coordinates to CIE (x,y) coordinates

    The CIE RGB colour space is one of many colour spaces, using three monochromatic 
    primary colours at standardized wavelengths of 700 nm (red), 546.1 nm (green) 
    and 435.8 nm (blue). Other RGB colour spaces uses different primary colours.

    This function converts from RGB coordinates (assumed CIE RGB) to CIE xy colour.
    The rgb array can have any number N of datasets in np.array[N,3].
    r, g, and b and in the first, second and third columns.

    https://en.wikipedia.org/wiki/CIE_1931_color_space
    https://en.wikipedia.org/wiki/RGB_color_space

    Args:
        | rgb (np.array[N,3]): CIE red/green/blue colour space component, N sets

    Returns:
        | xy (np.array[N,2]): color coordinates x, y.

    Raises:
        | No exception is raised.
    """

    # exact values for this conversion is specified in the CIE standard
 
    rgb = rgb.reshape(-1,3)

    XYZ = mRGBtoXYZ.dot(rgb.T)
    XYZ = XYZ.T
    x = XYZ[:,0] / ( XYZ[:,0] + XYZ[:,1] + XYZ[:,2] )
    y = XYZ[:,1] / ( XYZ[:,0] + XYZ[:,1] + XYZ[:,2] )
    return np.hstack((x.reshape(-1,1),y.reshape(-1,1)))



##############################################################################
##
def CIExyCIErgb(xy):
    """ Convert from CIE RGB coordinates to CIE (x,y) coordinates

    The CIE RGB colour space is one of many colour spaces, using three monochromatic 
    primary colours at standardized wavelengths of 700 nm (red), 546.1 nm (green) 
    and 435.8 nm (blue). Other RGB colour spaces uses different primary colours.

    This function converts from CIE xy colour to RGB coordinates (assumed CIE RGB).
    The xy array can have any number N of datasets in np.array[N,2]. 
    x is in the first column and y in the second column

    The rgb values are scaled such that the maximum value of any one
    component is 1, calculated separately per row. In other words,
    each rgb coordinate is normalised to 255 in one colour.

    https://en.wikipedia.org/wiki/CIE_1931_color_space
    https://en.wikipedia.org/wiki/RGB_color_space

    Args:
        | xy (np.array[N,2]): color coordinates x, y.

    Returns:
        | rgb (np.array[N,3]): CIE red/green/blue colour space component, N sets

    Raises:
        | No exception is raised.
    """

    # exact values for this conversion is specified in the CIE standard


    xy = xy.reshape(-1,2)
    Y = np.ones(xy[:,0].shape).reshape(-1,1)
    X = Y * (xy[:,0] / xy[:,1]).reshape(-1,1)
    Z = Y * (1 - (xy[:,0] + xy[:,1]).reshape(-1,1)) / xy[:,1].reshape(-1,1)
    XYZ = np.hstack((X,Y,Z))

    rgb =  (mXYZtoRGB.dot(XYZ.T)).T

    rgb /= np.max(rgb,axis=1).reshape(-1,1)

    return rgb


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


    doAll = False

    figtype = ".png"  # eps, jpg, png
    # figtype = ".eps"  # eps, jpg, png

    ## ----------------------- wavelength------------------------------------------
    #create the wavelength scale to be used in all spectral calculations,
    # wavelength is reshaped to a 2-D  (N,1) column vector
    wavelength=np.linspace(0.38, 0.72, 350).reshape(-1, 1)
    wavenum=np.linspace(13333, 27000, 350).reshape(-1, 1)

    if True:
        ## ----------------------load ciebar -----------------------------------

        ciebarwl = loadCIEbar(wavelength, stype='wl')
        ciebarwn = loadCIEbar(wavenum, stype='wn')
        cietriplt = ryplot.Plotter(1, 2, 2, figsize=(12,6))
        cietriplt.plot(1, ciebarwl[:,0], ciebarwl[:,1:4], "CIE tristimulus values, wl input",
                r'Wavelength $\mu$m', r'Response', plotCol = ['r','g','b'],
                label=['$\\bar{x}$', '$\\bar{y}$', '$\\bar{z}$'],legendAlpha=0.5);
        cietriplt.plot(2, 1e4/ciebarwl[:,0], ciebarwl[:,1:4], "CIE tristimulus values, wl input",
                r'Wavenumber cm$^{-1}$', r'Response', plotCol = ['r','g','b'],
                label=['$\\bar{x}$', '$\\bar{y}$', '$\\bar{z}$'],legendAlpha=0.5,maxNX=5);
        cietriplt.plot(3, 1e4/ciebarwn[:,0], ciebarwn[:,1:4], "CIE tristimulus values, wn input",
                r'Wavelength $\mu$m', r'Response', plotCol = ['r','g','b'],
                label=['$\\bar{x}$', '$\\bar{y}$', '$\\bar{z}$'],legendAlpha=0.5);
        cietriplt.plot(4, ciebarwn[:,0], ciebarwn[:,1:4], "CIE tristimulus values, wn input",
                r'Wavenumber cm$^{-1}$', r'Response', plotCol = ['r','g','b'],
                label=['$\\bar{x}$', '$\\bar{y}$', '$\\bar{z}$'],legendAlpha=0.5,maxNX=5);
        cietriplt.saveFig('cieBAR'+figtype)



    if doAll:

        ## -----------------------  test rgb to/from xy conversions---------------------

        print('mRGBtoXYZ:')
        print(mRGBtoXYZ)
        print('mXYZtoRGB:')
        print(mXYZtoRGB)
        print('mXYZtoRGB.dot(mRGBtoXYZ):')
        print(mXYZtoRGB.dot(mRGBtoXYZ))
        print(' ')
        rgb = np.asarray([[255,12,160],[255,0,0],[0,255,0],[0,0,255]])
        xy = CIErgbCIExy(rgb)
        print('RGB values:')
        print(rgb)
        print('xy values:')
        print(xy)
        rgbn = CIExyCIErgb(xy) * np.max(rgb)
        print('RGB values (recomputed from xy):')
        print(rgbn)






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
        #Use np.hstack to stack columns horizontally.

        sources = ryfiles.loadColumnTextFile('data/colourcoordinates/fluorescent.txt',
            abscissaOut=wavelength,comment='%', normalize=1).reshape(-1,1)
        sources = np.hstack((sources, ryplanck.planckel(wavelength,5900).reshape(-1,1)))
        sources = np.hstack((sources, ryplanck.planckel(wavelength,2850).reshape(-1,1)))
        sources = np.hstack((sources, ryfiles.loadColumnTextFile(
                                'data/colourcoordinates/LowPressureSodiumLamp.txt',
                                abscissaOut=wavelength, comment='%', normalize=1).reshape(-1,1)))
        #label sources in order of appearance
        sourcesTxt=['Fluorescent', 'Planck 5900 K', 'Planck 2850 K', 'Sodium']

        #normalize the source data (along axis-0, which is along columns)
        #this is not really necessary for CIE xy calc, which normalizes itself.
        #It is however useful for plotting the curves.
        sources /= np.max(sources,axis=0)

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
                    r'Reflectance', ['r-', 'g-', 'y-','g--', 'b-', 'm-'],label=samplesTxt,legendAlpha=0.5)
        smpleplt.saveFig('SampleReflectance'+figtype)

        ##------------------------- plot source spectra ------------------------------
        srceplt = ryplot.Plotter(2, 1, 1)
        srceplt.plot(1, wavelength, sources, "Normalized source radiance",
                    r'Wavelength $\mu$m', r'Radiance',
                    ['k:', 'k-.', 'k--', 'k-'],label=sourcesTxt,legendAlpha=0.5 )
        srceplt.saveFig('SourceRadiance'+figtype)

        ##------------------------- plot cie tristimulus spectra ---------------------
        cietriplt = ryplot.Plotter(3, 1, 1)
        cietriplt.plot(1, wavelength, bar,"CIE tristimulus values",r'Wavelength $\mu$m',
                r'Response', 'k--', label=['$\\bar{x}$','$\\bar{y}$','$\\bar{z}$'],legendAlpha=0.5)
        cietriplt.saveFig('tristimulus'+figtype)


        ##------------------------- calculate cie xy for samples and sources ---------
        xs = np.zeros((samples.shape[1],sources.shape[1]))
        ys = np.zeros((samples.shape[1],sources.shape[1]))
        for iSmpl in range(samples.shape[1]):
            for iSrc in range(sources.shape[1]):
                [ xs[iSmpl,iSrc], ys[iSmpl,iSrc], Y]=\
                    chromaticityforSpectralL(wavelength,
                    (samples[:,iSmpl]*sources[:,iSrc]).reshape(-1, 1),
                    bar[:,0], bar[:,1], bar[:,2])
                #print('{0:15s} {1:15s} ({2:.4f},{3:.4f})'.format(samplesTxt[iSmpl],
                #    sourcesTxt[iSrc], xs[iSmpl,iSrc], ys[iSmpl,iSrc]))

        ##---------------------- calculate cie xy for monochromatic  -----------------
        xm=np.zeros(wavelength.shape)
        ym=np.zeros(wavelength.shape)
        #create a series of data points with unity at specific wavelength
        for iWavel in range(wavelength.shape[0]):
            monospectrum=np.zeros(wavelength.shape)
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
                    [styleSample[iSmpl]], label=[samplesTxt[iSmpl]],legendAlpha=0.5 )
        #plot source markers
        styleSource=['bo', 'yo', 'ro', 'go']
        for iSmpl in range(samples.shape[1]):
            for iSrc in range(sources.shape[1]):
                if iSmpl==0:
                    legend=[sourcesTxt[iSrc]]
                else:
                    legend=''
                ciexyplt.plot(1,xs[iSmpl,iSrc],ys[iSmpl,iSrc],"CIE chromaticity diagram", r'x',r'y',\
                        [styleSource[iSrc]],label=legend,legendAlpha=0.5 )

        ciexyplt.saveFig('chromaticity'+figtype)

    print('module chroma done!')
