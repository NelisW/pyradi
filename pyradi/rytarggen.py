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

# The Initial Developer of the Original Code is CJ Willers, Copyright (C) 2006-2015
# All Rights Reserved.

# Contributor(s): ______________________________________.
################################################################

"""


For more detail see the documentation at

| ``http://nelisw.github.io/pyradi-docs/_build/html/index.html``,
| ``http://nelisw.github.io/pyradi-docs/_build/html/rytarggen.html``, or 
| ``pyradi/doc/rytarggen.rst``  

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= ""
__author__='CJ Willers'
__all__=['create_HDF5_image', 'hdf_Raw', 'hdf_Uniform', 'hdf_disk_photon', 'hdf_stairs',
    ]

import sys
import numpy as np
import scipy as sp
import scipy.signal as signal
import scipy.stats as stats
import scipy.constants as const
from scipy import interpolate
import re

import pyradi.ryfiles as ryfiles
import pyradi.ryutils as ryutils
import pyradi.ryplot as ryplot
import pyradi.ryprob as ryprob
import pyradi.ryplanck as ryplanck

######################################################################################
def hdf_Uniform(imghd5,rad_dynrange):
    r"""A generating function to create a uniform photon rate image.
    The uniform value in the image will have a value of rad_dynrange.
    The equivalent image value is expressed as in the same units as the input

    The function accepts radiant or photon rate dynamic range units inputs.

    This function must be called from rytarggen.create_HDF5_image

    Args:
        | imghd5 (handle to hdf5 file): file to which image must be added
        | rad_dynrange (float): uniform/max radiance value

    Returns:
        | nothing: as a side effect a set of photon radiance image files are written

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    
    # convert to radiance values in photon units
    imghd5['image/rad_dynrange'] = rad_dynrange * imghd5['image/conversion'].value
    imghd5['image/rad_min'] = 0. 

    # create photon rate radiance image from min to min+dynamic range, with no noise
    imghd5['image/PhotonRateRadianceNoNoise'][...] = \
                     rad_dynrange * np.ones((imghd5['image/imageSizePixels'].value)) 

    imghd5.flush()

    return imghd5 


######################################################################################
def hdf_disk_photon(imghd5,rad_min,rad_dynrange,fracdiameter,fracblur):
    r"""A generating function to create an image with illuminated circle with blurred boundaries.

    The function accepts radiance radiant or photon rate minimum and dynamic range units.
    The equivalent image value is expressed as in the same units as the input

    This function must be called from rytarggen.create_HDF5_image

    Args:
        | imghd5 (handle to hdf5 file): file to which image must be added
        | rad_min (float): additive minimum radiance value in the image
        | rad_dynrange (float): multiplicative radiance scale factor (max value)
        | fracdiameter (float):  diameter of the disk as fraction of minimum image size
        | fracblur (float):   blur of the disk as fraction of minimum image size

    Returns:
        | nothing: as a side effect a set of photon radiance image files are written

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """

    # convert to radiance values in photon units
    imghd5['image/rad_dynrange'] = rad_dynrange  * imghd5['image/conversion'].value
    imghd5['image/rad_min'] = rad_min * imghd5['image/conversion'].value
    # scale the disk to image size, as fraction
    maxSize = np.min((imghd5['image/imageSizeRows'].value, imghd5['image/imageSizeCols'].value))
    imghd5['image/disk_diameter'] = fracdiameter * maxSize
    imghd5['image/blur'] = fracblur * maxSize

    #create the disk, normalised to unity
    varx = np.linspace(-imghd5['image/imageSizeCols'].value/2, imghd5['image/imageSizeCols'].value/2, imghd5['image/imageSizePixels'].value[1])
    vary = np.linspace(-imghd5['image/imageSizeRows'].value/2, imghd5['image/imageSizeRows'].value/2, imghd5['image/imageSizePixels'].value[0])
    x1, y1 = np.meshgrid(varx, vary)
    delta_x = varx[1] - varx[0]
    delta_y = vary[1] - vary[0]
    Uin = ryutils.circ(x1,y1,imghd5['image/disk_diameter'].value) 
    #create blur disk normalised to unity
    dia = np.max((1, 2 * round(imghd5['image/blur'].value / np.max((delta_x,delta_y)))))
    varx = np.linspace(-dia, dia, 2 * dia)
    vary = np.linspace(-dia, dia, 2 * dia)
    x, y = np.meshgrid(varx, vary)
    H = ryutils.circ(x, y, dia)
    # convolve disk with blur
    NormLin = (np.abs(signal.convolve2d(Uin, H, mode='same'))/np.sum(H))  ** 2

    # create the photon rate radiance image from min to min+dynamic range, with no noise
    imghd5['image/PhotonRateRadianceNoNoise'][...] = \
                (imghd5['image/rad_min'].value + NormLin * imghd5['image/rad_dynrange'].value ) 

    imghd5.flush()

    return imghd5 



######################################################################################
def hdf_stairs(imghd5,rad_min,rad_dynrange,steps,imtype):
    r"""A generating function to create a staircase image, with log/linear and prescribed step count.

    The increment along stairs can be linear or logarithmic.

    The function accepts radiance radiant or photon rate minimum and dynamic range units.
    The equivalent image value is expressed as in lux units.


    This function must be called from rytarggen.create_HDF5_image

    Args:
        | imghd5 (handle to hdf5 file): file to which image must be added
        | rad_min (float): additive minimum radiance value in the image
        | rad_dynrange (float): radiance multiplicative scale factor (max value)
        | steps (int): number of steps in the image
        | imtype (string): string to define the type of image to be created ['stairslin','stairslog']

    Returns:
        | nothing: as a side effect a set of photon radiance image files are written

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """

    # convert to radiance values in photon units
    imghd5['image/rad_dynrange'] = rad_dynrange * imghd5['image/conversion'].value
    imghd5['image/rad_min'] = rad_min * imghd5['image/conversion'].value

    imghd5['image/steps'] = steps 
    imghd5['image/imtype'] = imtype

    #Create the stairs spatial definition
    size = imghd5['image/imageSizePixels'].value[1]
    if imtype in ['stairslin']:
        varx = np.linspace(0,size-1,size)
    else:
        varx = np.logspace(-1,np.log10(size-1),size)
    varx =  ((varx/(size/steps)).astype(int)).astype(float) / steps
    varx = varx / np.max(varx) 
    vary = np.linspace( - imghd5['image/imageSizeRows'].value/2, 
                            imghd5['image/imageSizeRows'].value/2,
                            imghd5['image/imageSizePixels'].value[0])
    vary = np.where(np.abs(vary)<imghd5['image/imageSizeRows'].value/3.,1.,0.)
    x, y = np.meshgrid(varx,vary)
    NormLin = y * x  * np.ones(x.shape)  


    # create the photon rate radiance image from min to min+dynamic range, with no noise
    imghd5['image/PhotonRateRadianceNoNoise'][...] =  \
            (imghd5['image/rad_min'].value + NormLin * imghd5['image/rad_dynrange'].value ) 
 
    imghd5.flush()

    return imghd5 


######################################################################################
def hdf_Raw(imghd5,filename,inputSize,outputSize,rad_min=-1,rad_dynrange=-1, imgNum=0,
inputOrigin=[0,0],blocksize=[1,1],sigma=0):
    r"""A generating function to create a photon rate image from raw image.

    The output image is extracted from the raw image, with blocks of raw image
    pixels averaged to single output image pixels. inputOrigin (lowest row,col values)
    defines from where in the raw input image the slicing takes place. blocksize defines
    how many raw image pixels must be averaged/aggregated together to define a single
    output image pixel, resolution is lowered by this factor.  sigma is the kernel 
    size to be used in scipy.filters.gaussian_filter.

    The subsampled image will be rescaled to rad_min + rad_dynrange.

    The raw image sequence must be of type np.float64 with no header or footer.

    The function accepts radiant or photon rate minimum and dynamic range units.
    The equivalent image value is expressed as in the same units as the output image

    This function must be called from rytarggen.create_HDF5_image

    Args:
        | imghd5 (handle to hdf5 file): file to which image must be added
        | filename (string):  Raw file filename, data must be np.float64
        | rad_min (float): additive minimum radiance  value in the image, -1 to not use scaling
        | inputSize ([int,int]): raw image size, number of rows,cols
        | outputSize ([int,int]): size of the output image row,cols
        | rad_dynrange (float): multiplicative  radiance scale factor (max value), -1 to not use scaling
        | imgNum (int): image number to be loaded from the image sequence
        | inputOrigin ([int,int]): raw image row,col where the image must be extracted from 
        | blocksize ([int,int]): row,col blocksize in raw image to be averaged to single output pixel 
        | sigma (float): gaussian spatial filter kernel rms size in raw image pixels

    Returns:
        | nothing: as a side effect a set of photon radiance image files are written
    Raises:
        | No exception is raised.

    Author: CJ Willers
    """

    # print(filename,inputSize,outputSize,rad_min,rad_dynrange, imgNum,inputOrigin,blocksize,sigma)

    imghd5['image/rad_dynrange'] = rad_dynrange * imghd5['image/conversion'].value
    imghd5['image/rad_min'] = rad_min * imghd5['image/conversion'].value
    imghd5['image/filename'] = filename 

    # read the imgNum'th raw image frame from file
    nfr,img = ryfiles.readRawFrames(filename, rows=inputSize[0], cols=inputSize[1],
                            vartype=np.float64, loadFrames=[imgNum])
    # print(nfr,img.shape)
    #extract the smaller raw image and coalesce/blur
    img = ryutils.blurryextract(img[0,:,:], inputOrigin=inputOrigin, 
            outputSize=outputSize, 
                sigma=sigma, blocksize=blocksize)

    if nfr > 0:
        img = img / imghd5['image/joule_per_photon'].value

        if imghd5['image/rad_min'].value < 0. and imghd5['image/rad_dynrange'].value < 0.:
            # don't scale the input image
            # create photon rate radiance image from input, no scaling, with no noise
            PhotonRateRadianceNoNoise = img 
        else:
            # scale the input image
            NormLin = (img - np.min(img)) / (np.max(img)- np.min(img))
            # create photon rate radiance image from min to min+dynamic range, with no noise
            PhotonRateRadianceNoNoise = imghd5['image/rad_min'].value  \
                        + NormLin * imghd5['image/rad_dynrange'].value
    else:
        print('Unknown image type or file not successfully read: {}\n no image file created'.format(filename))
        return imghd5

    # save the no noise image
    imghd5['image/PhotonRateRadianceNoNoise'][...] = PhotonRateRadianceNoNoise

    imghd5.flush()

    return imghd5 


######################################################################################
def create_HDF5_image(imageName, numPixels, fn, kwargs, wavelength,
    saveNoiseImage=False,saveEquivImage=False,
    equivalentSignalType='',equivalentSignalUnit='', LinUnits='', seedval=0,fintp=None):
    r"""This routine serves as calling function to a generating function to create images.
    This function expects that the calling function will return photon rate images,
    irrespective of the units of the min/max values used to create the image.
    Each generating function creates an image of a different type, taking as input
    radiant, photon rate, temperature or some other unit, as coded in the generating function.

    This calling function sets up the image and writes common information and then calls the 
    generating function of add the specific image type with radiometric units required.
    The calling function and its arguments must be given as arguments on this functions
    argument list.

    The image file is in HDF5 format, containing the 
    *  input parameters to the image creation process
    *  the image in photon rate units without photon noise
    *  the image in photon rate units with photon noise
    *  the image in some equivalent input unit radiant, photometric or photon rate units.

    The general procedure in the generating function is to  convert the radiance
    input values in units [W/m2] or  [q/m2.s)] to  photon rate radiance in units [q/m2.s)] 
    by converting by one photon's energy at the stated wavelength by 
    :math:`Q_p=\frac{h\cdot c}{\lambda}`,
    where :math:`\lambda` is wavelength, :math:`h` is Planck's constant and :math:`c` is
    the speed of light.  The conversion is done at a single wavelength, which is not very accurate.
    The better procedure is to create the photon rate image directly using a spectral integral.

    The following minimum HDF5 entries are required by pyradi.rystare:

        | ``'image/imageName'`` (string):  the image name  
        | ``'image/PhotonRateRadianceNoNoise'`` np.array[M,N]:  a float array with the image pixel values no noise 
        | ``'image/PhotonRateRadiance'`` np.array[M,N]:  a float array with the image pixel values with noise
        | ``'image/imageSizePixels'``:  ([int, int]): number of pixels [row,col]  
        | ``'image/imageFilename'`` (string):  the image file name  
        | ``'image/wavelength'`` (float):  where photon rate calcs are done  um
        | ``'image/imageSizeRows'`` (int):  the number of image rows
        | ``'image/imageSizeCols'`` (int):  the number of image cols
        | ``'image/imageSizeDiagonal'`` (float):  the FPA diagonal size in mm 
        | ``'image/equivalentSignal'`` (float):  the equivalent input signal, e.g. temperature or lux (optional)
        | ``'image/irradianceWatts'`` (float):  the exitance in the image W/m2 (optional)
        | ``'image/temperature'`` (float):  the maximum target temperature in the image K (optional)

    A few minimum entries are required, but you can add any information you wish to the generaring
    function, by adding the additional information to the generating function's kwargs. 


    Args:
        | imageName (string): the image name, used to form the filename.
        | numPixels ([int, int]): number of pixels [row,col].
        | fn (Python function): the generating function to be used to calculate the image.
        | kwargs (dictionary): kwargs to the passed to the generating function.
        | wavelength (float): wavelength where photon rate calcs are done in [m]
        | equivalentSignalType (str): type of the equivalent input scale (e.g., irradiance, temperature)
        | equivalentSignalUnit (str): units of the equivalent scale (e.g., W/m2, K, lux)
        | LinUnits (str): Lin units and definition separated with : (e.g., 'W/(m2.sr)', 'q/(s.m2.sr)')
        | seedval (int): a seed for the photon noise generator
        | saveNoiseImage (bool): save the noisy image to HDF5 file
        | saveEquivImage (bool): save the equivalent image to HDF5 file
        | fintp (function): interpolation function to map from radiance to equivalent unit

    Returns:
        | hdffilename (string): hdf5 filename
        |                     : as a side effect an image file is written

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """

    hdffilename = 'image-{}-{}-{}.hdf5'.format(imageName, numPixels[0], numPixels[1])
    imghd5 = ryfiles.erase_create_HDF(hdffilename)
    imghd5['image/imageName'] = imageName
    imghd5['image/imageSizePixels'] = numPixels
    imghd5['image/imageSizeRows'] = numPixels[0]
    imghd5['image/imageSizeCols'] = numPixels[1]
    imghd5['image/imageFilename'] = hdffilename
    imghd5['image/equivalentSignalType'] = equivalentSignalType
    imghd5['image/equivalentSignalUnit'] = equivalentSignalUnit
    imghd5['image/LinUnits'] = LinUnits
    imghd5['image/saveNoiseImage'] = saveNoiseImage
    imghd5['image/saveEquivImage'] = saveEquivImage
    dset = imghd5.create_dataset('image/equivalentSignal', numPixels, dtype='float', compression="gzip")
    dset = imghd5.create_dataset('image/PhotonRateRadianceNoNoise', numPixels, dtype='float', compression="gzip")
    dset = imghd5.create_dataset('image/PhotonRateRadiance', numPixels, dtype='float', compression="gzip")
    #photon rate radiance in the image ph/(m2.s), with no photon noise, will be filled by rendering function
    imghd5['image/PhotonRateRadianceNoNoise'][...] = \
              np.zeros((imghd5['image/imageSizePixels'].value[0],imghd5['image/imageSizePixels'].value[1]))

    imghd5['image/wavelength'] = wavelength
    # use units to determine if photon rate or watts
    # joule/photon factor to convert between W/m2 and q/(s.m2)
    if isinstance( imghd5['image/wavelength'], float):   
        imghd5['image/joule_per_photon'] = const.h * const.c / imghd5['image/wavelength'].value
    else:
        imghd5['image/joule_per_photon'] = const.h * const.c / np.mean(imghd5['image/wavelength'].value)
        
    imghd5['image/conversion'] = 1.0 if 'q/' in imghd5['image/LinUnits'].value[:3] \
                                        else 1. / imghd5['image/joule_per_photon'].value
        

    kwargs['imghd5'] = imghd5
    # call the function that actually generates the image
    imghd5 = fn(**kwargs)

    # add photon noise in the signal
    if imghd5['image/saveNoiseImage'].value:
        imghd5['image/PhotonRateRadiance'][...] = \
                 ryutils.poissonarray(imghd5['image/PhotonRateRadianceNoNoise'].value, seedval=seedval)

    # save equivalent signal
    if imghd5['image/saveEquivImage'].value:
        if fintp is not None:
            # save equivalent signal  (e.g., temperature or lux), by interpolation
            imghd5['image/equivalentSignal'][...] = fintp(imghd5['image/PhotonRateRadianceNoNoise'].value)
            # save the interpolation function to hdf5
            imghd5['image/interpolate_x'] = fintp.x
            imghd5['image/interpolate_y'] = fintp.y
        else:
            # save nonoise image as equivalent signal 
            imghd5['image/equivalentSignal'][...] = imghd5['image/PhotonRateRadianceNoNoise'].value
            

    imghd5.flush()
    imghd5.close()

    return hdffilename


######################################################################################
def analyse_HDF5_image(hdffilename,gwidh=12,gheit=8):
    r"""Summarise the image properties and statistics

    Args:
        | imghd5 (handle to hdf5 file): file to be analysed
        | gwidh (float): graph width in inches
        | gheit (float): graph height in inches

    Returns:
        | nothing: as a side effect a set properties are written and graphs created

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    from scipy import stats
    import pyradi.ryplot

    imghd5 = ryfiles.open_HDF(hdffilename)

    #calculate and display values of these variables
    elements = ['image/imageFilename','image/imageName','image/filename','image/rad_dynrange',
    'image/rad_min','image/irrad_dynrange','image/irrad_min','image/disk_diameter','image/blur',
    'image/blur','image/steps','image/imtype','image/imageSizePixels','image/pixelPitch',
    'image/imageSizeRows','image/imageSizeCols','image/imageSizeDiagonal',
    'image/equivalentSignalType','image/equivalentSignalUnit','image/LinUnits','image/EinUnits',
    'image/saveNoiseImage','image/saveEquivImage','image/joule_per_photon',
    'image/conversion',
    ]
    for item in elements:
        if item in imghd5:
            print('{:30s} : {}'.format(item,imghd5[item].value))

    # wavelength as scalar or vector
    if isinstance( imghd5['image/wavelength'], float):   
        print('{:30s} : {}'.format('wavelength',imghd5['image/wavelength'].value))
    else:
        print('{:30s} : {}'.format('wavelength (mean)',np.mean(imghd5['image/wavelength'].value)))

    #calculate and display statistics of these variables
    elements = ['image/PhotonRateRadianceNoNoise','image/PhotonRateRadiance',
    'image/PhotonRateIrradianceNoNoise','image/PhotonRateIrradiance','image/equivalentSignal'
    ]
    for item in elements:
        if item in imghd5:
            print('\nStatistics for {}:'.format(item))
            print(stats.describe(imghd5[item].value,axis=None))

    # plot the images
    p = ryplot.Plotter(1,3,1,hdffilename, figsize=(gwidh,gheit))
    for item in ['image/PhotonRateRadianceNoNoise','image/PhotonRateIrradianceNoNoise']:
        if item in imghd5:
            p.showImage(1,imghd5[item].value,item,cbarshow=True)

    for item in ['image/PhotonRateRadiance','image/PhotonRateIrradiance']:
        if item in imghd5:
            p.showImage(2,imghd5[item].value,item,cbarshow=True)

    if 'image/equivalentSignal' in imghd5:
        p.showImage(3,imghd5['image/equivalentSignal'].value,'image/equivalentSignal',cbarshow=True)

    p.saveFig('{}.png'.format(hdffilename[:-5]))

    # plot interpolation function
    if 'image/interpolate_x' in imghd5:
        q = ryplot.Plotter(1,1,1,hdffilename, figsize=(12,6))
        q.plot(1,imghd5['image/interpolate_x'].value,imghd5['image/interpolate_y'].value)
        q.saveFig('{}-lookup.png'.format(hdffilename[:-5]))

    print(50*'='+'\n\n')



######################################################################################
def calcTemperatureEquivalent(wavelength,sysresp,tmin,tmax):
    """Calc the interpolation function between temperature and photon rate radiance

    Args:
        | wavelength (np.array): wavelength vector
        | sysresp (np.array): system response spectral vector
        | tmin (float): minimum temperature in lookup table 
        | tmax (float): maximum temperature in lookup table 

    Returns:
        | interpolation function

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    wavelength = wavelength.reshape(-1, 1)
    sysresp = sysresp.reshape(-1, 1)
    
    temp = np.linspace(0.99*float(tmin), 1.01*float(tmax), 100).reshape(-1,1)
    # radiance in q/(s.m2)
    rad = np.trapz(sysresp * ryplanck.planck(wavelength, temp,
           type='ql'),wavelength, axis=0).reshape(-1,1) / np.pi
    fintpLE = interpolate.interp1d(rad.reshape(-1,), temp.reshape(-1,))
    fintpEL = interpolate.interp1d(temp.reshape(-1,), rad.reshape(-1,))
    return fintpLE,fintpEL


######################################################################################
def calcLuxEquivalent(wavelength,rad_min,rad_dynrange,units):
    """Calc the interpolation function between lux and photon rate radiance

    Assuming single wavelength colour, the specified wavelength value is used to 
    calculate the lux equivalent lux image for the radiance input range.

    Args:
        | wavelength (np.array): wavelength vector
        | sysresp (np.array): system response spectral vector
        | rad_min (float): minimum photon rate radiance lookup table 
        | rad_dynrange (float): maximum photon rate radiance in lookup table 
        | units (string): input radiance units q/s or W

    Returns:
        | interpolation function

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    if 'q' in units:
        conversion = wavelength / (const.h * const.c)
    else:
        conversion = 1.

    Wm2tolux = 683 * 1.019 * np.exp(-285.51 * (wavelength*1e6 - 0.5591)**2)
    # convert from q/s to W
    rad_minW = rad_min / conversion
    rad_dynrangeW = rad_dynrange / conversion
    radW = np.linspace(0.99*rad_minW, 1.01*(rad_minW+rad_dynrangeW), 1000) 
    lux =  Wm2tolux * radW
    # convert from W back to q/s when setting up the function    
    fintp = interpolate.interp1d((radW*wavelength / (const.h * const.c)).reshape(-1), lux.reshape(-1))
    return fintp


################################################################
################################################################
##
if __name__ == '__init__':
    pass

if __name__ == '__main__':

    import os.path
    import pyradi.ryfiles as ryfiles
    import pyradi.ryutils as ryutils

    doAll = False
    numPixels = [256, 256]  # [ROW, COLUMN] size
    wavelength = 0.55e-6

    #----------  create test images ---------------------
    if True:

        #create a zero uniform photon rate image
        # input in q/(s.m2),  output in q/(s.m2), equivalent in q/(s.m2) units 
        filename = create_HDF5_image(imageName='Zero',  
            numPixels=numPixels,wavelength=wavelength,
            saveNoiseImage=True,saveEquivImage=True,
            fn=hdf_Uniform, kwargs={'rad_dynrange':0},
            equivalentSignalType='Irradiance',equivalentSignalUnit='q/(s.m2.sr)', 
            LinUnits='q/(s.m2.sr)', seedval=0,fintp=None )
        analyse_HDF5_image(filename)

        #create a uniform photon rate image with nonzero value, from radiance input
        # input in q/(s.m2),  output in q/(s.m2), equivalent in q/(s.m2) units 
        filename = create_HDF5_image(imageName='Uniform',  
            numPixels=numPixels,wavelength=wavelength,
            saveNoiseImage=True,saveEquivImage=True,
            fn=hdf_Uniform, kwargs={'rad_dynrange':1.3e17},
            equivalentSignalType='Irradiance',equivalentSignalUnit='q/(s.m2.sr)', 
            LinUnits='q/(s.m2.sr)', seedval=0,fintp=None )
        analyse_HDF5_image(filename)


        # create a disk photon rate image, scaled from unity base, by min + dynamic range
        # input in q/(s.m2),  output in q/(s.m2), equivalent in q/(s.m2) units 
        filename = create_HDF5_image(imageName='Disk', 
            numPixels=numPixels,wavelength=wavelength,
            saveNoiseImage=True,saveEquivImage=True,
            fn=hdf_disk_photon, kwargs={'rad_min':0.0,'rad_dynrange':1.3e17,
                'fracdiameter':0.7,'fracblur':0.2},
            equivalentSignalType='Irradiance',equivalentSignalUnit='q/(s.m2.sr)', 
            LinUnits='q/(s.m2.sr)', seedval=0,fintp=None )
        analyse_HDF5_image(filename)

         
        # create stair photon rate image, scaled from unity base, by min + dynamic range
        # input in W/m2,  output in q/(s.m2), equivalent in lux units 
        rad_min = 9.659e-4
        rad_dynrange = 0.483
        LinUnits = 'W/(m2.sr)'
        fintp = calcLuxEquivalent(wavelength,rad_min,rad_dynrange,LinUnits)
        filename = create_HDF5_image(imageName='Stairslin-10',  
            numPixels=[250, 250], wavelength=wavelength,
            saveNoiseImage=True,saveEquivImage=True,
            fn=hdf_stairs, kwargs={'rad_min':rad_min,'rad_dynrange':rad_dynrange,
                'imtype':'stairslin','steps':10},
                equivalentSignalType='Irradiance',equivalentSignalUnit='lux',
                LinUnits=LinUnits, seedval=0,fintp=fintp )
        analyse_HDF5_image(filename)


        # create stair photon rate image, scaled from unity base, by min + dynamic range
        # input in W/m2,  output in q/(s.m2), equivalent in lux units 
        rad_min = 9.659e-4
        rad_dynrange = 0.483
        LinUnits = 'W/(m2.sr)'
        fintp = calcLuxEquivalent(wavelength,rad_min,rad_dynrange,LinUnits)
        filename = create_HDF5_image(imageName='Stairslin-40',  
            numPixels=[100,520],wavelength=wavelength,
            saveNoiseImage=True,saveEquivImage=True,
            fn=hdf_stairs, kwargs={'rad_min':rad_min,'rad_dynrange':rad_dynrange,
                'imtype':'stairslin','steps':40},
                equivalentSignalType='Irradiance',equivalentSignalUnit='lux', 
                LinUnits=LinUnits, seedval=0,fintp=fintp )
        analyse_HDF5_image(filename)

        # create stair photon rate image, scaled from unity base, by min + dynamic range
        # low light level input in W/m2,  output in q/(s.m2), equivalent in lux units 
        rad_min =9.659e-6
        rad_dynrange = 4.829e-3
        LinUnits = 'W/(m2.sr)'
        fintp = calcLuxEquivalent(wavelength,rad_min,rad_dynrange,LinUnits)
        filename = create_HDF5_image(imageName='Stairslin-LowLight-40',  
            numPixels=[100,520],wavelength=wavelength,
            saveNoiseImage=True,saveEquivImage=True,
            fn=hdf_stairs, kwargs={'rad_min':rad_min,'rad_dynrange':rad_dynrange,
                'imtype':'stairslin','steps':40},
            equivalentSignalType='Irradiance',equivalentSignalUnit='lux', 
            LinUnits='W/(m2.sr)', seedval=0,fintp=fintp )
        analyse_HDF5_image(filename)

        # create photon rate image from raw, unscaled 
        filename = create_HDF5_image(imageName='PtaInd-13Dec14h00X',
            numPixels=[512,512],wavelength=4.5e-6,
            saveNoiseImage=True,saveEquivImage=True,
            fn=hdf_Raw, kwargs={'filename':'data/PtaInd-13Dec14h00X.bin',
                'rad_min':-1,'rad_dynrange':-1,'imgNum':0},
            equivalentSignalType='Irradiance',equivalentSignalUnit='W/m2', 
            LinUnits='W/(m2.sr)', seedval=0,fintp=None )
        analyse_HDF5_image(filename)

        # create photon rate image from raw, unscaled 
        filename = create_HDF5_image(imageName='StairIR-raw',
            numPixels=[100,256],wavelength=4.5e-6,
            saveNoiseImage=True,saveEquivImage=True,
            fn=hdf_Raw, kwargs={'filename':'data/StairIR-raw.double',
                'rad_min':-1,'rad_dynrange':-1,'imgNum':0},
            equivalentSignalType='Irradiance',equivalentSignalUnit='W/m2', 
            LinUnits='W/(m2.sr)', seedval=0,fintp=None )
        analyse_HDF5_image(filename)

       #create an infrared image with lin stairs
        # work in temperature
        tmin = 293 # 20 deg C at minimum level
        tmax = 313 # 40 deg C at maximum level
        # do a wideband spectral integral
        wavelength = np.linspace(3.4,4.9,100)
        sysresp = np.ones(wavelength.shape)
        fintpLE,fintpEL = calcTemperatureEquivalent(wavelength,sysresp,tmin,tmax)

        print(tmin,np.min(fintpEL.x))
        print(tmax,np.max(fintpEL.x))

        filename = create_HDF5_image(imageName='StairslinIR-40',  
            numPixels=[100,520],wavelength=wavelength,
            saveNoiseImage=True,saveEquivImage=True,
            fn=hdf_stairs, kwargs={'rad_min':fintpEL(tmin),
                'rad_dynrange':fintpEL(tmax) -fintpEL(tmin),
                'imtype':'stairslin','steps':40},
            equivalentSignalType='Temperature',equivalentSignalUnit='K', 
            LinUnits='q/(s.m2.sr)', seedval=0,fintp=fintpLE )
        analyse_HDF5_image(filename,15,7)

        #create a scaled infrared image derived  from raw input image
        # use temperatures to define min and max values to which the
        # raw input image is scaled: minImg<->tmin and maxImg<->tmax
        tmin = 280 # K at minimum level
        tmax = 320 # K at maximum level
        # do a wideband spectral integral
        wavelength = np.linspace(3.7,4.8,100)
        sysresp = np.ones(wavelength.shape)
        fintpLE,fintpEL = calcTemperatureEquivalent(wavelength,sysresp,tmin,tmax)
        # create photon rate image from raw, scaled 
        filename = create_HDF5_image(imageName='PtaInd-13Dec14h00XScaled',
            numPixels=[512,512],wavelength=4.5e-6,
            saveNoiseImage=True,saveEquivImage=True,
            fn=hdf_Raw, kwargs={'filename':'data/PtaInd-13Dec14h00X.bin',
                'rad_min':fintpEL(tmin),'rad_dynrange':fintpEL(tmax) -fintpEL(tmin),
                'imgNum':0},
            equivalentSignalType='Temperature',equivalentSignalUnit='K', 
            LinUnits='q/(s.m2.sr)', seedval=0,fintp=fintpLE )
        analyse_HDF5_image(filename,15,7)

        #create a uniform photon rate image with nonzero value, for given temperature
        # input in q/(s.m2),  output in q/(s.m2), equivalent in q/(s.m2) units 
        tuniform = 295
        # do a wideband spectral integral
        wavelength = np.linspace(3.7,4.8,100)
        sysresp = np.ones(wavelength.shape)
        fintpLE,fintpEL = calcTemperatureEquivalent(wavelength,sysresp,tuniform-5,tuniform+5)
        # create photon rate image from raw, scaled 
        filename = create_HDF5_image(imageName='Uniform{:.0f}K'.format(tuniform),  
            numPixels=numPixels,wavelength=wavelength,
            saveNoiseImage=True,saveEquivImage=True,
            fn=hdf_Uniform, kwargs={'rad_dynrange':fintpEL(tuniform)},
            equivalentSignalType='Temperature',equivalentSignalUnit='K', 
            LinUnits='q/(s.m2.sr)', seedval=0,fintp=fintpLE )
        analyse_HDF5_image(filename)




    print('module rytarggen done!')
