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
http://nelisw.github.io/pyradi-docs/_build/html/index.html,
http://nelisw.github.io/pyradi-docs/_build/html/rytarggen.html, or 
pyradi/doc/rytarggen.rst  
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= ""
__author__='CJ Willers'
__all__=[
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
def hdf_Uniform_photon(imghd5,irrad_min,irrad_dynrange):
    r"""A generating function to create a uniform photon rate image.
    The unoform value in the image will be  irrad_min + irrad_dynrange.

    The function accepts radiant or photon rate minimum and dynamic range units.
    The equivalent image value is expressed as in the same units as the input

    This function must be called from rytarggen.create_HDF5_image

    Args:
        | imghd5 (handle to hdf5 file): file to which image must be added
        | irrad_min (float): additive minimum value in the image
        | irrad_dynrange (float): multiplicative scale factor (max value)
        | fracdiameter (float):  diameter of the disk as fraction of minimum image size
        | fracblurr (float):   blurr of the disk as fraction of minimum image size

    Returns:
        | nothing: as a side effect an image file is written

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """

    imghd5['image/irrad_dynrange'] = irrad_dynrange 
    imghd5['image/irrad_min'] = irrad_min 

    # create photon rate irradiance image from min to min+dynamic range, with no noise
    PhotonRateIrradianceNoNoise = \
                     (irrad_min  + np.ones((imghd5['image/imageSizeCols'].value.shape)) * irrad_dynrange) /\
                        imghd5['image/p_photon'].value

    # save the no noise image
    if imghd5['image/saveNoNoiseImage'].value:
        imghd5['image/PhotonRateIrradianceNoNoise'][...] = PhotonRateIrradianceNoNoise

    # add photon noise in the signal
    if imghd5['image/saveNoiseImage'].value:
        imghd5['image/PhotonRateIrradiance'][...] = ryutils.poissonarray(
                      PhotonRateIrradianceNoNoise, seedval=0)

    # save equivalent signal
    if imghd5['image/saveEquivImage'].value:
        imghd5['image/equivalentSignal'][...] = PhotonRateIrradianceNoNoise

    imghd5.flush()

    return imghd5 


######################################################################################
def hdf_disk_photon(imghd5,irrad_min,irrad_dynrange,fracdiameter,fracblurr):
    r"""A generating function to create an image with illuminated circle with blurred boundaries.

    The function accepts radiant or photon rate minimum and dynamic range units.
    The equivalent image value is expressed as in the same units as the input

    This function must be called from rytarggen.create_HDF5_image

    Args:
        | imghd5 (handle to hdf5 file): file to which image must be added
        | irrad_min (float): additive minimum value in the image
        | irrad_dynrange (float): multiplicative scale factor (max value)
        | fracdiameter (float):  diameter of the disk as fraction of minimum image size
        | fracblurr (float):   blurr of the disk as fraction of minimum image size

    Returns:
        | nothing: as a side effect an image file is written

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """

    imghd5['image/irrad_dynrange'] = irrad_dynrange 
    imghd5['image/irrad_min'] = irrad_min 
    # scale the disk to image size, as fraction
    maxSize = np.min((imghd5['image/imageSizeRows'].value, imghd5['image/imageSizeCols'].value))
    imghd5['image/disk_diameter'] = fracdiameter * maxSize
    imghd5['image/blurr'] = fracblurr * maxSize

    #create the disk, normalised to unity
    varx = np.linspace(-imghd5['image/imageSizeCols'].value/2, imghd5['image/imageSizeCols'].value/2, imghd5['image/imageSizePixels'].value[1])
    vary = np.linspace(-imghd5['image/imageSizeRows'].value/2, imghd5['image/imageSizeRows'].value/2, imghd5['image/imageSizePixels'].value[0])
    x1, y1 = np.meshgrid(varx, vary)
    delta_x = varx[1] - varx[0]
    delta_y = vary[1] - vary[0]
    Uin = ryutils.circ(x1,y1,imghd5['image/disk_diameter'].value) 
    #create blur disk normalised to unity
    dia = np.max((1, 2 * round(imghd5['image/blurr'].value / np.max((delta_x,delta_y)))))
    varx = np.linspace(-dia, dia, 2 * dia)
    vary = np.linspace(-dia, dia, 2 * dia)
    x, y = np.meshgrid(varx, vary)
    H = ryutils.circ(x, y, dia)
    # convolve disk with blur
    NormEin = (np.abs(signal.convolve2d(Uin, H, mode='same'))/np.sum(H))  ** 2

    # create the photon rate irradiance image from min to min+dynamic range, with no noise
    PhotonRateIrradianceNoNoise =  \
                (irrad_min  + NormEin * irrad_dynrange) / imghd5['image/p_photon'].value

    # save the no noise image
    if imghd5['image/saveNoNoiseImage'].value:
        imghd5['image/PhotonRateIrradianceNoNoise'][...] = PhotonRateIrradianceNoNoise

    # add photon noise in the signal
    if imghd5['image/saveNoiseImage'].value:
        imghd5['image/PhotonRateIrradiance'][...] = ryutils.poissonarray(
                      PhotonRateIrradianceNoNoise, seedval=0)

    # save equivalent signal
    if imghd5['image/saveEquivImage'].value:
        imghd5['image/equivalentSignal'][...] = PhotonRateIrradianceNoNoise

    imghd5.flush()

    return imghd5 



######################################################################################
def hdf_stairs_lux(imghd5,irrad_min,irrad_dynrange,steps,imtype):
    r"""A generating function to create a staircase image, with log/linear and precribed step count.

    The increment along stairs can be linear or logarithmic.

    The function accepts only radiant minimum and dynamic range units.
    The equivalent image value is expressed as in lux units

    This function must be called from rytarggen.create_HDF5_image

    Args:
        | imghd5 (handle to hdf5 file): file to which image must be added
        | irrad_min (float): additive minimum value in the image
        | irrad_dynrange (float): multiplicative scale factor (max value)
        | steps (int): number of steps in the image
        | imtype (string): string to define the type of image to be created ['stairslin','stairslog']

    Returns:
        | nothing: as a side effect an image file is written

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """

    imghd5['image/irrad_dynrange'] = irrad_dynrange 
    imghd5['image/irrad_min'] = irrad_min 
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
    NormEin = y * x  * np.ones(x.shape)  

    # create the photon rate irradiance image from min to min+dynamic range, with no noise
    PhotonRateIrradianceNoNoise = \
             (irrad_min  + NormEin * irrad_dynrange) / imghd5['image/p_photon'].value
    
    # save the no noise image
    if imghd5['image/saveNoNoiseImage'].value:
        imghd5['image/PhotonRateIrradianceNoNoise'][...] = PhotonRateIrradianceNoNoise

    # add photon noise in the signal
    if imghd5['image/saveNoiseImage'].value:
        imghd5['image/PhotonRateIrradiance'][...] = \
                 ryutils.poissonarray(PhotonRateIrradianceNoNoise, seedval=0)

    # save equivalent signal
    if imghd5['image/saveEquivImage'].value:
        # calc the interpolation between lux and photon rate irradiance
        Wm2tolux = 683 * 1.019 * np.exp(-285.51 * (imghd5['image/wavelength'].value*1e6 - 0.5591)**2)
        irrad = np.linspace(irrad_min, (irrad_min+irrad_dynrange), 1000) 
        lux =  Wm2tolux * irrad
        fintp = interpolate.interp1d((irrad/ imghd5['image/p_photon'].value).reshape(-1), lux.reshape(-1))
        # saveequivalent signal  (e.g., temperature or lux), by interpolation
        imghd5['image/equivalentSignal'][...] = PhotonRateIrradianceNoNoise

    imghd5.flush()

    return imghd5 



######################################################################################
def hdf_Raw(imghd5,filename,irrad_min=-1,irrad_dynrange=-1, imgNum=0):
    r"""A generating function to create a photon rate image from raw image.
    The raw image read in will be recaled to irrad_min + irrad_dynrange.

    The raw image sequence must be of type np.float64 with no header or footer.

    The function accepts radiant or photon rate minimum and dynamic range units.
    The equivalent image value is expressed as in the same units as the output image

    This function must be called from rytarggen.create_HDF5_image

    Args:
        | imghd5 (handle to hdf5 file): file to which image must be added
        | filename (string):  Raw file filename, data must be np.float64
        | irrad_min (float): additive minimum value in the image, -1 to not use scaling
        | irrad_dynrange (float): multiplicative scale factor (max value), -1 to not use scaling
        | imgNum (int): image numbmer to be loaded from the image sequence

    Returns:
        | nothing: as a side effect an image file is written

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """

    imghd5['image/irrad_dynrange'] = irrad_dynrange 
    imghd5['image/irrad_min'] = irrad_min 
    imghd5['image/filename'] = filename 

    # read the imgNum'th raw image frame from file
    nfr,img = ryfiles.readRawFrames(filename, rows=imghd5['image/imageSizePixels'].value[0], 
                            cols=imghd5['image/imageSizePixels'].value[1],
                            vartype=np.float64, loadFrames=[imgNum])

    if nfr > 0:
        if irrad_min < 0. and irrad_dynrange < 0.:
            # don't scale the input image
            # create photon rate irradiance image from input, no scaling, with no noise
            PhotonRateIrradianceNoNoise = img / imghd5['image/p_photon'].value
        else:
            # scale the input image
            img_min = np.min(img)
            img_dynrange = np.max(img)- np.min(img)
            NormEin = (img - img_min) / irrad_dynrange
            # create photon rate irradiance image from min to min+dynamic range, with no noise
            PhotonRateIrradianceNoNoise = (irrad_min  + NormEin * irrad_dynrange) /\
                                imghd5['image/p_photon'].value

    else:
        print('Unknown image type or file not successully read: {}\n no image file created'.format(filename))
        return imghd5

    # save the no noise image
    if imghd5['image/saveNoNoiseImage'].value:
        imghd5['image/PhotonRateIrradianceNoNoise'][...] = PhotonRateIrradianceNoNoise

    # add photon noise in the signal
    if imghd5['image/saveNoiseImage'].value:
        imghd5['image/PhotonRateIrradiance'][...] = \
                 ryutils.poissonarray(PhotonRateIrradianceNoNoise, seedval=0)

    # save equivalent signal
    if imghd5['image/saveEquivImage'].value:
        imghd5['image/equivalentSignal'][...] = PhotonRateIrradianceNoNoise

    imghd5.flush()

    return imghd5 


######################################################################################
def create_HDF5_image(imageName, pixelPitch, numPixels, fn, kwargs, wavelength,
    saveNoNoiseImage=True,saveNoiseImage=True,saveEquivImage=True,
    equivalentSignalType='',equivalentSignalUnit='', EinUnits=''):
    r"""This routine serves as calling function to a generating function to create images.
    Each generating function creates an image of a different type, taking as input
    radiant, photon rate, temperature or some other unit, as coded in the generating function.

    This calling function sets up the image and writes common information and then calls the 
    generating function of add the specific image type with radiometric units required.
    The calling function and its arguments must be given as arguments on this functions
    argument list.

    The image file is in HDF5 format, containing the 
    - input parameters to the image creation process
    - the image in photon rate units without photon niose
    - the image in photon rate units with photon noise
    - the image in some equivalent input unit radiant, photometric or photon rate units.

    The general procedur in the generating function is to  convert the irradiance
    input values in units [W/m2] to  photon rate irradiance in units [q/m2.s)] by relating 
    one photon's energy to power at the stated wavelength by :math:`Q_p=\frac{h\cdot c}{\lambda}`,
    where :math:`\lambda` is wavelength, :math:`h` is Planck's constant and :math:`c` is
    the speed of light.  The conversion is done at a single wavelength, which is not very accurate.
    The better procedure is to create the photon rate image direction in the spectral domain as
    a photon image.

    The following minimum HDF5 entries are required by pyradi.rystare:

        | ``'image/imageName'`` (string):  the image name  
        | ``'image/PhotonRateIrradianceNoNoise'`` np.array[M,N]:  a float array with the image pixel values no noise 
        | ``'image/PhotonRateIrradiance'`` np.array[M,N]:  a float array with the image pixel values with noise
        | ``'image/pixelPitch'``:  ([float, float]):  detector pitch in m [row,col]  
        | ``'image/imageSizePixels'``:  ([int, int]): number of pixels [row,col]  
        | ``'image/imageFilename'`` (string):  the image file name  
        | ``'image/wavelength'`` (float):  where photon rate calcs are done  um
        | ``'image/imageSizeRows'`` (int):  the number of image rows
        | ``'image/imageSizeCols'`` (int):  the number of image cols
        | ``'image/imageSizeDiagonal'`` (float):  the FPA diagonal size in mm 
        | ``'image/equivalentSignal'`` (float):  the equivalent input signal, e.g. temperaure or lux (optional)
        | ``'image/irradianceWatts'`` (float):  the exitance in the image W/m2 (optional)
        | ``'image/temperature'`` (float):  the maximum target temperature in the image K (optional)

    A few minimum entries are required, but you can add any information you wish to the generaring
    function, by adding the additional information to the generating function's kwargs. 


    Args:
        | imageName (string): the image name, used to form the filename.
        | pixelPitch ([float, float]):  detector pitch in m [row,col].
        | numPixels ([int, int]): number of pixels [row,col].
        | fn (Python function): the generating function to be used to calculate the image.
        | kwargs (dictionary): kwargs to the passed to the generating function.
        | wavelength (float): wavelength where photon rate calcs are done in [m]
        | equivalentSignalType (str): type of the equivalent input scale (e.g., irradiance, temperature)
        | equivalentSignalUnit (str): units of the equivalent input scale (e.g., W/m2, K)
        | EinUnits (str): Ein units and definition separated with : (e.g., 'W/m2 : on detector', 'q/(s.m2) : on detector')

        | saveNoNoiseImage (bool): save the noiseless image to HDF5 file
        | saveNoiseImage (bool): save the noisy image to HDF5 file
        | saveEquivImage (bool): save the equivalent image to HDF5 file

    Returns:
        | nothing: as a side effect an image file is written

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """

    hdffilename = 'image-{}-{}-{}.hdf5'.format(imageName, numPixels[0], numPixels[1])
    imghd5 = ryfiles.erase_create_HDF(hdffilename)
    imghd5['image/imageName'] = imageName
    imghd5['image/imageSizePixels'] = numPixels
    imghd5['image/pixelPitch'] = pixelPitch
    imghd5['image/imageSizeRows'] = pixelPitch[0] * numPixels[0]
    imghd5['image/imageSizeCols'] = pixelPitch[1] * numPixels[1]
    imghd5['image/imageSizeDiagonal'] = np.sqrt((pixelPitch[0] * numPixels[0]) ** 2. + (pixelPitch[1] * numPixels[1]) ** 2)
    imghd5['image/imageFilename'] = hdffilename
    imghd5['image/equivalentSignalUnit'] = equivalentSignalUnit
    imghd5['image/equivalentSignalType'] = equivalentSignalType
    imghd5['image/EinUnits'] = EinUnits
    imghd5['image/saveNoNoiseImage'] = saveNoNoiseImage
    imghd5['image/saveNoiseImage'] = saveNoiseImage
    imghd5['image/saveEquivImage'] = saveEquivImage
    dset = imghd5.create_dataset('image/equivalentSignal', numPixels, dtype='float', compression="gzip")
    dset = imghd5.create_dataset('image/PhotonRateIrradianceNoNoise', numPixels, dtype='float', compression="gzip")
    dset = imghd5.create_dataset('image/PhotonRateIrradiance', numPixels, dtype='float', compression="gzip")
    #photon rate irradiance in the image ph/(m2.s), with no photon noise, will be filled by rendering function
    imghd5['image/PhotonRateIrradianceNoNoise'][...] = \
              np.zeros((imghd5['image/imageSizePixels'].value[0],imghd5['image/imageSizePixels'].value[1]))

    imghd5['image/wavelength'] = wavelength
    p_photon = const.h * const.c / imghd5['image/wavelength'].value\
                     if 'W/' in imghd5['image/EinUnits'].value[:3] else 1.0
    imghd5['image/p_photon'] = p_photon

    kwargs['imghd5'] = imghd5
    imghd5 = fn(**kwargs)

    imghd5.flush()
    imghd5.close()


################################################################
################################################################
##
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    import os.path

    import pyradi.ryfiles as ryfiles
    import pyradi.ryutils as ryutils

    doAll = False

    numPixels = [256, 256]  # [ROW, COLUMN] size
    pixelPitch = [5e-6, 5e-6] # pixels pitch, in [m], [ROW, COLUMN] 
    wavelength = 0.55e-6

    #----------  create test images ---------------------

    if True:

        #create a uniform photon rate image, scaled from unity base, by min + dynamic range
        # input in q/(s.m2),  output in q/(s.m2), equivalent in q/(s.m2) units 
        create_HDF5_image(imageName='Uniform',  pixelPitch=pixelPitch, numPixels=numPixels,wavelength=wavelength,
            fn=hdf_Uniform_photon, kwargs={'irrad_min':0.0,'irrad_dynrange':1e16},
            equivalentSignalType='Irradiance',equivalentSignalUnit='q/(s.m2)', EinUnits='q/(s.m2): on detector' )

        # create a disk photon rate image, scaled from unity base, by min + dynamic range
        # input in q/(s.m2),  output in q/(s.m2), equivalent in q/(s.m2) units 
        create_HDF5_image(imageName='Disk',  pixelPitch=pixelPitch, numPixels=numPixels,wavelength=wavelength,
            fn=hdf_disk_photon, kwargs={'irrad_min':0.0,'irrad_dynrange':1e16,
            'fracdiameter':0.7,'fracblurr':0.2},
            equivalentSignalType='Irradiance',equivalentSignalUnit='q/(s.m2)', EinUnits='q/(s.m2): on detector' )

        # create stair photon rate image, scaled from unity base, by min + dynamic range
        # input in W/m2,  output in q/(s.m2), equivalent in lux units 
        create_HDF5_image(imageName='Stairslin-10',  pixelPitch=[5e-6, 5e-6], numPixels=[250, 250],
            wavelength=wavelength,
            fn=hdf_stairs_lux, kwargs={'irrad_min':74.08e-6,'irrad_dynrange':37.04e-3,
            'imtype':'stairslin','steps':10},
            equivalentSignalType='Irradiance',equivalentSignalUnit='lux', EinUnits='W/m2: on detector' )

        # create stair photon rate image, scaled from unity base, by min + dynamic range
        # input in W/m2,  output in q/(s.m2), equivalent in lux units 
        create_HDF5_image(imageName='Stairslin-40',  pixelPitch=[5e-6, 5e-6], numPixels=[100,520],
            wavelength=wavelength,
            fn=hdf_stairs_lux, kwargs={'irrad_min':74.08e-6,'irrad_dynrange':37.04e-3,
            'imtype':'stairslin','steps':40},
            equivalentSignalType='Irradiance',equivalentSignalUnit='lux', EinUnits='W/m2: on detector' )

        # create stair photon rate image, scaled from unity base, by min + dynamic range
        # low light level input in W/m2,  output in q/(s.m2), equivalent in lux units 
        create_HDF5_image(imageName='Stairslin-LowLight-40',  pixelPitch=[5e-6, 5e-6], numPixels=[100,520],
            wavelength=wavelength,
            fn=hdf_stairs_lux, kwargs={'irrad_min':74.08e-8,'irrad_dynrange':37.04e-5,
            'imtype':'stairslin','steps':40},
            equivalentSignalType='Irradiance',equivalentSignalUnit='lux', EinUnits='W/m2: on detector' )

        # create photon rate image from raw, scaled from unity base, by min + dynamic range
        # low light level input in W/m2,  output in q/(s.m2), equivalent in lux units 
        wavelength = 4.5e-6
        create_HDF5_image(imageName='E-W-m2-detector-raw',  pixelPitch=[12e-6, 12e-6], numPixels=[100,256],
            wavelength=wavelength,
            fn=hdf_Raw, kwargs={'filename':'data/E-W-m2-detector-100-256.double',
             'irrad_min':-1,'irrad_dynrange':-1,'imgNum':0},
            equivalentSignalType='Irradiance',equivalentSignalUnit='W/m2', EinUnits='W/m2: on detector' )




    if doAll:
        pass


        #create an infrared image with lin stairs
        # work in temperature
        tmin = 293 # 20 deg C at minimum level
        tmax = 313 # 40 deg C at maximum level
        # do all calcs at this wavelength, normally this would be a wideband spectral integral
        wavelength = 4.0e-6
        fno = 3.2
        omega = np.pi / ((2. * fno)**2.)
        deltalambda = 4.9 - 3.5
        irrad_min = deltalambda * omega * ryplanck.planck(wavelength*1e6, tmin, type='ql') / np.pi
        irrad_max = deltalambda * omega * ryplanck.planck(wavelength*1e6, tmax, type='ql') / np.pi
        irrad_dynrange = irrad_max - irrad_min
        tser = np.linspace(tmin, tmax, 100).reshape(-1)
        irrad = deltalambda * ryplanck.planck(wavelength*1e6, tser, type='ql').reshape(-1) * (omega / np.pi)
        fintp = interpolate.interp1d(irrad, tser)

        create_HDF5_imageZ(imageName='StairslinIR-40', imtype='stairslin', pixelPitch=[12e-6, 12e-6],
                numPixels=[100,520], wavelength=wavelength,
                irrad_dynrange=irrad_dynrange, irrad_min=irrad_min, 
                steps=40,fintp=fintp,equivalentSignalType='Temperature',
                equivalentSignalUnit='K',EinUnits='q/(s.m2) on detector plane')


