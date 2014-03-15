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

# The Initial Developer of the Original Code is MS Willers,
# Portions created by MS Willers are Copyright (C) 2006-2012
# All Rights Reserved.

# Contributor(s): _______________________________________________.
################################################################
"""
This module provides a set of functions to aid in the calculation of 3D noise parameters from
noise images. The functions are based on the work done by John D'Agostino and Curtis Webb.
For details see "3-D Analysis Framwork and Measurement Methodology for Imaging System
Nioise" p110-121 in "Infrared Imaging Systems: Design, Analysis, Modelling, and Testing II",
Holst, G. C., ed., Volume 1488, SPIE (1991).

See the __main__ function for examples of use.
"""
#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__='MS Willers'
__all__=['oprDT', 'oprDV', 'oprDH', 'oprSDT', 'oprSDV','oprSDH', 'getS','getNT','getNH',
    'getNV', 'getNVH','getNTV','getNTH', 'getNTVH', 'getTotal']

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)


import numpy
import ryfiles
import ryplot
import ryptw


################################################################
##
def oprDV(imgSeq):
    """ Operator DV averages over rows for each pixel.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | numpy array of dimensions (frames,1,cols)

    Raises:
        | No exception is raised.
    """

    frames,  rows,  cols = imgSeq.shape

    im = numpy.mean(imgSeq, axis=1, dtype=numpy.float64)
    imShaped = im.reshape(frames*cols)
    return imShaped.reshape(frames, 1, cols)


################################################################
##
def oprDH(imgSeq):
    """ Operator DH averages over columns for each pixel.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | numpy array of dimensions (frames,rows,1)

    Raises:
        | No exception is raised.
    """

    frames,  rows,  cols = imgSeq.shape

    im = numpy.mean(imgSeq, axis=2, dtype=numpy.float64)
    imShaped = im.reshape(frames*rows)
    return imShaped.reshape(frames, rows, 1)


################################################################
##
def oprDT(imgSeq):
    """ Operator DT averages over frames for each pixel.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | numpy array of dimensions (1,rows,cols)

    Raises:
        | No exception is raised.
    """

    frames,  rows,  cols = imgSeq.shape

    im = numpy.mean(imgSeq, axis=0, dtype=numpy.float64)
    imShaped = im.reshape(rows*cols)
    return imShaped.reshape(1, rows, cols)


################################################################
##
def oprSDT(imgSeq):
    """ Operator SDT first averages over frames for each pixel. The result is
    subtracted from all images.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | numpy array of dimensions (frames,rows,cols)

    Raises:
        | No exception is raised.
    """

    return imgSeq - oprDT(imgSeq)


################################################################
##
def oprSDH(imgSeq):
    """ Operator SDH first averages over columns for each pixel. The result is subtracted from all images.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | numpy array of dimensions (frames,rows,cols)

    Raises:
        | No exception is raised.
    """

    return imgSeq - oprDH(imgSeq)


################################################################
##
def oprSDV(imgSeq):
    """ Operator SDV first averages over rows for each pixel. The result is subtracted from all images.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | numpy array of dimensions (frames,rows,cols)

    Raises:
        | No exception is raised.
    """

    return imgSeq - oprDV(imgSeq)


################################################################
##
def getS(imgSeq):
    """ Average over all pixels.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): average of all pixels

    Raises:
        | No exception is raised.
    """
    im = oprDT(oprDV(oprDH(imgSeq)))

    return im[0, 0, 0]


################################################################
##
def getNT(imgSeq):
    """ Average for all pixels as a function of time/frames.
    Represents noise which consists of fluctuations in the temporal direction affecting
    the mean of each frame.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): frame-to-frame intensity variation

    Raises:
        | No exception is raised.
    """

    im = oprDV(oprDH(oprSDT(imgSeq)))

    # std = sqrt(mean(abs(x - x.mean())**2))
    # The average squared deviation is normally calculated as x.sum() / N, where N = len(x).
    # If ddof is specified, the divisor N - ddof is used instead.
    # In standard statistical practice, ddof=1 provides an unbiased estimator of the variance of the infinite population.
    # ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables.
    # The standard deviation computed in this function is the square root of the estimated variance, so even with ddof=1,
    # it will not be an unbiased estimate of the standard deviation per se.

    return (numpy.std(im, dtype=numpy.float64,  ddof=1),im)


################################################################
##
def getNVH(imgSeq):
    """ Average over all frames, for each pixel. Represents non-uniformity
    spatial noise that does not change from frame-to-frame.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): fixed sparial noise

    Raises:
        | No exception is raised.
    """

    im = oprDT(oprSDV(oprSDH(imgSeq)))

    return (numpy.std(im, dtype=numpy.float64,  ddof=1),im)


################################################################
##
def getNTV(imgSeq):
    """ Average for each row and frame over all columns .
    Represents variations in row averages that change from frame-to-frame.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): row temporal noise

    Raises:
        | No exception is raised.
    """

    im = oprDH(oprSDT(oprSDV(imgSeq)))

    return (numpy.std(im, dtype=numpy.float64,  ddof=1), im)


################################################################
##
def getNTH(imgSeq):
    """ Average for each column and frame over all rows.
    Represents variations in column averages that change from frame-to-frame.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): column temporal noise

    Raises:
        | No exception is raised.
    """

    im = oprDV(oprSDT(oprSDH(imgSeq)))

    return (numpy.std(im, dtype=numpy.float64,  ddof=1), im)


################################################################
##
def getNV(imgSeq):
    """ Average for each column over all frames and rows .
    Represents variations in row averages that are fixed in time.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): fixed row noise

    Raises:
        | No exception is raised.
    """

    im = oprDT(oprDH(oprSDV(imgSeq)))

    return (numpy.std(im, dtype=numpy.float64,  ddof=1), im)


################################################################
##
def getNH(imgSeq):
    """ Average for each row over all frames and cols.
    Represents variations in column averages that are fixed in time.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): fixed column noise

    Raises:
        | No exception is raised.
    """

    im = oprDT(oprDV(oprSDH(imgSeq)))

    return (numpy.std(im, dtype=numpy.float64,  ddof=1), im)


################################################################
##
def getNTVH(imgSeq):
    """ Noise for each row,  frame & column.
    Represents random noise in the detector and electronics.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): temporal pixel noise

    Raises:
        | No exception is raised.
    """

    im = oprSDT(oprSDV(oprSDH(imgSeq)))

    return (numpy.std(im, dtype=numpy.float64,  ddof=1), im)


################################################################
##
def getTotal(imgSeq):
    """ Total system noise.

    Args:
        | imgSeq (numpy.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): total system noise

    Raises:
        | No exception is raised.
    """

    nt,im    = getNT(imgSeq)
    nvh,im  = getNVH(imgSeq)
    ntv,im   = getNTV(imgSeq)
    nth,im   = getNTH(imgSeq)
    nv,im    = getNV(imgSeq)
    nh,im    = getNH(imgSeq)
    ntvh,im = getNTVH(imgSeq)

    return  numpy.sqrt(nt**2 + nvh**2 + ntv**2 + nth**2 + nv**2 + nh**2 + ntvh**2)


################################################################
##

if __name__ == '__main__':

    #--------------------- simulated sensor noise data from raw format ------------------------------------------------------------------------

    rows      = 100
    cols        = 100

    vartype   = numpy.uint16
    imagefile  = 'data/sensornoise.raw'
    outfilename = 'sensornoise.txt'

    # load images
    framesToLoad = range(1, 21, 1)
    frames, img    = ryfiles.readRawFrames(imagefile, rows, cols, vartype, framesToLoad)

    if frames > 0:

        # show something
        P = ryplot.Plotter(1, 1, 1,'Simulated noise', figsize=(12, 8))
        P.showImage(1, img[0])
        #P.getPlot().show()
        P.saveFig('rawframe0.png')

        outfile = open(outfilename, 'w')
        outfile.write('\n{0} Frames read from {1}\n'.format(frames, imagefile))
        outfile.write('\nImage average S       : {0:10.3f} \n'.format(getS(img)))
        outfile.write('Total system noise    : {0:10.3f} \n\n'.format(getTotal(img)))
        outfile.write('Fixed/spatial noise  | Temporal noise      | Variation effect\n')
        outfile.write('---------------------|---------------------|-----------------\n')
        outfile.write('Nh    : {0:10.3f}   | Nth   : {1:10.3f}  | Column \n'.format(getNH(img)[0],getNTH(img)[0]))
        outfile.write('Nv    : {0:10.3f}   | Ntv   : {1:10.3f}  | Row \n'.format(getNV(img)[0],getNTV(img)[0]))
        outfile.write('Nvh   : {0:10.3f}   | Ntvh  : {1:10.3f}  | Pixel \n'.format(getNVH(img)[0],getNTVH(img)[0]))
        outfile.write('                     | Nt    : {0:10.3f}  | Frame \n'.format(getNT(img)[0]))

        print('\n({0}, {3}, {4}) (frames, rows,cols) processed from {1} - see results in {2}\n'.format(frames, imagefile, outfilename,rows,cols))

    else:
        print('Error in reading noise images data')

    NH = getNH(img)[1].reshape(cols).reshape(1,cols)
    NTH = getNTH(img)[1].reshape(frames*cols).reshape(frames,cols)
    NV = getNV(img)[1].reshape(rows).reshape(rows,1)
    NTV = getNTV(img)[1].reshape(frames*rows).reshape(rows,frames)
    NVH = getNVH(img)[1].reshape(rows*cols).reshape(rows,cols)
    NTVH = getNTVH(img)[1][0,:,:]
    NT = getNT(img)[1].reshape(frames).reshape(frames,1)

    # print(NH.shape)
    # print(NTH.shape)
    # print(NV.shape)
    # print(NTV.shape)
    # print(NVH.shape)
    # print(NTVH.shape)
    # print(NT.shape)

    P = ryplot.Plotter(1, 3, 3,'Simulated Image Sequence 3-D Noise Components', figsize=(12, 12))
    P.showImage(1, NH, ptitle='Nh: Fixed Column: Avg(vt)', titlefsize=10)
    P.showImage(2, NTH, ptitle='Nth: Temporal Column: Avg(v)', titlefsize=10)
    P.showImage(4, NV, ptitle='Nv: Fixed Row: Avg(ht)', titlefsize=10)
    P.showImage(5, NTV, ptitle='Ntv: Temporal Row: Avg(h)', titlefsize=10)
    P.showImage(8, NVH, ptitle='Nvh: Fixed Pixel: Avg(t)\nPixel Non-uniformity', titlefsize=10)
    P.showImage(9, NTVH, ptitle='Ntvh: Temporal: Avg()\nRandom noise', titlefsize=10)
    P.showImage(7, NT, ptitle='Nt: Temporal Frame: Avg(hv)', titlefsize=10)
    P.showImage(3, img[0,:,:], ptitle='First Frame', titlefsize=10)
    P.showImage(6, img[-1,:,:], ptitle='Last Frame', titlefsize=10)
    P.getPlot().show()
    P.saveFig('simnoise.png')

    #--------------------- Jade ptw file with noise data ------------------------------------------------------------------------

    ptwfile  = 'data/PyradiSampleMWIR.ptw'
    outfilename = 'PyradiSampleMWIR.txt'

    header = ryptw.readPTWHeader(ptwfile)
    #ryptw.showHeader(header)

    rows = header.h_Rows
    cols = header.h_Cols

    framesToLoad = range(1, 101, 1)
    frames = len(framesToLoad)

    data = ryptw.getPTWFrame (header, framesToLoad[0])
    for frame in framesToLoad[1:]:
        f = (ryptw.getPTWFrame (header, frame))
        data = numpy.concatenate((data, f))

    img = data.reshape(frames, rows ,cols)

    # show something
    P = ryplot.Plotter(1, 1, 1,'Measured MWIR noise', figsize=(12, 8))
    P.showImage(1, img[0])
    P.getPlot().show()
    P.saveFig('ptwframe0.png')

    outfile = open(outfilename, 'w')

    outfile.write('\n{0} Frames read from {1}\n'.format(frames, ptwfile))
    outfile.write('\nImage average S       : {0:10.3f} \n'.format(getS(img)))
    outfile.write('Total system noise    : {0:10.3f} \n\n'.format(getTotal(img)))
    outfile.write('Fixed/spatial noise  | Temporal noise      | Variation effect\n')
    outfile.write('---------------------|---------------------|-----------------\n')
    outfile.write('Nh    : {0:10.3f}   | Nth   : {1:10.3f}  | Column \n'.format(getNH(img)[0],getNTH(img)[0]))
    outfile.write('Nv    : {0:10.3f}   | Ntv   : {1:10.3f}  | Row \n'.format(getNV(img)[0],getNTV(img)[0]))
    outfile.write('Nvh   : {0:10.3f}   | Ntvh  : {1:10.3f}  | Pixel \n'.format(getNVH(img)[0],getNTVH(img)[0]))
    outfile.write('                     | Nt    : {0:10.3f}  | Frame \n'.format(getNT(img)[0]))

    outfile.close()

    print('\n({0}, {3}, {4}) (frames, rows,cols) processed from {1} - see results in {2}\n'.format(frames, imagefile, outfilename,rows,cols))

    NH = getNH(img)[1].reshape(cols).reshape(1,cols)
    NTH = getNTH(img)[1].reshape(frames*cols).reshape(frames,cols)
    NV = getNV(img)[1].reshape(rows).reshape(rows,1)
    NTV = getNTV(img)[1].reshape(frames*rows).reshape(frames,rows)
    NVH = getNVH(img)[1].reshape(rows*cols).reshape(rows,cols)
    NTVH = getNTVH(img)[1][0,:,:]
    NT = getNT(img)[1].reshape(frames).reshape(frames,1)

    # print(NH.shape)
    # print(NTH.shape)
    # print(NV.shape)
    # print(NTV.shape)
    # print(NVH.shape)
    # print(NTVH.shape)
    # print(NT.shape)

    P = ryplot.Plotter(1, 3, 3,'PTW Image Sequence 3-D Noise Components', figsize=(12, 12))
    P.showImage(1, NH, ptitle='Nh: Fixed Column: Avg(vt)')
    P.showImage(2, NTH, ptitle='Nth: Temporal Column: Avg(v)')
    P.showImage(4, NV, ptitle='Nv: Fixed Row: Avg(ht)')
    P.showImage(5, NTV, ptitle='Ntv: Temporal Row: Avg(h)')
    P.showImage(8, NVH, ptitle='Nvh: Fixed Pixel: Avg(t)\nPixel Non-uniformity')
    P.showImage(9, NTVH, ptitle='Ntvh: Temporal: Avg()\nRandom noise')
    P.showImage(7, NT, ptitle='Nt: Temporal Frame: Avg(hv)')
    P.showImage(3, img[0,:,:], ptitle='First Frame')
    P.showImage(6, img[-1,:,:], ptitle='Last Frame')
    P.getPlot().show()

    P.saveFig('ptwnoise.png')

    print('ry3dnoise done!')
