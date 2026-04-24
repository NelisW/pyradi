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

This package was partly developed to provide additional material in support of students 
and readers of the book Electro-Optical System Analysis and Design: A Radiometry 
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""




__version__ = '0.1.0'
__author__='MS Willers'
__all__=['oprDT', 'oprDV', 'oprDH', 'oprSDT', 'oprSDV','oprSDH', 'getS','getNT','getNH',
    'getNV', 'getNVH','getNTV','getNTH', 'getNTVH', 'getTotal']

import sys
import numpy as np


################################################################
##
def oprDV(imgSeq):
    """ Operator DV averages over rows for each pixel.

    Args:
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | numpy array of dimensions (frames,1,cols)

    Raises:
        | No exception is raised.
    """

    frames,  rows,  cols = imgSeq.shape

    im = np.mean(imgSeq, axis=1, dtype=np.float64)
    imShaped = im.reshape(frames*cols)
    return imShaped.reshape(frames, 1, cols)


################################################################
##
def oprDH(imgSeq):
    """ Operator DH averages over columns for each pixel.

    Args:
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | numpy array of dimensions (frames,rows,1)

    Raises:
        | No exception is raised.
    """

    frames,  rows,  cols = imgSeq.shape

    im = np.mean(imgSeq, axis=2, dtype=np.float64)
    imShaped = im.reshape(frames*rows)
    return imShaped.reshape(frames, rows, 1)


################################################################
##
def oprDT(imgSeq):
    """ Operator DT averages over frames for each pixel.

    Args:
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | numpy array of dimensions (1,rows,cols)

    Raises:
        | No exception is raised.
    """

    frames,  rows,  cols = imgSeq.shape

    im = np.mean(imgSeq, axis=0, dtype=np.float64)
    imShaped = im.reshape(rows*cols)
    return imShaped.reshape(1, rows, cols)


################################################################
##
def oprSDT(imgSeq):
    """ Operator SDT first averages over frames for each pixel. The result is
    subtracted from all images.

    Args:
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

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
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

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
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

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
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

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
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): frame-to-frame intensity variation

    Raises:
        | No exception is raised.
    """

    im = oprDV(oprDH(oprSDT(imgSeq)))

    # ddof=1: unbiased variance estimator (divides by N-1).
    # ddof=0: maximum-likelihood estimate (divides by N).
    return (np.std(im, dtype=np.float64, ddof=1), im)


################################################################
##
def getNVH(imgSeq):
    """ Average over all frames, for each pixel. Represents non-uniformity
    spatial noise that does not change from frame-to-frame.

    Args:
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): fixed spatial noise

    Raises:
        | No exception is raised.
    """

    im = oprDT(oprSDV(oprSDH(imgSeq)))

    return (np.std(im, dtype=np.float64,  ddof=1),im)


################################################################
##
def getNTV(imgSeq):
    """ Average for each row and frame over all columns.
    Represents variations in row averages that change from frame-to-frame.

    Args:
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): row temporal noise

    Raises:
        | No exception is raised.
    """

    im = oprDH(oprSDT(oprSDV(imgSeq)))

    return (np.std(im, dtype=np.float64,  ddof=1), im)


################################################################
##
def getNTH(imgSeq):
    """ Average for each column and frame over all rows.
    Represents variations in column averages that change from frame-to-frame.

    Args:
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): column temporal noise

    Raises:
        | No exception is raised.
    """

    im = oprDV(oprSDT(oprSDH(imgSeq)))

    return (np.std(im, dtype=np.float64,  ddof=1), im)


################################################################
##
def getNV(imgSeq):
    """ Average for each column over all frames and rows.
    Represents variations in row averages that are fixed in time.

    Args:
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): fixed row noise

    Raises:
        | No exception is raised.
    """

    im = oprDT(oprDH(oprSDV(imgSeq)))

    return (np.std(im, dtype=np.float64,  ddof=1), im)


################################################################
##
def getNH(imgSeq):
    """ Average for each row over all frames and cols.
    Represents variations in column averages that are fixed in time.

    Args:
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): fixed column noise

    Raises:
        | No exception is raised.
    """

    im = oprDT(oprDV(oprSDH(imgSeq)))

    return (np.std(im, dtype=np.float64,  ddof=1), im)


################################################################
##
def getNTVH(imgSeq):
    """ Noise for each row,  frame & column.
    Represents random noise in the detector and electronics.

    Args:
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

    Returns:
        | noise (double): temporal pixel noise

    Raises:
        | No exception is raised.
    """

    im = oprSDT(oprSDV(oprSDH(imgSeq)))

    return (np.std(im, dtype=np.float64,  ddof=1), im)


################################################################
##
def getTotal(imgSeq):
    """ Total system noise.

    Args:
        | imgSeq (np.ndarray): numpy array of dimensions (frames,rows,cols)

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

    return  np.sqrt(nt**2 + nvh**2 + ntv**2 + nth**2 + nv**2 + nh**2 + ntvh**2)


################################################################
##

if __name__ == '__main__':

    import pyradi.ryfiles as ryfiles
    import pyradi.ryplot as ryplot
    import pyradi.ryptw as ryptw
    import pyradi.ryutils as ryutils

    #--------------------- simulated sensor noise data from raw format -------------------

    rows = 100
    cols = 100

    vartype     = np.uint16
    imagefile   = 'data/sensornoise.raw'
    outfilename = 'sensornoise.txt'

    # load images
    framesToLoad = list(range(1, 21, 1))
    frames, img = ryfiles.readRawFrames(imagefile, rows, cols, vartype, framesToLoad)

    if frames > 0:

        # show something
        P = ryplot.Plotter(1, 1, 1, 'Simulated noise', figsize=(12, 8))
        P.showImage(1, img[0])
        P.saveFig('rawframe0.png')

        with open(outfilename, 'w') as outfile:
            outfile.write(f'\n{frames} Frames read from {imagefile}\n')
            outfile.write(f'\nImage average S       : {getS(img):10.3f} \n')
            outfile.write(f'Total system noise    : {getTotal(img):10.3f} \n\n')
            outfile.write('Fixed/spatial noise  | Temporal noise      | Variation effect\n')
            outfile.write('---------------------|---------------------|-----------------\n')
            outfile.write(f'Nh    : {getNH(img)[0]:10.3f}   | Nth   : {getNTH(img)[0]:10.3f}  | Column \n')
            outfile.write(f'Nv    : {getNV(img)[0]:10.3f}   | Ntv   : {getNTV(img)[0]:10.3f}  | Row \n')
            outfile.write(f'Nvh   : {getNVH(img)[0]:10.3f}   | Ntvh  : {getNTVH(img)[0]:10.3f}  | Pixel \n')
            outfile.write(f'                     | Nt    : {getNT(img)[0]:10.3f}  | Frame \n')

        print(f'\n({frames}, {rows}, {cols}) (frames, rows, cols) processed from {imagefile} - see results in {outfilename}\n')

    else:
        print('Error in reading noise images data')

    NH = getNH(img)[1].reshape(cols).reshape(1,cols)
    NTH = getNTH(img)[1].reshape(frames*cols).reshape(frames,cols)
    NV = getNV(img)[1].reshape(rows).reshape(rows,1)
    NTV = getNTV(img)[1].reshape(frames*rows).reshape(rows,frames)
    NVH = getNVH(img)[1].reshape(rows*cols).reshape(rows,cols)
    NTVH = getNTVH(img)[1][0,:,:]
    NT = getNT(img)[1].reshape(frames).reshape(frames,1)

    P = ryplot.Plotter(1, 3, 3,'Simulated Image Sequence 3-D Noise Components', figsize=(12, 12))
    P.showImage(1, NH, ptitle=f'Nh: Fixed Column: Avg(vt) {ryutils.intify_tuple(NH.shape)}', titlefsize=10)
    P.showImage(2, NTH, ptitle=f'Nth: Temporal Column: Avg(v) {ryutils.intify_tuple(NTH.shape)}', titlefsize=10)
    P.showImage(4, NV, ptitle=f'Nv: Fixed Row: Avg(ht) {ryutils.intify_tuple(NV.shape)}', titlefsize=10)
    P.showImage(5, NTV, ptitle=f'Ntv: Temporal Row: Avg(h) {ryutils.intify_tuple(NTV.shape)}', titlefsize=10)
    P.showImage(8, NVH, ptitle=f'Nvh: Fixed Pixel: Avg(t)\nPixel Non-uniformity {ryutils.intify_tuple(NVH.shape)}', titlefsize=10)
    P.showImage(9, NTVH, ptitle=f'Ntvh: Temporal: Avg()\nRandom noise {ryutils.intify_tuple(NTVH.shape)}', titlefsize=10)
    P.showImage(7, NT, ptitle=f'Nt: Temporal Frame: Avg(hv) {ryutils.intify_tuple(NT.shape)}', titlefsize=10)
    P.showImage(3, img[0,:,:], ptitle=f'First Frame {ryutils.intify_tuple(img[0, :, :].shape)}', titlefsize=10)
    P.showImage(6, img[-1,:,:], ptitle=f'Last Frame {ryutils.intify_tuple(img[-1, :, :].shape)}', titlefsize=10)
    P.getPlot().show()
    P.saveFig('simnoise.png')

    #--------------------- Jade ptw file with noise data --------------------------------

    ptwfile     = 'data/PyradiSampleMWIR.ptw'
    outfilename = 'PyradiSampleMWIR.txt'

    header = ryptw.readPTWHeader(ptwfile)

    rows = header.h_Rows
    cols = header.h_Cols

    framesToLoad = list(range(1, 101, 1))
    frames = len(framesToLoad)
    data, _ = ryptw.getPTWFrame(header, framesToLoad[0])
    for frame in framesToLoad[1:]:
        f, _ = ryptw.getPTWFrame(header, frame)
        data = np.concatenate((data, f))

    img = data.reshape(frames, rows, cols)

    # show something
    P = ryplot.Plotter(1, 1, 1, 'Measured MWIR noise', figsize=(12, 8))
    P.showImage(1, img[0])
    P.getPlot().show()
    P.saveFig('ptwframe0.png')

    with open(outfilename, 'w') as outfile:
        outfile.write(f'\n{frames} Frames read from {ptwfile}\n')
        outfile.write(f'\nImage average S       : {getS(img):10.3f} \n')
        outfile.write(f'Total system noise    : {getTotal(img):10.3f} \n\n')
        outfile.write('Fixed/spatial noise  | Temporal noise      | Variation effect\n')
        outfile.write('---------------------|---------------------|-----------------\n')
        outfile.write(f'Nh    : {getNH(img)[0]:10.3f}   | Nth   : {getNTH(img)[0]:10.3f}  | Column \n')
        outfile.write(f'Nv    : {getNV(img)[0]:10.3f}   | Ntv   : {getNTV(img)[0]:10.3f}  | Row \n')
        outfile.write(f'Nvh   : {getNVH(img)[0]:10.3f}   | Ntvh  : {getNTVH(img)[0]:10.3f}  | Pixel \n')
        outfile.write(f'                     | Nt    : {getNT(img)[0]:10.3f}  | Frame \n')

    print(f'\n({frames}, {rows}, {cols}) (frames, rows, cols) processed from {ptwfile} - see results in {outfilename}\n')

    NH = getNH(img)[1].reshape(cols).reshape(1,cols)
    NTH = getNTH(img)[1].reshape(frames*cols).reshape(frames,cols)
    NV = getNV(img)[1].reshape(rows).reshape(rows,1)
    NTV = getNTV(img)[1].reshape(frames*rows).reshape(frames,rows)
    NVH = getNVH(img)[1].reshape(rows*cols).reshape(rows,cols)
    NTVH = getNTVH(img)[1][0,:,:]
    NT = getNT(img)[1].reshape(frames).reshape(frames,1)



    P = ryplot.Plotter(1, 3, 3,'PTW Image Sequence 3-D Noise Components', figsize=(12, 12))
    P.showImage(1, NH, ptitle=f'Nh: Fixed Column: Avg(vt) {ryutils.intify_tuple(NH.shape)}', titlefsize=10)
    P.showImage(2, NTH, ptitle=f'Nth: Temporal Column: Avg(v) {ryutils.intify_tuple(NTH.shape)}', titlefsize=10)
    P.showImage(4, NV, ptitle=f'Nv: Fixed Row: Avg(ht) {ryutils.intify_tuple(NV.shape)}', titlefsize=10)
    P.showImage(5, NTV, ptitle=f'Ntv: Temporal Row: Avg(h) {ryutils.intify_tuple(NTV.shape)}', titlefsize=10)
    P.showImage(8, NVH, ptitle=f'Nvh: Fixed Pixel: Avg(t)\nPixel Non-uniformity {ryutils.intify_tuple(NVH.shape)}', titlefsize=10)
    P.showImage(9, NTVH, ptitle=f'Ntvh: Temporal: Avg()\nRandom noise {ryutils.intify_tuple(NTVH.shape)}', titlefsize=10)
    P.showImage(7, NT, ptitle=f'Nt: Temporal Frame: Avg(hv) {ryutils.intify_tuple(NT.shape)}', titlefsize=10)
    P.showImage(3, img[0,:,:], ptitle=f'First Frame {ryutils.intify_tuple(img[0, :, :].shape)}', titlefsize=10)
    P.showImage(6, img[-1,:,:], ptitle=f'Last Frame {ryutils.intify_tuple(img[-1, :, :].shape)}', titlefsize=10)
    P.getPlot().show()

    P.saveFig('ptwnoise.png')

    print('ry3dnoise done!')
