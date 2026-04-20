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

# Contributor(s): MS Willers.
################################################################
"""
This module provides functions for file input/output. These are all wrapper
functions, based on existing functions in other Python classes. Functions are 
provided to save a two-dimensional array to a text file, load selected columns 
of data from a text file, load a column header line, compact strings to include 
only legal filename characters, and a function from the Python Cookbook to 
recursively match filename patterns.

See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students 
and readers of the book Electro-Optical System Analysis and Design: A Radiometry 
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""

__version__= "$Revision$"
__author__='pyradi team'
__all__=['open_HDF', 'erase_create_HDF', 'print_HDF5_text', 'print_HDF5_dataset_value', 
         'get_HDF_branches', 'plotHDF5Bitmaps', 'plotHDF5Images', 'plotHDF5Histograms'
]

import sys
import numpy as np
import os.path, fnmatch
from matplotlib import cm as mcm
import h5py
from skimage.io import imread, imsave



######################################################################################
def open_HDF(filename, mode='r'):
    """Open and return an HDF5 file with the given filename.

    See https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md
    for more information on using HDF5 as a data structure.

    Default open mode is read, cam be overridden by any of these:
    'r', 'r+', 'w', 'w-'/'x', 'a'
    https://h5py.readthedocs.io/en/stable/quick.html
    https://h5py.readthedocs.io/en/stable/high/file.html#opening-creating-files

    Args:
        | filename (string): name of the file to be opened

    Returns:
        | HDF5 file.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    f = h5py.File(filename,mode)
    return f


######################################################################################
def erase_create_HDF(filename):
    """Create and return a new HDS5 file with the given filename, erase the file if existing.

    See https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md
    for more information on using HDF5 as a data structure.

    open for writing, truncate if exists
    https://h5py.readthedocs.io/en/stable/high/file.html#opening-creating-files

    Args:
        | filename (string): name of the file to be created

    Returns:
        | HDF5 file.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    if os.path.isfile(filename):
        os.remove(filename)
    f = h5py.File(filename,'w')
    return f

######################################################################################
def get_HDF_DS(hdf5File, path):
    """Check to see if a path is present in an HDF5 file and then return the dataset.

    Args:
        | hdf5File (string): HDF5 file to be read
        | path (string): path to the dataset

    Returns:
        | HDF5 contents along the path, or None.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    if path in hdf5File:
        rtn = hdf5File[path]
    else:
        rtn = None

    return rtn


######################################################################################
def print_HDF5_text(vartext):
    """Prints text in visiting algorithm in HDF5 file.

    See https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md
    for more information on using HDF5 as a data structure.

    Args:
        | vartext (string): string to be printed

    Returns:
        | HDF5 file.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    print(vartext)

######################################################################################
def print_HDF5_dataset_value(var, obj):
    """Prints a data set in visiting algorithm in HDF5 file.

    See https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md
    for more information on using HDF5 as a data structure.

    Args:
        | var (string): path to a dataset
        | obj (h5py dataset): dataset to be printed

    Returns:
        | HDF5 file.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    if type(obj.file[var]) is h5py._hl.dataset.Dataset:
        print(var, obj.file[var].name)


######################################################################################
def get_HDF_branches(hdf5File):
    """Print list of all the branches in the file

    See https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md
    for more information on using HDF5 as a data structure.

    Args:
        | hdf5File (H5py file): the file to be opened

    Returns:
        | HDF5 file.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    return hdf5File.visit(get_HDF_branches)

######################################################################################
def plotHDF5Bitmaps(hfd5f, prefix, pformat='png', lstimgs=None, debug=False):
    """Plot arrays in the HFD5 as scaled bitmap images.

    See https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md
    for more information on using HDF5 as a data structure.

    Retain zero in the array as black in the image, only scale the max value to 255

    Args:
        | hfd5f (H5py file): the file to be opened
        | prefix (string): prefix to be prepended to filename
        | pformat (string): type of file to be created png/jpeg
        | lstimgs ([string]): list of paths to image in the HFD5 file

    Returns:
        | Nothing.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    from . import ryplot

    for lstimg in lstimgs:
        arr = hfd5f['{}'.format(lstimg)][...]
        if debug:
            print('data set {} has shape {} '.format(lstimg,arr.shape))

        # if arr.shape is not ():
        if arr.shape != ():
            if np.max(arr) != 0.:
                arr = 255 * arr/np.max(arr)
                imsave('{}-{}.{}'.format(prefix,lstimg.replace('/','-'),pformat), arr.astype(np.uint8))


######################################################################################
def plotHDF5Images(hfd5f, prefix, colormap=mcm.jet, cbarshow=True, lstimgs=None, logscale=False, debug=False):
    """Plot images contained in hfd5f with colour map to show magnitude.

    See https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md
    for more information on using HDF5 as a data structure.

    http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps

    Args:
        | hfd5f (H5py file): the file to be opened
        | prefix (string): prefix to be prepended to filename
        | colormap (Matplotlib colour map): colour map to be used in plot
        | cbarshow (boolean): indicate if colour bar must be shown
        | lstimgs ([string]): list of paths to image in the HFD5 file
        | logscale (boolean): True if display must be on log scale

    Returns:
        | Nothing.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    from . import ryplot

    for lstimg in lstimgs:
        arr = hfd5f['{}'.format(lstimg)][...]

        if debug:
            print('data set {} has shape {} '.format(lstimg,arr.shape))

        # if arr.shape is not ():
        if arr.shape != ():
            if logscale:
                filename = '{}-plot-{}-log.png'.format(prefix,lstimg.replace('/','-'))
                with ryplot.savePlot(1,1,1,figsize=(8,8), saveName=[filename]) as p:
                    p.showImage(1, np.log10(arr), ptitle=lstimg, cmap=colormap, cbarshow=cbarshow);
            else:
                filename = '{}-plot-{}.png'.format(prefix,lstimg.replace('/','-'))
                with ryplot.savePlot(1,1,1,figsize=(8,8), saveName=[filename]) as p:
                    p.showImage(1, arr, ptitle=lstimg, cmap=colormap, cbarshow=cbarshow);


######################################################################################
def plotHDF5Histograms(hfd5f, prefix, format='png', lstimgs=None, bins=50):
    """Plot histograms of images contained in hfd5f

    See https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md
    for more information on using HDF5 as a data structure.

    Retain zero in the array as black in the image, only scale the max value to 255

    Args:
        | hfd5f (H5py file): the file to be opened
        | prefix (string): prefix to be prepended to filename
        | format (string): type of file to be created png/jpeg
        | lstimgs ([string]): list of paths to image in the HFD5 file
        | bins ([int]): Number of bins to be used in histogram

    Returns:
        | Nothing.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    from . import ryplot

    for lstimg in lstimgs:
        arr = hfd5f['{}'.format(lstimg)][...]
        his, bin = np.histogram(arr,bins=bins)
        filename = '{}-hist-plot-{}.{}'.format(prefix,lstimg.replace('/','-'),format)
        with ryplot.savePlot(1,1,1,figsize=(8,4), saveName=[filename]) as p:
            p.plot(1, (bin[1:]+bin[:-1])/2, his, '{}, {} bins'.format(lstimg, bins), 'Magnitude','Counts / bin',maxNX=5)


   

################################################################
################################################################
##
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':
    pass