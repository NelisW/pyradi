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
__all__=['saveHeaderArrayTextFile', 'loadColumnTextFile', 'loadHeaderTextFile', 
         'cleanFilename', 'listFiles','readRawFrames','rawFrameToImageFile',
         'arrayToLaTex','epsLaTexFigure',
         'read2DLookupTable', 
         'downloadFileUrl', 'unzipGZipfile', 'untarTarfile', 'downloadUntar',
         'open_HDF', 'erase_create_HDF', 'print_HDF5_text', 'print_HDF5_dataset_value', 
         'get_HDF_branches', 'plotHDF5Bitmaps', 'plotHDF5Images', 'plotHDF5Histograms']

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)


from scipy.interpolate import interp1d
import numpy as np
import os.path, fnmatch
from matplotlib import cm as mcm
import h5py
from skimage.io import imread, imsave

################################################################
def saveHeaderArrayTextFile(filename,dataArray, header=None,
        comment=None, delimiter=None):
    """Save a numpy array to a file, included header lines.

    This function saves a two-dimensional array to a text file, with
    an optional user-defined header. This functionality will be part of
    numpy 1.7, when released.

    Args:
        | filename (string): name of the output ASCII flatfile.
        | dataArray (np.array[N,M]): a two-dimensional array.
        | header (string): the optional header.
        | comment (string): the symbol used to comment out lines, default value is None.
        | delimiter (string): delimiter used to separate columns, default is whitespace.

    Returns:
        | Nothing.

    Raises:
        | No exception is raised.
    """
    #open required file
    file=open(filename, 'wt')

    #write the header info to the output file
    if (header is not None):
        for line in header.split('\n'):
            file.write(comment+line+'\n')

    #then write the array, using the file handle (and not filename)
    np.savetxt(file, dataArray, delimiter=delimiter)
    #neatly close the file
    file.close()


################################################################
def loadColumnTextFile(filename, loadCol=[1],  \
        comment=None, normalize=0, skiprows=0, delimiter=None,\
        abscissaScale=1,ordinateScale=1, abscissaOut=None, returnAbscissa=False):
    """Load selected column data from a text file, processing as specified.

    This function loads column data from a text file, scaling and interpolating 
    the read-in data, according to user specification. The first 0'th column has 
    special significance: it is considered the abscissa (x-values) of the data 
    set, while the remaining columns are any number of ordinate (y-value) vectors.
    The user passes a list of columns to be read (default is [1]) - only these 
    columns are read, processed and returned when the function exits.The user 
    also passes an abscissa vector to which the input data is interpolated and 
    then subsequently amplitude scaled or normalised.  

    Note: leave only single separators (e.g. spaces) between columns!
    Also watch out for a single space at the start of line.

    Args:
        | filename (string): name of the input ASCII flatfile.
        | loadCol ([int]): the M =len([]) column(s) to be loaded as the ordinate, default value is column 1
        | comment (string): string, the symbol used to comment out lines, default value is None
        | normalize (int): integer, flag to indicate if data must be normalized.
        | skiprows (int): integer, the number of rows to be skipped at the start of the file (e.g. headers)
        | delimiter (string): string, the delimiter used to separate columns, default is whitespace.
        | abscissaScale (float): scale by which abscissa (column 0) must be multiplied
        | ordinateScale (float): scale by which ordinate (column >0) must be multiplied
        | abscissaOut (np.array[N,] or [N,1]): abscissa vector on which output variables are interpolated.
        | returnAbscissa (bool): return the abscissa vector as second item in return tuple.

    Returns:
        | ordinatesOut (np.array[N,M]): The interpolated, M columns of N rows, processed array.
        | abscissaOut (np.array[N,M]): The ascissa where the ordinates are interpolated

    Raises:
        | No exception is raised.
    """

    #prepend the 0'th column to the rest of the list, make local copy first
    ldCol = loadCol[:]
    ldCol.insert(0, 0)

    #load first column as well as user-specified column from the
    # given file, scale as prescribed
    coldata = np.loadtxt(filename, usecols=ldCol,
            comments=comment,  skiprows=skiprows,
            delimiter=delimiter)

    abscissa = abscissaScale * coldata[:,0]
    ordinate = ordinateScale * coldata[:,1:]

    if  abscissaOut is not None:
        #convert to [N, ] array
        abscissaOut=abscissaOut.reshape(-1,)
        #inpterpolate read values with the given inut vec
        f=interp1d(abscissa,  ordinate, axis=0)
        interpValue=f(abscissaOut)
    else:
        interpValue = ordinate
        abscissaOut = abscissa

    #if read-in values must be normalised.
    if normalize != 0:
        interpValue /= np.max(interpValue,axis=0)

    if returnAbscissa:
        return interpValue, abscissaOut.reshape(-1,1)
    else:
        return interpValue


################################################################################
def loadHeaderTextFile(filename, loadCol=[1], comment=None):
    """Loads column header data in the first string of a text file.

    loads column header data from a file, from the first row.
    Headers must be delimited by commas. The function [LoadColumnTextFile] provides
    more comprehensive capabilties.

    Args:
        | filename (string): the name of the input ASCII flatfile.
        | loadCol ([int]): list of numbers, the column headers to be loaded , default value is column 1
        | comment (string): the symbol to comment out lines

    Returns:
        | [string]: a list with selected column header entries

    Raises:
        | No exception is raised.
    """

    if isinstance(filename, str):
        infile = open(filename, 'rb')
    else:
        infile = filename

    line = infile.readline()
    line = line.lstrip(' '+comment).split(',')
    #get rid of leading and trailing whitespace
    list=[x.strip() for x in line]
    #select only those column headers required
    rtnList =[list[i] for i in loadCol ]

    return rtnList


################################################################
def cleanFilename(sourcestring,  removestring =" %:/,.\\[]<>"):
    """Clean a string by removing selected characters.

    Creates a legal and 'clean' source string from a string by removing some 
    clutter and  characters not allowed in filenames.
    A default set is given but the user can override the default string.

    Args:
        | sourcestring (string): the string to be cleaned.
        | removestring (string): remove all these characters from the string (optional).

    Returns:
        | (string): A cleaned-up string.

    Raises:
        | No exception is raised.
    """
    #remove the undesireable characters
    return [c for c in sourcestring if c not in removestring]



################################################################
def downloadUntar(tgzFilename, url, destinationDir=None,  tarFilename=None):
    """Download and untar a compressed tar archive, and save all files to the specified directory.

    The tarfilename is used to open the tar file, extracting to the destinationDir specified.
    If no destinationDir is given, the local directory '.' is used.
    Before downloading, a check is done to determine if the file was already downloaded
    and exists in the local file system.

    Args:
        | tgzFilename (string): the name of the tar archive file
        | url (string): url where to look for the file (not including the filename)
        | destinationDir (string): to where the files must be extracted (optional)
        | tarFilename (string): downloaded tar filename (optional)

    Returns:
        | ([string]): list of filenames saved, or None if failed.
 
    Raises:
        | Exceptions are handled internally and signaled by return value.
    """

    import os

    if destinationDir is None:
        dirname = '.'
    else:
        dirname = destinationDir

    if tarFilename is None:
        tarname = tgzFilename + '.tar'
    else:
        tarname = tarFilename

    tgzPath = os.path.join(destinationDir, tgzFilename)
    if  os.path.isfile(tgzPath):
        tgzAvailable = True
        # print('{} is already available, download not required'.format(tgzPath))
    else:    
        urlfile = url+tgzFilename
        # print("Attempting to download the data file {}".format(urlfile))
        if downloadFileUrl(urlfile) is None:
            print('\ndownload failed, please check url or internet connection')
            tgzAvailable = False
        else:
            tgzAvailable = True
    result = []
    if tgzAvailable == True:
        if unzipGZipfile(tgzPath,tarname) is None:
            print('Unzipping the tgz file {} to dir {} failed'.format(tgzPath,tarname))
        else:
            result = untarTarfile(tarname,destinationDir)
            if result is None:
                print('untarTarfile failed for {} to {}'.format(tarname,destinationDir))
                filesAvailable = False
            else:
                filesAvailable = True
                # print('Sucessfully extracted:\n{}'.format(result))

    return result


################################################################
def untarTarfile(tarfilename, saveDirname=None):
    """Untar a tar archive, and save all files to the specified directory.

    The tarfilename is used to open a file, extraxting to the saveDirname specified.
    If no saveDirname is given, the local directory '.' is used.

    Args:
        | tarfilename (string): the name of the tar archive.
        | saveDirname (string): to where the files must be extracted

    Returns:
        | ([string]): list of filenames saved, or None if failed.

    Raises:
        | Exceptions are handled internally and signaled by return value.
    """

    if saveDirname is None:
        dirname = '.'
    else:
        dirname = saveDirname

    import tarfile
    import os
    import errno

    try:
        os.makedirs(dirname)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            print('Unable to create directory {}'.format(dirname))
            return None


    f = tarfile.open(tarfilename, 'r')
    filenames = f.getnames()
    f.extractall(dirname)
    f.close()

    # filexextracted = []
    # for filename in filenames:
    #     if not os.path.isfile(os.path.join(dirname, filename)):
    #         filexextracted.append(filename)
    return filenames


################################################################
def unzipGZipfile(zipfilename, saveFilename=None):
    """Unzip a file that was compressed using the gzip format.

    The zipfilename is used to open a file, to the saveFilename specified.
    If no saveFilename is given, the basename of the zipfilename is used, 
    but with the file extension removed.

    Args:
        | zipfilename (string): the zipfilename to be decompressed.
        | saveFilename (string): to where the file must be saved (optional).

    Returns:
        | (string): Filename saved, or None if failed.

    Raises:
        | Exceptions are handled internally and signaled by return value.
    """

    if saveFilename is None:
        filename = os.path.basename(zipfilename)[:-4]
    else:
        filename = saveFilename

    import gzip
    #get file handle
    f = gzip.open(zipfilename, 'rb')
    try:
        # Open file for writing
        with open(filename, "wb") as file:
            file.write(f.read())
    except:
        print('Unzipping of {} failed'.format(zipfilename))
        return None
    finally:
        f.close()

    return filename


################################################################
def downloadFileUrl(url,  saveFilename=None):
    """Download a file, given a URL.

    The URL is used to download a file, to the saveFilename specified.
    If no saveFilename is given, the basename of the URL is used.
    Before doownloading, first test to see if the file already exists.

    Args:
        | url (string): the url to be accessed.
        | saveFilename (string): path to where the file must be saved (optional).

    Returns:
        | (string): Filename saved, or None if failed.

    Raises:
        | Exceptions are handled internally and signaled by return value.
    """

    if saveFilename is None:
        filename = os.path.basename(url)
    else:
        filename = saveFilename

    if os.path.exists(filename):
        pass
    else:
        import urllib.request, urllib.error, urllib.parse
        from urllib.error import HTTPError

        try:
            #get file handle
            f = urllib.request.urlopen(url)
            # Open file for writing
            with open(filename, "wb") as file:
                file.write(f.read())
        #handle errors
        except urllib.error.HTTPError as e:
           print('HTTP Error: {} for {}'.format(e.code, url))
           return None
        except urllib.error. URLError as e:
           print('URL Error: {} for {}'.format(e.reason, url))
           return None

    return filename


################################################################
#lists the files in a directory and subdirectories
#this code is adapted from a recipe in the Python Cookbook
def listFiles(root, patterns='*', recurse=1, return_folders=0, useRegex=False):
    """Lists the files/directories meeting specific requirement

        Returns a list of file paths to files in a file system, searching a 
        directory structure along the specified path, looking for files 
        that matches the glob pattern. If specified, the search will continue 
        into sub-directories.  A list of matching names is returned. The 
        function supports a local or network reachable filesystem, but not URLs.

    Args:
        | root (string): directory root from where the search must take place
        | patterns (string): glob/regex pattern for filename matching
        | recurse (unt): flag to indicate if subdirectories must also be searched (optional)
        | return_folders (int): flag to indicate if folder names must also be returned (optional)
        | useRegex (bool): flag to indicate if patterns areregular expression strings (optional)

    Returns:
        | A list with matching file/directory names

    Raises:
        | No exception is raised.
    """
    if useRegex:
        import re

    # Expand patterns from semicolon-separated string to list
    pattern_list = patterns.split(';')
    # Collect input and output arguments into one bunch
    class Bunch:
        def __init__(self, **kwds): self.__dict__.update(kwds)
    arg = Bunch(recurse=recurse, pattern_list=pattern_list,
        return_folders=return_folders, results=[])

    def visit(arg, dirname, files):
        # Append to arg.results all relevant files (and perhaps folders)
        for name in files:
            fullname = os.path.normpath(os.path.join(dirname, name))
            if arg.return_folders or os.path.isfile(fullname):
                for pattern in arg.pattern_list:
                    if useRegex:
                        regex = re.compile(pattern)
                        #search returns None is pattern not found
                        if regex.search(name):
                            arg.results.append(fullname)
                            break
                    else:
                        if fnmatch.fnmatch(name, pattern):
                            arg.results.append(fullname)
                            break
        # Block recursion if recursion was disallowed
        if not arg.recurse: files[:]=[]
    os.path.walk(root, visit, arg)
    return arg.results

################################################################
##
def rawFrameToImageFile(image, filename):
    """Writes a single raw image frame to image file.
    The file type must be given, e.g. png or jpg.
    The image need not be scaled beforehand, it is done prior 
    to writing out the image. Could be one of
    BMP, JPG, JPEG, PNG, PPM, TIFF, XBM, XPM)
    but the file types available depends
    on the QT imsave plugin in use.

    Args:
        | image (np.ndarray): two-dimensional array representing an image
        | filename (string): name of file to be written to, with extension

    Returns:
        | Nothing

    Raises:
        | No exception is raised.
    """
    #normalise input image (img) data to between 0 and 1
    from scipy import ndimage

    image = (image - ndimage.minimum(image)) / (ndimage.maximum(image) - ndimage.minimum(image))

    # http://scikit-image.org/docs/dev/api/skimage.io.html#imsave
    import skimage.io as io
    io.imsave(filename, image) 


################################################################
##
def readRawFrames(fname, rows, cols, vartype, loadFrames=[]):
    """ Loading multi-frame two-dimensional arrays from a raw data file of known data type.

        The file must consist of multiple frames, all with the same number of rows and columns.
        Frames of different data types can be read, according to the user specification.  
        The user can specify which frames must be loaded (if not the whole file).

    Args:
        | fname (string): filename
        | rows (int): number of rows in each frame
        | cols (int): number of columns in each frame
        | vartype (np.dtype): numpy data type of data to be read
        |                                      int8, int16, int32, int64
        |                                      uint8, uint16, uint32, uint64
        |                                      float16, float32, float64
        | loadFrames ([int]): optional list of frames to load, zero-based , empty list (default) loads all frames

    Returns:
        | frames (int) : number of frames in the returned data set,
        |                      0 if error occurred
        | rawShaped (np.ndarray): vartype numpy array of dimensions (frames,rows,cols),
        |                                              None if error occurred

    Raises:
        | No exception is raised.
    """

    frames = 0
    rawShaped = None

    # load all frames in the file

    if not loadFrames:
        try:
            with open(fname, 'rb') as fin:
                data = np.fromfile(fin, vartype,-1)

        except IOError:
            #print('  File not found, returning {0} frames'.format(frames))
            return int(frames), rawShaped

    # load only frames requested

    else:
        try:
            framesize = rows * cols;
            lastframe = max(loadFrames)
            data = None

            with open(fname, 'rb') as fin:
                for frame in range(0, lastframe+1, 1):
                    dataframe = np.fromfile(fin, vartype,framesize)
                    if frame in loadFrames:
                        if data is None:
                            data = dataframe
                        else:
                            data = np.concatenate((data, dataframe))

        except IOError:
            #print('  File not found, returning {0} frames'.format(frames))
            return int(frames), rawShaped

    frames = data.size / (rows * cols)
    sizeCheck = frames * rows * cols

    if sizeCheck == data.size:
        rawShaped = data.reshape(int(frames), int(rows) ,int(cols))
        #print('  Returning {0} frames of size {1} x {2} and data type {3} '.format(  \
        #rawShaped.shape[0],rawShaped.shape[1],rawShaped.shape[2],rawShaped.dtype))
    else:
        #print('  Calculated size = {0}, actual size = {1}, returning  {3} frames '.format(sizeCheck,data.size,frames) )
        pass

    return int(frames), rawShaped


################################################################
##
def epsLaTexFigure(filename, epsname, caption, scale=None, vscale=None, filemode='a', strPost=''):
    """ Write the LaTeX code to include an eps graphic as a latex figure.
        The text is added to an existing file.


    Args:
        | fname (string):  text writing output path and filename.
        | epsname (string): filename/path to eps file (relative to where the LaTeX document is built).
        | caption (string): figure caption
        | scale (double): figure scale to textwidth [0..1]
        | vscale (double): figure scale to textheight [0..1]
        | filemode (string): file open mode (a=append, w=new file) (optional)
        | strPost (string): string to write to file after latex figure block (optional)

    Returns:
        | None, writes a file to disk

    Raises:
        | No exception is raised.
    """

    with open(filename, filemode) as outfile:
        outfile.write('\\begin{figure}[tb]\n')
        outfile.write('\\centering\n')
        if scale is not None or vscale is not None:
            scale = 1 if scale is None else scale
            vscale = 1 if vscale is None else vscale
            outfile.write('\\includegraphics[width={0}\\textwidth,height={1}\\textheight,keepaspectratio]{{{2}}}\n'.\
                format(scale, vscale, epsname))
        else:
            outfile.write('\\includegraphics{{{0}}}\n'.format(epsname))
        outfile.write('\\caption{{{0}. \label{{fig:{1}}}}}\n'.format(caption,epsname))
        outfile.write('\\end{figure}\n')
        outfile.write('{}\n'.format(strPost))
        outfile.write('\n')

################################################################
##
def arrayToLaTex(filename, arr, header=None, leftCol=None,formatstring='%1.4e', filemode='wt'):
    """ Write a numpy array to latex table format in output file.

        The table can contain only the array data (no top header or
        left column side-header), or you can add either or both of the
        top row or side column headers. Leave 'header' or 'leftcol' as
        None is you don't want these.

        The output format of the array data can be specified, i.e.
        scientific notation or fixed decimal point.

    Args:
        | fname (string): text writing output path and filename
        | arr (np.array[N,M]): array with table data
        | header (string): column header in final latex format (optional)
        | leftCol ([string]): left column each row, in final latex format (optional)
        | formatstring (string): output format precision for array data (see np.savetxt) (optional)
        | filemode (string): file open mode (a=append, w=new file) (optional)

    Returns:
        | None, writes a file to disk

    Raises:
        | No exception is raised.
    """

    #is seems that savetxt does not like unicode strings
    formatstring = formatstring.encode('ascii')

    if leftCol is None:
        numcols = arr.shape[1]
    else:
        numcols = arr.shape[1] + 1

    file=open(filename, filemode)
    file.write('\\begin{{tabular}}{{ {0} }}\n\hline\n'.format('|'+ numcols*'c|'))

    #write the header
    if header is not None:
        # first column for header
        if leftCol is not None:
            file.write('{0} & '.format(leftCol[0]))
        #rest of the header
        file.write('{0}\\\\\hline\n'.format(header))

    #write the array data
    if leftCol is None:
        #then write the array, using the file handle (and not filename)
        np.savetxt(file, arr, fmt=formatstring,  delimiter='&',newline='\\\\\n')
    else:
        # first write left col for each row, then array data for that row
        for i,entry in enumerate(leftCol[1:]):
            file.write(entry+'&')
            np.savetxt(file, arr[i].reshape(1,-1), fmt=formatstring, delimiter='&',newline='\\\\\n')

    file.write('\hline\n\end{tabular}\n\n')
    file.close()


################################################################
##
def read2DLookupTable(filename):
    """ Read a 2D lookup table and extract the data.

        The table has the following format: ::

            line 1: xlabel ylabel title
            line 2: 0 (vector of y (col) abscissa)
            lines 3 and following: (element of x (row) abscissa), followed
            by table data.

        From line/row 3 onwards the first element is the x abscissa value
        followed by the row of data, one point for each y abscissa value.
        
        The file format can depicted as follows: ::

            x-name y-name ordinates-name
            0 y1 y2 y3 y4
            x1 v11 v12 v13 v14
            x2 v21 v22 v23 v24
            x3 v31 v32 v33 v34
            x4 v41 v42 v43 v44
            x5 v51 v52 v53 v54
            x6 v61 v62 v63 v64

        This function reads the file and returns the individual data items.

    Args:
        | fname (string): input path and filename

    Returns:
        | xVec ((np.array[N])): x abscissae
        | yVec ((np.array[M])): y abscissae
        | data ((np.array[N,M])): data corresponding the x,y
        | xlabel (string): x abscissa label
        | ylabel (string): y abscissa label
        | title (string): dataset title

    Raises:
        | No exception is raised.
    """
    import numpy  as np

    with open(filename,'r') as f:
        lines = f.readlines()
        xlabel, ylabel, title = lines[0].split()
    aArray = np.loadtxt(filename, skiprows=1, dtype=float)
    xVec = aArray[1:, 0]
    yVec = aArray[0, 1:] 
    data = aArray[1:, 1:]
    return(xVec, yVec, data, xlabel, ylabel, title)

######################################################################################
def open_HDF(filename):
    """Open and return an HDF5 file with the given filename.

    See https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md
    for more information on using HDF5 as a data structure.

    Args:
        | filename (string): name of the file to be opened

    Returns:
        | HDF5 file.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    f = h5py.File(filename)
    return f


######################################################################################
def erase_create_HDF(filename):
    """Create and return a new HDS5 file with the given filename, erase the file if existing.

    See https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md
    for more information on using HDF5 as a data structure.

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
    f = h5py.File(filename)
    return f

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
def plotHDF5Bitmaps(hfd5f, prefix, format='png', lstimgs=None):
    """Plot arrays in the HFD5 as scaled bitmap images.

    See https://github.com/NelisW/pyradi/blob/master/pyradi/hdf5-as-data-format.md
    for more information on using HDF5 as a data structure.

    Retain zero in the array as black in the image, only scale the max value to 255

    Args:
        | hfd5f (H5py file): the file to be opened
        | prefix (string): prefix to be prepended to filename
        | format (string): type of file to be created png/jpeg
        | lstimgs ([string]): list of paths to image in the HFD5 file

    Returns:
        | Nothing.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    from . import ryplot

    for lstimg in lstimgs:
        arr = hfd5f['{}'.format(lstimg)].value
        if np.max(arr) != 0.:
            arr = 255 * arr/np.max(arr)
            imsave('{}-{}.{}'.format(prefix,lstimg.replace('/','-'),format), arr.astype(np.uint8))


######################################################################################
def plotHDF5Images(hfd5f, prefix, colormap=mcm.jet, cbarshow=True, lstimgs=None, logscale=False):
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
        arr = hfd5f['{}'.format(lstimg)].value
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
        arr = hfd5f['{}'.format(lstimg)].value
        his, bin = np.histogram(arr,bins=bins)
        filename = '{}-hist-plot-{}.{}'.format(prefix,lstimg.replace('/','-'),format)
        with ryplot.savePlot(1,1,1,figsize=(8,4), saveName=[filename]) as p:
            p.plot(1, (bin[1:]+bin[:-1])/2, his, '{}, {} bins'.format(lstimg, bins), 'Magnitude','Counts / bin')


################################################################
################################################################
##
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    from . import ryplot
    from . import ryutils

    # read two-dimensional lookup table
    xVec,yVec,data,xlabel, ylabel, title = read2DLookupTable('data/OTBMLSNavMar15Nov4_10-C1E.txt')
  
    p = ryplot.Plotter(1)
    for azim in [0,18,36]:
        p.plot(1,yVec,data[azim,:],xlabel='Zenith [rad]',ylabel='Irradiance [W/m$^2$]',
            ptitle='3-5 {}m, Altitude 10 m'.format(ryutils.upMu(False)),
            label=['Azim={0:.0f} deg'.format(yVec[azim])])
    p.saveFig('OTBMLSNavMar15Nov4_10-C1E.png')

    print ('Test writing latex format arrays:')
    arr = np.asarray([[1.0,2,3],[4,5,6],[7,8,9]])
    arrayToLaTex('array.txt', arr)
    arrayToLaTex('array.txt', arr, formatstring='%.1f')
    headeronly = 'Col1 & Col2 & Col3'
    arrayToLaTex('array.txt', arr, headeronly, formatstring='%.3f')
    header = 'Col 1 & Col 2 & Col 3'
    leftcol = ['XX','Row 1','Row 2','Row 3']
    #with \usepackage{siunitx} you can even do this:
    arrayToLaTex('array.txt', arr, header, leftcol, formatstring=r'\num{%.6e}')

    print ('Test writing eps file figure latex fragments:')
    epsLaTexFigure('eps.txt', 'picture.eps', 'This is the caption', 0.75)

    print ('Test writing and reading numpy array to text file, with header:')
    #create a two-dimensional array of 25 rows and 7 columns as an outer product
    twodA=np.outer(np.arange(0, 5, .2),np.arange(1, 8))
    #write this out as a test file
    filename='ryfilestesttempfile.txt'
    saveHeaderArrayTextFile(filename,twodA, header="line 1 header\nline 2 header", \
                       delimiter=' ', comment='%')

    #create a new range to be used for interpolation
    tim=np.arange(1, 3, .3).reshape(-1, 1)
    #read the test file and interpolate the selected columns on the new range tim
    # the comment parameter is superfluous, since there are no comments in this file

    print(loadColumnTextFile(filename, [0,  1,  2,  4],abscissaOut=tim,  comment='%').shape)
    print(loadColumnTextFile(filename, [0,  1,  2,  4],abscissaOut=tim,  comment='%'))
    os.remove(filename)

    ##------------------------- samples ----------------------------------------
    # read space separated file containing wavelength in um, then samples.
    # select the samples to be read in and then load all in one call!
    # first line in file contains labels for columns.
    wavelength=np.linspace(0.38, 0.72, 350).reshape(-1, 1)
    samplesSelect = [1,2,3,8,10,11]
    samples = loadColumnTextFile('data/colourcoordinates/samples.txt', abscissaOut=wavelength, \
                loadCol=samplesSelect,  comment='%')
    samplesTxt=loadHeaderTextFile('data/colourcoordinates/samples.txt',\
                loadCol=samplesSelect, comment='%')
    #print(samples)
    print(samplesTxt)
    print(samples.shape)
    print(wavelength.shape)

    ##------------------------- plot sample spectra ------------------------------
    smpleplt = ryplot.Plotter(1, 1, 1)
    smpleplt.plot(1, wavelength, samples, "Sample reflectance", r'Wavelength $\mu$m',
                r'Reflectance', 
                ['r-', 'g-', 'y-','g--', 'b-', 'm-'],label=samplesTxt,legendAlpha=0.5)
    smpleplt.saveFig('SampleReflectance'+'.png')

    ##===================================================
    print ('\nTest CleanFilename function:')
    inString="aa bb%cc:dd/ee,ff.gg\\hh[ii]jj"
    print('{0}\n{1}'.format(inString,cleanFilename(inString) ))
    inString="aa bb%cc:dd/ee,ff.gg\\hh[ii]jj"
    print('{0}\n{1}'.format(inString,cleanFilename(inString, "") ))

    print ('\nTest listFiles function:')
    print(listFiles('./', patterns='*.py', recurse=1, return_folders=1))

    ##------------------------- load frames from binary & show ---------------------------
    import matplotlib.pyplot as plt

    imagefile = 'data/sample.ulong'
    rows = 100
    cols = 100
    vartype = np.uint32
    framesToLoad =  [1, 3, 5, 7]
    frames, img = readRawFrames(imagefile, rows, cols, vartype, framesToLoad)

    if frames == len(framesToLoad):

        #first plot using ryplot, using matplotlib
        P = ryplot.Plotter(1, 2, 2,'Sample frames from binary file', figsize=(4, 4))
        P.showImage(1, img[0], 'frame {0}'.format(framesToLoad[0]))
        P.showImage(2, img[1], 'frame {0}'.format(framesToLoad[1]), cmap=plt.cm.autumn)
        P.showImage(3, img[2], 'frame {0}'.format(framesToLoad[2]), cmap=plt.cm. bone)
        P.showImage(4, img[3], 'frame {0}'.format(framesToLoad[3]), cmap=plt.cm.gist_rainbow)
        P.getPlot().show()
        P.saveFig('sample.png', dpi=300)
        print('\n{0} frames of size {1} x {2} and data type {3} read from binary file {4}'.format(  \
        img.shape[0],img.shape[1],img.shape[2],img.dtype, imagefile))

        #now write the raw frames to image files
        type = ['png','png','png','png']
        for i in range(frames):
            print(i)
            filename = 'rawIm{0}.{1}'.format(i,type[i])
            rawFrameToImageFile(img[i],filename)

    else:
        print('\nNot all frames read from file')

    #######################################################################
    print("Test the glob version of listFiles")
    filelist = listFiles('.', patterns=r"ry*.py", recurse=0, return_folders=0)
    for filename in filelist:
        print('  {0}'.format(filename))

    print("Test the regex version of listFiles")
    filelist = listFiles('.', patterns=r"[a-z]*p[a-z]*\.py[c]*", \
        recurse=0, return_folders=0, useRegex=True)
    for filename in filelist:
        print('  {0}'.format(filename))


    #######################################################################
    print("Test downloading a file from internet given a URL")
    url = 'https://pyradi.googlecode.com/svn/trunk/pyradi/' + \
          'data/colourcoordinates/samplesVis.txt'
    if downloadFileUrl(url) is not None:
       print('success')
    else:
       print('download failed')

    #######################################################################
    print("Test unzipping a gzip file, then untar the file")
    if unzipGZipfile('./data/colourcoordinates/colourcoordinates.tgz','tar') is not None:
        print('success')
    else:
        print('unzip failed')

    print("Test untarring a tar file")
    result = untarTarfile('tar','.')
    if result is not None:
        print(result)
    else:
        print('untarTarfile failed')


    tgzFilename = 'colourcoordinates.tgz'
    destinationDir = '.'
    tarFilename = 'colourcoordinates.tar'
    url = 'https://pyradi.googlecode.com/svn/trunk/pyradi/' + \
                                        'data/colourcoordinates/'
    names = downloadUntar(tgzFilename, url, destinationDir, tarFilename)
    if names:
        print('Files downloaded and untarred {}!'.format(tgzFilename))
        print(names)
    else:
        print('Failed! unable to downloaded and untar {}'.format(tgzFilename))


    #######################################################################
    print('\nmodule ryfiles done!')
