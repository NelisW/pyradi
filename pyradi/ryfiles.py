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

__version__ = "$Revision$"
__author__ = 'pyradi team'
__all__ = [
    'saveHeaderArrayTextFile', 'loadColumnTextFile', 'loadHeaderTextFile',
    'cleanFilename', 'listFiles', 'readRawFrames', 'writeRawFrames',
    'rawFrameToImageFile',
    'arrayToLaTex', 'epsLaTexFigure', 'execOnFiles',
    'read2DLookupTable',
    'downloadFileUrl', 'unzipGZipfile', 'untarTarfile', 'downloadUntar',
    'mergeDFS', 'latex_escape',
]

import errno
import fnmatch
import gzip
import io
import os
import os.path
import re
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from numbers import Number

import numpy as np


################################################################
def latex_escape(str_):
    """Escape special LaTeX characters in a string.

    Args:
        | str_ (string): input string potentially containing LaTeX special chars.

    Returns:
        | (string): string with special characters replaced by LaTeX escape sequences.

    Raises:
        | No exception is raised.
    """
    CHARS = {
        '&':  r'\&',
        '%':  r'\%',
        '$':  r'\$',
        '#':  r'\#',
        '_':  r'\_',
        '{':  r'\{',
        '}':  r'\}',
        '~':  r'\textasciitilde{}',
        '^':  r'\^{}',
        '\\': r'\textbackslash{}',
        '>':  r'\textgreater{}',
        '<':  r'\textless{}',
    }
    return u"".join([CHARS.get(c, c) for c in str_])


################################################################
def saveHeaderArrayTextFile(filename, dataArray, header=None,
        comment=None, delimiter=None):
    """Save a numpy array to a file, with optional header lines.

    This function saves a two-dimensional array to a text file, with
    an optional user-defined header.

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
    with open(filename, 'wb') as f:
        if header is not None:
            for line in header.split('\n'):
                f.write((comment + line + '\n').encode('utf-8'))
        # np.savetxt does not accept delimiter=None in numpy 2.x; use ' ' as the default.
        np.savetxt(f, dataArray, delimiter=delimiter if delimiter is not None else ' ')


################################################################
def loadColumnTextFile(filename, loadCol=[1],
        comment=None, normalize=0, skiprows=0, delimiter=None,
        abscissaScale=1, ordinateScale=1, abscissaOut=None, returnAbscissa=False):
    """Load selected column data from a text file, processing as specified.

    This function loads column data from a text file, scaling and interpolating
    the read-in data, according to user specification. The first 0'th column has
    special significance: it is considered the abscissa (x-values) of the data
    set, while the remaining columns are any number of ordinate (y-value) vectors.
    The user passes a list of columns to be read (default is [1]) - only these
    columns are read, processed and returned when the function exits. The user
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
        | abscissaOut (np.array[N,M]): The abscissa where the ordinates are interpolated

    Raises:
        | No exception is raised.
    """
    # Prepend the 0'th column to the rest of the list; make a local copy first.
    ldCol = loadCol[:]
    ldCol.insert(0, 0)

    coldata = np.loadtxt(filename, usecols=ldCol,
            comments=comment, skiprows=skiprows,
            delimiter=delimiter)

    abscissa = abscissaScale * coldata[:, 0]
    ordinate = ordinateScale * coldata[:, 1:]

    if abscissaOut is not None:
        from scipy.interpolate import interp1d
        abscissaOut = abscissaOut.reshape(-1,)
        f = interp1d(abscissa, ordinate, axis=0)
        interpValue = f(abscissaOut)
    else:
        interpValue = ordinate
        abscissaOut = abscissa

    if normalize != 0:
        interpValue /= np.max(interpValue, axis=0)

    if returnAbscissa:
        return interpValue, abscissaOut.reshape(-1, 1)
    else:
        return interpValue


################################################################################
def loadHeaderTextFile(filename, loadCol=[1], comment=None):
    """Load column header data from the first line of a text file.

    Headers must be delimited by commas. The function loadColumnTextFile provides
    more comprehensive capabilities.

    Args:
        | filename (string): the name of the input ASCII flatfile.
        | loadCol ([int]): list of numbers, the column headers to be loaded, default value is column 1
        | comment (string): the symbol to comment out lines

    Returns:
        | [string]: a list with selected column header entries

    Raises:
        | No exception is raised.
    """
    if isinstance(filename, str):
        infile = open(filename, 'r')
    else:
        infile = filename

    line = infile.readline()
    strip_chars = ' ' + (comment or '')
    header_items = [x.strip() for x in line.lstrip(strip_chars).split(',')]
    return [header_items[i] for i in loadCol]


################################################################
def cleanFilename(sourcestring, removestring=" %:/,.\\[]<>*?"):
    """Clean a string by removing selected characters.

    Creates a legal and 'clean' source string from a string by removing some
    clutter and characters not allowed in filenames.
    A default set is given but the user can override the default string.

    Args:
        | sourcestring (string): the string to be cleaned.
        | removestring (string): remove all these characters from the string (optional).

    Returns:
        | (string): A cleaned-up string.

    Raises:
        | No exception is raised.
    """
    return ''.join([c for c in sourcestring if c not in removestring])


################################################################
def downloadUntar(tgzFilename, url, destinationDir=None, tarFilename=None, proxy=None):
    """Download and untar a compressed tar archive, saving all files to the specified directory.

    The tarfilename is used to open the tar file, extracting to the destinationDir specified.
    If no destinationDir is given, the local directory '.' is used.
    Before downloading, a check is done to determine if the file was already downloaded
    and exists in the local file system.

    Args:
        | tgzFilename (string): the name of the tar archive file
        | url (string): url where to look for the file (not including the filename)
        | destinationDir (string): to where the files must be extracted (optional)
        | tarFilename (string): downloaded tar filename (optional)
        | proxy (string): path to proxy server (optional).

        The proxy string is something like this
        proxy = {'https':r'https://username:password@proxyname:portnumber'}

    Returns:
        | ([string]): list of filenames saved, or None if failed.

    Raises:
        | Exceptions are handled internally and signaled by return value.
    """
    dirname = destinationDir if destinationDir is not None else '.'
    tarname = tarFilename if tarFilename is not None else tgzFilename + '.tar'

    tgzPath = os.path.join(dirname, tgzFilename)
    if os.path.isfile(tgzPath):
        tgzAvailable = True
    else:
        urlfile = url + tgzFilename
        if downloadFileUrl(url=urlfile, proxy=proxy) is None:
            print('\ndownload failed, please check url or internet connection')
            tgzAvailable = False
        else:
            tgzAvailable = True

    result = []
    if tgzAvailable:
        if unzipGZipfile(tgzPath, tarname) is None:
            print(f'Unzipping the tgz file {tgzPath} to dir {tarname} failed')
        else:
            result = untarTarfile(tarname, dirname)
            if result is None:
                print(f'untarTarfile failed for {tarname} to {dirname}')

    return result


################################################################
def untarTarfile(tarfilename, saveDirname=None):
    """Untar a tar archive, saving all files to the specified directory.

    The tarfilename is used to open a file, extracting to the saveDirname specified.
    If no saveDirname is given, the local directory '.' is used.

    Args:
        | tarfilename (string): the name of the tar archive.
        | saveDirname (string): to where the files must be extracted

    Returns:
        | ([string]): list of filenames saved, or None if failed.

    Raises:
        | Exceptions are handled internally and signaled by return value.
    """
    dirname = saveDirname if saveDirname is not None else '.'

    try:
        os.makedirs(dirname)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            print(f'Unable to create directory {dirname}')
            return None

    with tarfile.open(tarfilename, 'r') as f:
        filenames = f.getnames()
        f.extractall(dirname, filter='data')

    return filenames


################################################################
def unzipGZipfile(zipfilename, saveFilename=None):
    """Unzip a file that was compressed using the gzip format.

    The zipfilename is used to open a file, saving to the saveFilename specified.
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
    filename = saveFilename if saveFilename is not None else os.path.basename(zipfilename)[:-4]

    try:
        with gzip.open(zipfilename, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                f_out.write(f_in.read())
    except Exception:
        print(f'Unzipping of {zipfilename} failed')
        return None

    return filename


################################################################
def downloadFileUrl(url, saveFilename=None, proxy=None):
    """Download a file, given a URL.

    The URL is used to download a file, to the saveFilename specified.
    If no saveFilename is given, the basename of the URL is used.
    Before downloading, first test to see if the file already exists.

    Args:
        | url (string): the url to be accessed.
        | saveFilename (string): path to where the file must be saved (optional).
        | proxy (string): path to proxy server (optional).

        The proxy string is something like this
        proxy = {'https':r'https://username:password@proxyname:portnumber'}

    Returns:
        | (string): Filename saved, or None if failed.

    Raises:
        | Exceptions are handled internally and signaled by return value.
    """
    filename = saveFilename if saveFilename is not None else os.path.basename(url)

    if os.path.exists(filename):
        return filename

    try:
        if proxy is not None:
            h_proxy = urllib.request.ProxyHandler(proxy)
            auth = urllib.request.HTTPBasicAuthHandler()
            opener = urllib.request.build_opener(h_proxy, auth, urllib.request.HTTPHandler)
            urllib.request.install_opener(opener)

        with urllib.request.urlopen(url) as response:
            with open(filename, 'wb') as f:
                f.write(response.read())
    except urllib.error.HTTPError as e:
        print(f'HTTP Error: {e.code} for {url}')
        return None
    except urllib.error.URLError as e:
        print(f'URL Error: {e.reason} for {url}')
        return None

    return filename


################################################################
def execOnFiles(cmdline, root, patterns='*', recurse=1, return_folders=0,
                useRegex=False, printTask=False):
    """Execute a program on a list of files/directories meeting specific requirements.

    Seek files recursively and then execute a program on those files.
    The program is defined as a command line string as would be typed on
    a terminal, except that a token '{0}' is given in place of the filename.
    During execution the token is replaced with the filename found in the recursive search::

        task = cmdline.format(filename)

    Example: cmdline = 'bmpp -l eps.object {0}'

    Args:
        | cmdline (str): string that defines the program to be executed
        | root (string): directory root from where the search must take place
        | patterns (string): glob/regex pattern for filename matching
        | recurse (int): flag to indicate if subdirectories must also be searched (optional)
        | return_folders (int): flag to indicate if folder names must also be returned (optional)
        | useRegex (bool): flag to indicate if patterns are regular expression strings (optional)
        | printTask (bool): flag to indicate if the commandline must be printed (optional)

    Returns:
        | A list with matching file/directory names

    Raises:
        | No exception is raised.
    """
    if cmdline is None:
        return

    filenames = listFiles(root, patterns, recurse, return_folders, useRegex)
    for filename in filenames:
        filename = os.path.join(*filename.split('\\'))
        task = cmdline.format(filename)
        if printTask:
            print(task)
        p = subprocess.Popen(task)
        while p.poll() is None:
            time.sleep(0.5)


################################################################
def listFiles(root, patterns='*', recurse=1, return_folders=0, useRegex=False):
    """List files/directories matching a specific pattern.

    Returns a list of file paths in a file system, searching a directory structure
    along the specified path, looking for files that match the glob pattern. If
    specified, the search will continue into sub-directories. The function supports
    a local or network reachable filesystem, but not URLs.

    Args:
        | root (string): directory root from where the search must take place
        | patterns (string): glob/regex pattern for filename matching. Multiple patterns
          may be present, each one separated by ;
        | recurse (int): flag to indicate if subdirectories must also be searched (optional)
        | return_folders (int): flag to indicate if folder names must also be returned (optional)
        | useRegex (bool): flag to indicate if patterns are regular expression strings (optional)

    Returns:
        | A list with matching file/directory names

    Raises:
        | No exception is raised.
    """
    pattern_list = patterns.split(';')
    filenames = []
    filertn = []

    for dirpath, dirnames, files in os.walk(root):
        if dirpath == root or recurse:
            for filen in files:
                filenames.append(os.path.abspath(os.path.join(dirpath, filen)))
            if return_folders:
                for dirn in dirnames:
                    filenames.append(os.path.abspath(os.path.join(dirpath, dirn)))

    for name in filenames:
        if return_folders or os.path.isfile(name):
            for pattern in pattern_list:
                # Both glob and regex are matched against the basename only,
                # so that results are consistent regardless of the CWD.
                basename = os.path.basename(name)
                if useRegex:
                    if re.compile(pattern).search(basename):
                        filertn.append(name)
                        break
                else:
                    if fnmatch.fnmatch(basename, pattern):
                        filertn.append(name)
                        break

    return filertn


################################################################
def QueryDelete(recurse, tdir, patn, instr=""):
    """Delete files matching a pattern, after prompting the user for confirmation.

    Args:
        | recurse (int): flag to indicate if the search must be recursive (0=no, 1=yes)
        | tdir (string): directory to search
        | patn (string): file pattern to match
        | instr (string): pre-supplied answer ('y' to skip the prompt)

    Returns:
        | Nothing.

    Raises:
        | No exception is raised.
    """
    thefiles = listFiles(tdir, patn, recurse)
    if len(thefiles) > 0:
        for filename in thefiles:
            print(filename)
        if len(instr) == 0:
            instr = input("Delete these files? (y/n)")
        if instr == 'y':
            for filename in thefiles:
                os.remove(filename)


################################################################
def rawFrameToImageFile(image, filename):
    """Write a single raw image frame to an image file.

    The file type is determined by the extension (e.g. png or jpg).
    The image is normalised to [0, 255] prior to writing.

    Args:
        | image (np.ndarray): two-dimensional array representing an image
        | filename (string): name of file to be written to, with extension

    Returns:
        | Nothing

    Raises:
        | No exception is raised.
    """
    import imageio
    imin = image.min()
    imax = image.max()
    image = 255 * (image.astype(np.float64) - imin) / (imax - imin)
    imageio.imwrite(filename, image.astype(np.uint8))


################################################################
def writeRawFrames(fname, img, vartype, writeFrames=[]):
    """Write selected multiple 2D frames from a 3D array to a raw data file.

    The array must be two-dimensional or three-dimensional.
    Frames increase over the first dimension of the 3D array.

    Args:
        | fname (string): filename
        | img (np.array(:,:,:) or np.array(:,:)): array to be written to disk
        | vartype (np.dtype): numpy data type of data to be written
        |                                      int8, int16, int32, int64
        |                                      uint8, uint16, uint32, uint64
        |                                      float16, float32, float64
        | writeFrames ([int]): optional list of frames to write, zero-based;
        |                      empty list (default) writes all frames

    Returns:
        | message (string): empty if successful, fail message otherwise

    Raises:
        | No exception is raised.
    """
    if len(img.shape) == 2:
        img = img[None, ...]
    elif len(img.shape) == 3:
        pass
    else:
        return f'Input array rank inappropriate: {img.shape}'

    if not writeFrames:
        writeFrames = range(0, img.shape[0])

    try:
        with open(fname, 'wb') as fout:
            img[writeFrames, :, :].astype(vartype).tofile(fout)
    except IOError:
        return 'Error when writing file'

    return None


################################################################
def readRawFrames(fname, rows, cols, vartype, loadFrames=[]):
    """Load multi-frame two-dimensional arrays from a raw data file of known data type.

    The file must consist of multiple frames, all with the same number of rows and columns.
    Frames of different data types can be read according to the user specification.

    Args:
        | fname (string): filename
        | rows (int): number of rows in each frame
        | cols (int): number of columns in each frame
        | vartype (np.dtype): numpy data type of data to be read
        |                                      int8, int16, int32, int64
        |                                      uint8, uint16, uint32, uint64
        |                                      float16, float32, float64
        | loadFrames ([int]): optional list of frames to load, zero-based;
        |                     empty list (default) loads all frames

    Returns:
        | frames (int): number of frames in the returned data set, 0 if error occurred
        | rawShaped (np.ndarray): vartype numpy array of dimensions (frames, rows, cols),
        |                         None if error occurred

    Raises:
        | Exception is raised if IOError
    """
    frames = 0
    rawShaped = None

    if not loadFrames:
        try:
            with open(fname, 'rb') as fin:
                data = np.fromfile(fin, vartype, -1)
        except IOError:
            return int(frames), rawShaped
    else:
        try:
            framesize = rows * cols
            lastframe = max(loadFrames)
            data = None

            with open(fname, 'rb') as fin:
                for frame in range(0, lastframe + 1):
                    dataframe = np.fromfile(fin, vartype, framesize)
                    if frame in loadFrames:
                        data = dataframe if data is None else np.concatenate((data, dataframe))
        except IOError:
            return int(frames), rawShaped

    frames = data.size / (rows * cols)
    sizeCheck = frames * rows * cols

    if sizeCheck == data.size:
        rawShaped = data.reshape(int(frames), int(rows), int(cols))

    return int(frames), rawShaped


################################################################
def epsLaTexFigure(filename, epsname, caption, scale=None, vscale=None,
                   filemode='a', strPost=''):
    """Write the LaTeX code to include an eps graphic as a latex figure.

    The text is appended to an existing file.

    Args:
        | filename (string): text writing output path and filename.
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
            outfile.write(
                f'\\includegraphics[width={scale}\\textwidth,height={vscale}\\textheight,'
                f'keepaspectratio]{{{epsname}}}\n'
            )
        else:
            outfile.write(f'\\includegraphics{{{epsname}}}\n')
        outfile.write(f'\\caption{{{caption}. \\label{{fig:{epsname}}}}}\n')
        outfile.write('\\end{figure}\n')
        outfile.write(f'{strPost}\n')
        outfile.write('\n')


################################################################
def arrayToLaTex(filename, arr, header=None, leftCol=None,
                 formatstring='%10.4e', filemode='w'):
    """Write a numpy array to latex table format in an output file.

    The table can contain only the array data (no top header or left column
    side-header), or you can add either or both of the top row or side column
    headers. Leave 'header' or 'leftCol' as None if you don't want these.

    The output format of the array data can be specified, i.e.
    scientific notation or fixed decimal point.

    Args:
        | filename (string): text writing output path and filename
        | arr (np.array[N,M]): array with table data
        | header (string): column header in final latex format (optional)
        | leftCol ([string]): left column each row, in final latex format (optional)
        | formatstring (string): output format precision for array data (see np.savetxt) (optional)
        | filemode (string): file open mode — 'w' (default, new file) or 'a' (append)

    Returns:
        | None, writes a file to disk

    Raises:
        | No exception is raised.
    """
    # np.savetxt always writes bytes internally, so we capture it via BytesIO
    # and decode to str — this way the outer file stays in clean text mode.
    def _savetxt_str(data):
        buf = io.BytesIO()
        np.savetxt(buf, data, fmt=formatstring, delimiter='&', newline='\\\\\n')
        return buf.getvalue().decode('utf-8')

    # Normalise filemode: strip any 'b' so we always open in text mode.
    filemode = filemode.replace('b', '') or 'w'

    numcols = arr.shape[1] if leftCol is None else arr.shape[1] + 1

    with open(filename, filemode) as f:
        f.write(f"\\begin{{tabular}}{{ {'|' + numcols * 'c|'} }}\n\\hline\n")

        if header is not None:
            if leftCol is not None:
                f.write(f'{leftCol[0]} & ')
            f.write(f'{header}\\\\\\hline\n')

        if leftCol is None:
            f.write(_savetxt_str(arr))
        else:
            for i, entry in enumerate(leftCol[1:]):
                f.write(entry + '&')
                f.write(_savetxt_str(arr[i].reshape(1, -1)))

        f.write('\\hline\n\\end{tabular}\n\n')


################################################################
def read2DLookupTable(filename):
    """Read a 2D lookup table and extract the data.

    The table has the following format::

        line 1: xlabel ylabel title
        line 2: 0 (vector of y (col) abscissa)
        lines 3 and following: (element of x (row) abscissa), followed by table data.

    The file format can be depicted as follows::

        x-name y-name ordinates-name
        0 y1 y2 y3 y4
        x1 v11 v12 v13 v14
        x2 v21 v22 v23 v24
        x3 v31 v32 v33 v34

    Args:
        | filename (string): input path and filename

    Returns:
        | xVec (np.array[N]): x abscissae
        | yVec (np.array[M]): y abscissae
        | data (np.array[N,M]): data corresponding to x, y
        | xlabel (string): x abscissa label
        | ylabel (string): y abscissa label
        | title (string): dataset title

    Raises:
        | No exception is raised.
    """
    with open(filename, 'r') as f:
        xlabel, ylabel, title = f.readlines()[0].split()
    aArray = np.loadtxt(filename, skiprows=1, dtype=float)
    xVec = aArray[1:, 0]
    yVec = aArray[0, 1:]
    data = aArray[1:, 1:]
    return xVec, yVec, data, xlabel, ylabel, title


################################################################
def mergeDFS(df1, df2, leftPre=None, rightPre=None, bounds_error=False,
             mergeOn=None):
    """Merge two pandas DataFrames on a common column, returning the merged DataFrame.

    By default the merging takes place on columns named 'time' or 'Time',
    but the merge column name can be specified in mergeOn.

    Args:
        | df1 (DataFrame): first dataframe to be merged
        | df2 (DataFrame): second dataframe to be merged
        | leftPre (string): prefix to prepend to df1 column names, except time
        | rightPre (string): prefix to prepend to df2 column names, except time
        | bounds_error (boolean): passed through to the interpolation function
        | mergeOn (string): if not merging on time, use this column name instead

    Returns:
        | (pd.DataFrame): A Pandas DataFrame with the merged data

    Raises:
        | No exception is raised.
    """
    import pandas as pd

    if mergeOn is not None:
        if 'time' in df1.columns or 'time' in df2.columns or \
                'Time' in df1.columns or 'Time' in df2.columns:
            print('The "time" columns may not already be present in a dataframe')
            return pd.DataFrame()
        df1 = df1.rename(columns={mergeOn: 'time'})
        df2 = df2.rename(columns={mergeOn: 'time'})

    df1 = df1.rename(columns={'Time': 'time'})
    df2 = df2.rename(columns={'Time': 'time'})

    def _add_prefix(df, prefix):
        return df.rename(columns={
            col: (prefix + '_' + col if 'time' not in col else col)
            for col in df.columns
        })

    if leftPre is not None:
        df1 = _add_prefix(df1, leftPre)
    if rightPre is not None:
        df2 = _add_prefix(df2, rightPre)

    merged = False

    # Case: both frames have a single row
    if df1.shape[0] == 1 and df2.shape[0] == 1:
        if df1['time'].iloc[0] == df2['time'].iloc[0]:
            df1 = df1.merge(df2, on='time')
        else:
            print(f'mergeDFS failed to merge two single-row frames on different times:\n{df1}\n{df2}')
            df1 = None
        merged = True

    # Case: equal number of rows with identical time values
    if not merged and df1.shape[0] == df2.shape[0]:
        if np.all(np.where(df1['time'] == df2['time'], True, False)):
            df1 = df1.merge(df2, on='time')
            merged = True

    # General case: interpolate the longer frame onto the shorter one's time axis
    if not merged:
        if (np.max(df1['time']) - np.min(df1['time'])) > \
                (np.max(df2['time']) - np.min(df2['time'])):
            df1, df2 = df2, df1

        from scipy.interpolate import interp1d
        for col in df2.columns:
            if isinstance(df2[col].iloc[0], Number):
                fint = interp1d(df2['time'], df2[col], bounds_error=bounds_error)
                df1[col] = fint(df1['time'])

    dfRtn = df1.copy() if df1 is not None else None
    if mergeOn is not None and dfRtn is not None:
        dfRtn = dfRtn.rename(columns={'time': mergeOn})

    return dfRtn


################################################################
################################################################

if __name__ == '__init__':
    pass

if __name__ == '__main__':
    import os as _os
    _clutter = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'clutter')
    _os.makedirs(_clutter, exist_ok=True)

    def _c(fname):
        """Redirect an output filename into the clutter folder."""
        return _os.path.join(_clutter, _os.path.basename(str(fname)))


    import ryplot
    import ryutils

    rit = ryutils.intify_tuple

    doAll = False

    if False:
        # This example requires the DKTools bmpp executable http://dktools.sourceforge.net/bmpp.html
        execOnFiles(cmdline='bmpp -l eps.object {0}', root='./ref/', patterns='*.png',
            recurse=1, return_folders=0, useRegex=False, printTask=True)

    if doAll:
        # Read two-dimensional lookup table
        xVec, yVec, data, xlabel, ylabel, title = read2DLookupTable('data/OTBMLSNavMar15Nov4_10-C1E.txt')

        p = ryplot.Plotter(1)
        for azim in [0, 18, 36]:
            p.plot(1, yVec, data[azim, :], xlabel='Zenith [rad]', ylabel='Irradiance [W/m$^2$]',
                ptitle=f'3-5 {ryutils.upMu(False)}m, Altitude 10 m',
                label=[f'Azim={yVec[azim]:.0f} deg'])
        p.saveFig(_c('OTBMLSNavMar15Nov4_10-C1E.png'))

        print('Test writing latex format arrays:')
        arr = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
        arrayToLaTex('array.txt', arr)
        arrayToLaTex('array.txt', arr, formatstring='%.1f', filemode='a')
        headeronly = 'Col1 & Col2 & Col3'
        arrayToLaTex('array.txt', arr, headeronly, formatstring='%.3f', filemode='a')
        header = 'Col 1 & Col 2 & Col 3'
        leftcol = ['XX', 'Row 1', 'Row 2', 'Row 3']
        arrayToLaTex('array.txt', arr, header, leftcol, formatstring=r'\num{%.6e}', filemode='a')

        print('Test writing eps file figure latex fragments:')
        epsLaTexFigure('eps.txt', 'picture.eps', 'This is the caption', 0.75)

        print('Test writing and reading numpy array to text file, with header:')
        twodA = np.outer(np.arange(0, 5, .2), np.arange(1, 8))
        filename = 'ryfilestesttempfile.txt'
        saveHeaderArrayTextFile(filename, twodA, header="line 1 header\nline 2 header",
                           delimiter=' ', comment='%')

        tim = np.arange(1, 3, .3).reshape(-1, 1)
        tabl = loadColumnTextFile(filename, [0, 1, 2, 4], abscissaOut=tim, comment='%')
        print(rit(tabl.shape))
        print(loadColumnTextFile(filename, [0, 1, 2, 4], abscissaOut=tim, comment='%'))
        os.remove(filename)

        wavelength = np.linspace(0.38, 0.72, 350).reshape(-1, 1)
        samplesSelect = [1, 2, 3, 8, 10, 11]
        samples = loadColumnTextFile('data/colourcoordinates/samples.txt', abscissaOut=wavelength,
                    loadCol=samplesSelect, comment='%')
        samplesTxt = loadHeaderTextFile('data/colourcoordinates/samples.txt',
                    loadCol=samplesSelect, comment='%')
        print(samplesTxt)
        print(rit(samples.shape))
        print(rit(wavelength.shape))

        smpleplt = ryplot.Plotter(1, 1, 1)
        smpleplt.plot(1, wavelength, samples, "Sample reflectance", r'Wavelength $\mu$m',
                    r'Reflectance',
                    ['r', 'g', 'y', 'k', 'b', 'm'], label=samplesTxt, legendAlpha=0.5)
        smpleplt.saveFig(_c('SampleReflectance.png'))

        print('\nTest cleanFilename function:')
        inString = "aa bb%cc:dd/ee,ff.gg\\hh[ii]jj"
        print(f'{inString}\n{cleanFilename(inString)}')
        print(f"{inString}\n{cleanFilename(inString, ' ')}")
        print(f"{inString}\n{cleanFilename(inString, '')}")

        print('\nTest listFiles function - only python files in current dir:')
        print(listFiles('./', patterns='*.py', recurse=0, return_folders=1))
        print('\nTest listFiles function - only python files in nested dirs:')
        print(listFiles('./', patterns='*.py', recurse=1, return_folders=1))

        import matplotlib.pyplot as plt

        imagefile = 'data/sample.ulong'
        rows = 100
        cols = 100
        vartype = np.uint32
        framesToLoad = [1, 3, 5, 7]
        frames, img = readRawFrames(imagefile, rows, cols, vartype, framesToLoad)

        if frames == len(framesToLoad):
            P = ryplot.Plotter(1, 2, 2, 'Sample frames from binary file', figsize=(4, 4))
            P.showImage(1, img[0], f'frame {framesToLoad[0]}')
            P.showImage(2, img[1], f'frame {framesToLoad[1]}', cmap=plt.cm.autumn)
            P.showImage(3, img[2], f'frame {framesToLoad[2]}', cmap=plt.cm.bone)
            P.showImage(4, img[3], f'frame {framesToLoad[3]}', cmap=plt.cm.gist_rainbow)
            P.getPlot().show()
            P.saveFig(_c('sample.png'), dpi=300)
            print(f'\n{img.shape[0]} frames of size {img.shape[1]} x {img.shape[2]} and data type {img.dtype} read from binary file {imagefile}')

            img_types = ['png', 'png', 'png', 'png']
            for i in range(frames):
                filename = f'rawIm{i}.{img_types[i]}'
                print(f'  saving image {i} file to {filename}')
                rawFrameToImageFile(img[i], filename)
        else:
            print('\nNot all frames read from file')

        imagefile = 'data/sample.ulong'
        rows = 100
        cols = 100
        vartype = np.uint32
        frames, img = readRawFrames(imagefile, rows, cols, vartype, loadFrames=[])

        writeRawFrames('sample_all.double', img, 'double', writeFrames=[])
        writeRawFrames('sample_first.double', img, 'double', writeFrames=[1, 6])
        img1 = np.squeeze(img[0, :, :])
        writeRawFrames('sample_only.double', img1, 'double')

        patrn = r'*.py'
        print(f"Test the glob version of listFiles: {patrn}")
        for filename in listFiles('.', patterns=patrn, recurse=0, return_folders=0):
            print(f'  {filename}')

        patrn = r'*.py;*.pyc'
        print(f"Test the glob version of listFiles with two patterns: {patrn}")
        for filename in listFiles('.', patterns=patrn, recurse=0, return_folders=0):
            print(f'  {filename}')

        patrn = r"^[a-z]{2}p[a-z]*\.py[c]*"
        print(f"Test the regex version of listFiles: {patrn}")
        for filename in listFiles('.', patterns=patrn, recurse=0, return_folders=0, useRegex=True):
            print(f'  {filename}')

        patrn = r"^[a-z]{2}p[a-z]*\.py;^[a-z]{2}p[a-z]*\.pyc"
        print(f"Test the regex version of listFiles with two patterns: {patrn}")
        for filename in listFiles('.', patterns=patrn, recurse=0, return_folders=0, useRegex=True):
            print(f'  {filename}')

        print("Test downloading a file from internet given a URL")
        url = ('https://raw.githubusercontent.com/NelisW/pyradi/master/pyradi/'
               'data/colourcoordinates/samplesVis.txt')
        if downloadFileUrl(url) is not None:
            print('success')
        else:
            print('download failed')

        print("Test unzipping a gzip file, then untar the file")
        if unzipGZipfile('./data/colourcoordinates/colourcoordinates.tgz', 'tar') is not None:
            print('success')
        else:
            print('unzip failed')

        print("Test untarring a tar file")
        result = untarTarfile('tar', '.')
        if result is not None:
            print(result)
        else:
            print('untarTarfile failed')

        tgzFilename = 'colourcoordinates.tgz'
        destinationDir = '.'
        tarFilename = 'colourcoordinates.tar'
        url = ('https://raw.githubusercontent.com/NelisW/pyradi/master/pyradi/'
               'data/colourcoordinates/')
        names = downloadUntar(tgzFilename, url, destinationDir, tarFilename)
        if names:
            print(f'Files downloaded and untarred {tgzFilename}!')
            print(names)
        else:
            print(f'Failed! unable to downloaded and untar {tgzFilename}')

    mystr = 'a&b%c$d#e_f{g}h~i^j\\k>l<m'
    print(latex_escape(mystr))

    print('\nmodule ryfiles done!')
