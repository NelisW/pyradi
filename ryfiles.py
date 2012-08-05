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
This module provides functions for file input/output. These are all wrapper functions, based on
existing functions in other Python classes.

See the __main__ function for examples of use.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__version__= "$Revision$"
__author__='CJ Willers'
__all__=['SaveHeaderArrayTextFile', 'LoadColumnTextFile', 'LoadHeaderTextFile', 'CleanFilename', 
         'listFiles']

from scipy.interpolate import interp1d
import numpy
import os.path, fnmatch
import csv
import ryplot

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
    numpy.savetxt(file, dataArray, delimiter=delimiter)
    #neatly close the file
    file.close()


################################################################
def loadColumnTextFile(filename, loadCol=[1],  \
        comment=None, normalize=0, skiprows=0, delimiter=None,\
        abscissaScale=1,ordinateScale=1, abscissaOut=None):
    """Load selected column data from a text file, processing as specified.
    
    This function loads column data from a text file, 
    manipulating the data read in. The individual vector data 
    must be given in columns in the file, with the 
    abscissa (x-value) in first column (col 0 in Python)
    and any number of ordinate (y-value) vectors in second and 
    later columns.
    
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
        
    Returns:
        | (np.array[N,M]): The interpolated, M columns of N rows, processed array. 
        
    Raises:
        | No exception is raised.
    """

    #numpy.loadtxt(fname, dtype=<type 'float'>, comments='#', \
    #   delimiter=None, converters=None, skiprows=0, \
    #   usecols=None, unpack=False, ndmin=0)
    
    #load first column as well as user-specified column from the 
    # given file, scale as prescribed
    abscissa=abscissaScale*numpy.loadtxt(filename, usecols=[0],\
            comments=comment,  skiprows=skiprows, \
            delimiter=delimiter,  unpack=True)
    ordinate = ordinateScale*numpy.loadtxt(filename, \
            usecols=loadCol,comments=comment,skiprows=skiprows,\
            delimiter=delimiter, unpack=True)    
    
    if  abscissaOut is not None:
        #convert to [N, ] array
        abscissaOut=abscissaOut[:,0]
        #inpterpolate read values with the given inut vec
        f=interp1d(abscissa,  ordinate)
        interpValue=f(abscissaOut)
    else:
        interpValue=ordinate
    
    # read more than one column, get back into required shape.
    if interpValue.ndim > 2:
        interpValue = interpValue.squeeze().T

    #if read-in values must be normalised.
    if normalize != 0:
        interpValue /= numpy.max(interpValue,axis=0)
    
    #return in a form similar to input
    return interpValue.reshape(len(loadCol),  -1 ).T


################################################################################
def loadHeaderTextFile(filename, loadCol=[1], comment=None):
    """Loads column data from a text file, using the csv package.
    
    Using the csv package, loads column header data from a file, from the firstrow. 
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

    with open(filename, 'rb') as infile:
        #read from CVS file, must be comma delimited
        lstHeader = csv.reader(infile,quoting=csv.QUOTE_ALL)
        #get rid of leading and trailing whitespace
        list=[x.strip() for x in lstHeader.next()]
        #select only those required
        rtnList =[list[i] for i in loadCol ]    
        infile.close()
        
    return rtnList


################################################################
def cleanFilename(sourcestring,  removestring =" %:/,.\\[]"):
    """Clean a string by removing selected characters.
    
    Creates a legal and 'clean' sourcestring from a string by removing some clutter and illegals. 
    A default set is given but the user can override the default string.

    Args:
        | sourcestring (string): the string to be cleaned.
        | removestring (string): remove all these characters from the source.
    
    Returns:
        | (string): A cleaned-up string.
        
    Raises:
        | No exception is raised.
    """
    #remove spaces,comma, ":.%/\[]"
    return filter(lambda c: c not in removestring, sourcestring)


################################################################

#lists the files in a directory and subdirectories
#this code comes from the Python Cookbook
def listFiles(root, patterns='*', recurse=1, return_folders=0):
    """Lists the files/directories meeting specific requirement
    
    Searches a directory structure along the specified path, looking
    for files that matches the glob pattern. If specified, the search will
    continue into sub-directories.  A list of matching names is returned.
     
    Args:
        | root (string): root directory from where the search must take place
        | patterns (string): glob pattern for filename matching
        | recurse (unt): should the search extend to subdirs of root?
        | return_folders (int): should foldernames also be returned?
        
    Returns:
        | A list with matching file/directory names
        
    Raises:
        | No exception is raised.
    """
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
                    if fnmatch.fnmatch(name, pattern):
                        arg.results.append(fullname)
                        break
        # Block recursion if recursion was disallowed
        if not arg.recurse: files[:]=[]
    os.path.walk(root, visit, arg)
    return arg.results




################################################################
################################################################
##
## 

if __name__ == '__init__':
    pass

if __name__ == '__main__':
   
    print ('Test writing and reading numpy array to text file, with header:')
    #create a two-dimensional array of 25 rows and 7 columns as an outer product
    twodA=numpy.outer(numpy.arange(0, 5, .2),numpy.arange(1, 8))
    #write this out as a test file
    filename='ryfilestesttempfile.txt'
    saveHeaderArrayTextFile(filename,twodA, header="line 1 header\nline 2 header", \
                       delimiter=' ', comment='%')
    
    #create a new range to be used for interpolation
    tim=numpy.arange(1, 3, .3).reshape(-1, 1)
    #read the test file and interpolate the selected columns on the new range tim
    # the comment parameter is superfluous, since there are no comments in this file
    
    print(loadColumnTextFile(filename, [0,  1,  2,  4],abscissaOut=tim,  comment='%').shape)
    print(loadColumnTextFile(filename, [0,  1,  2,  4],abscissaOut=tim,  comment='%'))
    os.remove(filename)

    ##------------------------- samples ----------------------------------------
    # read space separated file containing wavelength in um, then samples.
    # select the samples to be read in and then load all in one call!
    # first line in file contains labels for columns.
    wavelength=numpy.linspace(0.38, 0.72, 350).reshape(-1, 1)
    samplesSelect = [1,2,3,8,10,11]
    samples = loadColumnTextFile('data/samples.txt', abscissaOut=wavelength, \
                loadCol=samplesSelect,  comment='%')
    samplesTxt=loadHeaderTextFile('data/samples.txt',\
                loadCol=samplesSelect, comment='%')
    #print(samples)
    print(samplesTxt)
    print(samples.shape)
    print(wavelength.shape)

    ##------------------------- plot sample spectra ------------------------------
    smpleplt = ryplot.Plotter(1, 1, 1)
    smpleplt.plot(1, "Sample reflectance", r'Wavelength $\mu$m',\
                r'Reflectance', wavelength, samples, \
                ['r-', 'g-', 'y-','g--', 'b-', 'm-'],samplesTxt,0.5)
    smpleplt.saveFig('SampleReflectance'+'.png')

    ##===================================================
    print ('\nTest CleanFilename function:')
    inString="aa bb%cc:dd/ee,ff.gg\\hh[ii]jj"
    print('{0}\n{1}'.format(inString,cleanFilename(inString) ))
    inString="aa bb%cc:dd/ee,ff.gg\\hh[ii]jj"
    print('{0}\n{1}'.format(inString,cleanFilename(inString, "") ))
 
    print ('\nTest listFiles function:')
    print(listFiles('./', patterns='*.py', recurse=1, return_folders=1))
    


    
    print('module ryfiles done!')
