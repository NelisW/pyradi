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

# The Initial Developer of the Original Code is AE Mudau,
# Portions created by AE Mudau are Copyright (C) 2012
# All Rights Reserved.

# Contributor(s):  CJ Willers, PJ Smit.
################################################################
"""
This module provides MODTRAN tape7 file reading.
Future development may include a class to write tape5 files, but that is 
a distant target at present.

See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students 
and readers of the book Electro-Optical System Analysis and Design: A Radiometry 
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""


from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['fixHeaders', 'loadtape7','fixHeadersList','runModtranAndCopy','variationTape5File']

import re
import sys
import numpy as np
if sys.version_info[0] > 2:
    # from io import StringIO
    from io import BytesIO
    # from str import maketrans
else:
    from string import maketrans
    import StringIO


##############################################################################
##http://stackoverflow.com/questions/1324067/how-do-i-get-str-translate-to-work-with-unicode-strings
def fixHeaders(instr):
    """
    Modifies the column header string to be compatible with numpy column lookup.

    Args:
        | list columns (string): column name.


    Returns:
        | list columns (string): fixed column name.

    Raises:
        | No exception is raised.
    """

    # intab = "+-[]@"
    # outtab = "pmbba"

    # if isinstance(instr, str):
    #     translate_table = dict((ord(char), str(outtab)) for char in intab)
    # else:
    #     assert isinstance(instr, str)
    #     translate_table = maketrans(intab, outtab)
    # print(instr, type(translate_table),translate_table )
    # return instr.translate(translate_table)

    if sys.version_info[0] > 2:
        intab = "+--[]@"
        outtab = "pmmbba"

        # print(instr)
        # if isinstance(instr, unicode):
        #     translate_table = dict((ord(char), unicode(outtab)) for char in intab)
        # else:
        #     assert isinstance(instr, str)
        translate_table = str.maketrans(intab, outtab)
    else:
        intab = u"+-[]@"
        outtab = u"pmbba"

        # print(instr)
        if isinstance(instr, unicode):
            translate_table = dict((ord(char), unicode(outtab)) for char in intab)
        else:
            assert isinstance(instr, str)
            translate_table = maketrans(intab, outtab)

    rtnstring = instr.translate(translate_table)
    return rtnstring




##############################################################################
##
def fixHeadersList(headcol):
    """
    Modifies the column headers to be compatible with numpy column lookup.

    Args:
        | list columns ([string]): list of column names.


    Returns:
        | list columns ([string]): fixed list of column names.

    Raises:
        | No exception is raised.
    """
    headcol = [fixHeaders(strn) for strn in headcol]
    return headcol

##############################################################################
##
def loadtape7(filename, colspec = []):

    """
    Read the Modtran tape7 file. This function was tested with Modtran5 files.

    Args:
        | filename (string): name of the input ASCII flatfile.
        | colspec ([string]): list of column names required in the output the spectral transmittance data.

    Returns:
        | np.array: an array with the selected columns. Col[0] is the wavenumber.

    Raises:
        | No exception is raised.



    This function reads in the tape7 file from MODerate spectral resolution
    atmospheric TRANsmission (MODTRAN) code, that is used to model the
    propagation of the electromagnetic radiation through the atmosphere. tape7
    is a primary file that contains all the spectral results of the MODTRAN
    run. The header information in the tape7 file contains portions of the
    tape5 information that will be deleted. The header section in tape7 is
    followed by a list of spectral points with corresponding transmissions.
    Each column has a different component of the transmission or radiance. 
    For more detail, see the modtran documentation.

    The user selects the appropriate columns by listing the column names, as
    listed below.

    The format of the tape7 file changes for different IEMSCT values. For
    the most part the differences are hidden in the details.  
    The various column headers used in the tape7 file are as follows:

    IEMSCT = 0 has two column header lines.  Different versions of modtran
    has different numbers of columns. In order to select the column, you
    must concatenate the two column headers with an underscore in between. All
    columns are available with the following column names: ['FREQ_CM-1',
    'COMBIN_TRANS', 'H2O_TRANS', 'UMIX_TRANS', 'O3_TRANS', 'TRACE_TRANS',
    'N2_CONT', 'H2O_CONT', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'HNO3_TRANS',
    'AER+CLD_abTRNS', '-LOG_COMBIN', 'CO2_TRANS', 'CO_TRANS', 'CH4_TRANS',
    'N2O_TRANS', 'O2_TRANS', 'NH3_TRANS', 'NO_TRANS', 'NO2_TRANS',
    'SO2_TRANS', 'CLOUD_TRANS', 'CFC11_TRANS', 'CFC12_TRANS', 'CFC13_TRANS',
    'CFC14_TRANS', 'CFC22_TRANS', 'CFC113_TRANS', 'CFC114_TRANS',
    'CFC115_TRANS', 'CLONO2_TRANS', 'HNO4_TRANS', 'CHCL2F_TRANS',
    'CCL4_TRANS', 'N2O5_TRANS','H2-H2_TRANS','H2-HE_TRANS','H2-CH4_TRANS',
    'CH4-CH4_TRANS']

    IEMSCT = 1 has single line column headers. A number of columns have
    headers, but with no column numeric data.  In the following list the
    columns with header names ** are empty and hence not available: ['FREQ',
    'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS', *SOL_SCAT*,
    *SING_SCAT*, 'GRND_RFLT', *DRCT_RFLT*, 'TOTAL_RAD', *REF_SOL*, *SOL@OBS*,
    'DEPTH', 'DIR_EM', *TOA_SUN*, 'BBODY_T[K]']. Hence, these columns do not
    have valid data: ['SOL_SCAT', 'SING_SCAT', 'DRCT_RFLT', 'REF_SOL',
    'SOL@OBS', 'TOA_SUN']

    IEMSCT = 2 has single line column headers. All the columns are available:
    ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS', 'SOL_SCAT',
    'SING_SCAT', 'GRND_RFLT', 'DRCT_RFLT', 'TOTAL_RAD', 'REF_SOL', 'SOL@OBS',
    'DEPTH', 'DIR_EM', 'TOA_SUN', 'BBODY_T[K]']

    IEMSCT = 3 has single line column headers.  One of these seems to be two
    words, which, in this code must be concatenated with an underscore. There
    is also  additional column (assumed to be depth in this code).  The
    columns available are ['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH']

    The tape7.scn file has missing columns, so this function does not work for
    tape7.scn files.  If you need a tape7.scn file with all the columns populated
    you would have to use the regular tape7 file and convolve this to lower resolution.

    UPDATE:
    Different versions of Modtran have different columns present in the tape7 file, 
    also not all the columns are necessarily filled, some have headings but no data.
    To further complicate matters, the headings are not always separated by spaces,
    sometime (not often) heading text runs into each other with no separator.

    It seems that the column headings are right aligned with the column data itself,
    so if we locate the right most position of the column data, we can locate the
    headings with valid data - even in cases with connected headings.  

    The algorithm used is as follows:

    1. step down to a data line beyond the heading lines (i.e., the data)
    2. locate the position of the last character of every discrete colummn
    3. From the last column of the previous col, find start of next col
    4. move up into the header again and isolate the header column text for each column

    In sptite of all the attempts to isolate special cases, reading of tape7 files
    remain a challange and may fail in newer versions of Modtran, until fixed.
    """

    infile = open(filename, 'r')
    idata = {}
    colHead = []
    lines = infile.readlines()#.strip()
    infile.close()
    if len(lines) < 10:
        print(f'Error reading file {filename}: too few lines!')
        return None

    #determine values for MODEL, ITYPE, IEMSCT, IMULT from card 1
    #tape5 input format (presumably also tape7, line 1 format?)
    #format Lowtran7  (13I5, F8.3, F7.0) = (MODEL, ITYPE, IEMSCT, IMULT)
    #format Modtran 4 (2A1, I3, 12I5, F8.3, F7.0) = (MODTRN, SPEED, MODEL, ITYPE, IEMSCT, IMULT)
    #format Modtran 5 (3A1, I2, 12I5, F8.0, A7) = (MODTRN, SPEED, BINARY, MODEL, ITYPE, IEMSCT, IMULT)
    #MODEL = int(lines[0][4])
    #ITYPE = int(lines[0][9])
    IEMSCT = int(lines[0][14])
    #IMULT = int(lines[0][19])
    #print('filename={0}, IEMSCT={1}'.format(filename,IEMSCT))

    #skip the first few rows that contains tape5 information and leave the
    #header for the different components of transimissions.
    #find the end of the header.
    headline = 0
    while lines[headline].find('FREQ') < 0:
        headline = headline + 1

    #some files has only a single text column head, while others have two
    # find out what the case is for this file and concatenate if necessary
    line1 = lines[headline] # alway header 1
    line2 = lines[headline+1] # maybe data, maybe header 2
    line3 = lines[headline+2] # definately data

    #see if there is a second header line
    p = re.compile('[a-df-zA-DF-Z]+')
    line2found = True if p.search(line2) is not None else False

    #modtran 4 does not use underscores
    line1 = line1.replace('TOT TRANS','TOT_TRANS')
    line1 = line1.replace('PTH THRML','PTH_THRML')
    line1 = line1.replace('THRML SCT','THRML_SCT')
    line1 = line1.replace('SURF EMIS','SURF_EMIS')
    line1 = line1.replace('SOL SCAT','SOL_SCAT')
    line1 = line1.replace('SING SCAT','SING_SCAT')
    line1 = line1.replace('GRND RFLT','GRND_RFLT')
    line1 = line1.replace('DRCT RFLT','DRCT_RFLT')
    line1 = line1.replace('TOTAL RAD','TOTAL_RAD')
    line1 = line1.replace('REF SOL','REF_SOL')

    colcount = 0
    colEnd = []
    #strip newline from the data line
    linet = line3.rstrip()

    idx = 0
    while idx < len(linet):
        while linet[idx].isspace():
            idx += 1
            if idx == len(linet):
                break
        while not linet[idx].isspace():
            idx += 1
            if idx == len(linet):
                break
        colEnd.append(idx+1)

    colSrt = [0] + [v-1 for v in colEnd[:-1]]
    if sys.version_info[0] > 2:
        # http://www.diveintopython3.net/porting-code-to-python-3-with-2to3.html
        # zip returns an iterator, not a list
        collim = list(zip(colSrt,colEnd))
    else:
        collim = zip(colSrt,colEnd)

    # iemsct=3 has a completely messed up header, replace with this
    if IEMSCT == 3:
        colHead1st = ' '.join(['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH'])
    else:
        # each entry in collim defines the slicing start and end for each col, including leading whitepace
        # missing columns may have headers that came through as single long string, 
        # now remove by splitting on space and taking the last one
        colHead1st = [line1[lim[0]:lim[1]-1].split()[-1].strip() for lim in collim]
    colHead2nd = [line2[lim[0]:lim[1]-1].split()[-1].strip() for lim in collim]

    # if colHead2nd[0].find('CM') >= 0:
    if line2found:
        colHead = [h1+'_'+h2 for (h1,h2) in zip(colHead1st,colHead2nd)]
        deltaHead = 1
    else:
        colHead = colHead1st
        deltaHead = 0

    # print(colHead)

    #different IEMSCT values have different column formats
    # some cols have headers and some are empty.
    # for IEMSCT of 0 and 2 the column headers are correct and should work as is.
    #for IEMSCT of 1 the following columns are empty and must be deleted from the header

    if IEMSCT == 1:
        removeIEMSCT1 = ['SOL_SCAT', 'SING_SCAT', 'DRCT_RFLT', 'REF_SOL', 'SOL@OBS', 'TOA_SUN']
        colHead = [x for x in colHead if x not in removeIEMSCT1]

    if IEMSCT == 3:
        colHead = ['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH']

    # build a new data set with appropriate column header and numeric data
    #change all - and +  to alpha to enable table lookup
    colHead = fixHeadersList(colHead)

    s = ' '.join(colHead) + '\n'
    # now append the numeric data, ignore the original header and last row in the file
    s = s + ''.join(lines[headline+1+deltaHead:-1])

    #read the string in from a StringIO in-memory file
    # lines = np.ndfromtxt(io.StringIO(s), dtype=None,  names=True)
    if sys.version_info[0] > 2:
        # lines = np.ndfromtxt(StringIO(s), dtype=None,  names=True)
        lines = np.ndfromtxt(BytesIO(s.encode('utf-8')), dtype=None,  names=True)
    else:
        lines = np.ndfromtxt(StringIO.StringIO(s), dtype=None,  names=True)

    # print('lines=',lines)

    #extract the wavenumber col as the first column in the new table
    # print(lines)
    coldata= lines[fixHeaders(colspec[0])].reshape(-1, 1)
    # then append the other required columns
    for colname in colspec[1:]:
        coldata = np.hstack((coldata, lines[fixHeaders(colname)].reshape(-1, 1)))

    return coldata



##############################################################################
##
def variationTape5File(scenario, variation, tape5,varFunc,fmtstr, srcstr,outfile='tape5' ):
    """Read tape5 in specified folder, modify one or more variable, write out new tape5 in new folder

    One tape5 is written with as many changes as are present in the four lists:
        <variation, varFunc,fmtstr, srcstr> with corresponding values.
    
    If only one value is to be changed, the four values may be scalar (not lists).
    If the values are scalar, lists are formed before processing.

    Replace a search string <srcstr> (must be unique in the file) with the value 
    of a variation <variation> formatted  by <fmtstr> as required by the tape5 format. 
    The variation input value <variation> is processed by the function <varFunc>.
    All occurrences of the search string <srcstr> are blindly replaced, 
    and must occur only once in the file and be of the same width as the field
    to be replaced.

    The variation folder name is constructed by joining with '-' the Python str() 
    representation of the variables.

    The variation variable must be of the form that can be used to create sub-folders in
    the master scenario folder. The variation can be processed (scaled, etc.) by <varFunc>
    before writing out to file. For example, variation can be an integer in metre, 
    then scaled to kilometer. If varFunc is None, the value is used as is.

    The variation tape5 is written to a new sub folder with the name variation, constructed
    in the scenario folder.  The template tape5 file must be in the scenario root file.

    

    dir structure:
    |dir root
        | file domodtran.py
        | dir scenario 1
            |dir variation 1
            |dir variation 2
            |dir variation 3
            |file tape5 template for scenario 1
        | dir scenario 2
            |dir variation 1
            |dir variation 2
            |dir variation 3
            |file tape5 template for scenario 2
    The scenario directories must exist, but the variation scenarios are created.

    Args:
        | scenario (string): path to folder with the scenario tape5
        | variation ([int/float]): number that defines the variation, will be modified by varFunc
        | tape5 (string): tape5 filename, must be in the scenario folder
        | varFunc ([fn]): Function to process the variation number before writing, can be None
        | fmtstr ([string]): print format string to be used when writing to tape5
        | srcstr ([string]): the string in the tape5 file to be replaced
        | outfile (]string]): the name to be used when writing the modified tape5

    Returns:
        | side effect creates new folders and files.

    Raises:
        | No exception is raised.
    """

    import os

    # if not lists, make lists
    if not isinstance(variation, list): 
        variation = [variation]
    if not isinstance(varFunc, list): 
        varFunc = [varFunc]
    if not isinstance(fmtstr, list): 
        fmtstr = [fmtstr]
    if not isinstance(srcstr, list): 
        srcstr = [srcstr]

    # check lists same length
    if len(variation)!= len(varFunc):
        print('len(variation)!= len(varFunc)')
        return None
    if len(variation)!= len(fmtstr):
        print('len(variation)!= len(fmtstr)')
        return None
    if len(variation)!= len(srcstr):
        print('len(variation)!= len(srcstr)')
        return None

    #read the tape5 base file
    tape5base = os.path.join(scenario,tape5)
    with open(tape5base) as fin:
        lines = fin.readlines()

    varcomb = '-'.join([str(i) for i in variation])
    # print(tape5base, varcomb, scenario)

    #change the srcstr to new value with specfied format for all variations
    outlines = []
    for line in lines:
        for i,_ in enumerate(variation):
            # use varFunc if not None
            newVal = varFunc[i](variation[i])if varFunc[i] else variation[i]
            line = line.replace(srcstr[i],f'{newVal:{fmtstr[i]}}')
        outlines.append(line)

    #create new scenario and write file to new dir
    dirname = os.path.join('.',scenario,'{}'.format(varcomb))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(dirname, outfile)
    with open(filename,'w') as fout:
        # print('writing',filename)
        fout.writelines(outlines)

    return varcomb


##############################################################################
##
def runModtranAndCopy(root, research, pathToModtranBin,execname):
    """
    Look for input files in directories, run modtran and copy results to dir.

    Finds all files below the root directory that matches the 
    regex pattern in research. Then runs modtran on these files
    and write the tape5/6/7/8 back to where the input file was.

    Each input file must be in a separate directory, because the results are
    all written to files with the names 'tape5', etc.

    Args:
        | root ([string]): path to root dir containing the dirs with modtran input files.
        | research ([string]): regex to use when searching for input files ('.*.ltn' or 'tape5')
        | pathToModtranBin ([string]): path to modtran executable directory.
        | execname ([string]): modtran executable filename.

    Returns:
        | List of all the files processed.

    Raises:
        | No exception is raised.
    """
    import os
    import shutil
    import subprocess
    import time
    import pyradi.ryfiles as ryfiles

    filepaths = ryfiles.listFiles(root, patterns=research, recurse=1, return_folders=0, useRegex=True)

    # print('**********************',root,research,filepaths)
 
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        dirname = os.path.dirname(filepath)

        #get rid of clutter to make sure that our tape5 will be used
        for rfile in ['tape5','modin','modin.ontplt','mod5root.in','.ezl20ck']:
            rpath = os.path.join(pathToModtranBin, rfile)
            if os.path.exists(rpath):
                os.remove(rpath)

        #copy our tape5 across
        tape5path = os.path.join(pathToModtranBin, 'tape5')
        shutil.copy2(filepath, tape5path)

        #run modtran on the tape5 file in its bin directory
        if os.path.exists(tape5path):

            p = subprocess.Popen(os.path.join(pathToModtranBin, execname), 
                    shell=True, stdout=None, stderr=None, cwd=pathToModtranBin)
            while  p.poll() == None:
                time.sleep(0.5)

            # #copy the tape5/6/7 files back to appropriate directory
            for outname in ['tape5', 'tape6', 'tape7','tape8']:
                    outpath = os.path.join(pathToModtranBin,outname)
                    if os.path.exists(outpath):
                        shutil.copy2(outpath, dirname)


################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    import math
    import sys
    import numpy as np

    import pyradi.ryplot as ryplot
    import pyradi.ryutils as ryutils


    doAll = True
    
    if doAll:
        pathToModtranBin = r'C:\PcModWin5\bin'
        research = '.*.ltn'
        root = r'D:\work\ISP\SWIR-tradeoff\data\atmo\TropicalRural\slant'
        execname = 'OntarMod5_3_2.exe'
        runModtranAndCopy(root=root, research=research, pathToModtranBin=pathToModtranBin, execname=execname)

    if doAll:
        figtype = ".png"  # eps, jpg, png
        #figtype = ".eps"  # eps, jpg, png

        ## ----------------------- -----------------------------------------
        tape7= loadtape7("data/tape7-01", ['FREQ_CM-1', 'COMBIN_TRANS', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'AER+CLD_abTRNS'] )
        np.savetxt('tape7-01a.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-01b", ['FREQ_CM-1', 'COMBIN_TRANS', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'AER+CLD_abTRNS'] )
        np.savetxt('tape7-01b.txt', tape7, fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-01", ['FREQ_CM-1', 'COMBIN_TRANS', 'H2O_TRANS', 'UMIX_TRANS', 'O3_TRANS', 'TRACE_TRANS', 'N2_CONT', 'H2O_CONT', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'HNO3_TRANS', 'AER+CLD_abTRNS', '-LOG_COMBIN', 'CO2_TRANS', 'CO_TRANS', 'CH4_TRANS', 'N2O_TRANS', 'O2_TRANS', 'NH3_TRANS', 'NO_TRANS', 'NO2_TRANS', 'SO2_TRANS', 'CLOUD_TRANS', 'CFC11_TRANS', 'CFC12_TRANS', 'CFC13_TRANS', 'CFC14_TRANS', 'CFC22_TRANS', 'CFC113_TRANS', 'CFC114_TRANS', 'CFC115_TRANS', 'CLONO2_TRANS', 'HNO4_TRANS', 'CHCL2F_TRANS', 'CCL4_TRANS', 'N2O5_TRANS'] )
        np.savetxt('tape7-01.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-02", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS', 'GRND_RFLT', 'TOTAL_RAD', 'DEPTH', 'DIR_EM', 'BBODY_T[K]'] )
        np.savetxt('tape7-02.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-02b", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'GRND_RFLT', 'TOTAL_RAD', 'DEPTH', 'DIR_EM', 'BBODY_T[K]'] )
        np.savetxt('tape7-02.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-02c", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'GRND_RFLT', 'TOTAL_RAD', 'DEPTH', 'DIR_EM', 'BBODY_T[K]'] )
        np.savetxt('tape7-02.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-03", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS', 'SOL_SCAT', 'SING_SCAT', 'GRND_RFLT', 'DRCT_RFLT', 'TOTAL_RAD', 'REF_SOL', 'SOL@OBS', 'DEPTH', 'DIR_EM', 'TOA_SUN', 'BBODY_T[K]'] )
        np.savetxt('tape7-03.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-04", ['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH'] )
        np.savetxt('tape7-04.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-05", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'TOTAL_RAD'] )
        np.savetxt('tape7-05.txt', tape7,fmt=str('%.6e'))

        tape7= loadtape7("data/tape7-05b", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'SURF_EMIS', 'TOTAL_RAD'] )
        np.savetxt('tape7-05b.txt', tape7,fmt=str('%.6e'))


    if doAll:

        colSelect =  ['FREQ_CM-1', 'COMBIN_TRANS', 'H2O_TRANS', 'UMIX_TRANS', \
              'O3_TRANS', 'H2O_CONT', 'MOLEC_SCAT', 'AER+CLD_TRANS']
        tape7= loadtape7("data/tape7VISNIR5kmTrop23Vis", colSelect )
        wavelen = ryutils.convertSpectralDomain(tape7[:,0],  type='nl')
        mT = ryplot.Plotter(1, 1, 1,"Modtran Tropical, 23 km Visibility (Rural)"\
                           + ", 5 km Path Length",figsize=(12,6))
        mT.plot(1, wavelen, tape7[:,1:], "","Wavelength [$\mu$m]", "Transmittance",
               label=colSelect[1:],legendAlpha=0.5, pltaxis=[0.4,1, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.saveFig('ModtranPlot.png')
        #mT.saveFig('ModtranPlot.eps')

        # this example plots the individual transmittance components
        colSelect =  ['FREQ_CM-1', 'COMBIN_TRANS', 'MOLEC_SCAT', 'CO2_TRANS', 'H2O_TRANS', 'H2O_CONT', 'CH4_TRANS',\
           'O3_TRANS', 'O2_TRANS', 'N2O_TRANS', 'AER+CLD_TRANS', 'SO2_TRANS']
        tape7= loadtape7("data/horizon5kmtropical.fl7", colSelect )
        wavelen = ryutils.convertSpectralDomain(tape7[:,0],  type='nl')
        mT = ryplot.Plotter(1, 9, 1,"Modtran Tropical, 23 km Visibility (Rural)"\
                           + ", 5 km Path Length",figsize=(6,12))
        mT.semilogX(1, wavelen, tape7[:,1], '','', '',
               label=colSelect[1:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(2, wavelen, tape7[:,2], '','', '',
               label=colSelect[2:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(3, wavelen, tape7[:,10], '','', '',
               label=colSelect[10:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(4, wavelen, tape7[:,4] , '','', '',
               label=colSelect[4:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])

        mT.semilogX(5, wavelen, tape7[:,5] , '','', '',
               label=colSelect[5:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])

        mT.semilogX(6, wavelen, tape7[:,3]  , '','', '',
               label=colSelect[3:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(7, wavelen, tape7[:,6]  , '','', '',
               label=colSelect[6:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(8, wavelen, tape7[:,7] * tape7[:,8] , '','', '',
               label=colSelect[7:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
        mT.semilogX(9, wavelen, tape7[:,9]  , '','', '',
               label=colSelect[9:],legendAlpha=0.5, pltaxis=[0.2,15, 0, 1],
               maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])


        mT.saveFig('ModtranSpec.png')
        mT.saveFig('ModtranSpec.eps')

        # calculate the total path radiance over a spectral band
        #read the tape7 file with path radiance components, plot and integrate.
        colSelect =  ['FREQ', 'PTH_THRML','SOL_SCAT','SING_SCAT', 'TOTAL_RAD']
        skyrad= loadtape7("data/NIRscat.fl7", colSelect )
        sr = ryplot.Plotter(1, 4,1,"Path Radiance in NIR, Path to Space from 3 km",figsize=(12,8))
        # plot the components separately
        for i in [1,2,3,4]:
          Lpath = 1.0e4 * skyrad[:,i]
          sr.plot(i,  skyrad[:,0], Lpath, "","Wavenumber [cm$^{-1}$]", "L [W/(m$^2$.sr.cm$^{-1}$)]",
                 label=[colSelect[i][:]],legendAlpha=0.5, #pltaxis=[0.4,1, 0, 1],
                 maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])

          #convert from /cm^2 to /m2 and integrate using the wavenumber vector
          #normally you would multiply with a sensor spectral response before integration
          #this calculation is over the whole band, equally weighted.
          totinband = np.trapz(Lpath.reshape(-1, 1),skyrad[:,0], axis=0)[0]
          print('{0} sum is {1} [W/(m^2.sr)]'.format(colSelect[i][:],totinband))
        sr.saveFig('NIRPathradiance.png')
        print('Note that multiple scatter contributes significantly to the total path radiance')

        #repeat the same calculation, but this time do in wavelength domain
        colSelect =  ['FREQ', 'PTH_THRML','SOL_SCAT','SING_SCAT', 'TOTAL_RAD']
        skyrad= loadtape7("data/NIRscat.fl7", colSelect )
        sr = ryplot.Plotter(1, 4,1,"Path Radiance in NIR, Path to Space from 3 km",
                            figsize=(12,8))
        # plot the components separately
        for i in [1,2,3,4]:
          wl, Lpath = ryutils.convertSpectralDensity(skyrad[:,0], skyrad[:,i],'nl')
          Lpath *= 1.0e4
          sr.plot(i,  wl, Lpath, "","Wavelength [$\mu$m]","L [W/(m$^2$.sr.$\mu$m)]",
                 label=[colSelect[i][:]],legendAlpha=0.5, #pltaxis=[0.4,1, 0, 1],
                 maxNX=10, maxNY=4, powerLimits = [-4,  4, -5, 5])
          totinband = - np.trapz(Lpath.reshape(-1, 1),wl,axis=0)[0]
          print('{0} integral is {1:.6e} [W/(m^2.sr)]'.format(colSelect[i][:],totinband))

        sr.saveFig('NIRPathradiancewl.png')
        print('Note that multiple scatter contributes significantly to total path radiance')


    print('\n\nrymodtran is done!')
