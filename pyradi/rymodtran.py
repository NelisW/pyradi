# -*- coding: utf-8 -*-


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

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

__version__= "$Revision$"
__author__= 'pyradi team'
__all__= ['fixHeaders', 'loadtape7','fixHeadersList']

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)


import numpy as np
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

    intab = u"+-[]@"
    outtab = u"pmbba"

    if isinstance(instr, unicode):
        translate_table = dict((ord(char), unicode(outtab)) for char in intab)
    else:
        assert isinstance(instr, str)
        translate_table = maketrans(intab, outtab)
    return instr.translate(translate_table)


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

    headcol = [fixHeaders(str) for str in headcol]
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
        | numpy.array: an array with the selected columns. Col[0] is the wavenumber.

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

    IEMSCT = 0 has two column header lines.  In order to select the column, you
    must concatenate the two column headers with an underscore in between. All
    columns are available with the following column names: ['FREQ_CM-1',
    'COMBIN_TRANS', 'H2O_TRANS', 'UMIX_TRANS', 'O3_TRANS', 'TRACE_TRANS',
    'N2_CONT', 'H2O_CONT', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'HNO3_TRANS',
    'AER+CLD_abTRNS', '-LOG_COMBIN', 'CO2_TRANS', 'CO_TRANS', 'CH4_TRANS',
    'N2O_TRANS', 'O2_TRANS', 'NH3_TRANS', 'NO_TRANS', 'NO2_TRANS',
    'SO2_TRANS', 'CLOUD_TRANS', 'CFC11_TRANS', 'CFC12_TRANS', 'CFC13_TRANS',
    'CFC14_TRANS', 'CFC22_TRANS', 'CFC113_TRANS', 'CFC114_TRANS',
    'CFC115_TRANS', 'CLONO2_TRANS', 'HNO4_TRANS', 'CHCL2F_TRANS',
    'CCL4_TRANS', 'N2O5_TRANS']

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

    """

    infile = open(filename, 'r')
    idata = {}
    lines = infile.readlines()#.strip()
    infile.close()

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
    colHead1st = lines[headline].split()
    colHead2nd = lines[headline+1].split()
    if colHead2nd[0].find('CM') >= 0:
        colHead = [h1+'_'+h2 for (h1,h2) in zip(colHead1st,colHead2nd)]
        deltaHead = 1
    else:
        colHead = colHead1st
        deltaHead = 0

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
    lines = np.ndfromtxt(StringIO.StringIO(s), dtype=None,  names=True)

    #extract the wavenumber col as the first column in the new table
    coldata= lines[fixHeaders(colspec[0])].reshape(-1, 1)
    # then append the other required columns
    for colname in colspec[1:]:
        coldata = np.hstack((coldata, lines[fixHeaders(colname)].reshape(-1, 1)))

    return coldata


################################################################
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    import math
    import sys
    import numpy

    import pyradi.ryplot as ryplot
    import pyradi.ryutils as ryutils

    figtype = ".png"  # eps, jpg, png
    #figtype = ".eps"  # eps, jpg, png

    ## ----------------------- -----------------------------------------
    tape7= loadtape7("data/tape7-01", ['FREQ_CM-1', 'COMBIN_TRANS', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'AER+CLD_abTRNS'] )
    np.savetxt('tape7-01a.txt', tape7,fmt='%.6e')

    tape7= loadtape7("data/tape7-01", ['FREQ_CM-1', 'COMBIN_TRANS', 'H2O_TRANS', 'UMIX_TRANS', 'O3_TRANS', 'TRACE_TRANS', 'N2_CONT', 'H2O_CONT', 'MOLEC_SCAT', 'AER+CLD_TRANS', 'HNO3_TRANS', 'AER+CLD_abTRNS', '-LOG_COMBIN', 'CO2_TRANS', 'CO_TRANS', 'CH4_TRANS', 'N2O_TRANS', 'O2_TRANS', 'NH3_TRANS', 'NO_TRANS', 'NO2_TRANS', 'SO2_TRANS', 'CLOUD_TRANS', 'CFC11_TRANS', 'CFC12_TRANS', 'CFC13_TRANS', 'CFC14_TRANS', 'CFC22_TRANS', 'CFC113_TRANS', 'CFC114_TRANS', 'CFC115_TRANS', 'CLONO2_TRANS', 'HNO4_TRANS', 'CHCL2F_TRANS', 'CCL4_TRANS', 'N2O5_TRANS'] )
    np.savetxt('tape7-01.txt', tape7,fmt='%.6e')

    tape7= loadtape7("data/tape7-02", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS', 'GRND_RFLT', 'TOTAL_RAD', 'DEPTH', 'DIR_EM', 'BBODY_T[K]'] )
    np.savetxt('tape7-02.txt', tape7,fmt='%.6e')

    tape7= loadtape7("data/tape7-03", ['FREQ', 'TOT_TRANS', 'PTH_THRML', 'THRML_SCT', 'SURF_EMIS', 'SOL_SCAT', 'SING_SCAT', 'GRND_RFLT', 'DRCT_RFLT', 'TOTAL_RAD', 'REF_SOL', 'SOL@OBS', 'DEPTH', 'DIR_EM', 'TOA_SUN', 'BBODY_T[K]'] )
    np.savetxt('tape7-03.txt', tape7,fmt='%.6e')

    tape7= loadtape7("data/tape7-04", ['FREQ', 'TRANS', 'SOL_TR', 'SOLAR', 'DEPTH'] )
    np.savetxt('tape7-04.txt', tape7,fmt='%.6e')

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
      totinband = numpy.trapz(Lpath.reshape(-1, 1),skyrad[:,0], axis=0)[0]
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
      totinband = - numpy.trapz(Lpath.reshape(-1, 1),wl,axis=0)[0]
      print('{0} integral is {1} [W/(m^2.sr)]'.format(colSelect[i][:],totinband))

    sr.saveFig('NIRPathradiancewl.png')
    print('Note that multiple scatter contributes significantly to total path radiance')


    print('Done!')
