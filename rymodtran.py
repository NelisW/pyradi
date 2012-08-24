#  $Id: rychroma.py 43 2012-08-07 20:14:44Z neliswillers@gmail.com $
#  $HeadURL: https://pyradi.googlecode.com/svn/trunk/rychroma.py $

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

# Contributor(s): ______________________________________.
################################################################
"""
This module provides rudimentary MODTRAN file reading.

See the __main__ function for examples of use.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals

__version__= "$Revision: 43 $"
__author__= 'pyradi team'
__all__= ['chromaticityforSpectralL']

import numpy as np
from string import maketrans

##############################################################################
##
def fixHeaders(str):
    intab = "+-"
    outtab = "pm"
    trantab = maketrans(intab, outtab)
    str=str.translate(trantab)
    return str


def loadtape7SCNFile(filename, colspec = [], delimiter=None ):

    """
    This funciton reads in the tape7 file from MODerate spectral resolution atmospheric 
    TRANsmission (MODTRAN) code, that is used to model the propagation of the 
    electromagnetic radiation through the atmosphere. tape7 is a primary file that contains all the spectral  
    results of the MODTRAN run. The header information in the tape7 file contains portions of the tape5 
    information that will be deleted. The header section in tape7 is followed by a list of spectral points 
    with corresponding transmissions. Each column has a different component of the transmission and 
    the total transmittance is always found in the second column.

            | filename (string): name of the input ASCII flatfile. 
            | colspec output the spectral transmittance data
    """

    infile = open(filename, 'r')
    idata = {}
    data = infile.readlines()#.strip()
    infile.close()
    s = ""

    #skip the first 10 row that contains tape5 information and leave the header for the different components
    #of transimissions
    #remove the last row in the file
    for counter in range(10,len(data)-1):
        if counter==10:
            data[counter]=fixHeaders(data[counter]) 
            
        columns = data[counter].strip("\n").split()

        for colcount in range(0, len(columns)):
            s = s + columns[colcount] + "  "  
        s = s + "\n"

    #output a newfile containing each column with different component of the transmission. 
    allData = open('outputfile.txt', 'w')
    allData.write(s)
    allData.close()
    #read in the cleaned text file without the header and last line found in tape7 
    
    data = np.ndfromtxt('outputfile.txt', delimiter=' ', dtype=None,  names=True)
    data = np.delete(data, (0), axis=0)
    
    coldata= data[colspec[0]].reshape(-1, 1)
    for colname in colspec[1:]:
        coldata = np.hstack((coldata, data[fixHeaders(colname)].reshape(-1, 1)))
       
    return coldata
 
def nparrayToString(coldata): 
    s=np.str(coldata)
    s=s.replace("'", "")
    s=s.replace("[[", " ")
    s=s.replace("[", "")
    s=s.replace("]", "")
    return s

################################################################
##

if __name__ == '__init__':
    pass
    
if __name__ == '__main__':
        
    import math
    import sys

    #import pyradi.ryplot as ryplot

    figtype = ".png"  # eps, jpg, png
    #figtype = ".eps"  # eps, jpg, png

    ## ----------------------- wavelength------------------------------------------
    #create the wavelength scale to be used in all spectral calculations, 
    # wavelength is reshaped to a 2-D  (N,1) column vector
    wavelength=np.linspace(0.38, 0.72, 350).reshape(-1, 1)

    tape7= loadtape7SCNFile("ModTrantape7file", ['FREQ', 'COMBIN', 'MOLEC', 'AER+CLD', 'AER-CLD'] )
    print(tape7)
    print(tape7.shape)
    
    s=nparrayToString(tape7)
    file=open('SpectralTransmittance.txt', 'w')
    file.write(s)
    file.close()
