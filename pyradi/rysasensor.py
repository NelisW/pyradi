#  $Id$
#  $HeadURL$
################################################################
# The contents of this file are subject to the BSD 3Clause (New)cense
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

# Contributor(s): MS Willers

################################################################
"""

http://nbviewer.jupyter.org/github/NelisW/ComputationalRadiometry/blob/master/09b-StaringArrayDetectors.ipynb

See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students
and readers of the book Electro-Optical System Analysis and Design: A Radiometry
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""

__version__ = "$Revision$"
__author__ = 'pyradi team'
__all__ = ['']

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)

import numpy as np
import math
import sys
import itertools
import pandas as pd
import pyradi.ryutils as ryutils
import pyradi.ryplanck as ryplanck
import pyradi.rypflux as rypflux
import pyradi.rymodtran as rymodtran

from numbers import Number


        
################################################################
################################################################
##

if __name__ == '__main__':
 
    doAll = False

    if True:  



    print('\n---------------------\n')
    print('module rydecount done!')


        # wl = np.linspace(0.43,0.69, 300)
        # # photLumEff,_ = ryutils.luminousEfficiency(vlamtype='photopic', wavelen=wl, eqnapprox=True)
        # print('\nRadiance of sunlit surface: {} q/(s.m2.sr)'.format(pf.nElecCntReflSun(wl, tauSun=0.6)))



# # to calculate the electron count in the detector from reflected sunlight only
# pf = rypflux.PFlux()
# wl = np.linspace(0.43,0.69, 300)
# print('Radiance of sunlit surface: {:.2e} q/(s.m2.sr), {} to {} um'.format(
#         pf.nElecCntReflSun(wl, tauSun=0.6),np.min(wl),np.max(wl)))

# nsun = pf.nElecCntReflSun(wl, tauSun=0.6, tauAtmo=0.8,  tauOpt=0.9, quantEff=0.5, 
#         rhoTarg=.5, cosTarg=1, inttime=0.01, pfrac=1, detarea=(10e-6)**2, fno=3.2)
# print('{:.0f} electrons'.format(nsun))





