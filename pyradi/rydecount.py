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

##############################################################################################
##############################################################################################
##############################################################################################
class EOSystem:
    """ 


    """

    ############################################################
    ##
    def __init__(self):
        """Class constructor

        The constructor defines the 

            Args:
                | fignumber (int): the plt figure number, must be supplied

            Returns:
                | Nothing. Sets up the class for use

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', 'lllPhotonrates','nElecCntReflSun']



    ############################################################
    ##
    def lllPhotonrates(self, specranges=None ):
        """

            Args:
                | specranges (dictionary): User-supplied dictionary defining the spectral
                |   responses. See the dictionary format above and an example in the code.

            Returns:
                | Pandas dataframe with radiance in the specified spectral bands.

            Raises:
                | No exception is raised.
        """


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



# # to calculate the scene electron count from the low light table
# def nEcntLLight(tauAtmo, tauFilt, tauOpt, quantEff, rhoTarg, cosTarg, 
#              inttime, pfrac, detarea, fno, scenario, specBand, dfPhotRates):
#     """ Calculate the number of electrons in a detector
#     All values in base SI units
#     rhoTarg is the target diffuse reflectance
#     cosTarg the cosine of the sun incidence angle
#     scenario must be one of the dfPhotRates index values
#     specBand must be one of the dfPhotRates column values
#     """
    
#     L =  quantEff * tauOpt * tauFilt *  tauAtmo * \
#             rhoTarg * dfPhotRates.ix[scenario][specBand]
#     n = np.pi * inttime * pfrac * detarea * L * cosTarg / (4 * fno**2)
#     return n
    
    
# # to calculate the electron count in the detector from a thermal source only
# def nElecCntThermalScene(wl, tmptr, emis, tauAtmo, tauFilt, tauOpt, quantEff, inttime, pfrac, detarea, fno):
#     """ Calculate the number of electrons in a detector
#     All values in base SI units
#     """
    
#     L = emis * tauAtmo * tauFilt * tauOpt * quantEff * \
#             ryplanck.planck(wl, tmptr, type='ql')/np.pi
#     L = np.trapz( L, x=wl,axis=0)
#     n = np.pi * inttime * pfrac * detarea * L / (4 * fno**2)
#     return n
    

# # to calculate the electron count in the detector from a thermal source only
# def nEcntThermalOptics(wl, tmptrOpt, tauFilt, tauOpt, quantEff, inttime, pfrac, detarea, fno):
#     """ Calculate the number of electrons in a detector
#     All values in base SI units
#     """
    
#     L = tauFilt * (1.0 - tauOpt) * quantEff * \
#             ryplanck.planck(wl, tmptrOpt, type='ql')/np.pi
#     L = np.trapz( L, x=wl,axis=0)
#     n = np.pi * inttime * pfrac * detarea * L / (4 * fno**2)
#     return n    



    # ############################################################
    # ##
    # def nElecCntReflSun(self, wl, tauSun, tauAtmo=1, tauFilt=1, tauOpt=1, quantEff=1, 
    #     rhoTarg=1, cosTarg=1, inttime=1, pfrac=1, detarea=1, fno=0.8862269255, emissun=1.0, tmprt=6000.):
    #     """ Calculate the number of electrons in a detector or photon radiance for reflected sunlight

    #         All values in base SI units.

    #         By using the default values when calling the function the radiance at the 
    #         source can be calculated.

    #         Args:
    #             | wl (np.array (N,) or (N,1)): wavelength 
    #             | tauAtmo (np.array (N,) or (N,1)): transmittance between the scene and sensor 
    #             | tauSun (np.array (N,) or (N,1)): transmittance between the scene and sun 
    #             | tauFilt (np.array (N,) or (N,1)): sensor filter transmittance 
    #             | tauOpt (np.array (N,) or (N,1)): sensor optics transmittance 
    #             | quantEff (np.array (N,) or (N,1)): detector quantum efficiency 
    #             | rhoTarg (np.array (N,) or (N,1)): target diffuse surface reflectance 
    #             | cosTarg (scalar): cosine between surface normal and sun/moon direction
    #             | inttime (scalar): detector integration time
    #             | pfrac (scalar):  fraction of optics clear aperture
    #             | detarea (scalar): detector area
    #             | fno (scalar): optics fnumber
    #             | emissun (scalar): sun surface emissivity
    #             | tmprt (scalar): sun surface temperature

    #         Returns:
    #             | n (scalar): number of electrons accumulated during integration time

    #         Raises:
    #             | No exception is raised.
    #     """
        
    #     L =  emissun * tauAtmo * tauFilt * tauOpt * tauSun * quantEff * \
    #             rhoTarg * ryplanck.planck(wl, tmprt, type='ql')/np.pi
    #     L = np.trapz( L, x=wl,axis=0)
    #     n = np.pi * inttime * pfrac * detarea * L * 2.17e-5 * cosTarg / (4 * fno**2)

    #     return n