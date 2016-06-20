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
"""Provides a simple, order of magnitude estimate of the photon flux and 
electron count in a detector for various sources and scene lighting.  
All models are based on published information or derived herein, so you 
can check their relevancy and suitability for your work.  

For a detailed theoretical derivation and more examples of use see:
http://nbviewer.jupyter.org/github/NelisW/ComputationalRadiometry/blob/master/07-Optical-Sources.ipynb

See the __main__ function for examples of use.

This package was partly developed to provide additional material in support of students
and readers of the book Electro-Optical System Analysis and Design: A Radiometry
Perspective,  Cornelius J. Willers, ISBN 9780819495693, SPIE Monograph Volume
PM236, SPIE Press, 2013.  http://spie.org/x648.html?product_id=2021423&origin_id=x646
"""

__version__ = "$Revision$"
__author__ = 'pyradi team'
__all__ = ['PFlux']

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
from numbers import Number

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
class PFlux:
    """ 
    See here: 
    https://github.com/NelisW/ComputationalRadiometry/blob/master/07-Optical-Sources.ipynb
    for mathemetical derivations and more detail.


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


        # low light level lux, colour temperature and fraction photopic
        # the fraction predicts the ratio between photopic and scotopic
        # this is used later to weigh spectrally
        # source: RCA/Burle electro-optics handbook
        self.lllux = {'Sun light': [107527,5700,1.0], 
              'Full sky light': [10752,12000,1.0],
              'Overcast day':[1075,6000,1.0],
              'Very dark day':[107,7000,1.0],
              'Twilight': [10.8,10000,1.0],
              'Deep twilight': [1.08,10000,0.8],
              'Full moon': [0.108,4150,0.6],
              'Quarter moon':[0.0108,4150,0.4],
              'Star light': [0.0011,5000,0.2],
              'Overcast night':[0.0001,5000, 0.],
            }
        self.llluxCols = ['Irradiance-lm/m2','ColourTemp','FracPhotop']






    ############################################################
    ##
    def lllPhotonrates(self, specranges=None ):
        """Calculate the approximate photon rate radiance for low light conditions

            The colour temperature of various sources are used to predict the 
            photon flux.  The calculation uses the colour temperature of the source and
            the ratio of real low light luminance to the luminance of a Planck radiator 
            at the same temperature as the source colour temperature.

            This procedure critically depends on the sources' spectral radiance in the
            various different spectral bands.  For this calculation the approach is taken 
            that for natural scenes the spectral shape can be modelled by a Planck curve 
            at the appropriate colour temperature.

            The steps followed are as follows:

            1.  Calculate the photon rate for the scene at the appropriate colour temperature,
                spectrally weighted by the eye's luminous efficiency response.  Do this for
                photopic and scotopic vision.

            2.  Weigh the photopic and scotopic photon rates according to illumination level

            3.  Determine the ratio k of low light level scene illumination to photon irradiance.
                This factor k is calculated in the visual band, but then applied to scale the
                other spectral bands by the same scale.

            4.  Use Planck radiation at the appropriate colour temperature to calculate the radiance
                in any spectral band, but then scale the value with the factor k.

            The specranges format is as follows::

                numpts = 300
                specranges = {
                    key: [wavelength vector, response vector ], 
                    'VIS': [np.linspace(0.43,0.69,numpts).reshape(-1,1),np.ones((numpts,1)) ], 
                    'NIR': [np.linspace(0.7, 0.9,numpts).reshape(-1,1),np.ones((numpts,1)) ], 
                    'SWIR': [np.linspace(1.0, 1.7,numpts).reshape(-1,1),np.ones((numpts,1)) ], 
                    'MWIR': [np.linspace(3.6,4.9,numpts).reshape(-1,1),np.ones((numpts,1)) ], 
                    'LWIR': [np.linspace(7.5,10,numpts).reshape(-1,1),np.ones((numpts,1)) ], 
                    }

            If specranges is None, the predefined values are used, as shown above.

            The function returns scene radiance in a Pandas datatable with the 
            following columns::

                u'Irradiance-lm/m2', u'ColourTemp', u'FracPhotop', u'k',
                u'Radiance-q/(s.m2.sr)-NIR', u'Radiance-q/(s.m2.sr)-VIS',
                u'Radiance-q/(s.m2.sr)-MWIR', u'Radiance-q/(s.m2.sr)-LWIR',
                u'Radiance-q/(s.m2.sr)-SWIR'

            and rows with the following index::

                u'Overcast night', u'Star light', u'Quarter moon', u'Full moon',
                u'Deep twilight', u'Twilight', u'Very dark day', u'Overcast day',
                u'Full sky light', u'Sun light'

            Args:
                | specranges (dictionary): User-supplied dictionary defining the spectral
                |   responses. See the dictionary format above and an example in the code.

            Returns:
                | Pandas dataframe with radiance in the specified spectral bands.

            Raises:
                | No exception is raised.
        """

        self.dfPhotRates = pd.DataFrame(self.lllux).transpose()
        self.dfPhotRates.columns = ['Irradiance-lm/m2','ColourTemp','FracPhotop']
        self.dfPhotRates.sort_values(by='Irradiance-lm/m2',inplace=True)

        wl = np.linspace(0.3, 0.8, 100)
        photLumEff,wl = ryutils.luminousEfficiency(vlamtype='photopic', wavelen=wl)
        scotLumEff,wl = ryutils.luminousEfficiency(vlamtype='scotopic', wavelen=wl)


        self.dfPhotRates['k'] = (self.dfPhotRates['Irradiance-lm/m2']) / (\
                        self.dfPhotRates['FracPhotop'] * 683 * np.trapz(photLumEff.reshape(-1,1) * \
                                    ryplanck.planckel(wl, self.dfPhotRates['ColourTemp']),wl, axis=0)\
                        + \
                        (1-self.dfPhotRates['FracPhotop']) * 1700 * np.trapz(scotLumEff.reshape(-1,1) * \
                                    ryplanck.planckel(wl, self.dfPhotRates['ColourTemp']),wl, axis=0))                           \

        if specranges is None:
            numpts = 300
            specranges = {
            'VIS': [np.linspace(0.43,0.69,numpts).reshape(-1,1),np.ones((numpts,1)) ], 
            'NIR': [np.linspace(0.7, 0.9,numpts).reshape(-1,1),np.ones((numpts,1)) ], 
            'SWIR': [np.linspace(1.0, 1.7,numpts).reshape(-1,1),np.ones((numpts,1)) ], 
            'MWIR': [np.linspace(3.6,4.9,numpts).reshape(-1,1),np.ones((numpts,1)) ], 
            'LWIR': [np.linspace(7.5,10,numpts).reshape(-1,1),np.ones((numpts,1)) ], 
            }

        for specrange in specranges.keys():
            wlsr = specranges[specrange][0]
            self.dfPhotRates['Radiance-q/(s.m2.sr)-{}'.format(specrange)] = (self.dfPhotRates['k'] /np.pi ) * \
                np.trapz(specranges[specrange][1] * ryplanck.planck(wlsr, self.dfPhotRates['ColourTemp'],'ql'),wlsr, axis=0)

        self.dfPhotRates.sort_values(by='Irradiance-lm/m2',inplace=True)

        return self.dfPhotRates

################################################################
################################################################
##

if __name__ == '__main__':
    import pyradi.rymodtran as rymodtran

    doAll = True

    if doAll:  
        pf = PFlux()
        print(pf.lllux)

        dfPhotRates = pd.DataFrame(pf.lllux).transpose()
        dfPhotRates.columns = ['Irradiance-lm/m2','ColourTemp','FracPhotop']
        dfPhotRates.sort_values(by='Irradiance-lm/m2',inplace=True)
        print(dfPhotRates)
        print(pf.lllPhotonrates())


    print('\n---------------------\n')
    print('module rypflux done!')


        # wl = np.linspace(0.43,0.69, 300)
        # # photLumEff,_ = ryutils.luminousEfficiency(vlamtype='photopic', wavelen=wl, eqnapprox=True)
        # print('\nRadiance of sunlit surface: {} q/(s.m2.sr)'.format(pf.nElecCntReflSun(wl, tauSun=0.6)))

