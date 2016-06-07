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


For more examples of use see:
https://github.com/NelisW/ComputationalRadiometry

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

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.font_manager import FontProperties
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.dates as mdates
# from mpl_toolkits.axes_grid1 import make_axes_locatable


class PFlux:
    """ 


    See here: 
    https://github.com/NelisW/ComputationalRadiometry/blob/master/07-Optical-Sources.ipynb
    for more detail.


    """

    ############################################################
    ##
    def __init__(self):
        """Class constructor

        The constructor defines the 

            Args:
                | fignumber (int): the plt figure number, must be supplied

            Returns:
                | Nothing. Creates the figure for subequent use.

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', 'lllPhotonrates']

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




    ############################################################
    ##
    def lllPhotonrates(self, specranges=None ):
        """Calculate the photon rate radiance in spectral bands for low light conditions

            Args:
                | var1 ([strings]): User-supplied list
                |    of plotting styles(can be empty []).
                | var2 (int): Length of required sequence.

            Returns:
                | Whatever.

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
            specranges = {
            'VIS': [np.linspace(0.43,0.69,300).reshape(-1,1) ], 
            'NIR': [np.linspace(0.7, 0.9,300).reshape(-1,1) ], 
            'SWIR': [np.linspace(1.0, 1.7,300).reshape(-1,1) ], 
            'MWIR': [np.linspace(3.6,4.9,300).reshape(-1,1) ], 
            'LWIR': [np.linspace(7.5,10,300).reshape(-1,1) ], 
            }

        for specrange in specranges.keys():
            wlsr = specranges[specrange][0]
            self.dfPhotRates['Radiance-q/(s.m2.sr)-{}'.format(specrange)] = (self.dfPhotRates['k'] /np.pi ) * \
                        np.trapz(ryplanck.planck(wlsr, self.dfPhotRates['ColourTemp'],'ql'),wlsr, axis=0)

        self.dfPhotRates.sort_values(by='Irradiance-lm/m2',inplace=True)

        return self.dfPhotRates

################################################################
################################################################
##

if __name__ == '__main__':

    doAll = False

    if True:  
        pf = PFlux()
        print(pf.lllux)

        # dfPhotRates = pd.DataFrame(pf.lllux).transpose()
        # dfPhotRates.columns = ['Irradiance-lm/m2','ColourTemp','FracPhotop']
        # dfPhotRates.sort_values(by='Irradiance-lm/m2',inplace=True)
        # print(dfPhotRates)

        print(pf.lllPhotonrates())
    print('module rypflux done!')
