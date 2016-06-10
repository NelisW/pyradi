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


##############################################################################################
##############################################################################################
##############################################################################################
class Spectral(object):
    """Generic spectral can be used for any spectral vector
    """
    ############################################################
    ##
    def __init__(self, ID, val, val2=None, wl=None, wn=None, desc=None):
        """Defines a spectral variable of property vs wavelength iof wavenumber

        One of wavelength or wavenunber must be supplied, the other is calculated.
        No assumption is made of the sampling interval on eiher wn or wl.

        A second  value vector could optionally be supplied. This is used for atmo
        transmittance and path radiance.

        The constructor defines the 

            Args:
                | ID (str): identification string
                | val (np.array (N,) or (N,1)): vector of property values
                | val2 (np.array (N,) or (N,1)): second vector of property values
                | wl (np.array (N,) or (N,1)): vector of wavelength values
                | wn (np.array (N,) or (N,1)): vector of wavenumber values
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        self.ID = ID
        self.desc = desc
        self.val = val.reshape(-1,1)
        if val2 is not None:
            val2 = val2.reshape(-1,1)
        self.val2 = val2

        if wn is not None:
            self.wn =  wn.reshape(-1,1)
            self.wl = 1e4 /  self.wn
        elif wl is not None:
            self.wl =  wl.reshape(-1,1)
            self.wn = 1e4 /  self.wl
        else:
            print('Spectral {} has both wn and wl as None'.format(ID))

    ############################################################
    ##
    def __str__(self):
        """Returns string representation of the object

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        strn = 'Spectral ID: {}\n'.format(self.ID)
        strn += 'desc: {}\n'.format(self.desc)
        strn += 'wl: {}\n'.format(self.wl)
        strn += 'wn: {}\n'.format(self.wn)
        strn += 'val: {}\n'.format(self.val)
        strn += 'val2: {}\n'.format(self.val2)

        return strn



##############################################################################################
##############################################################################################
##############################################################################################
class Atmo(Spectral):
    """Atmospheric spectral such as transittance or attenuation coefficient
    """
    ############################################################
    ##
    def __init__(self, ID, tau, rad=None, distance=None, wl=None, wn=None, desc=None):
        """Defines a spectral variable of property vs wavelength or wavenumber

        One of wavelength or wavenunber must be supplied, the other is calculated.
        No assumption is made of the sampling interval on either wn or wl.

        If distance is not None, tau=transmittance, rad=path radiance, at distance

        If distance is None, tau must be the attenuation coefficients in m^{-1}, and
        rad must be None or Lpath/(1-tauPath)

        If distance is not None, val is the transmittance for given distance in m.
        In this case the attenuation coefficient is calculated and stored in val

            Args:
                | ID (str): identification string
                | val (np.array (N,) or (N,1)): transmittance or attenuation coeff
                | distance (scalar): distance in m if transmittance, or None if att coeff
                | wl (np.array (N,) or (N,1)): vector of wavelength values
                | wn (np.array (N,) or (N,1)): vector of wavenumber values
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        acoeff = tau

        if distance is not None:
            acoeff = -np.log(tau)/distance
            if rad is not None:
                rad = rad / (1-tau)

        Spectral.__init__(self, ID=ID, val=acoeff, val2=rad, wl=wl, wn=wn, desc=desc)
           
    ############################################################
    ##
    def __str__(self):
        """Returns string representation of the object

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        strn =  'Atmo ID: {}\n'.format(self.ID)
        strn += 'desc: {}\n'.format(self.desc)
        strn += 'wl: {}\n'.format(self.wl)
        strn += 'wn: {}\n'.format(self.wn)
        strn += 'attcoeff: {}\n'.format(self.val)
        strn += 'pathrad: {}\n'.format(self.val2)

        return strn

    ############################################################
    ##
    def tauR(self, distance):
        """Calculates the transmittance at distance 

        Distance is in m

            Args:
                | distance (scalar or np.array (M,)): distance in m if transmittance, or None if att coeff

            Returns:
                | transmittance (np.array (N,M) ): transmittance along N at distance along M

            Raises:
                | No exception is raised.
        """
        distance = np.array(distance).reshape(1,-1)
        return np.exp(-distance * self.val)

    ############################################################
    ##
    def pathR(self, distance):
        """Calculates the path radiance at distance

        Distance is in m

            Args:
                | distance (scalar or np.array (M,)): distance in m if transmittance, or None if att coeff

            Returns:
                | transmittance (np.array (N,M) ): transmittance along N at distance along M

            Raises:
                | No exception is raised.
        """
        distance = np.array(distance).reshape(1,-1)
        tau = np.exp(-distance * self.val)
        return self.val2 / (1-tau)


##############################################################################################
##############################################################################################
##############################################################################################
class Sensor(Spectral):
    """Sensor characteristics
    """
    ############################################################
    ##
    def __init__(self, ID, fno, detarea, inttime, wl, tauOpt=1, quantEff=1, pfrac=1,
                desc=''):
        """Sensor characteristics

            Args:
                | ID (str): identification string
                | fno (scalar): optics fnumber
                | detarea (scalar): detector area
                | inttime (scalar): detector integration time
                | wl (np.array (N,) or (N,1)): wavelength  in um
                | tauOpt (np.array (N,) or (N,1)): sensor optics transmittance 
                | quantEff (np.array (N,) or (N,1)): detector quantum efficiency 
                | pfrac (scalar):  fraction of optics clear aperture
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        self.ID = ID
        self.fno = fno
        self.detarea = detarea
        self.inttime = inttime
        self.wl = wl
        self.tauOpt = tauOpt
        self.quantEff = quantEff
        self.pfrac = pfrac
        self.desc = desc
           
    ############################################################
    ##
    def __str__(self):
        """Returns string representation of the object

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        strn =  'Sensor ID: {}\n'.format(self.ID)
        strn += 'desc: {}\n'.format(self.desc)
        strn += 'fno: {}\n'.format(self.fno)
        strn += 'detarea: {}\n'.format(self.detarea)
        strn += 'inttime: {}\n'.format(self.inttime)
        strn += 'pfrac: {}\n'.format(self.pfrac)
        strn += 'wl: {}\n'.format(self.wl)
        strn += 'tauOpt: {}\n'.format(self.tauOpt)
        strn += 'quantEff: {}\n'.format(self.quantEff)

        return strn

    ############################################################
    ##


##############################################################################################
##############################################################################################
##############################################################################################
class Target(Spectral):
    """Target / Source characteristics
    """
    ############################################################
    ##
    def __init__(self, ID, wl, emis, tmprt, refl=1, cosTarg=1, taumed=1, scale=1, desc=''):
        """Source characteristics

            Target transmittance = 1 - emis - refl.



            Args:
                | ID (str): identification string
                | wl (np.array (N,) or (N,1)): wavelength 
                | emis (np.array (N,) or (N,1)): surface emissivity
                | tmprt (scalar): surface temperature
                | refl (np.array (N,) or (N,1)): surface reflectance
                | cosTarg (scalar): cosine between surface normal and illumintor direction
                | taumed (np.array (N,) or (N,1)): transmittance between the surface and illumintor 
                | scale (scalar): surface radiance scale factor, sun: 2.17e-5, otherwise 1.
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        self.ID = ID
        self.wl = wl
        self.emis = emis
        self.tmprt = tmprt
        self.refl = refl
        self.cosTarg = cosTarg
        self.taumed = taumed
        self.scale = scale
        self.desc = desc

          
    ############################################################
    ##
    def __str__(self):
        """Returns string representation of the object

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        strn =  'Sensor ID: {}\n'.format(self.ID)
        strn += 'desc: {}\n'.format(self.desc)
        strn += 'tmprt: {}\n'.format(self.tmprt)
        strn += 'cosTarg: {}\n'.format(self.cosTarg)
        strn += 'scale: {}\n'.format(self.scale)
        strn += 'wl: {}\n'.format(self.wl)
        strn += 'emis: {}\n'.format(self.emis)
        strn += 'refl: {}\n'.format(self.refl)
        strn += 'taumed: {}\n'.format(self.taumed)

        return strn


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

        self.spectrals = {}
        self.atmos = {}
        self.sensors = {}
        self.targets = {}



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

    doAll = False

    if True:  
        pf = PFlux()
        print(pf.lllux)

        dfPhotRates = pd.DataFrame(pf.lllux).transpose()
        dfPhotRates.columns = ['Irradiance-lm/m2','ColourTemp','FracPhotop']
        dfPhotRates.sort_values(by='Irradiance-lm/m2',inplace=True)
        print(dfPhotRates)

        # test loading of spectrals
        pf.spectrals['ID1'] = Spectral('ID1',val=np.linspace(0,1,2),wl=np.linspace(.4,.7,2),desc="some description")
        pf.spectrals['ID2'] = Spectral('ID2',val=np.linspace(1,0,2),wl=np.linspace(7,12,2),desc="some description")
        print('\n---------------------\nSpectrals:')
        for key in pf.spectrals:
            print(pf.spectrals[key])

        # test loading of atmospheric spectrals
        pf.atmos['A1'] = Atmo(ID='A1', tau=np.array([.2,.5,1]), distance=1000, wl=np.array([1,2,3]), 
            wn=None, desc='My super atmo')
        pf.atmos['A2'] = Atmo(ID='A1', tau=np.array([.2,.5,1]), rad=np.array([.2,.5,1]), wl=np.array([1,2,3]), 
            wn=None, desc='My second atmo')
        print('\n---------------------\Atmos:')
        for key in pf.atmos:
            print(pf.atmos[key])

        for distance in [1000,2000]:
            print('distance={} m, tau={}'.format(distance,pf.atmos['A1'].tauR(distance)))

        distances =np.array([1000,2000])
        print('distances={} m, tau={}'.format(distances,pf.atmos['A1'].tauR(distances)))


        pf.sensors['S1'] = Sensor(ID='S1', fno=3.2, detarea=(10e-6)**2, inttime=0.01, 
            wl=np.array([1,2,3]), tauOpt=np.array([0.4, 0.9, 0.2]), quantEff=np.array([0.6, 0.7, 0.4]), 
            pfrac=0.4, desc='Sensor one')
        pf.sensors['S2'] = Sensor(ID='S2', fno=4, detarea=(15e-6)**2, inttime=0.02, 
            wl=np.array([1,2,3]), tauOpt=np.array([0.45, 0.95, 0.5]), quantEff=np.array([0.3, 0.4, 0.5]), 
            desc='Sensor two')
        print('\n---------------------\Sensors:')
        for key in pf.sensors:
            print(pf.sensors[key])


        pf.targets['T1'] = Target(ID='T1', wl=np.array([1,2,3]), emis=1, tmprt=300, refl=0, cosTarg=0.5,
            taumed=np.array([0.45, 0.95, 0.5]), scale=1, desc='Source one')
        pf.targets['T2'] = Target(ID='T2', wl=np.array([1,2,3]), emis=0., tmprt=6000, refl=1,cosTarg=1,
            scale=2.17e-5,taumed=0.5,desc='Source two')
        print('\n---------------------\Sources:')
        for key in pf.targets:
            print(pf.targets[key])

        # print(pf.lllPhotonrates())

        # wl = np.linspace(0.43,0.69, 300)
        # # photLumEff,_ = ryutils.luminousEfficiency(vlamtype='photopic', wavelen=wl, eqnapprox=True)
        # print('\nRadiance of sunlit surface: {} q/(s.m2.sr)'.format(pf.nElecCntReflSun(wl, tauSun=0.6)))


       

    print('\n---------------------\n')
    print('module rypflux done!')





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