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
class Spectral(object):
    """Generic spectral can be used for any spectral vector
    """
    ############################################################
    ##
    def __init__(self, ID, wl=None, wn=None, value=None, emis=None, tran=None, refl=None, atco=None, prad=None, desc=None):
        """Defines a spectral variable of property vs wavelength iof wavenumber

        One of wavelength or wavenunber must be supplied, the other is calculated.
        No assumption is made of the sampling interval on eiher wn or wl.

        A second  value vector could optionally be supplied. This is used for atmo
        transmittance and path radiance.

        The constructor defines the 

            Args:
                | ID (str): identification string
                | wl (np.array (N,) or (N,1)): vector of wavelength values
                | wn (np.array (N,) or (N,1)): vector of wavenumber values
                | value (np.array (N,) or (N,1)): vector of property values
                | tran (np.array (N,) or (N,1)): transmittance vector 
                | emis (np.array (N,) or (N,1)): emissivity vector 
                | refl (np.array (N,) or (N,1)): reflectance vector 
                | atco (np.array (N,) or (N,1)): attenuation coeff vector  m-1
                | prad (np.array (N,) or (N,1)): normalised path radiance vector 
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        self.ID = ID
        self.desc = desc
        if value is not None:
            value = value.reshape(-1,1)
        self.value = value
        if emis is not None:
            emis = emis.reshape(-1,1)
        self.emis = emis
        if tran is not None:
            tran = tran.reshape(-1,1)
        self.tran = tran
        if refl is not None:
            refl = refl.reshape(-1,1)
        self.refl = refl
        if atco is not None:
            atco = atco.reshape(-1,1)
        self.atco = atco
        if prad is not None:
            prad = prad.reshape(-1,1)
        self.prad = prad

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
        numpts = self.wn.shape[0]
        stride = numpts / 3
        strn = 'ID: {}\n'.format(self.ID)
        strn += 'desc: {}\n'.format(self.desc)

        # for all numpy arrays, provide subset of values
        for var in self.__dict__:
            # then see if it is an array
            if isinstance(eval('self.{}'.format(var)), np.ndarray):
                svar = (np.vstack((eval('self.{}'.format(var))[0::stride], eval('self.{}'.format(var))[-1] ))).T
                strn += '{} (subsampled.T): {}\n'.format(var, svar)


        return strn

    ############################################################
    ##
    def plot(self, vars, filename, heading, ytitle=''):
        """Do a simple plot of spectral variable(s)

            Args:
                | vars ([str]): list of variables to type (in string form)
                | filename (str): filename for png graphic
                | heading (str): graph heading
                | ytitle (str): graph y-axis title

            Returns:
                | Nothing, writes png file to disk

            Raises:
                | No exception is raised.
        """
        import pyradi.ryplot as ryplot
        p = ryplot.Plotter(1,2,1,figsize=(8,5))

        for var in vars:
            # first see if it is in the current class
            if var in self.__dict__:
                # then see if it is an array
                if isinstance(eval('self.{}'.format(var)), np.ndarray):
                    p.plot(1,self.wl,eval('self.{}'.format(var)),heading,'Wavelength $\mu$m',
                        ytitle,label=[var])
                    p.plot(2,self.wn,eval('self.{}'.format(var)),heading,'Wavenumber cm$^{-1}$',
                        ytitle,label=[var])

        p.saveFig(filename)




##############################################################################################
##############################################################################################
##############################################################################################
class Atmo(Spectral):
    """Atmospheric spectral such as transittance or attenuation coefficient
    """
    ############################################################
    ##
    def __init__(self, ID, distance=None, wl=None, wn=None,  tran=None, atco=None, prad=None, desc=None):
        """Defines a spectral variable of property vs wavelength or wavenumber

        One of wavelength or wavenunber must be supplied, the other is calculated.
        No assumption is made of the sampling interval on either wn or wl.

        If distance is not None, tran=transmittance, prad=path radiance, at distance,
        then the atco and normalised path radiance is calculated.

        If distance is None, atco (attenuation coefficients in m^{-1}), and
        normalised prad (Lpath/(1-tauPath)) are used as supplied


            Args:
                | ID (str): identification string
                | distance (scalar): distance in m if transmittance, or None if att coeff
                | wl (np.array (N,) or (N,1)): vector of wavelength values 
                | wn (np.array (N,) or (N,1)): vector of wavenumber values
                | tran (np.array (N,) or (N,1)): transmittance
                | atco (np.array (N,) or (N,1)): attenuation coeff in m-1
                | prad (np.array (N,) or (N,1)): path radiance over distance in W/(sr.m2.cm-1)
                | desc (str): description string

            Returns:
                | None

            Raises:
                | No exception is raised.
        """

        __all__ = ['__init__', ]

        if distance is not None:
            if atco is None:
                atco = -np.log(tran)/distance
            if prad is not None:
                prad = prad / (1 - np.exp(- atco * distance))

        atco = atco.reshape(-1,1)
        prad = prad.reshape(-1,1)

        Spectral.__init__(self, ID=ID, atco=atco, prad=prad, wl=wl, wn=wn, desc=desc)
           
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

        strn = Spectral.__str__(self)

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
        return np.exp(-distance * self.atco)

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
        tran = np.exp(-distance * self.atco)
        return self.prad * (1 - tran)


##############################################################################################
##############################################################################################
##############################################################################################
class Sensor():
    """Sensor characteristics
    """
    ############################################################
    ##
    def __init__(self, ID, fno, detarea, inttime, wl, tauOpt=1, quantEff=1, 
                pfrac=1, desc=''):
        """Sensor characteristics

            Args:
                | ID (str): identification string
                | fno (scalar): optics fnumber
                | detarea (scalar): detector area
                | inttime (scalar): detector integration time
                | tauOpt (scalar or Spectral): sensor optics transmittance 
                | tauOpt (scalar or Spectral): sensor optics transmittance 
                | quantEff (scalar or np.array): detector quantum efficiency 
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
        self.pfrac = pfrac
        self.desc = desc

        if isinstance(tauOpt, Spectral) or isinstance(tauOpt, Number):
            self.tauOptVal = tauOpt
        else:
            print('\n\n{} tauOpt must be of type Spectral or scalar number\n\n'.format(self.ID))
            self.quantEffVal = None

        if isinstance(quantEff, Spectral) or isinstance(quantEff, Number):
            self.quantEffVal = quantEff
        else:
            print('\n\n{} quantEff must be of type Spectral scalar number\n\n'.format(self.ID))
            self.quantEffVal = None


           
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
        strn += 'tauOpt {}\n'.format(self.tauOptVal)
        strn += 'quantEff {}\n'.format(self.quantEffVal)


        return strn


    ############################################################
    ##
    def tauOpt(self):
        """Returns scaler or np.array for optics transmittance

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.tauOptVal, Spectral):
             rtnVal = self.tauOptVal.value
           
        if isinstance(self.tauOptVal, Number):
            rtnVal = self.tauOptVal 

        return rtnVal

    ############################################################
    ##
    def QE(self):
        """Returns scaler or np.array for detector quantEff

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.quantEffVal, Spectral):
             rtnVal = self.quantEffVal.value
           
        if isinstance(self.quantEffVal, Number):
            rtnVal = self.quantEffVal 

        return rtnVal

##############################################################################################
##############################################################################################
##############################################################################################
class Target(Spectral):
    """Target / Source characteristics
    """
    ############################################################
    ##
    def __init__(self, ID, emis, tmprt, refl=1, cosTarg=1, taumed=1, scale=1, desc=''):
        """Source characteristics

            Target transmittance = 1 - emis - refl.



            Args:
                | ID (str): identification string
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
        self.tmprt = tmprt
        self.cosTarg = cosTarg
        self.scale = scale
        self.desc = desc

        if isinstance(emis, Spectral) or isinstance(emis, Number):
            self.emisVal = emis
        else:
            print('\n\n{} emis must be of type Spectral or scalar number\n\n'.format(self.ID))
            self.emisVal = None

        if isinstance(refl, Spectral) or isinstance(refl, Number):
            self.reflVal = refl
        else:
            print('\n\n{} refl must be of type Spectral scalar number\n\n'.format(self.ID))
            self.reflVal = None

        if isinstance(taumed, Spectral) or isinstance(taumed, Number):
            self.taumedVal = taumed
        else:
            print('\n\n{} taumed must be of type Spectral scalar number\n\n'.format(self.ID))
            self.taumedVal = None
          
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

        strn += 'emis: {}\n'.format(self.emis())
        strn += 'refl: {}\n'.format(self.refl())
        strn += 'taumed: {}\n'.format(self.taumed())

        return strn

    ############################################################
    ##
    def emis(self):
        """Returns scaler or np.array for optics transmittance

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.emisVal, Spectral):
             rtnVal = self.emisVal.value
           
        if isinstance(self.emisVal, Number):
            rtnVal = self.emisVal 

        return rtnVal


    ############################################################
    ##
    def refl(self):
        """Returns scaler or np.array for optics transmittance

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.reflVal, Spectral):
             rtnVal = self.reflVal.value
           
        if isinstance(self.reflVal, Number):
            rtnVal = self.reflVal 

        return rtnVal


    ############################################################
    ##
    def taumed(self):
        """Returns scaler or np.array for optics transmittance

            Args:
                | None

            Returns:
                | str

            Raises:
                | No exception is raised.
        """
        if isinstance(self.taumedVal, Spectral):
             rtnVal = self.taumedVal.value
           
        if isinstance(self.taumedVal, Number):
            rtnVal = self.taumedVal 

        return rtnVal


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
    import pyradi.rymodtran as rymodtran

    doAll = False

    if True:  
        pf = PFlux()
        print(pf.lllux)

        dfPhotRates = pd.DataFrame(pf.lllux).transpose()
        dfPhotRates.columns = ['Irradiance-lm/m2','ColourTemp','FracPhotop']
        dfPhotRates.sort_values(by='Irradiance-lm/m2',inplace=True)
        print(dfPhotRates)
        print(pf.lllPhotonrates())

        # test loading of spectrals
        print('\n---------------------Spectrals:')
        spectral = np.loadtxt('data/MWIRsensor.txt')
        pf.spectrals['ID1'] = Spectral('ID1',value=spectral[:,1],wl=spectral[:,0],desc="MWIR transmittance")
        pf.spectrals['ID2'] = Spectral('ID2',value=1-spectral[:,1],wl=spectral[:,0],desc="MWIR absorption")
        for key in pf.spectrals:
            print(pf.spectrals[key])
        for key in pf.spectrals:
            filename ='{}-{}'.format(key,pf.spectrals[key].desc)
            pf.spectrals[key].plot(['value'],filename=filename,heading=pf.spectrals[key].desc,ytitle='Value')

        # test loading of atmospheric spectrals
        print('\n---------------------Atmos:')
        tape72 = rymodtran.loadtape7("data/tape7-02", ['FREQ', 'TOT_TRANS', 'PTH_THRML'] )
        pf.atmos['A1'] = Atmo(ID='tape7-02', tran=tape72[:,1], prad=1e4*tape72[:,2], distance=2000, wl=None, 
            wn=tape72[:,0], desc='tape7-02 raw data input')
        tape72 = rymodtran.loadtape7("data/tape7-02", ['FREQ', 'TOT_TRANS', 'PTH_THRML'] )
        pf.atmos['A2'] = Atmo(ID='tape7-02', atco=-np.log(tape72[:,1])/2000., prad=1e4*tape72[:,2] / (1 - tape72[:,1]),
            wl=1e4/tape72[:,0], wn=None, desc='tape7-02 normalised input')
        for key in pf.atmos:
            print(pf.atmos[key])
        for key in pf.atmos:
            pf.atmos[key].plot(['prad'],filename='{}-{}-{}'.format(key,pf.atmos[key].desc,'prad'),
                heading=pf.atmos[key].desc,ytitle='Norm Lpath W/(sr.m2.cm-1)')
            pf.atmos[key].plot(['atco'],filename='{}-{}-{}'.format(key,pf.atmos[key].desc,'atco'),
                heading=pf.atmos[key].desc,ytitle='Attenuation m$^{-1}$')

        numpts = pf.atmos['A1'].wn.shape[0]
        stride = numpts / 3

        for distance in [1000,2000]:
            print('distance={} m, tran={}'.format(distance,pf.atmos['A1'].tauR(distance)[0::stride].T))
            ID = 'Atau{:.0f}'.format(distance)
            pf.spectrals[ID] = Spectral(ID,tran=pf.atmos['A1'].tauR(distance),wl=pf.atmos['A1'].wl,desc="MWIR transmittance "+ID)
            pf.spectrals[ID].plot(['tran'],filename='{}-{}-tau'.format(key,pf.spectrals[ID].desc),
                heading=pf.spectrals[ID].desc,ytitle='Transmittance')
            ID = 'Aprad{:.0f}'.format(distance)
            pf.spectrals[ID] = Spectral(ID,prad=pf.atmos['A1'].pathR(distance),wl=pf.atmos['A1'].wl,desc="MWIR path radiance "+ID)
            pf.spectrals[ID].plot(['prad'],filename='{}-{}-prad'.format(key,pf.spectrals[ID].desc),
                heading=pf.spectrals[ID].desc,ytitle='Lpath W/(sr.m2.cm-1)')

        distances =np.array([1000,2000])
        print('distances={} m, tran={}'.format(distances,pf.atmos['A1'].tauR(distances)[0::stride].T))
        print('distances={} m, atco={}'.format(distances,(-np.log(pf.atmos['A1'].tauR(distances))/distances)[0::stride].T))


        # test loading of sensors
        print('\n---------------------Sensors:')
        pf.sensors['S1'] = Sensor(ID='S1', fno=3.2, detarea=(10e-6)**2, inttime=0.01, 
            wl=np.array([1,2,3]), tauOpt=.9, quantEff=.6, 
            pfrac=0.4, desc='Sensor one')
        spectral = np.loadtxt('data/MWIRsensor.txt')
        pf.spectrals['ID1S2'] = Spectral('ID1',value=spectral[:,1],wl=spectral[:,0],desc="MWIR transmittance")
        pf.sensors['S2'] = Sensor(ID='S2', fno=4, detarea=(15e-6)**2, inttime=0.02, 
            wl=np.array([1,2,3]), tauOpt=pf.spectrals['ID1S2'], quantEff=pf.spectrals['ID1S2'], 
            desc='Sensor two')
        for key in pf.sensors:
            print(pf.sensors[key])
        #     print(pf.sensors[key].tauOpt())
        #     print(pf.sensors[key].QE())


        # test loading of targets/sources
        print('\n---------------------Sources:')
        emisx = Spectral('ID1',value=spectral[:,1],wl=spectral[:,0],desc="MWIR emissivity")
        pf.targets['T1'] = Target(ID='T1', emis=emisx, tmprt=300, refl=0, cosTarg=0.5,
            taumed=.5, scale=1, desc='Source one')
        pf.targets['T2'] = Target(ID='T2', emis=0., tmprt=6000, refl=1,cosTarg=1,
            scale=2.17e-5,taumed=0.5,desc='Source two')
        for key in pf.targets:
            print(pf.targets[key])



    print('\n---------------------\n')
    print('module rypflux done!')


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